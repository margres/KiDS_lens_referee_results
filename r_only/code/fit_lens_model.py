"""
Full lens+mass+source model fit for a single KiDS cutout, via Nautilus nested
sampling. One call = one array task in the SLURM templates under slurm/.

Model: lens light (Sersic core) + lens mass (Isothermal + external shear) +
source light (Sersic core), fit with af.Nautilus. Source effective_radius is
capped at 0.4" -- an uncapped prior let the sampler stretch the source into
an unphysical elongated streak that overfits extended residual structure
instead of reconstructing a compact lensed arc (seen on 2/3 local validation
objects at effective_radius upper_limit=1.0", n_live=100; chi2/pix 1.5-1.7
with residuals showing the model over-predicting flux along a fake diagonal
feature not present in the data). n_live=150 here vs. 100 in the original
local test.

Usage:
    python3 fit_lens_model.py --dataset KiDSDR4_J010127.840-334319.40 \
        --cutout_dir dataset/lens_cutouts_fits --number_of_cores 1

Writes results/<dataset>.json (summary: einstein_radius, snr_arc_model,
snr_arc_peak, chi2_per_pix, log_likelihood, time_s) plus the full PyAutoFit
output tree under output/<path_prefix>/<dataset>/, which includes
pre-rendered FITS (data, model_image, residual_map,
normalized_residual_map, source_subtracted_image, signal_to_noise_map, ...)
under output/.../image/fit_dataset/fits/ -- no need to reload the model to
inspect fit quality, just read those FITS files directly.
"""
import argparse
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel

warnings.filterwarnings('ignore')

import autofit as af
import autolens as al

PSF_FWHM = 0.7
# Fixed [0.5, 3.5]" annulus, superseded 2026-08-02: was independent of each
# object's fitted einstein_radius (0.3-4.0" across the sample), so for small
# lenses it swept in unrelated field flux and for large ones the real arc sat
# past the outer edge entirely. See ANNULUS_INNER_FRAC/OUTER_FRAC below.
ANNULUS_INNER_FRAC, ANNULUS_OUTER_FRAC, ANNULUS_MIN_WIDTH = 0.6, 1.4, 0.5
# Point-source masking: a real (but unmodelled) compact source -- a
# foreground star, AGN core, or interloper -- sitting inside the fit mask
# contaminates both the mass-model fit and the arc-S/N annulus. Detected via
# photutils DAOStarFinder (FWHM matched to the known PSF) and excluded with a
# small circular hole in the fit mask. A central exclusion radius protects
# the lens galaxy (and any compact arc knot near it) from being flagged.
POINT_SOURCE_CENTER_EXCLUSION = 1.0  # arcsec; don't flag anything this close to centre
POINT_SOURCE_MASK_RADIUS = 1.5  # x PSF FWHM, radius of the masked hole per detection


def detect_point_sources(data_sub, pixel_scale, noise_sigma, center_exclusion=POINT_SOURCE_CENTER_EXCLUSION):
    """Return [(x_pix, y_pix), ...] centroids of compact point-like sources
    in data_sub, excluding anything within center_exclusion arcsec of the
    cutout centre (the lens galaxy)."""
    from photutils.detection import DAOStarFinder
    psf_fwhm_pix = PSF_FWHM / pixel_scale
    finder = DAOStarFinder(fwhm=psf_fwhm_pix, threshold=8 * noise_sigma,
                            sharplo=0.3, sharphi=1.0, roundlo=-0.7, roundhi=0.7)
    sources = finder(data_sub)
    if sources is None:
        return []
    ny, nx = data_sub.shape
    cy, cx = ny / 2, nx / 2
    out = []
    for s in sources:
        r_arcsec = np.hypot(s['xcentroid'] - cx, s['ycentroid'] - cy) * pixel_scale
        if r_arcsec > center_exclusion:
            out.append((float(s['xcentroid']), float(s['ycentroid'])))
    return out


def apply_point_source_holes(mask_bool, centers_xy, pixel_scale):
    """mask_bool: True = masked out (PyAutoLens Mask2D convention). Adds a
    small circular True (excluded) region at each detected point source."""
    ny, nx = mask_bool.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    hole_radius_pix = POINT_SOURCE_MASK_RADIUS * (PSF_FWHM / pixel_scale)
    out = mask_bool.copy()
    for x, y in centers_xy:
        out |= (np.hypot(xx - x, yy - y) <= hole_radius_pix)
    return out


def fit_one(fits_path, obj_id, n_live=150, number_of_cores=1, path_prefix='hpc_lens_models',
            source_re_max=0.4, mask_radius=7.0, n_networks=4, use_jax_vmap=True,
            mask_point_sources=False, source_re_prior='uncapped'):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
    pixel_scale = float(WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value)
    cutout_size = data.shape[0]

    nan_mask = np.isnan(data)
    if nan_mask.any():
        data[nan_mask] = np.nanmedian(data)
    ny, nx = data.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2, nx / 2
    r_pix = np.hypot(yy - cy, xx - cx)
    bg_region = data[r_pix > 0.35 * min(ny, nx)]
    _, median, std = sigma_clipped_stats(bg_region, sigma=3.0, maxiters=5)
    noise_sigma = std
    data_sub = data - median

    noise_map_arr = np.clip(np.full_like(data_sub, noise_sigma), 1e-6, None)
    data_ag = al.Array2D.no_mask(values=data_sub, pixel_scales=pixel_scale)
    noise_ag = al.Array2D.no_mask(values=noise_map_arr, pixel_scales=pixel_scale)
    # PSF: a normalized Gaussian kernel of FWHM=PSF_FWHM arcsec. The old
    # al.Kernel2D.from_gaussian(...) API was removed in autolens 2026.7.x; the
    # PSF is now an al.Convolver wrapping an al.Array2D kernel. Build the
    # Gaussian in pixel units (sigma_arcsec / pixel_scale) with astropy so the
    # result is identical to the old from_gaussian call.
    _sigma_pix = (PSF_FWHM / 2.3548) / pixel_scale
    _kernel_arr = Gaussian2DKernel(x_stddev=_sigma_pix, x_size=21, y_size=21).array.astype(np.float64)
    _kernel_arr /= _kernel_arr.sum()
    psf = al.Convolver(kernel=al.Array2D.no_mask(values=_kernel_arr, pixel_scales=pixel_scale), normalize=True)
    dataset = al.Imaging(data=data_ag, noise_map=noise_ag, psf=psf, check_noise_map=False)
    # Fixed physical radius, not a fraction of cutout size: the cutout is 151px
    # (~14.5" at 0.9x half-size), way beyond any plausible arc/lens-light extent
    # for einstein_radius in [0.3, 4.0]". A mask that wide lets an uncapped-Re
    # source profile reach out and fit unrelated field interlopers (companion
    # stars/galaxies near the cutout edge) as if they were part of the lensed
    # arc, producing streak-shaped unphysical solutions. 7" keeps ~1.75x the
    # max einstein_radius prior as margin while excluding the field.
    mask = al.Mask2D.circular(shape_native=dataset.shape_native, pixel_scales=pixel_scale, radius=mask_radius)
    if mask_point_sources:
        point_sources = detect_point_sources(data_sub, pixel_scale, noise_sigma)
        if point_sources:
            new_bool = apply_point_source_holes(np.asarray(mask), point_sources, pixel_scale)
            mask = al.Mask2D(mask=new_bool, pixel_scales=pixel_scale)
            print(f"[mask_point_sources] {obj_id}: masked {len(point_sources)} detected point source(s) "
                  f"at {point_sources}", flush=True)
    dataset = dataset.apply_mask(mask=mask)

    lens_bulge = af.Model(al.lp_linear.SersicCore)
    lens_mass = af.Model(al.mp.Isothermal)
    lens_mass.centre = lens_bulge.centre
    lens_mass.einstein_radius = af.UniformPrior(lower_limit=0.3, upper_limit=4.0)
    lens_shear = af.Model(al.mp.ExternalShear)
    lens_galaxy = af.Model(al.Galaxy, redshift=0.4, bulge=lens_bulge, mass=lens_mass, shear=lens_shear)

    source_bulge = af.Model(al.lp_linear.SersicCore)
    source_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
    source_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
    if source_re_prior == 'tight':
        # Found 2026-08-02: the uncapped UniformPrior(0.02, 30) lets the
        # sampler wander to either extreme for a large fraction of the
        # sample -- 35% of parametric fits landed at effective_radius > 2"
        # (a diffuse blob mimicking the whole field, not a compact arc) and
        # 8.6% collapsed to < 0.05" (a point source, unable to reproduce
        # extended/multiply-imaged structure). Both extremes fit noticeably
        # worse on average than the physically-plausible middle group. This
        # confirms, at full-sample scale, the single-object "streak" risk
        # flagged but left unresolved on 2026-07-25. Recentered on a
        # physically motivated compact-source scale (user-specified, not a
        # guess): GaussianPrior(mean=0.2, sigma=0.15) truncated to [0.02, 0.8].
        source_bulge.effective_radius = af.TruncatedGaussianPrior(
            mean=0.2, sigma=0.15, lower_limit=0.02, upper_limit=0.8)
    else:
        source_bulge.effective_radius = af.UniformPrior(lower_limit=0.02, upper_limit=source_re_max)
    source_galaxy = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

    model = af.Collection(galaxies=af.Collection(lens=lens_galaxy, source=source_galaxy))
    # Parametric source (SersicCore, no pixelization) -> JAX is the correct
    # accelerator per PyAutoLens HPC guidance (it vectorises the likelihood; it
    # is NOT GPU-only, it also accelerates CPU runs). BUT PyAutoLens only
    # supports use_jax on Python >= 3.11 -- on 3.10 it errors, so gate on both
    # the version AND jax being importable. Falls back to the plain (validated)
    # CPU path otherwise, matching the shipped behaviour.
    _use_jax = False
    if sys.version_info >= (3, 11):
        try:
            import jax  # noqa: F401
            _use_jax = True
        except Exception:
            _use_jax = False
    # BUG (found 2026-07-30): al.AnalysisImaging's own default for the
    # use_jax kwarg is True when omitted -- the previous version of this
    # line omitted it entirely in the else branch, silently running with
    # JAX enabled on this Python 3.10 venv despite the version gate above
    # concluding it shouldn't be. Must pass use_jax=_use_jax explicitly in
    # both branches.
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=_use_jax)

    # RETRY-ON-CRASH (added 2026-07-30): Nautilus's own ellipsoid-splitting
    # bound computation (nautilus/bounds/basic.py:minimum_volume_enclosing_ellipsoid)
    # has a rare, seed-dependent numerical-stability failure -- confirmed via
    # a live-checkpoint inspection that the live points involved had NaN
    # physical parameters with spuriously high log-likelihoods, causing the
    # point cloud's covariance to go singular. This is a known upstream bug
    # class (nautilus issues #34/#35, "matrix not positive definite" during
    # ellipsoid fitting; not fully fixed as of nautilus 1.0.6, no config knob
    # avoids it). The upstream author's own recommendation for this failure
    # is simply to retry with a different random seed -- af.Nautilus doesn't
    # fix a seed here, so a fresh search object naturally gets a new one.
    # Retrying is the appropriate handling for a confirmed third-party
    # numerical instability, not a symptom patch: the root cause lives inside
    # nautilus's own bound-fitting algorithm, well outside this pipeline.
    max_attempts = 3
    t0 = time.time()
    for attempt in range(1, max_attempts + 1):
        search = af.Nautilus(path_prefix=path_prefix, name=obj_id, unique_tag=obj_id, n_live=n_live, number_of_cores=number_of_cores, n_networks=n_networks, use_jax_vmap=use_jax_vmap)
        try:
            result = search.fit(model=model, analysis=analysis)
            break
        except np.linalg.LinAlgError:
            if attempt == max_attempts:
                raise
            print(f"[retry] Nautilus hit a singular-matrix crash on {obj_id} "
                  f"(attempt {attempt}/{max_attempts}); clearing output dir and "
                  f"retrying with a fresh search.", flush=True)
            shutil.rmtree(search.paths.output_path, ignore_errors=True)
    dt = time.time() - t0

    fit = result.max_log_likelihood_fit
    sn_maps = list(fit.subtracted_signal_to_noise_maps_of_galaxies_dict.values())
    subtracted_imgs = list(fit.subtracted_images_of_galaxies_dict.values())
    source_sn_map = np.asarray(sn_maps[1].native)
    source_subtracted_image = np.asarray(subtracted_imgs[1].native)
    residual = np.asarray(fit.data.native) - np.asarray(fit.model_data.native)

    einstein_radius = result.instance.galaxies.lens.mass.einstein_radius

    fny, fnx = source_sn_map.shape
    fyy, fxx = np.mgrid[0:fny, 0:fnx]
    fcy, fcx = fny / 2, fnx / 2
    r_arcsec = np.hypot(fyy - fcy, fxx - fcx) * pixel_scale
    annulus_inner = max(0.1, ANNULUS_INNER_FRAC * einstein_radius)
    annulus_outer = max(annulus_inner + ANNULUS_MIN_WIDTH, ANNULUS_OUTER_FRAC * einstein_radius)
    region = (r_arcsec > annulus_inner) & (r_arcsec < annulus_outer)

    snr_model = source_subtracted_image[region].sum() / (noise_sigma * np.sqrt(region.sum()))
    snr_peak = np.nanmax(source_sn_map[region])
    chi2 = np.mean((residual[region] / noise_sigma) ** 2)
    # Source Sersic effective radius (arcsec). Recorded so we can tell whether
    # the fit is railing against source_re_max -- a binding cap means the source
    # size (and hence the arc-S/N derived from the source-subtracted image) is
    # prior-limited, not data-driven.
    source_effective_radius = float(result.instance.galaxies.source.bulge.effective_radius)

    return {
        'ID': obj_id, 'time_s': dt, 'einstein_radius': einstein_radius,
        'snr_arc_model': snr_model, 'snr_arc_peak': snr_peak, 'chi2_per_pix': chi2,
        'log_likelihood': result.log_likelihood,
        'source_effective_radius': source_effective_radius,
        'source_re_max': source_re_max,
        'mask_radius': mask_radius,
        'annulus_inner_arcsec': annulus_inner,
        'annulus_outer_arcsec': annulus_outer,
        'mask_point_sources': mask_point_sources,
        'source_re_prior': source_re_prior,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Target ID, matches a filename in --cutout_dir')
    parser.add_argument('--cutout_dir', default='dataset/lens_cutouts_fits')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--n_live', type=int, default=150)
    parser.add_argument('--number_of_cores', type=int, default=1)
    parser.add_argument('--source_re_max', type=float, default=0.4,
                        help='Upper limit (arcsec) of the source Sersic effective_radius '
                             'UniformPrior. Default 0.4 (the reactive cap under review); '
                             'pass 30.0 for the PyAutoLens default (effectively uncapped).')
    parser.add_argument('--results_suffix', default='',
                        help='Optional suffix appended to the result JSON filename, so '
                             'multiple prior settings can be compared without clobbering.')
    parser.add_argument('--mask_radius', type=float, default=7.0,
                        help='Circular fit-mask radius in arcsec (default 7.0, ~1.75x the max '
                             'einstein_radius prior). Was previously 0.9x the cutout half-size '
                             '(~14.5"), which let an uncapped source profile reach field '
                             'interlopers near the cutout edge and fit them as streak-shaped '
                             'unphysical arcs.')
    parser.add_argument('--mask_point_sources', action='store_true',
                        help='Auto-detect compact point-like sources (photutils DAOStarFinder, '
                             'excluding a 1.0" radius around the cutout centre) and add a small '
                             'circular hole in the fit mask at each detection, so an unmodelled '
                             'foreground star/interloper does not contaminate the mass-model fit '
                             'or the arc-S/N annulus.')
    parser.add_argument('--source_re_prior', choices=['uncapped', 'tight'], default='uncapped',
                        help='"uncapped" (default, unchanged): UniformPrior(0.02, source_re_max). '
                             '"tight": TruncatedGaussianPrior(mean=0.2, sigma=0.15, [0.02, 0.8]) -- '
                             'found 2026-08-02 that the uncapped prior lets ~44%% of parametric fits '
                             'land at an implausible source size (either a diffuse >2" blob or a '
                             'collapsed <0.05" point), both fitting worse on average.')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    fits_path = os.path.join(args.cutout_dir, f'{args.dataset}.fits')
    result = fit_one(fits_path, args.dataset, n_live=args.n_live,
                     number_of_cores=args.number_of_cores, source_re_max=args.source_re_max,
                     mask_radius=args.mask_radius, mask_point_sources=args.mask_point_sources,
                     source_re_prior=args.source_re_prior)
    out_name = f'{args.dataset}{args.results_suffix}.json'
    with open(os.path.join(args.results_dir, out_name), 'w') as f:
        json.dump(result, f, indent=2)
    print(result, flush=True)
