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
ANNULUS_INNER, ANNULUS_OUTER = 0.5, 3.5


def fit_one(fits_path, obj_id, n_live=150, number_of_cores=1, path_prefix='hpc_lens_models',
            source_re_max=0.4, mask_radius=7.0):
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
    analysis = al.AnalysisImaging(dataset=dataset, use_jax=_use_jax) if _use_jax \
        else al.AnalysisImaging(dataset=dataset)
    search = af.Nautilus(path_prefix=path_prefix, name=obj_id, unique_tag=obj_id, n_live=n_live, number_of_cores=number_of_cores)

    t0 = time.time()
    result = search.fit(model=model, analysis=analysis)
    dt = time.time() - t0

    fit = result.max_log_likelihood_fit
    sn_maps = list(fit.subtracted_signal_to_noise_maps_of_galaxies_dict.values())
    subtracted_imgs = list(fit.subtracted_images_of_galaxies_dict.values())
    source_sn_map = np.asarray(sn_maps[1].native)
    source_subtracted_image = np.asarray(subtracted_imgs[1].native)
    residual = np.asarray(fit.data.native) - np.asarray(fit.model_data.native)

    fny, fnx = source_sn_map.shape
    fyy, fxx = np.mgrid[0:fny, 0:fnx]
    fcy, fcx = fny / 2, fnx / 2
    r_arcsec = np.hypot(fyy - fcy, fxx - fcx) * pixel_scale
    region = (r_arcsec > ANNULUS_INNER) & (r_arcsec < ANNULUS_OUTER)

    snr_model = source_subtracted_image[region].sum() / (noise_sigma * np.sqrt(region.sum()))
    snr_peak = np.nanmax(source_sn_map[region])
    chi2 = np.mean((residual[region] / noise_sigma) ** 2)
    einstein_radius = result.instance.galaxies.lens.mass.einstein_radius
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
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    fits_path = os.path.join(args.cutout_dir, f'{args.dataset}.fits')
    result = fit_one(fits_path, args.dataset, n_live=args.n_live,
                     number_of_cores=args.number_of_cores, source_re_max=args.source_re_max,
                     mask_radius=args.mask_radius)
    out_name = f'{args.dataset}{args.results_suffix}.json'
    with open(os.path.join(args.results_dir, out_name), 'w') as f:
        json.dump(result, f, indent=2)
    print(result, flush=True)
