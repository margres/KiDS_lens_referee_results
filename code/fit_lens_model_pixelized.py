"""
Pixelized-source variant of fit_lens_model.py, for the small subset of known
lenses (~6/107) whose residuals show a real concentric-ring structure that a
single parametric SersicCore source cannot represent (see
NOTES_ring_residual_issue.md). Same lens light + mass model as
fit_lens_model.py; only the source changes, from a parametric SersicCore to
a pixelized mesh (al.Pixelization + al.mesh.RectangularAdaptDensity +
al.reg.Constant), following PyAutoLens's own
notebooks/imaging/features/pixelization/modeling.ipynb example.

Deliberately NOT the full 5-stage SLaM pipeline from that repo's
cpu_fast_modeling.ipynb (MGE light profiles, adapt regularization chained
across 2 pixelized stages, PowerLaw mass upgrade) -- that is a much bigger
lift for 6 objects. This is a single-stage swap: same Isothermal mass +
SersicCore lens light as the parametric fit, source replaced with a
Constant-regularized pixelized mesh.

PyAutoLens hard-requires a PositionsLH for pixelization fits (raises
AnalysisException otherwise -- unconstrained pixelized sources are prone to
demagnified/degenerate solutions, the same failure family as the streak bug
fixed in fit_lens_model.py). We have no manual position-picking pipeline, so
positions_from_parametric_residual() derives candidate image positions
automatically from the ALREADY-COMPLETED parametric fit's residual map:
peaks in normalized-residual (data > model) mark where the single-Sersic
source failed to capture real arc flux, which is exactly where a pixelized
source's images should be. Requires fit_lens_model.py to have already been
run for the object (reads its output/hpc_lens_models/<ID>/.../fit.fits).

Usage:
    python3 fit_lens_model_pixelized.py --dataset KiDSDR4_J120940.293-015050.60 \
        --cutout_dir lens_cutouts_fits --number_of_cores 8

Writes results_pixelized/<dataset>.json plus the full PyAutoFit output tree
under output/hpc_lens_models_pixelized/<dataset>/... (separate path_prefix
from the parametric fits so PyAutoFit's identifier caching can't collide
with an existing parametric-model output dir for the same object).
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
from scipy.ndimage import maximum_filter, gaussian_filter

PSF_FWHM = 0.7
ANNULUS_INNER, ANNULUS_OUTER = 0.5, 3.5


def positions_from_parametric_residual(obj_id, grid, pixel_scale, sigma_thresh=3.0, max_positions=4,
                                        min_sep_arcsec=0.3, parametric_path_prefix='hpc_lens_models'):
    """
    PyAutoLens requires a PositionsLH for pixelization fits (unconstrained
    pixelized sources are prone to demagnified/degenerate solutions -- the
    same failure class as the streak bug fixed in fit_lens_model.py). We
    have no manual position-picking pipeline, so derive candidate multiple
    -image positions automatically: the parametric fit's normalized
    residual map is high (data > model) exactly where the real arc flux
    wasn't captured by the single-Sersic source -- i.e. at the image
    positions a pixelized source needs to reconstruct. Peak-find on that.
    """
    import glob
    from astropy.io import fits as _fits
    p = glob.glob(f"output/{parametric_path_prefix}/{obj_id}/{obj_id}/*/image/fit.fits")
    if not p:
        raise FileNotFoundError(
            f"No completed parametric fit found for {obj_id} under "
            f"output/{parametric_path_prefix}/ -- run fit_lens_model.py for "
            f"this object first, positions are derived from its residual."
        )
    with _fits.open(p[0]) as hdul:
        mask = hdul["MASK"].data.astype(bool)
        nr = hdul["NORMALIZED_RESIDUAL_MAP"].data.copy()
    nr[mask] = 0.0
    smoothed = gaussian_filter(nr, sigma=1.0)

    ny, nx = smoothed.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = ny / 2, nx / 2
    r_pix = np.hypot(yy - cy, xx - cx)
    candidate = (~mask) & (r_pix > ANNULUS_INNER / pixel_scale) & (smoothed > sigma_thresh)

    local_max = (maximum_filter(smoothed, size=5) == smoothed) & candidate
    peak_rc = np.argwhere(local_max)
    peak_vals = smoothed[local_max]
    order = np.argsort(-peak_vals)
    peak_rc = peak_rc[order]

    grid_native = np.asarray(grid.native)
    positions = []
    for row, col in peak_rc:
        y_arcsec, x_arcsec = grid_native[row, col]
        if any(np.hypot(y_arcsec - py, x_arcsec - px) < min_sep_arcsec for py, px in positions):
            continue
        positions.append((float(y_arcsec), float(x_arcsec)))
        if len(positions) >= max_positions:
            break

    if len(positions) < 2:
        raise ValueError(
            f"Only found {len(positions)} distinct residual peak(s) above "
            f"{sigma_thresh}sigma for {obj_id} -- need >=2 for a PositionsLH. "
            f"Lower sigma_thresh or inspect the residual map manually."
        )
    return positions


def fit_one_pixelized(fits_path, obj_id, n_live=100, number_of_cores=1,
                       path_prefix='hpc_lens_models_pixelized', mask_radius=7.0,
                       mesh_pixels=12, positions_threshold=0.5):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header
    pixel_scale = float(WCS(header).proj_plane_pixel_scales()[0].to('arcsec').value)

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
    _sigma_pix = (PSF_FWHM / 2.3548) / pixel_scale
    _kernel_arr = Gaussian2DKernel(x_stddev=_sigma_pix, x_size=21, y_size=21).array.astype(np.float64)
    _kernel_arr /= _kernel_arr.sum()
    psf = al.Convolver(kernel=al.Array2D.no_mask(values=_kernel_arr, pixel_scales=pixel_scale), normalize=True)
    dataset = al.Imaging(data=data_ag, noise_map=noise_ag, psf=psf, check_noise_map=False)
    mask = al.Mask2D.circular(shape_native=dataset.shape_native, pixel_scales=pixel_scale, radius=mask_radius)
    dataset = dataset.apply_mask(mask=mask)
    # Required for pixelization fits (denser sub-gridding when mapping image
    # pixels to source-plane mesh pixels) -- not needed for parametric fits.
    dataset = dataset.apply_over_sampling(over_sample_size_pixelization=4)

    lens_bulge = af.Model(al.lp_linear.SersicCore)
    lens_mass = af.Model(al.mp.Isothermal)
    lens_mass.centre = lens_bulge.centre
    lens_mass.einstein_radius = af.UniformPrior(lower_limit=0.3, upper_limit=4.0)
    lens_shear = af.Model(al.mp.ExternalShear)
    lens_galaxy = af.Model(al.Galaxy, redshift=0.4, bulge=lens_bulge, mass=lens_mass, shear=lens_shear)

    mesh = af.Model(al.mesh.RectangularAdaptDensity, shape=(mesh_pixels, mesh_pixels))
    regularization = af.Model(al.reg.Constant)
    pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
    source_galaxy = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens_galaxy, source=source_galaxy))

    position_coords = positions_from_parametric_residual(obj_id, dataset.grid, pixel_scale)
    positions = al.Grid2DIrregular(position_coords)
    positions_likelihood = al.PositionsLH(positions=positions, threshold=positions_threshold)

    _use_jax = False
    if sys.version_info >= (3, 11):
        try:
            import jax  # noqa: F401
            _use_jax = True
        except Exception:
            _use_jax = False
    analysis = al.AnalysisImaging(dataset=dataset, positions_likelihood_list=[positions_likelihood], use_jax=_use_jax) if _use_jax \
        else al.AnalysisImaging(dataset=dataset, positions_likelihood_list=[positions_likelihood])
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
    regularization_coefficient = float(result.instance.galaxies.source.pixelization.regularization.coefficient)

    return {
        'ID': obj_id, 'time_s': dt, 'einstein_radius': einstein_radius,
        'snr_arc_model': snr_model, 'snr_arc_peak': snr_peak, 'chi2_per_pix': chi2,
        'log_likelihood': result.log_likelihood,
        'mask_radius': mask_radius, 'mesh_pixels': mesh_pixels,
        'regularization_coefficient': regularization_coefficient,
        'source_model': 'pixelized',
        'positions': position_coords, 'positions_threshold': positions_threshold,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cutout_dir', default='dataset/lens_cutouts_fits')
    parser.add_argument('--results_dir', default='results_pixelized')
    parser.add_argument('--n_live', type=int, default=100)
    parser.add_argument('--number_of_cores', type=int, default=1)
    parser.add_argument('--mask_radius', type=float, default=7.0)
    parser.add_argument('--mesh_pixels', type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    fits_path = os.path.join(args.cutout_dir, f'{args.dataset}.fits')
    result = fit_one_pixelized(fits_path, args.dataset, n_live=args.n_live,
                                number_of_cores=args.number_of_cores,
                                mask_radius=args.mask_radius, mesh_pixels=args.mesh_pixels)
    out_name = f'{args.dataset}.json'
    with open(os.path.join(args.results_dir, out_name), 'w') as f:
        json.dump(result, f, indent=2)
    print(result, flush=True)
