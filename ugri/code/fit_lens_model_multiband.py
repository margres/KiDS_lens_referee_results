"""
Multi-band variant of fit_lens_model.py: fits the SAME lens+mass+source
model jointly across several KiDS bands (default r+g+i), using PyAutoLens's
official multi-wavelength API (af.AnalysisFactor + af.FactorGraphModel --
see autolens_workspace/notebooks/multi/start_here.ipynb). Mass and light
*shape* parameters are shared/tied across bands (one joint posterior, not
independent per-band fits); each band's light-profile amplitude is solved
independently per band automatically since we use linear light profiles
(al.lp_linear.SersicCore), so no explicit per-band amplitude parameter is
needed.

Why multi-band: color separates lens (typically red/passive) from source
(often bluer) light more robustly than a single-band mask/prior tweak can --
directly relevant to the streak-type failures documented in
NOTES_ring_residual_issue.md, where a same-color-as-lens interloper gets
fit as if it were part of the arc.

u-band deliberately excluded by default: KiDS u is the shallowest/noisiest
of the four bands by survey design (much less exposure time than g/r/i).
Including noisy pixels in a joint chi2 mostly dilutes the fit rather than
constraining it -- the color-separation goal is already served by g+r+i.
Pass --bands u,g,r,i to include it if ever wanted; nothing structural
prevents it, this is a data-quality judgment call, not a hard limitation.

Requires per-band cutouts already fetched: lens_cutouts_fits/ (r, from
fetch_cutouts.py) and lens_cutouts_fits_g/, lens_cutouts_fits_i/ (from
fetch_multiband_cutouts.py --band g / --band i).

Usage:
    python3 fit_lens_model_multiband.py --dataset KiDSDR4_J010127.840-334319.40 \
        --bands r,g,i --number_of_cores 8

Writes results_multiband/<dataset>.json (per-band snr_arc/chi2 + the shared
einstein_radius) plus the full PyAutoFit output tree under
output/hpc_lens_models_multiband/<dataset>/... (separate path_prefix from
the single-band fits, so PyAutoFit's identifier caching can't collide).
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

# TODO: measured per-band, not just a single reused constant -- KiDS seeing
# varies by band (g is typically best, u/i often worse). Using one PSF for
# all bands is a known simplification for this first cut.
BAND_CUTOUT_DIRS = {
    'r': 'lens_cutouts_fits',
    'g': 'lens_cutouts_fits_g',
    'u': 'lens_cutouts_fits_u',
    'i': 'lens_cutouts_fits_i',
}


def load_band_dataset(fits_path, mask_radius):
    """Mirrors fit_lens_model.fit_one's preprocessing exactly, for one band."""
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
    return dataset, pixel_scale, noise_sigma


def fit_one_multiband(obj_id, bands, n_live=150, number_of_cores=1,
                       path_prefix='hpc_lens_models_multiband',
                       source_re_max=0.4, mask_radius=7.0):
    datasets, pixel_scales, noise_sigmas = {}, {}, {}
    for band in bands:
        p = os.path.join(BAND_CUTOUT_DIRS[band], f'{obj_id}.fits')
        if not os.path.exists(p):
            raise FileNotFoundError(f"No {band}-band cutout for {obj_id} at {p}")
        ds, ps, ns = load_band_dataset(p, mask_radius)
        datasets[band] = ds
        pixel_scales[band] = ps
        noise_sigmas[band] = ns

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

    _use_jax = False
    if sys.version_info >= (3, 11):
        try:
            import jax  # noqa: F401
            _use_jax = True
        except Exception:
            _use_jax = False

    analysis_factor_list = []
    for band in bands:
        analysis = al.AnalysisImaging(dataset=datasets[band], use_jax=_use_jax) if _use_jax \
            else al.AnalysisImaging(dataset=datasets[band])
        # model.copy() per band: PyAutoFit links priors by identity, not by
        # value, so this keeps the mass/light SHAPE parameters joint across
        # bands (one shared posterior) while letting each band's linear
        # light-profile amplitude solve independently, as intended.
        analysis_factor_list.append(af.AnalysisFactor(prior_model=model.copy(), analysis=analysis))

    factor_graph = af.FactorGraphModel(*analysis_factor_list)
    search = af.Nautilus(path_prefix=path_prefix, name=obj_id, unique_tag=obj_id,
                          n_live=n_live, number_of_cores=number_of_cores)

    t0 = time.time()
    result_list = search.fit(model=factor_graph.global_prior_model, analysis=factor_graph)
    dt = time.time() - t0

    einstein_radius = result_list[0].instance.galaxies.lens.mass.einstein_radius

    per_band = {}
    for band, result in zip(bands, result_list):
        fit = result.max_log_likelihood_fit
        sn_maps = list(fit.subtracted_signal_to_noise_maps_of_galaxies_dict.values())
        subtracted_imgs = list(fit.subtracted_images_of_galaxies_dict.values())
        source_sn_map = np.asarray(sn_maps[1].native)
        source_subtracted_image = np.asarray(subtracted_imgs[1].native)
        residual = np.asarray(fit.data.native) - np.asarray(fit.model_data.native)

        fny, fnx = source_sn_map.shape
        fyy, fxx = np.mgrid[0:fny, 0:fnx]
        fcy, fcx = fny / 2, fnx / 2
        r_arcsec = np.hypot(fyy - fcy, fxx - fcx) * pixel_scales[band]
        region = (r_arcsec > ANNULUS_INNER) & (r_arcsec < ANNULUS_OUTER)

        per_band[band] = {
            'snr_arc_model': float(source_subtracted_image[region].sum() / (noise_sigmas[band] * np.sqrt(region.sum()))),
            'snr_arc_peak': float(np.nanmax(source_sn_map[region])),
            'chi2_per_pix': float(np.mean((residual[region] / noise_sigmas[band]) ** 2)),
        }

    return {
        'ID': obj_id, 'time_s': dt, 'bands': bands, 'einstein_radius': float(einstein_radius),
        'log_likelihood_per_band': [float(r.log_likelihood) for r in result_list],
        'mask_radius': mask_radius, 'source_re_max': source_re_max,
        'per_band': per_band,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--bands', default='r,g,i',
                         help='Comma-separated band list, e.g. "r,g,i". u excluded by default (noisy) -- see module docstring.')
    parser.add_argument('--results_dir', default='results_multiband')
    parser.add_argument('--n_live', type=int, default=150)
    parser.add_argument('--number_of_cores', type=int, default=1)
    parser.add_argument('--source_re_max', type=float, default=0.4)
    parser.add_argument('--mask_radius', type=float, default=7.0)
    args = parser.parse_args()

    bands = args.bands.split(',')
    os.makedirs(args.results_dir, exist_ok=True)
    result = fit_one_multiband(args.dataset, bands, n_live=args.n_live,
                                number_of_cores=args.number_of_cores,
                                source_re_max=args.source_re_max, mask_radius=args.mask_radius)
    with open(os.path.join(args.results_dir, f'{args.dataset}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(result, flush=True)
