# KiDS strong-lens arc-S/N modelling (referee response support material)

Supporting code, data, and diagnostics for a KiDS strong-lens paper referee
response point: a recovered-vs-missed known-lens comparison based on
arc-level signal-to-noise, derived from a full lens+mass+source model fit.

**2026-07-28 correction:** an earlier version of this repo fit the wrong
107-object sample (a stale early-draft candidate list, `targets.csv`, that
had since been superseded). That content is kept for reference under
`archive_2026-07-28_wrong_sample/` — the fix methodology in it (mask-radius
bug, etc.) is still valid, just applied to the wrong objects. Everything
below is against the correct, current target set.

## Method

Each lens is fit with [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens)
— lens Sersic-core light + isothermal mass + external shear, Sersic-core
source — via `af.Nautilus` nested sampling. Arc S/N is computed from the
lens-light-subtracted image in an annulus around the Einstein radius (not
the full residual, so source-model quality doesn't directly bias it — see
`r_only/code/NOTES_ring_residual_issue.md`).

Two parallel fit tracks, kept in separate top-level folders:
- **`r_only/`** — single-band (KiDS r) fits. Primary/most complete track.
- **`ugri/`** — multi-band (r+g+i jointly, u excluded as too noisy — KiDS u
  is the shallowest of the four bands by survey design and mostly dilutes
  a joint fit rather than constraining it) fits. Color helps separate
  lens/source/interloper light more robustly than single-band mask/prior
  tweaks alone — motivated directly by streak-type failures documented in
  `r_only/code/NOTES_ring_residual_issue.md`. Uses PyAutoLens's official
  multi-wavelength API (`af.AnalysisFactor` + `af.FactorGraphModel`): one
  shared lens+source model posterior across bands, not independent
  per-band fits. In progress — g/u/i tile downloads were still running as
  of this push, fits not yet started.

## Target set (223 unique objects)

- **107 known lenses** (`reference_csvs/known_lenses_AB.csv`: 15 grade A +
  92 grade B, TEGLIE/LinKS origin) — the ground truth this project is
  checking recovery against.
- **140 new grade A/B discoveries** from this work's own AL/NN search
  (`reference_csvs/KiDS_SGL_candidates.csv`, `Grade` column; 1032 grade-C
  candidates also in that file are not being fit).
- 24 objects overlap (re-discoveries), giving 223 unique cutouts.

## Recovered vs. missed (the actual referee question)

Of the 107 known lenses, cross-referencing against
`KiDS_SGL_candidates.csv`'s `Recovered_known` column
(`reference_csvs/known_lenses_recovered_vs_missed.csv` has the full
per-object breakdown):

| | count | grade A | grade B |
|---|---|---|---|
| Recovered (found by this work's search) | 24 | 4 | 20 |
| Missed (never appear in the candidate output at all) | 83 | 11 | 72 |

*Why* the 83 were missed is the open question this arc-S/N modelling is
meant to help answer (e.g. is arc S/N systematically lower for missed
lenses) — needs fits for the full 107 to check, in progress (see Status).

## Contents

- `r_only/code/` — `fit_lens_model.py` (main parametric fit),
  `fit_lens_model_pixelized.py` (in-progress pixelized-source variant for
  ring-residual objects), `array_all_AB.sbatch`, `NOTES_ring_residual_issue.md`.
- `r_only/fits_cutouts/` — 151x151px KiDS DR4 r-band cutouts (~0.214"/px)
  for all 223 known+new grade A/B objects.
- `r_only/results/` — per-object `.json` fit summaries (point estimates),
  partial set, updated as the cluster job progresses (see Status). Also
  `parameter_uncertainties.csv` — 3-sigma error bars on `einstein_radius`
  and `source_effective_radius` per lens (pulled from PyAutoFit's
  posterior, `errors_at_sigma_3` in each object's internal
  `samples_summary.json` — not otherwise exposed anywhere else in this
  repo; the per-object `.json` files are point estimates only).
- `r_only/residual_images/` — diagnostic residual-map grids.
- `r_only/per_lens_diagnostics/<ID>.png` — one 4-panel figure per fit
  lens: observed data, model reconstruction, normalized residual, and the
  source-plane (de-lensed) reconstruction, all image-plane panels on the
  same flux scale. 198/223 present (all lenses with a completed fit as of
  this push).
- `r_only/raw_fits/<ID>_fit.fits` (MASK/MODEL_DATA/RESIDUAL_MAP/
  NORMALIZED_RESIDUAL_MAP/CHI_SQUARED_MAP extensions) and
  `<ID>_source_plane.fits` (MASK/SOURCE_PLANE_IMAGE_1) — the raw data
  behind `per_lens_diagnostics/`, for anyone who wants their own plots or
  to pull numbers directly rather than being stuck with the PNG rendering.
  198/223 present.
- `ugri/code/` — `fetch_multiband_cutouts.py`, `fit_lens_model_multiband.py`,
  `array_multiband.sbatch`.
- `ugri/results/` — multi-band fit summaries, not yet populated (see Status).
- `reference_csvs/` — the known-lens ground truth, the candidate list, and
  the recovered-vs-missed breakdown (shared by both tracks).
- `archive_2026-07-28_wrong_sample/` — the earlier (wrong-sample) push,
  kept for reference.

## Status (as of this push)

**r-only: 198 / 223 objects fit so far** (90/107 known lenses covered),
**177 clean (chi2/pix<=5), 21 elevated** — see
`r_only/residual_images/worst20_by_chi2_2026-07-29.png`. 25 objects
remain, including a handful stuck in an intermittent silent
nested-sampling deadlock (being retried).

**ugri: 0 / 223 fit yet.** g-band fetch complete, u in progress, i not yet
started (first fetch attempt hit an 8h timeout mid-g-band — per-tile time
for these full-coadd files is much longer than the r-band cutout tiles;
resubmitted with a 48h budget, resuming from cache). A dependent SLURM job
is queued to auto-start the multi-band fit the moment the fetch completes
— no manual trigger needed.

Known open issues in the r-only fitting pipeline (methodology, not
sample-specific — see `r_only/code/NOTES_ring_residual_issue.md` for
detail), relevant background for the multi-band effort too:
- A mask-radius bug that caused streak-shaped unphysical fits and
  12h-wall-clock non-convergence hangs on some objects — **fixed** (mask
  tightened from ~14.5" to a fixed 7"), though at least one object
  (J085156) still streaks even at 7" — the interloper is closer than that.
- A separate, likely-root-caused (corrupted HDF5 nested-sampling checkpoint,
  probably an NFS file-locking issue) intermittent hang affects a small
  fraction of fits; those are cleared and retried.
- A subset of objects show a genuine concentric ring-shaped residual a
  single Sersic-core source profile can't represent (real source
  complexity, not a bug) — a pixelized-source reconstruction
  (`r_only/code/fit_lens_model_pixelized.py`) is in progress for these.

This is in-progress work, not a final/published result set.
