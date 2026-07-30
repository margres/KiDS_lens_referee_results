# KiDS strong-lens arc-S/N modelling (referee response support material)

Supporting code, data, and diagnostics for a KiDS strong-lens paper referee
response point: a recovered-vs-missed known-lens comparison based on
arc-level signal-to-noise, derived from a full lens+mass+source model fit.

**2026-07-28 correction:** an earlier version of this repo fit the wrong
107-object sample (a stale early-draft candidate list, `targets.csv`, that
had since been superseded) — the fix methodology (mask-radius bug, etc.)
was still valid, just applied to the wrong objects. That earlier push has
since been removed (still recoverable from git history if ever needed).
Everything below is against the correct, current target set.

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
- `r_only/residual_images/` — diagnostic residual-map grids, including
  `all_lenses_full_diagnostic_combined.png` — every fit lens's data/model/
  residual/source-plane panels stacked into one single (very tall) image,
  row order matching `all_202_id_chi2_mapping.txt`'s numbering.
- `r_only/per_lens_diagnostics/<NNN>_<ID>.png` — one 4-panel figure per
  fit lens (same 4 panels as above): observed data, model reconstruction,
  normalized residual, and the source-plane (de-lensed) reconstruction,
  all image-plane panels on the same flux scale. Filename-prefixed with
  the same grid number used in the residual-grid images, so you can cross
  -reference a flagged number back to its individual file. 202/223 present.
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

## Status (as of this push)

**r-only: 202 / 223 objects fit so far** (92/107 known lenses covered),
**180 clean (chi2/pix<=5), 22 elevated** — see
`r_only/residual_images/worst20_by_chi2_2026-07-29.png`. 25 objects
remain, including a handful stuck in an intermittent silent
nested-sampling deadlock (being retried).

**ugri: paused, 0 / 223 fit.** g/u/i fetch completed successfully (all 223
targets have g+r+i cutouts now). But the fit itself hit a real performance
problem, diagnosed and the job killed 2026-07-29: `af.FactorGraphModel`/
`af.AnalysisFactor` (the official PyAutoLens multi-wavelength API used in
`ugri/code/fit_lens_model_multiband.py`) appears not to be JAX-vmap-
compatible — production single-band fits do ~200-300k likelihood calls in
~25min via batched JAX evaluation, but the multiband factor-graph path
managed only ~900 calls in 4.5h (confirmed via a standalone timing script:
a single warmed-up likelihood call takes ~1s through the factor graph vs.
the ~7ms/call the batched single-band path achieves — not a "3 bands = 3x
slower" problem, a ~1000x one). Checked and ruled out a simpler
explanation first (parameter-count blowup from `model.copy()` not sharing
priors across bands) before concluding it's the vmap fallback.
**Deprioritized** until this is fixed — either find why the factor graph
breaks vmap, or rewrite as a single custom `Analysis` that sums chi-squared
across bands directly (bypassing `AnalysisFactor` entirely). r-only remains
the primary track in the meantime.

Known open issues in the r-only fitting pipeline (methodology, not
sample-specific — see `r_only/code/NOTES_ring_residual_issue.md` for
detail), relevant background for the multi-band effort too:
- A mask-radius bug that caused streak-shaped unphysical fits and
  12h-wall-clock non-convergence hangs on some objects — **fixed** (mask
  tightened from ~14.5" to a fixed 7"). At least one object (J085156) still
  streaked even at 7" — the interloper was closer than that; refit with a
  5" mask for that object specifically, chi2/pix dropped 83.6 -> 2.5.
- The separate intermittent silent deadlock (nested-sampling stage hangs
  with zero progress, no error) is **root-caused and fixed as of 2026-07-30**.
  `py-spy` couldn't attach (no ptrace permission on this cluster), so used
  Python's stdlib `faulthandler` (`SIGUSR1` -> live stack dump, no special
  permission needed) instead, and found `fit_lens_model.py`'s own JAX
  version-gate had a real bug: the intent was "only enable JAX on Python
  >=3.11," but the fallback branch omitted the `use_jax` kwarg entirely,
  and PyAutoLens's own default for an omitted `use_jax` is `True` — so this
  Python-3.10 pipeline was silently running an unsupported JAX
  configuration the whole time, which is the likely cause of the hangs
  (confirmed 3 different internal stall points across debug attempts, all
  JAX/Nautilus-internal, all showing ~0% CPU/GPU utilization). **Fix**:
  explicitly pass `use_jax=False` always on this venv. Verified: worker
  processes now show genuine 75-79% CPU utilization instead of ~0%.
  Tried the "do it properly" alternative too (real JAX on the Python-3.11
  GPU venv) -- that also stalled, so the fix is disabling JAX for this
  pipeline entirely, not using it correctly instead.
- A subset of objects show a genuine concentric ring-shaped residual a
  single Sersic-core source profile can't represent (real source
  complexity, not a bug) — a pixelized-source reconstruction
  (`r_only/code/fit_lens_model_pixelized.py`) is in progress for these.

This is in-progress work, not a final/published result set.
