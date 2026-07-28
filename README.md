# KiDS strong-lens arc-S/N modelling (referee response support material)

Supporting code, data, and diagnostics for a KiDS strong-lens paper referee
response point: a recovered-vs-missed known-lens comparison based on
arc-level signal-to-noise, derived from a full lens+mass+source model fit.

**2026-07-28 correction:** an earlier version of this repo fit the wrong
107-object sample (a stale early-draft candidate list, `targets.csv`, that
had since been superseded). That content is kept for reference under
`archive_2026-07-28_wrong_sample/` — the fix methodology in it (mask-radius
bug, etc.) is still valid, just applied to the wrong objects. Everything at
the top level now is against the correct, current target set.

## Method

Each lens is fit with [PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens)
— lens Sersic-core light + isothermal mass + external shear, Sersic-core
source — via `af.Nautilus` nested sampling. Arc S/N is computed from the
lens-light-subtracted image in an annulus around the Einstein radius (not
the full residual, so source-model quality doesn't directly bias it — see
`code/NOTES_ring_residual_issue.md`).

## Target set (223 unique objects)

- **107 known lenses** (`reference_csvs/known_lenses_AB.csv`: 15 grade A +
  92 grade B, TEGLIE/LinKS origin) — the ground truth this project is
  checking recovery against.
- **140 new grade A/B discoveries** from this work's own AL/NN search
  (`reference_csvs/KiDS_SGL_candidates.csv`, `Grade` column; 1032 grade-C
  candidates also in that file are not being fit).
- 24 objects overlap (re-discoveries), giving 223 unique cutouts in
  `fits_cutouts/`.

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

- `code/` — `fit_lens_model.py` (main parametric fit),
  `fit_lens_model_pixelized.py` (in-progress pixelized-source variant for
  ring-residual objects), `slurm/array_all_AB.sbatch` (the array-job
  template used to fit the 223-object set), `NOTES_ring_residual_issue.md`.
- `fits_cutouts/` — 151x151px KiDS DR4 r-band cutouts (~0.214"/px) for all
  223 known+new grade A/B objects.
- `results/` — per-object `.json` fit summaries, for whichever objects have
  completed so far (see Status — this is a partial set, updated as the
  cluster job progresses).
- `reference_csvs/` — the known-lens ground truth, the candidate list, and
  the recovered-vs-missed breakdown.
- `archive_2026-07-28_wrong_sample/` — the earlier (wrong-sample) push,
  kept for reference.

## Status (as of this push)

**16 / 223 objects fit so far** — a cluster array job (223 tasks, 20
concurrent) is running to fit the rest. This repo will be updated as it
progresses.

Known open issues in the fitting pipeline (methodology, not sample-specific
— see `code/NOTES_ring_residual_issue.md` for detail):
- A mask-radius bug that caused streak-shaped unphysical fits and
  12h-wall-clock non-convergence hangs on some objects — **fixed** (mask
  tightened from ~14.5" to a fixed 7").
- A separate, still-unresolved intermittent silent deadlock in the
  nested-sampling stage (not caused by the mask bug, not yet root-caused)
  affects a small fraction of fits; those are retried.
- A subset of objects show a genuine concentric ring-shaped residual a
  single Sersic-core source profile can't represent (real source
  complexity, not a bug) — a pixelized-source reconstruction is in
  progress for these.

This is in-progress work, not a final/published result set.
