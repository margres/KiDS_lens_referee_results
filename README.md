# KiDS strong-lens arc-S/N modelling (referee response support material)

Supporting code, data, and diagnostics for a KiDS strong-lens paper referee
response point: a recovered-vs-missed known-lens comparison based on
arc-level signal-to-noise, derived from a full lens+mass+source model fit.

## Method

Each of the 107 known KiDS DR4 lenses is fit with
[PyAutoLens](https://github.com/PyAutoLabs/PyAutoLens) — lens Sersic-core
light + isothermal mass + external shear, Sersic-core source — via
`af.Nautilus` nested sampling. Arc S/N is computed from the
lens-light-subtracted image in an annulus around the Einstein radius (not
the full residual, so source-model quality doesn't directly bias it — see
`code/NOTES_ring_residual_issue.md`).

## Contents

- `code/` — `fit_lens_model.py` (the main parametric fit, produces the
  `results/*.json` summaries), `fit_lens_model_pixelized.py` (an
  in-progress pixelized-source variant for objects where a single Sersic
  source isn't enough), `NOTES_ring_residual_issue.md` (open-issue writeup).
- `fits_cutouts/` — 151x151px KiDS DR4 r-band cutouts (~0.214"/px) for each
  known lens, the direct input to the fits.
- `results/` — per-object `.json` summaries (Einstein radius, arc S/N,
  chi2/pix, source parameters) from the fits.
- `residual_images/` — diagnostic plots: `all_107_normalized_residuals_numbered.png`
  (every object's normalized-residual map, numbered, chi2/pix in the
  corner — cross-reference IDs via `numbered_id_chi2_mapping.txt`),
  before/after comparisons from debugging a mask-radius bug that was
  producing streak-shaped unphysical fits on some objects.

## Status / known issues

- A bug where the fit mask was much larger (~14.5") than needed for this
  sample's Einstein radii (0.7-4") let some fits reach unrelated bright
  objects near the cutout edge and produce streak-shaped unphysical
  solutions — fixed by tightening the mask to a fixed 7" radius. See the
  before/after images in `residual_images/`.
- A small subset of objects (~6/107) still show a real concentric
  ring-shaped residual at the lens centre after that fix — likely genuine
  source structure a single Sersic-core profile can't represent. A
  pixelized-source reconstruction (`code/fit_lens_model_pixelized.py`) is
  in progress for these.
- A small fraction of fits intermittently hang (silent, no progress) in the
  nested-sampling stage for reasons not yet root-caused; affected objects
  are retried.

This is in-progress work, not a final/published result set.
