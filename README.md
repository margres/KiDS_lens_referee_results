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
- `ugri/results/` — multi-band fit summaries, 199/223 present (see Status).
- `reference_csvs/` — the known-lens ground truth, the candidate list, and
  the recovered-vs-missed breakdown (shared by both tracks).

## Status (as of this push)

**Update (2026-08-02): two real methodology bugs found via user visual QA on
the numbered chi2-bin mosaics, both fixed and rolled out.**

1. **Annulus bug**: `chi2_per_pix` and the arc-S/N were both computed inside
   a *fixed* `[0.5,3.5]"` annulus, independent of each object's fitted
   Einstein radius (0.3-4.0" across the sample) — biasing not just the S/N
   number but *which fits counted as well-converged*. Fixed: annulus now
   scales `[0.6, 1.4] x einstein_radius` (min width 0.5"), recomputed for
   every completed fit directly from saved outputs (`galaxy_images.fits` +
   `dataset.fits` + `fit.fits`), no re-fitting needed. This flips the
   recovered-vs-missed S/N direction (still not significant either way) —
   the result below is reframed as a straightforward null result rather
   than "missed lenses are marginally brighter, contrary to expectation."
2. **Source-size bug**: `source_effective_radius` (background source size)
   was bimodal across ~44% of all parametric fits under the old uncapped
   `Uniform(0.02", 30")` prior — 35% landed at an implausible >2" diffuse
   blob (up to 30"), 8.6% collapsed to <0.05" (a point source unable to
   reproduce extended arc structure). Both extremes fit measurably worse on
   average. New `--source_re_prior tight` option in `fit_lens_model.py`
   (`TruncatedGaussianPrior(mean=0.2", sigma=0.15", [0.02",0.8"])`, a
   physically-motivated compact-source scale) was tested on every object
   showing this pathology; **kept only where it genuinely improved chi2**
   (roughly half the time — the rest reverted to the original fit, since a
   "more physical" source size doesn't guarantee a better fit). Same
   before/after comparison-and-revert policy applied to a point-source
   auto-masking feature (`--mask_point_sources`, photutils DAOStarFinder)
   for objects with a real unmodelled contaminant in the fit mask.
- `r_only/chi2_bin_mosaics/refit_review_mosaic.png` — new: RGB + new-attempt
  r-band data/model/residual for every object touched by these two fixes,
  labelled with old/new chi2 and the automatic keep/revert decision, using
  the *same* `#N` reference numbers as `chi2_bin_id_mapping.txt`.
- Result counts moved to 205/223 (93/107 known lenses); "good" tier grew
  111->119 objects.  **Referee-response comparison (chi2/pix<=2, n=46 of 93
  converged known lenses)**: recovered (n=15) median arc S/N=38.7 (IQR
  18.6-60.3); missed (n=31) median arc S/N=32.0 (IQR 19.0-56.4).
  Mann-Whitney p=0.74 (p=0.66 after trimming top/bottom 10% of each group)
  — no significant difference in either direction; the sign of the small
  residual gap is itself sensitive to reasonable methodology choices,
  underscoring this is a genuine null result.
- Not all fit batches from this fix are finished yet (the decent-tier
  extension of the source-size-prior test was still running as of this
  push) — expect another update.

**Update (2026-08-01): r_only/results/ now published as "best-of"
(parametric vs. pixelized), the actual referee-response arc-S/N result
computed, and new combined RGB+data+model+residual mosaics added.**

- `r_only/results/` now holds, per object, whichever of the parametric or
  pixelized-source fit has the lower reduced chi2 (`results_best/` on the
  cluster side; `source_model` field in each JSON says which one won) —
  204/223 total, 92/107 known lenses. This supersedes the parametric-only
  results from the previous push for the ~38 ring-residual objects where
  pixelized won.
- `reference_csvs/known_lenses_recovered_vs_missed_snr.csv` — the per-known
  -lens table behind the referee-response comparison: Grade, recovered/
  missed, chi2/pix, snr_arc_model, source_model, and a
  `well_converged_chi2le2` flag.
- **The actual referee-response result**, restricted to well-converged fits
  (reduced chi2/pix <= 2, n=46 of the 92 converged known lenses): recovered
  lenses (n=12) have median arc S/N = 34.2 (IQR 24.9-51.6); missed lenses
  (n=34) have median arc S/N = 40.0 (IQR 21.3-67.0). Mann-Whitney U p=0.74
  (p=0.68 after trimming the top/bottom 10% of each group) — missed lenses
  are *not* systematically fainter; the difference is not significant in
  either direction. No evidence arc detectability limits the search's
  recovery rate.
- `r_only/chi2_bin_mosaics/mosaic_chi2_{good,decent,bad}.png` — new: one
  big image per chi2 tier (good <=2, decent 2-5, bad >5), one row per
  object sorted by chi2 ascending, four panels per row (RGB cutout from the
  KiDS DR4 PNG cache / observed data / model / normalized residual in
  sigma). This is the "diagnostic images regenerated against the latest
  fits" follow-up flagged as pending in the previous push.
- Several fit batches (missing-object backfill, mask-tighten retries, the
  pixelized-source queue, and a 5-object two-deflector group-model pilot)
  were still running on the cluster at the time of this push — counts above
  are a snapshot, expect another push once they land.

**Update (2026-08-02): all in-flight batches finished, results_best now
205/223 (93/107 known lenses); mosaics carry reference numbers.** The
2-deflector group model (`fit_lens_model_group.py`, not yet in this
repo's `code/`) won the best-of comparison for 3 of its 5 pilot objects
(e.g. J233430: chi2/pix 5.75 -> 1.94, moving it into the "good" tier) —
`source_model` in a result JSON can now say `group` as well as `parametric`
/`pixelized`. Each row in `r_only/chi2_bin_mosaics/mosaic_chi2_*.png` is now
labelled with a `#N` reference number (continuous across all three bin
images, ordered by chi2/pix ascending); `r_only/chi2_bin_mosaics/
chi2_bin_id_mapping.txt` maps every number back to its object ID, tier, and
chi2/pix — use that number to flag a specific object for a fix. The
recovered-vs-missed arc-S/N numbers above are unchanged by this update (the
new completions didn't add/remove anything from the chi2<=2 tier).

**r-only: 204 / 223 objects fit so far** (92/107 known lenses covered).
Remaining objects mostly hitting an upstream Nautilus numerical-instability
bug (see below) even through the retry wrapper; backfill still running.

**ugri: 199 / 223 objects fit** (94/107 known lenses covered) — **up from
0/223 at the last push.** The multiband track was not actually blocked by
a JAX-vmap incompatibility as previously believed (see below) — once the
real root cause was found and fixed, a full r+g+i joint fit dropped from
"doesn't converge in days" to **~26 minutes/object**, using the same
`number_of_cores=8` as the single-band pipeline (no extra CPU scaling
needed). 24 objects remain, running now.

**Root cause found 2026-07-30/31, and it explains BOTH the r-only silent
deadlock AND the multiband slowdown as the same bug**, not two separate
problems: `fit_lens_model.py`'s (and `fit_lens_model_multiband.py`'s)
version-gate for JAX was silently broken. Intent was "only enable JAX on
Python >=3.11 with jax importable, else run without it" — but the fallback
branch omitted the `use_jax` kwarg to `al.AnalysisImaging` entirely, and
PyAutoLens's own default when that kwarg is omitted is `True`. So this
Python-3.10 pipeline was silently running an unsupported JAX configuration
the whole time. `py-spy` couldn't attach (no ptrace permission on this
cluster), so used Python's stdlib `faulthandler` (`SIGUSR1` -> live stack
dump) instead to catch it stuck inside JAX/Nautilus-internal code showing
~0% CPU/GPU utilization. **Fix**: always explicitly pass
`use_jax=_use_jax` (never omit the kwarg). Verified via direct worker
`ps aux` inspection: genuine 75-79% CPU utilization across all 8 workers,
instead of ~0%. Tried the "do it properly" alternative too (real JAX on a
Python-3.11 GPU venv) — that also stalled, so the fix is disabling JAX for
this pipeline entirely, not using it correctly instead. All the earlier
24-core/60-core/100GB-memory scaling attempts for multiband
(`test_multiband_60cpu.sbatch` etc.) were solving the wrong problem and
are now obsolete — 8 cores was always enough once JAX was off.

A second, separate, upstream Nautilus bug also affects a handful of
objects across both tracks: `numpy.linalg.LinAlgError: Singular matrix`
inside Nautilus's own ellipsoid bound-fitting code
(`nautilus/bounds/basic.py:minimum_volume_enclosing_ellipsoid`) — a known,
previously-reported, not-fully-closed issue upstream (nautilus-sampler
issues #34/#35). No pip upgrade or config knob avoids it; the maintainer's
own advice is to retry with a different random seed. **Mitigation**:
`fit_one()` now wraps the fit in a retry loop (max 3 attempts, clearing
the crashed attempt's output first) — not a guaranteed fix, since a small
number of objects have hit the same crash on all 3 retries.

Other known issues in the r-only fitting pipeline (methodology, not
sample-specific — see `r_only/code/NOTES_ring_residual_issue.md` for
detail), relevant background for the multi-band effort too:
- A mask-radius bug that caused streak-shaped unphysical fits and
  12h-wall-clock non-convergence hangs on some objects — **fixed** (mask
  tightened from ~14.5" to a fixed 7"). Several objects needed a further
  tighten to 5" where the interloper sat closer than 7" (e.g. J085156,
  chi2/pix 83.6 -> 2.5). A systematic residual-pattern classification (peak
  structure near the mask edge = likely interloper, vs. peak at the lens
  centre = likely a genuine complex source) was run across all chi2>2
  objects on 2026-07-31 to route each one to the right fix automatically.
- A subset of objects show a genuine concentric ring-shaped residual a
  single Sersic-core source profile can't represent (real source
  complexity, not a bug) — a pixelized-source reconstruction
  (`r_only/code/fit_lens_model_pixelized.py`) is now running at scale for
  this class (was a 5-object pilot as of the last push).
- A small number of objects have a genuine second massive deflector near
  the primary lens (not just a foreground light source to subtract) — a
  two-deflector group-scale model is being piloted for these, based on
  PyAutoLens's own official group-lens example.

This is in-progress work, not a final/published result set. Per-lens
diagnostic images (`per_lens_diagnostics/`, `raw_fits/`,
`residual_images/`) are not yet regenerated against the latest fits in
this push — the `results/` JSONs above are current as of 2026-07-31, the
diagnostic images will follow in a subsequent push.
