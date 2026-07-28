# Open issue: ~6/107 known lenses have a ring-shaped residual, single-Sersic source isn't enough

Cluster-session note for whoever picks this repo up next (2026-07-26).

## What's fixed already, don't re-investigate
- The 14.5" fit mask was letting an uncapped-Re source reach field
  interlopers near the cutout edge -> streak-shaped unphysical fits, and
  (same root cause) some Nautilus runs hanging for the full 12h wall clock
  without converging (`N_eff` stuck at 1 for 300k+ likelihood calls).
- Fixed: `fit_lens_model.py` mask radius is now a fixed `--mask_radius`
  arg, default **7.0"** (was `0.9 * cutout_half_size` ≈ 14.5"). Confirmed
  on the two worst cases: chi2/pix 2151->5.45 and 3782->0.88.
- Rerunning all previously-bad/timed-out objects with this fix now
  (job 797876 on the cluster).

## What's still open -- this is the ask
After the mask fix, ~6 of 107 known lenses still have chi2/pix 7-25 with a
**concentric ring-shaped residual right at the lens centre** (not a streak
toward the field edge -- structurally different from the fixed bug).
Quantified via smoothed-normalized-residual central peak amplitude and
`%pixels with |residual|>3sigma` (should be ~0.3% for a clean fit):

| ID | chi2/pix | central_peak (sigma) | % pixels >3sigma |
|---|---|---|---|
| KiDSDR4_J120940.293-015050.60 | 25.55 | 13.11 | 19.0% |
| KiDSDR4_J115708.669-003646.33 | 12.87 | 9.62 | 10.0% |
| KiDSDR4_J121327.635-012023.85 | 11.97 | 7.84 | 8.0% |
| KiDSDR4_J091906.613-012949.31 | 9.26 | 5.69 | 10.4% |
| KiDSDR4_J133554.535-005014.06 | 7.32 | 4.73 | 8.2% |
| KiDSDR4_J133701.430-004057.97 | 3.52 | 4.33 | 4.8% (source_re railed at 29.82"/30" cap even with the tighter mask -- worth a second look, may still have a reachable interloper within 7", not yet checked) |

Read: these look like genuinely complex/ring-like real source structure
that a single analytic `SersicCore` light profile can't represent -- not a
bug, a model-flexibility limit.

**The documented PyAutoLens fix for this is a pixelized source
reconstruction**, not another prior/mask tweak:
`autolens_workspace/notebooks/imaging/features/pixelization/cpu_fast_modeling.ipynb`
(CPU-only, no GPU needed -- relevant since this cluster's venv is Python
3.10, no JAX acceleration available). It's a full SLaM chain:
1. `source_lp` -- parametric-source fit (MGE light profiles for lens+source)
   to get a robust starting mass model. Close to what `fit_lens_model.py`
   already does.
2. `source_pix_1` -- pixelized source, `mesh.RectangularAdaptDensity`,
   `reg.Adapt` regularization, mass priors chained from step 1.
3. `source_pix_2` -- refit with `mesh.RectangularAdaptImage` using adapt
   images from step 1.
4. `light_lp` -- refit lens light with mass+source fixed from step 3.
5. `mass_total` -- `PowerLaw` mass model (upgrade from Isothermal), lens
   light fixed from step 4.

Each stage also wires in `PositionsLH` (position-based likelihood penalty)
using multiple-image positions -- we don't currently supply positions
anywhere in `fit_lens_model.py`, would need a way to get those (peak
detection on the residual, or manual).

This is a materially bigger change than anything fixed on the cluster side
so far (new model classes, multi-stage chaining, adapt-image bookkeeping,
positions input) -- flagging rather than unilaterally rewriting the shared
fit script. Could use a second pair of eyes / a decision on whether it's
worth the complexity for ~6 objects, or whether "real complex lens, model
underfits it" is an acceptable caveat to note in the paper instead.

Full context/history: `project_kids_pyautolens_handoff.md` memory on the
cluster side (not in this repo). Current cluster state: 18/25 rerun batch
back, mask fix confirmed working for the streak/hang class, this ring
issue is what's left.
