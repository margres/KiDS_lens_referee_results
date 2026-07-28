# Known issues in these results

Of 107 known lenses: **87 ok, 16 elevated chi2/pix, 4 missing entirely.**
Full per-object list in `results_issues.csv`. "ok" here means chi2/pix <= 5,
not a guarantee of a perfect fit — treat as a triage flag, not a final QA pass.

## Missing (4) — never completed

`KiDSDR4_J144950.700+005536.65`, `KiDSDR4_J144357.047-011030.17`,
`KiDSDR4_J121131.442-011118.67`, `KiDSDR4_J122456.016+005048.05`

The Nautilus non-linear search hangs on these with **zero progress ever**
(no sampling status output at all, even after 10+ hours) — a silent,
intermittent, not-yet-root-caused deadlock, most likely in the
multiprocessing worker pool. Reproducible on retry for the same objects
(same 3 hang every time; not random). Not a data problem — the input FITS
cutouts for these look normal (no NaN/Inf/zero-flux issues). Still being
investigated.

## Elevated chi2/pix (16)

Highest first: J114811 (130), J141741 (86), J085156 (84), J091940 (45),
J120940 (26), J115708 (13), J121327 (12), J121339 (11), J091906 (9.3),
J133554 (7.3), J085207 (6.9), J090150 (6.0), J141829 (5.7), J120743 (5.7),
J120906 (5.7), J122800 (5.4).

For the objects visually reviewed so far (J120940, J115708, J121327,
J091906, J133554, J120743, J120906, J122800 — see
`residual_images/all_107_normalized_residuals_numbered.png`), the elevated
chi2 comes from a **real concentric ring-shaped residual at the lens
centre**, not a fitting bug: a single Sersic-core source profile can't
represent the true (more complex/ring-like) source structure. A
pixelized-source reconstruction is in progress
(`code/fit_lens_model_pixelized.py`) to address this for the worst cases.

J114811, J141741, J085156, J091940, and J121339 have **not yet been
visually reviewed** — they're flagged here purely on the chi2 number, cause
unconfirmed.

## What this does and doesn't affect

Arc S/N (`snr_arc_model`/`snr_arc_peak`, the paper's headline quantity) is
computed from the **data** with only the fitted *lens light* subtracted,
not the full lens+source model — so it's mostly robust to a poorly-fit
*source* shape (the elevated-chi2 class above). A bad *lens light* fit
would bias it directly; none of the flagged objects are known to have that
specific problem, but it hasn't been explicitly checked object-by-object.

_Snapshot as of this push — see the parent repo's commit history for
anything more recent than this file._
