# Pipeline known issues and next steps

Cross-cutting concerns that span lenses, or belong to the pipeline code rather than any one
system's data. Per-lens results, convergence status and modelling judgements stay in
`claude/<LENS>/README.md` / `RESULTS_*.md`, which link here for anything that follows the
tooling or the dataset rather than one fit.

---

## 1. `run_fit.py --stages hmc` overwrites `metadata.json` with argparse defaults

A run that re-runs only the sampling stage (`--stages hmc`) writes `metadata.json` fresh from
its own argument list, which never mentioned the MAP/SVI stages it did not re-run — so those
stages' fields get overwritten with argparse **defaults**, not the values actually used.

**Confirmed instance:** `claude/DESI257/runs/r2_s2l2/metadata.json` currently claims
`svi_peak_lr: 0.002` (the argparse default). The SVI that actually produced the checkpoint
used **1e-3**, per `runs/r2_s2l2.log`'s own printed schedule line — the log, not the metadata,
is authoritative for that run. See `claude/DESI257/RESULTS_v2.md` for the full record.

Metadata that silently misdescribes how a result was produced is a provenance hazard as
serious as a wrong number in the result itself, since every read-only tool
(`diagnose.rebuild_model` and everything built on it) trusts `metadata.json` to reconstruct
the model. **Not yet fixed.**

**Fix:** stages not re-run in a given invocation should preserve the previous run's recorded
metadata values for those stages, rather than emitting argparse defaults over them.

---

## 2. MCLMC frozen-chain failure mode

An 80,000-draw / 10,000-burnin MCLMC re-sample on DESI257 (`claude/DESI257/runs/r2_s2l2`,
attempted from the same MAP+SVI checkpoint as the working 10,000-draw run) came back with
every chain frozen:

| | good 10,000-draw run | frozen 80,000-draw run |
|---|---|---|
| mean within-chain sd | 0.08155 | **0.0006021** (135x smaller) |
| between-chain sd of chain means | — | 0.09816 (unchanged in scale) |
| all values finite | yes | yes |
| R-hat | — | **7.86e15** |
| bulk-ESS | — | 50 (= chain count; one effective sample per chain) |

All values finite, and the between-chain spread stayed at the real posterior's scale — the
signature of 50 independent point-samples with no internal exploration, not 50 chains
converging to a point. The frozen checkpoint is preserved, not deleted:
`claude/DESI257/runs/r2_s2l2/hmc_80k_FROZEN_do_not_use.npz`.

**Cause unattributed.** Two variables changed at once relative to DESI234's successful
40,000-draw run (`claude/DESI234/RESULTS_v2.md`, burnin 5,000, two GPUs): this attempt used
burnin **10,000** and a **single** GPU. Either the longer burn-in degenerated MCLMC's
step-size adaptation, or the single-device chain layout did (recall from DESI238 round 5 that
HMC and MCLMC/MAMS use different chain-axis layouts, and a silent mismatch there produces
plausible-looking wrong output rather than an error).

**Next step, and it is a one-variable test:** 80,000 draws, burnin 5,000, on two GPUs — i.e.
DESI257's failed configuration with only the burn-in and GPU count changed to match DESI234's
working configuration. This isolates which of the two changed variables is the cause. This is
a pipeline robustness question, since MCLMC is used for every lens; it happened to surface on
DESI257 and could recur on any of them.

---

## 3. The mask-aware chi-squared bug, and its fix

`diagnose.residual_figure` and `plot_overview.py`'s panel used to average normalised squared
residuals over **every** pixel, including masked ones. Since a masked interloper carries the
field's largest residuals, this inflated reported chi2_nu badly: DESI234's round-1 value read
4.225 instead of a true 1.560; DESI257's read 7.112 instead of 1.533.

Both call sites now share one helper, `diagnose.masked_chisq`, which sums `(residual/sigma)^2`
over unmasked pixels only, divided by the unmasked count — matching `ProbModel.log_prob`'s own
second return value exactly, per `CLAUDE.md`'s rule that gigalens is the ground truth.
`claude/pipeline/tests/test_masked_chisq.py` (moved out of `verify.py`, was check 7, on
2026-08-09 — it asserts pipeline correctness, not something per-lens) exercises the fixed
function directly (not a reimplementation of the formula) and would fail if either call
site regressed.

Per-lens files may keep a one-line note that their own chi-squared values use the corrected
definition — that is a fact about their numbers — but this is the general lesson: **a
chi-squared computed outside `diagnose.masked_chisq` (or `ProbModel`'s own return value) is
suspect until shown otherwise.**

---

## 4. The four-panel `residuals.png` format and the `plot_overview.py` consolidation

`residuals.png` is now a four-panel figure: observed image + critical curve, model + critical
curve, unlensed source + caustics (with the ray-traced cutout border overlaid, dashed), and
the mask-aware normalised residual. `diagnose.residual_figure` and `plot_overview.py` used to
be two separate implementations of the same panel; they now share `diagnose.masked_chisq` (see
item 3) and the same critical-curve/caustic machinery (`LensSolver.critical_and_caustics`,
`image_border_in_source_plane`, both from `gigalens.jax.analysis` — nothing here reimplements
a Jacobian or a caustic by hand). See `diagnose.residual_figure`'s docstring for the full
rationale of each panel.

---

## 5. Cross-lens: a pixel-scale checkerboard at the deflector core

**The observation, per lens:** DESI234's and DESI257's `residuals.png` both show a pixel-scale
alternating +/-sigma checkerboard at the deflector core, reaching roughly +/-5 sigma. See each
system's own `RESULTS_v2.md` for its residual panel and chi-squared.

**The cross-cutting inference, stated here because it spans both systems:** a checkerboard
that alternates sign pixel-to-pixel is the signature of a **PSF mismatch**, not an inadequate
light profile — a light-profile error smooths out over several pixels, it does not alternate
sign at the pixel scale. The same pattern at the same location relative to the deflector core,
appearing independently in two different lenses fitted with two different delivered PSF
kernels, points at a problem shared by the dataset rather than a per-lens light-profile
failure.

**Precedent and recommended remedy:** `claude/DESI238/PSF_REPLICATION.md` documents rebuilding
that system's PSF directly from its own parent frame with `claude/pipeline/make_psf.py`,
rather than trusting an undocumented delivered kernel — and found three defects in the
delivered PSF that way, including a real 0.008" astrometric bias with no prior record.
**Rebuilding the PSF for DESI234 and DESI257, following that validated procedure, is the
highest-value next step for both systems** — ahead of adding more source-light freedom to
either fit, since a PSF mismatch at the core could itself be inflating the apparent need for
arc-level structure by biasing the fit's flux allocation.

---

## Next steps, ordered by value

1. **Rebuild the PSF for DESI234 and DESI257** (item 5) — the highest-value remaining step for
   both systems' fits, following the validated `claude/DESI238/PSF_REPLICATION.md` procedure.
2. **Isolate the MCLMC frozen-chain cause** (item 2) with the one-variable test: 80,000 draws,
   burnin 5,000, two GPUs on DESI257.
3. **Fix `run_fit.py`'s metadata-overwrite bug** (item 1): a `--stages hmc`-only re-run should
   preserve prior stages' recorded metadata rather than overwrite it with argparse defaults.
