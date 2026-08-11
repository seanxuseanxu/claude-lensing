# Pipeline known issues and next steps

Cross-cutting concerns that span lenses, or belong to the pipeline code rather than any one
system's data. Per-lens results, convergence status and modelling judgements stay in
`claude/<LENS>/README.md` / `RESULTS_*.md`, which link here for anything that follows the
tooling or the dataset rather than one fit.

**How to read this file.**

- **The item numbers are permanent.** Around thirty places across `claude/` cite them as
  "`KNOWN_ISSUES.md` item N" — three lens directories' results files, `config.py` comment
  blocks, `analysis/` scripts, `run_fit.py` and the guide. An item is never renumbered and a
  closed one is never deleted, only condensed. New findings take the next free number.
- **A closed item is a summary, not a transcript.** Once a defect is fixed, what is kept is
  the problem, the fix, and where the regression test lives. The measurement that found it
  lives in the lens file that made it, cited here.
- **The measurements belong to the lens.** This file states the cross-cutting inference and
  names the system it was measured on; the numbers themselves are in that system's own
  `RESULTS_*.md`. If you want the evidence, follow the pointer.

Open items, needing work: **2** (MCLMC frozen chains), **5** (residual reporting change),
**6** (MAP step count), **9** (a sizing rule to apply, not a bug). Everything else is closed.

---

## 1. `run_fit.py --stages hmc` overwrote `metadata.json` with argparse defaults — FIXED

A run that re-ran only the sampling stage wrote `metadata.json` fresh from its own argument
list, which never mentioned the MAP/SVI stages it did not re-run — so those stages' fields
were overwritten with argparse **defaults** rather than the values actually used. The record
of *how the surrogate was optimised* was corrupted; the physical decoding of the run was not,
since `rebuild_model` reads none of the affected fields.

**Fixed 2026-08-09.** `run_fit.py` merges through `merge_prior_metadata(config,
prev_metadata, stages)`: fields map to stages in one auditable dict literal
(`STAGE_METADATA_FIELDS`), and any stage not in this invocation's `--stages` keeps the
previous file's recorded value. A field absent from the previous metadata is not fabricated.
`stages` always records this invocation's own stages; `timings` merges per stage. Tested in
`tests/test_diagnose_and_metadata.py`.

The confirmed instance is DESI257 `r2_s2l2`, five fields wrong. The corrupted-versus-true
table, and the reasoning that `n_devices` is *not* part of the bug, are in
`claude/DESI257/RESULTS_v2.md` §2. That file is left as it stands, wrong fields and all —
rewriting a finished run's provenance by hand is a worse habit than carrying a documented
error.

---

## 2. MCLMC frozen-chain failure mode — OPEN, cause unattributed

MCLMC runs sometimes return chains that never move: every value finite, between-chain spread
at the real posterior's scale, within-chain sd collapsed by two orders of magnitude. That is
the signature of N independent point-samples with no internal exploration, not of N chains
converging on a point.

**`bulk ESS ≈ n_chains` is the diagnostic to key on**, not R-hat's magnitude. Two instances:

| | DESI257 80k (`RESULTS_v2.md` §4) | DESI234 `r3_unmasked` (`RESULTS_v3.md` §5.4) |
|---|---|---|
| burn-in / devices | 10,000 / 1 | **5,000 / 2** |
| between-chain sd ÷ within-chain sd | 163 | 15.4 on `theta_E`; 20.9 at worst |
| bulk ESS | 50 (= chain count) | 50.7 (= chain count) |
| max R-hat | 7.9e15 | 13.9 |

**Both originally suspected variables are excluded by counterexample.** DESI234's freeze
happened at burn-in 5,000 on two GPUs — the configuration nominated as the control for
DESI257's failure — so neither the long burn-in nor the single-device chain layout is the
cause. DESI234's freeze is far milder in R-hat yet identical in kind, which is why the ESS
signature and not R-hat is the thing to test on.

**The discriminating test, and it is cheap:** partition the chains by their per-chain medians
and recompute R-hat within each cluster. Frozen chains keep `bulk ESS ≈ n_chains_in_cluster`
under *every* partition; genuinely multimodal chains produce at least one cluster with R-hat
near 1. This separates two failures with opposite fixes, and it costs one script reading a
finished checkpoint (`claude/DESI234/analysis/mode_structure.py`).

**What the counterexample points at.** `r3_fg` — same lens, same launcher, same burn-in,
same devices, 46 parameters instead of 36 — did not freeze. So the trigger is neither the
configuration nor the machine but the posterior geometry the sampler was handed;
`r3_unmasked` is the model with a 110-sigma object in the field and no component to put on
it, which the data rejects by Δchi² ≈ 3700. A plausible reading is that MCLMC's step-size
adaptation degenerates on a badly misspecified target. That is a hypothesis, not a
measurement. **Next step: capture MCLMC's adapted step size and trajectory length from a
frozen run and compare against a healthy one on the same lens** — the freeze is almost
certainly visible there rather than in the samples. Try float64 first; see the next-steps
list.

---

## 3. The mask-aware chi-squared bug — FIXED

`diagnose.residual_figure` and `plot_overview.py` averaged normalised squared residuals over
**every** pixel, including masked ones. Since a masked interloper carries the field's largest
residuals, this inflated reported `chi2_nu` badly — DESI234 round 1 read 4.225 against a true
1.560 (`DESI234/RESULTS_v2.md`), DESI257 read 7.112 against 1.533 (`DESI257/RESULTS_v2.md`).

Both call sites now share `diagnose.masked_chisq`, which sums `(residual/sigma)^2` over
unmasked pixels only, divided by the unmasked count — matching `ProbModel.log_prob`'s own
second return value exactly, per `CLAUDE.md`'s rule that gigalens is the ground truth.
`tests/test_masked_chisq.py` exercises the fixed function itself, not a reimplementation of
the formula, and would fail if either call site regressed.

The general lesson: **a chi-squared computed outside `diagnose.masked_chisq` (or
`ProbModel`'s own return value) is suspect until shown otherwise.**

---

## 4. The four-panel `residuals.png` format — CLOSED, this is now just the format

`residuals.png` is a four-panel figure: observed image + critical curve, model + critical
curve, unlensed source + caustics (with the ray-traced cutout border overlaid, dashed), and
the mask-aware normalised residual. `diagnose.residual_figure` and `plot_overview.py` used to
be two implementations of the same panel; they now share `masked_chisq` (item 3) and the same
critical-curve machinery (`LensSolver.critical_and_caustics`, `image_border_in_source_plane`,
both from `gigalens.jax.analysis` — nothing here reimplements a Jacobian or a caustic by
hand). `residual_figure`'s docstring carries the rationale for each panel.

---

## 5. The pixel-scale checkerboard at the deflector core — CLOSED as an anomaly, open as a reporting change

DESI234's and DESI257's residuals both show a pixel-scale alternating +/-sigma checkerboard at
the deflector core, reaching roughly +/-5 sigma (each system's `RESULTS_v2.md`). Two rounds of
work went into explaining it, and the explaining was the mistake.

**The resolution (2026-08-10), from the repository owner:** a pixel-scale residual pattern at
the core is common when modelling deflectors with Sersic profiles, and is taken by their
modelling team to be a PSF artifact — never having been given a deep dive, because in practice
it does not need one. **The published convention is to mask the centre when computing
chi-squared, but not when modelling.** The mass model still sees those pixels and is
constrained by them; only the reported goodness-of-fit excludes them.

**What was measured before that, and still stands.** Both leading explanations were tested and
neither survived, which was worth having:

- *Not a stale PSF.* `psf234.json` records `psf234.fits` as already rebuilt from DESI234's own
  parent frame by the *fixed* `make_psf.py` on 2026-08-08. The checkerboard persists.
- *Not a supersampling artifact.* `claude/DESI234/analysis/undersampling_check.py` compares
  `diagnose_undersampling`'s predicted `delta/sigma` map against the observed residual at the
  same parameter point: observed peak 6.582 sigma, predicted 2.446 sigma, **Pearson r = −0.109**.
  The predicted quadrature error is a smooth single-signed blob; the observation is sharp
  pixel-scale sign alternation. They share a location and nothing else.
- *But DESI234 is genuinely undersampled at `supersample = 2`* — a separate finding, and the
  reason round 3 moved to ss=4. Full numbers in `DESI234/RESULTS_v3.md` §3.

**The lesson is about novelty, not about pixels.** Deciding that something needs explaining is
a judgement made before any computation, and it was made here against no reference for what a
normal residual map looks like — this repository is three systems fitted by one pipeline, so
its idiosyncrasies read as anomalies. One question to someone with field experience closed an
item that two rounds of measurement had left open. **Ask what is normal; compute what is
specific.**

**Consequences to carry:**

- `chi2_nu` as currently reported includes the core pixels and is therefore inflated on every
  run of every system. Cross-run comparisons remain valid — the contamination is common to all
  of them — but the absolute values are not a fit quality anyone else would recognise.
  **Reporting a centre-masked `chi2_nu` alongside the current one is the fix, and is the one
  piece of this item still outstanding.**
- Do **not** mask the core during modelling. That discards the pixels that most constrain the
  deflector's centroid and inner slope.
- Rebuilding DESI257's PSF, and the per-pixel kernel diagnostic that replaced it, are both
  withdrawn as priorities. Legitimate curiosities, not blockers.

---

## 6. The MAP stage can stop hundreds of nats short of the mode, silently — CHECKED, not solved

**The failure.** An optimiser can finish far below the best draw of the chain it seeds, while
its own loss curve still looks like it is descending. On DESI257 `r2_s2l2` the stored
`best_lp` is **681.8 nats below** the maximum scored sampler draw, and below the *typical*
draw, with a `chisq_hist` tail slope of −0.00018/step — still descending when it stopped
(`DESI257/RESULTS_v3.md` §1). DESI234 round 2 showed the same thing an order of magnitude
smaller, 86.5 nats (`DESI234/RESULTS_v3.md` §2).

**Why it matters more than a bad point estimate.** SVI starts from the MAP point and the
sampler is seeded from SVI, so a MAP that stopped in the foothills propagates into where every
chain begins. On DESI257 that is measurable: 48 of 50 chains sit on one side of a mode boundary
at post-burn-in draw 0, so round 2's quoted 73.4% "dominant mode" fraction is substantially a
measurement of the seed rather than of the posterior (`DESI257/RESULTS_v2.md` §3).

**Now checked, not assumed.** `diagnose.map_convergence_check` (called from `diagnose.main`)
recomputes the MAP point's log posterior, scores a thinned set of sampler draws through the
same `log_prob`, prints the `chisq_hist` tail slope, and FAILS loudly with the deficit in nats
when `map best_lp < max(sampler log posterior)`. `map_hmc_verdict` is the pure predicate behind
it, tested in `tests/test_diagnose_and_metadata.py` on both the passing case and this measured
failing one.

**The remedy is `--map-steps`, and it has to be measured per model.** 350 (the argparse
default) and 1500 both fell short at 36 parameters. Raise the count until `best_lp` plateaus
*and* clears the best draw of the sampler it seeds, and record which value did it. This stays
open because there is no automatic step-count selection — only a check that tells you the
number you chose was too small.

---

## 7. `boundary_pinning_check` saw only `Uniform` priors — FIXED, with a residual limit

The check inspected only `tfd.Uniform`. `gamma`'s prior is `TruncatedNormal(2.0, 0.25, 1.0,
3.0)`, so a posterior pressing on the 3.0 ceiling was never reported; round 1's `n_sersic`
pinning was caught only because that parameter's prior happened to be Uniform.

Fixed 2026-08-09: `TruncatedNormal` bounds are read off `.low`/`.high` and reported alongside
the Uniform ones, each row labelled with its family so a hard bound is distinguishable from a
truncation. Unbounded priors stay excluded on purpose — see 7a.

**The residual limitation, stated because it bit on the run that motivated the fix.** The check
tests the **median** of the pooled draws, so it cannot see a mode it is not centred on. DESI257
`r2_s2l2`'s `gamma` prints at 76.8% of range and is not flagged — correctly, for a median —
while its minority mode reaches 2.885 against a 3.0 ceiling. Read the pinning table together
with the R-hat column: `gamma`'s R-hat there is 1.453, which is the warning that a single
summary statistic is not describing that parameter.

### 7a. The larger blind spot: unbounded priors — FIXED 2026-08-10

Excluding `Normal` and `LogNormal` is defensible for a *pinning* test — there is no bound to
pin against — but it left the pipeline with no check at all on the parameters carrying those
priors, and that is where the worst fit on this project hid. DESI234 `r3_fg` fitted a deflector
axis ratio of `q = 0.143`, a 7:1 needle, under a `Normal(0, 0.15)` ellipticity prior, alongside
a near-singular `gamma = 2.951 ± 0.005`. Nothing reported either. `chi2_nu` was 1.082, the best
on the system; R-hat was 1.92, which was read as a sampling problem
(`DESI234/RESULTS_v3.md`, correction banner).

`diagnose.prior_posterior_tension` closes this. It measures the **prior quantile of the
posterior median**, converted to equivalent Gaussian sigma, so it works for every prior family
this pipeline uses — bounded, unbounded and skewed alike — and reports the posterior/prior sd
ratio beside it.

**Pinning and tension are different questions, and the difference is worth holding: pinning
means the prior set the value; tension means the data set it, against the prior's advice.** The
second is the commoner signature of misspecification — a parameter absorbing an error made
somewhere else — and it is invisible to every aggregate statistic.

Reported, never raised. An atypical system is a thing that exists, and plausibility bands stay
flexible here by the same reasoning as `priors.py`'s. What the check buys is that the atypical
case must be noticed and argued for instead of passing silently into a results table.

---

## 8. `mask_circles` is a per-run choice, and is now read back from the run — FIXED

`diagnose.rebuild_model` used to build the likelihood mask from the lens config as it stands
*today*, on the reasoning that a mask is a fact about the field like the crop. DESI257 broke
that: its SW object was masked in rounds 1–2 and unmasked from round 3, so the same config
cannot describe both, and diagnosing an old run against the new config would silently compare
a different likelihood than the one that was fitted.

As of 2026-08-09 `rebuild_model` uses the **recorded** `mask_circles` when present, falls back
to the config only for runs predating the key, prints a `NOTE:` when the two differ, and
raises if the rebuilt mask's kept-pixel count disagrees with the recorded `n_masked_pixels`.
That last check is what makes the reconstruction verifiable rather than hopeful.

General lesson: **anything that changes the *likelihood* — the mask, the noise model, the crop
— must be recorded per run and read back from the run, not from today's config.** A prior's
support is already handled this way (`priors.py`'s versioning); the mask was the gap.

---

## 9. Raising `supersample` costs memory as ss², and the MAP/SVI batch buffer is per-device

Not a bug — a sizing rule, found the expensive way on DESI234 round 3 and stated here because
it applies to every lens. With `supersample` raised 2 → 4, a previously working
`--n-samples 4096` died with `RESOURCE_EXHAUSTED ... 178.02GiB [jit_run_map]` on **two** 80 GB
A100s; halving to 1536 asked 67.07 GiB and still failed.

**The second failure is the informative one.** 67 GiB is comfortably under 2 × 80 GB, so the
allocation is not sharded across devices — the MAP/SVI batch buffer is allocated **per
device**, and the ceiling is one card after JAX's 75% preallocation, about **61 GiB**. Adding
GPUs does not raise it. The scaling is clean enough to size a run by arithmetic:

| | cost/sample | 4096 | 2048 | 1024 |
|---|---|---|---|---|
| ss=2 | 10.9 MiB | 44.6 GiB | 22.3 GiB | 11.2 GiB |
| ss=4 | **43.6 MiB** | 178 GiB ✗ | 89 GiB ✗ | **43.6 GiB** ✓ |

**The rule: raising `supersample` by a factor k divides the affordable `--n-samples` and
`--n-vi` by k².** DESI234 round 3 went 4096 → 1024 and 2048 → 1024, landing on the same
43.6 GiB footprint round 2 ran at, and paid for the smaller batch with `--map-steps`
1500 → 5000. **Smoke-test `--stages map,svi` with `--map-steps 3 --svi-steps 3` on the
largest model variant before launching a long fit**; it takes under a minute and it is the
difference between finding this at launch and finding it an hour in. This applies to DESI257
whenever it follows DESI234 to ss=4.

---

## Next steps, ordered by value

Closed items are not repeated here; see the item bodies. What remains:

1. **Report a centre-masked `chi2_nu` alongside the current one** (item 5). Every `chi2_nu` in
   this repository is inflated by core pixels the field excludes when reporting fit quality.
   This is a reporting change in `diagnose.masked_chisq`'s callers, not a modelling change —
   the core stays in the likelihood.

2. **Re-fit DESI234 round 4 with the deflector ellipticity constrained** (items 5, 7a).
   `r3_fg` is rejected: `q = 0.143`, `gamma = 2.951`. The question round 4 asks is whether a
   foreground component survives when the deflector is required to be round. Until it is
   answered, nothing about the interloper's mass is established.

3. **Re-run `prior_posterior_tension` over every archived run on all three systems.** New as of
   2026-08-10 and applied only to DESI234. DESI238's converged result and DESI257's rounds have
   never been checked for a parameter sitting far out on an unbounded prior, which is exactly
   the failure it was built to catch.

4. **Isolate the MCLMC freeze** (item 2) by capturing the adapted step size and trajectory
   length from a frozen run and a healthy one on the same lens. Run `mode_structure.py`-style
   within-cluster R-hat on any future R-hat failure before calling it multimodal.

   **Try float64 first.** MCLMC is meant to work with little tuning beyond sampling lengths,
   and freezing usually indicates something else — poor preconditioning or bad geometry. The
   repository owner reports that float64 helps MCLMC adaptation specifically. Every run in this
   project has been float32: `data.py` pins `likelihood_precision="float32"` deliberately, to
   hold arithmetic fixed across the scene-API port. Raising it costs roughly 2x forward memory
   and compute and needs `JAX_ENABLE_X64=1` process-wide, and it is a cheaper experiment than
   instrumenting the sampler.

   Note from the MCLMC paper (`papers/mclmc/NOTES.md`): it says nothing about floating-point
   precision, so float64 is an empirical bet, not a prescription. It does make one sharp,
   checkable claim that bears on this — drift and diffusion each separately preserve the
   stationary distribution, so **the step size is the only source of bias**, never the
   decoherence parameter. Its geometric-ergodicity proof assumes a globally log-convex target
   with bounded gradients, which a lens posterior is not, so the paper's convergence-rate
   guarantee does not cover our runs at all. And the algorithm as published has no mass matrix;
   gigalens seeds one from the SVI surrogate, so the implementation is already outside the
   paper.

5. **Measure `--map-steps` per model variant** (item 6). The check reports the deficit; nothing
   chooses the step count. 350 and 1500 both fell short at 36 parameters.

---

## Closed since this file was last reorganised (2026-08-11)

- **An ESS floor is now enforced.** `diagnose.ess_floor_report` checks Vehtari et al.'s three
  conditions (`papers/r-hat_convergence/NOTES.md` §4) — ≥ 50 per split half, ≥ 400 total, ≥ 4
  chains — and prints a pass/fail line for each *before* the R-hat verdict it qualifies. When
  R-hat passes and the floor does not, the summary says the run is undetermined rather than
  converged, since below the floor a passing R-hat is uninformative too. Constants
  `ESS_FLOOR_TOTAL`, `ESS_FLOOR_PER_SPLIT_HALF`, `MIN_CHAINS` in `diagnose.py`; tested in
  `tests/test_ess_floor_and_plots.py`, including the case the floor exists for — 4 chains × 32
  iid draws, R-hat 1.025, floor failed.

  It correctly refuses DESI234 `r3_unmasked` (min ESS 51) and DESI257 `r2_s2l2` (min ESS 66,
  0.7 per split half against 50 chains). The second was not expected: `r2_s2l2` was assumed on
  first writing to clear the floor comfortably, and direct measurement — max R-hat 1.461, min
  bulk ESS 95, min tail ESS 66 over 50 × 10,000 draws — says otherwise. Its R-hat failure was
  already known; what is new is that the run is below the floor at which R-hat can be read at
  all, so it is undetermined rather than merely unconverged.

- **The two diagnostic plots Vehtari et al. recommend now exist**, both run by `diagnose.main`
  on every invocation, both computed from chains already on disk (`NOTES.md` §5, §7).

  - **ESS vs draws** (`ess_vs_draws.png`): min(bulk, tail) ESS at eight geometrically spaced
    draw prefixes, for the six worst-R-hat parameters, against a linear-growth reference. The
    log-log slope is fitted over the upper half of the range and the verdict printed in words,
    so the decision does not depend on eyeballing the figure. ~0.3x the cost of the full ESS
    pass, because it carries six columns rather than all 36.
  - **Rank plots** (`rank_plots.png`): pooled-rank histograms, one panel per chain, for the
    same six parameters. Above ten chains it draws the ten least uniform by chi-square — a
    first-N slice would drop the single bad chain whenever it is not near the front, which is
    the one case the figure exists for. Ranks are always pooled over all chains; only the
    drawing is subset. Hand-computed rather than via `arviz.plot_rank`, which in arviz 1.2
    draws a rank Δ-ECDF and returns a `PlotCollection`, not this histogram.

  Item 2's frozen-chain investigation was the first customer and both delivered: `r3_unmasked`
  gives ESS 51 flat (slope −0.02) and one rank spike per chain — the chains did not move, and
  no additional draws would have helped. `r2_s2l2` gives slope −0.4 on all six worst
  parameters, which is the same verdict from the other direction: a fit whose problem is the
  model or the surrogate, not the run length. Together they add ~30 s to a 172 s diagnosis on
  the 2M-draw run.

- **Redundant pipeline code removed 2026-08-11.** `data.build_prob_model` is now the one place
  the `ImageData` → `ProbModel` construction happens; `run_fit.py` (which fits), `verify.py`
  (which pre-flights) and `diagnose.rebuild_model` (which reconstructs a finished run) each
  reach it by a different route but build the same likelihood, which is the property that
  matters — a pre-flight that checks a different likelihood than the fit uses is worse than no
  pre-flight. `make_cutout.first_image_hdu` is likewise shared with `make_psf.py`, so a cutout
  and a PSF cannot be cut from different extensions of the same frame. `run_fit.load_hmc_checkpoint`
  was dead and is gone. `priors.py`'s near-duplicate `_lens_light_priors_v1`/`_v2` were
  deliberately **not** collapsed: `README.md`'s versioning rule requires a superseded prior
  builder be kept verbatim, since editing one in place silently rewrites every old run's
  physical values.
