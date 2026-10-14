# DESI234, round 3: the interloper, on a converged MAP

> ## CORRECTION, 2026-08-10 — this round's positive conclusions are withdrawn
>
> **`r3_fg`'s mass model is unphysical, and every claim below that rests on it falls with it.**
> Its main deflector has axis ratio `q = 0.143` — a 7:1 needle — and `gamma = 2.951 ± 0.005`,
> nearly singular, sitting 3.95 sigma out on a `TruncatedNormal(2.0, 0.25, 1, 3)` prior with a
> posterior 50x narrower than that prior. Galaxy-scale deflectors are round and near-isothermal.
> Every other run on this system is sane: `q = 0.736 / 0.900 / 0.943` and
> `gamma = 2.677 / 2.199 / 2.130` for `r1_s1l1` / `r2_s2l2` / `r3_unmasked`. `r3_fg` is the
> outlier, and adding the foreground component is what broke it.
>
> Specifically withdrawn:
>
> - **"The interloper is a foreground mass, and modelling it matters"** (§5.2, §5.4). The SIS
>   at `theta_E = 0.370"`, 12.2 sd from zero, is what a needle-shaped deflector requires nearby
>   to stay consistent with the arc. The statistic was computed correctly — the prior is a
>   `HalfNormal` with mode at zero, so zero was available and was excluded — but a sound
>   statistic on a rejected model measures nothing.
> - **The 0.066" shift in the main `theta_E`** (§5.2). That is the gap between a sane mass model
>   and a broken one, not the interloper's effect on the deflector.
> - **"The model without it is rejected by Δchi² ≈ 3700"** (§5.2, §5.4). The comparison is not
>   between two competitive models; it is between a physical model and an unphysical one that
>   fits better. `chi2_nu = 1.082`, the best on this system, was bought with `q = 0.143`.
> - **"Neither is a model problem"** and **"the fix is sampling, not modelling"** (§5.4, header).
>   True for `r3_unmasked`, whose chains are frozen with a sane mass model. False for `r3_fg`.
>
> **What survives.** §1's residual anatomy, §2's MAP-convergence work and §3's preflight are
> independent of `r3_fg`'s posterior and stand. So does `r3_unmasked`'s `theta_E = 1.5386`,
> which reproduces round 2 to 0.0007" — support for round 2, not a replacement for it, since
> its chains are frozen.
>
> **How it was missed, and how it is caught now.** The deflector's `e1`/`e2` priors are
> unbounded `Normal(0, 0.15)`, which `boundary_pinning_check` excludes by construction, so no
> check in the pipeline looked at them. The fitted values are `e1 = -0.689`, `|e| = 0.749` — a
> 7:1 needle where galaxy-scale deflectors are round. `diagnose.prior_posterior_tension` was
> added on 2026-08-10 and flags all three: `e1` at **4.59 sigma**, `src0/center_x` at **4.05**,
> `gamma` at **3.95**, the last with a posterior 50x narrower than its prior. See
> `claude/guide/05-diagnostics.md` §5 on why `chi2_nu` did not and could not catch this.
>
> The body below is left as written, per this project's habit of banner-then-keep rather than
> rewriting a finished run's record. Read it as the round-3 report plus this correction.

**Status: complete, and largely negative. Round 3 produces no `theta_E` and settles nothing
about the interloper.**

Both fits ran and **neither converged** (§5.1). §5.4 says why, and the two runs failed for two
different reasons: `r3_unmasked`'s 50 chains are **frozen**, `r3_fg`'s are **under-mixed along a
`theta_E` ridge**. Neither is the bimodality §4(b) predicted.

Beyond that, **`r3_fg` is rejected as unphysical**: its deflector is a 7:1 needle, axis ratio
`q = 0.143` (16/84: 0.134–0.153), with `gamma = 2.951 ± 0.005` against a
`TruncatedNormal(2.0, 0.25, 1, 3)` prior — 3.95 prior σ out, posterior 50× narrower than the
prior. Galaxy-scale deflectors are neither. So `r3_fg`'s non-convergence is the lesser problem;
the model it converged *towards* is wrong.

What the round establishes:

- **`r3_unmasked`'s mass model is sane** (`q = 0.943`, `gamma = 2.130`) and it lands at
  `theta_E = 1.5386"`, reproducing round 2 to **0.0007"** despite frozen chains. Two
  conditionings agreeing to under a milliarcsecond is the round's one durable result.
- **Round 2's `1.539 +0.003/-0.002"` stands** as the system's number, qualified by the
  unmodelled interloper mass — a known, unquantified systematic.

What the round does **not** establish — these were claimed in the original version of this
section and are **retracted** (2026-08-10); see the banner above:

- ~~The interloper is a foreground mass, and modelling it matters.~~ `r3_fg`'s foreground SIS
  at `theta_E = 0.370"`, 12.2 σ from zero, is a statistic computed on a rejected model. The
  statistic itself was sound — the prior is a `HalfNormal` with mode at zero
  (`pipeline/priors.py:190-212`), so zero was available and excluded — but soundness does not
  survive the model. A needle-shaped deflector needs extra nearby mass to stay consistent with
  the arc, which is a likelier reading of the detection than the interloper.
- ~~Masking biased the main deflector by 0.066".~~ That is the gap between a sane mass model
  and a broken one, not a measurement of the interloper's effect.
- ~~The model without it is rejected by Δchi² ≈ 3700.~~ `r3_fg` does fit better
  (`chi2_nu` 1.082 vs 1.662, the best on the system), but it bought that fit with an unphysical
  mass distribution. A likelihood-ratio argument presupposes both models are admissible.
- ~~Round 2 is superseded.~~ It is not.

**Round 4** is therefore a re-run of the foreground model, not a sampling fix: constrain the
deflector's ellipticity to a physical range, refit, and ask whether the foreground component
survives. §5.4's reading that the fix is sampling rather than modelling was made before
`r3_fg`'s mass model was examined and is wrong for that run; it remains right for
`r3_unmasked`, whose frozen chains are a genuine sampler problem (float64 is the first thing to
vary — `pipeline/data.py:118` pins float32).

Round 2 (`runs/r2_s2l2`, 36 parameters) converged at R-hat <= 1.008 with
`theta_E = 1.539 +0.003/-0.002"` but `chi2_nu = 1.300`, and `residuals.png` showed three
unmodelled features. This round asks one question — **is the masked object at 2.30" a
foreground interloper or a lensed image?** — and asks it on a MAP that has actually reached
the mode, with the round-2 residual features measured rather than described.

Every number below was produced by a script in `analysis/`, listed in `analysis/README.md`,
so it is reproducible rather than asserted.

---

## 1. Stage 0: what the round-2 residuals actually are

Three features were visible in `runs/r2_s2l2/residuals.png`. Round 2 attributed each to a
cause; Stage 0 tested those attributions before spending a fit on them. Two of the three
attributions did not survive.

### 1.1 The core checkerboard is not a supersampling artifact — but the model *is* undersampled

*(`analysis/undersampling_check.py`, `analysis/undersampling_certify.py`)*

`KNOWN_ISSUES.md` item 5 attributes the +/-5 sigma pixel-scale checkerboard at the deflector
core to a PSF mismatch and recommends rebuilding the PSF. **That recommendation is stale for
this system**: `psf234.json` records `psf234.fits` as already built by the *fixed*
`make_psf.py` from DESI234's own parent frame on 2026-08-08 (16 stars, 21 extended objects
rejected by the size cut, full provenance) — i.e. after the defects
`DESI238/PSF_REPLICATION.md` found were fixed. The pattern persists anyway, and persists
identically on DESI257, which has an independently built kernel.

The competing explanation was gigalens' `supersample >= 2` subgrid-kernel rendering
convention, whose quadrature error is worst where the model is cuspy — and DESI234's bulge is
`n_sersic = 5.27` at `R_sersic = 0.43"`. `gigalens.jax.analysis.diagnose_undersampling` tests
this directly, on the MAP point plus 8 posterior draws.

**The decisive test**, in a 10x10 px box at the deflector core (pixel (39, 39) =
`cfg.centroid_pix`), at the HMC-median point `residuals.png` itself was rendered at:

| | |
|---|---|
| observed normalised-residual peak | 6.582 sigma |
| ss=2 predicted `delta/sigma` peak | 2.446 sigma |
| Pearson r, pixel-by-pixel over the box | **-0.109** |

The predicted quadrature error at the core is a smooth, single-signed blob; the observation is
a sharp pixel-scale sign alternation. They share a rough location and nothing else. **The
checkerboard is not a rendering choice.** Its cause is now genuinely open — the PSF-mismatch
reading survives, but the one remedy `KNOWN_ISSUES.md` proposes for it has already been
applied to this system.

**Separately, and this is a second finding rather than a caveat on the first: DESI234 is
undersampled at `supersample = 2`.** Against the diagnostic's 0.2 sigma/pixel tolerance:

Re-measured against a `supersample=32` reference on the same 9-sample batch
(`analysis/undersampling_certified.txt`):

| rung | PSF convention | max \|delta\|/sigma | % pixels > 0.2 sigma | `delta_chi2` mean (spread) |
|---|---|---|---|---|
| ss=1 | native | 12.579 | 10.42% | +356.9 (+64.3 .. +834.0) |
| **ss=2 (round 2)** | subgrid | **3.132** | 8.17% | −340.2 (−414.2 .. −215.1) |
| **ss=4 (round 3)** | subgrid | **0.997** | 1.97% | −109.3 (−164.3 .. −64.4) |

`recommended_supersample: None` — **no rung reaches tolerance**, and the ss=32 reference is
itself formally uncertified (self-check vs ss=16: 0.402 sigma native / 0.451 sigma subgrid,
against a `tolerance/4 = 0.05 sigma` bar, ~9x over). Certifying to that standard needs
ss_ref >= 64. What the reference doubling *does* establish is that the ss=4 rung is stable:
0.988 at ss_ref=16, 0.997 at ss_ref=32, a 1% shift across a doubling. Round 3 therefore
raises `supersample` 2 -> 4 as the better of two imperfect renders, not as a fix — §4.

Two things make that change cheap and clean:

- **It does not move the PSF model.** ss=2 and ss=4 are both the `subgrid` convention, so
  2 -> 4 refines quadrature only. It is 2 -> 1 that crosses into `native` and changes the PSF
  by `convention_gap_max = 5.02 sigma` — a gap between conventions, not an error in either,
  and not reducible by supersampling.
- **It is free in wall time.** Measured on this 80x80 image, post-JIT: ss=4/ss=2 = 1.05x at
  batch 50 (this fit's chain count) and 0.90x at batch 1. Kernel-launch overhead dominates
  the O(ss^2) compute at this image size, so the naive 4x is not what this costs.

### 1.1n Why the first ladder was not certified, and what the second one refuted

Recorded because it is a trap in reading this diagnostic, not because it changes a result.
The first run used `reference_supersample=16` and reported `reference_self_error` of 10.6
sigma (native) / 8.9 sigma (subgrid) — so `reference_converged` was **False** and no rung's
`delta_chi2` was certified. Two specific artifacts followed:

- The ss=8 rung's `max |delta|/sigma` of 8.943 equals the subgrid `reference_self_error`
  (8.9429) to four figures. At ss=8 against an ss=16 reference the rung comparison *is* the
  self-check, so that number measures reference noise, not the ss=8 quadrature error.
  The certification run confirms this: at ss_ref=32, ss=4 (two doublings below the old
  reference, so never subject to the coincidence) reads 0.997 against its old 0.988, while
  the ss=8 number vanishes with the reference that produced it. **A rung one doubling below
  an unconverged reference is not a measurement of that rung.**
- The natural second explanation — that one outlier draw in the 9-sample batch dominated the
  8.9 sigma self-error (the ss=8 rung's `delta_chi2` spread ran -9.5 to +1665 on a mean of
  +435) — is **refuted**. Per-sample self-errors at ss_ref=32 run 0.070 to 0.451 (subgrid),
  worst the MAP point at 0.4515, only ~2.3x the next worst, and the batch maximum equals the
  MAP's value to 10 significant figures. The error is spread across the batch. That old
  `delta_chi2` spread was reference-resolution noise, per the bullet above, not a pathological
  draw.

The shape/correlation test above does not depend on the reference being converged to
sub-tolerance precision, which is why it was reported as a result while the ladder was re-run,
and why it stands unchanged now that the ladder is.

### 1.2 The arc residual is substructure, not a displaced arc

*(`analysis/arc_residual_anatomy.py`, `analysis/arc_width_test.py`)*

`RESULTS_v2.md` §5(b) read the paired -/+ lobes along the S arc (+/-4-5 sigma, blue
immediately adjacent to red) as discrete star-forming clumps. But adjacent opposite-sign pairs
are also the signature of a *displaced* arc, which a pixelated source would absorb without
diagnosing — so the reading was tested rather than carried forward.

Over 118 arc pixels (selected from `config.py`'s documented ridge: radius 1.66-1.82",
array-space PA 125-235 deg, intersected with the run's own likelihood mask and a 3 sigma
model-brightness cut), at the HMC median, arc `chi2_nu` = 9.64 before correction. Fitting the
normalised residual against the model's own first spatial derivative:

| decomposition | implied shift | fraction of arc chi2 explained |
|---|---|---|
| global (row, col) | (-0.062, -0.071) px = 0.09 px = 0.006" | **0.5%** |
| tangential / radial | (-0.023, +0.076) px = 0.08 px = 0.005" | **0.7%** |

Both are consistent with zero. The gradient-projection model can only vary as smoothly as the
model's own gradient, and cannot reproduce the several sign flips the actual residual shows;
the leftover after subtracting it is visually indistinguishable from the raw residual.
**The arc is not measurably displaced**, and §5(b)'s clump reading survives the test designed
to overturn it.

A first-derivative projection tests only *displacement*, not *mis-shaping*: an arc too narrow
or too wide in the radial direction puts its residual on the model's **second** derivative and
would pass the test above as cleanly as genuine knots do. That second-order test is
`analysis/arc_width_test.py`: over the *same* 118 pixels and the same tangential/radial frame,
the normalised residual is fitted against
`a·dm/dt + b·dm/dr + c·d²m/dr² + d·d²m/dt² + e·d²m/dt·dr`, and the three second-order terms are
judged by the incremental R² they buy.

| | value |
|---|---|
| first-order R² (a, b) | −0.064 |
| full second-order R² (a-e) | 0.007 |
| **incremental R², the three 2nd-order terms** | **0.071** |
| fraction of arc chi2 explained, 1st / 2nd order | 0.6% / 7.2% |
| implied radial width error from `c` | ~0.09 px |
| **null: incremental R², mean / 95th pct** | **0.210 / 0.476** |

**The null is the whole test.** Five free parameters on 118 spatially correlated pixels
always explain *something*; 7.2% of arc chi2 sounds like a result until it is put beside what
three arbitrary smooth regressors buy on the same footprint. Against 200 Gaussian-smoothed
white-noise triples matched in correlation length to the real second-derivative fields, the
measured 0.071 sits **below the null's mean** and far below its 95th percentile. **There is no
evidence the arc is mis-shaped.**

Recorded because it nearly went wrong: the first null was built from random linear
combinations of the same five derivative fields, and after projecting out the first-order
terms it collapsed onto exactly the 3-D subspace the real second-order terms span — returning
null mean = 95th percentile = the measured value, identically. A null that reproduces the
measurement to the digit is not a strict null, it is the measurement wearing a different hat.
It was replaced with independent smoothed noise fields.

**Taken together, §1.2's two tests close the question.** Neither a displacement (0.09 px, 0.5%
of arc chi2) nor a radial width error (below its own null) accounts for the paired lobes.
`RESULTS_v2.md` §5(b)'s reading — genuine source-plane substructure — survives both tests
built to overturn it, and round 4's lever is source freedom, not astrometry.

### 1.3 `src1` remains an unresolved large-scale flux gradient

Not re-tested this round. `RESULTS_v2.md` §6 established that `src1` — a huge, smooth blob
2.5" outside the caustic — is not multiply imaged, so it is not a source component; it is the
model's only handle on extended light it otherwise cannot fit. This round does not address it,
and it stays in the caveats.

---

## 2. Stage 0: the MAP had not converged, and the SVI surrogate was never the problem

*(`analysis/map_deficit.py`, `analysis/map_deficit.txt`)*

`diagnose.map_convergence_check` had never been run on this system. Both rounds fail it:

| run | `best_lp` (stored = recomputed) | max sampler lp | median sampler lp | deficit |
|---|---|---|---|---|
| `r1_s1l1` | 15796.226 | 16713.512 | 16683.344 | **917.3 nats** |
| `r2_s2l2` | 17376.092 | 17462.760 | 17448.764 | **86.5 nats** |

A MAP is by definition the highest-density point of the posterior, so an optimiser finishing
below the *typical* draw of the chain it seeds has not converged. `r2_s2l2`'s deficit is an
order of magnitude smaller than DESI257's 681.8 nats (`KNOWN_ISSUES.md` item 6), which is
consistent with round 2 having run 1500 MAP steps rather than 350 — but 86.5 nats is still a
failure, and the `chisq_hist` tail slope confirms why:

| run | tail window | slope |
|---|---|---|
| `r2_s2l2` | last 500 of 1500 | **-4.374e-05/step** (still descending) |
| `r2_s2l2` | last 150 of 1500 | -2.697e-05/step (still descending) |
| `r1_s1l1` | all 350 | -1.144e-02/step (still descending, 2 orders steeper) |

`r1_s1l1`'s slope is two orders of magnitude steeper than `r2_s2l2`'s, consistent with its 10x
larger deficit: that MAP was cut off far earlier in its descent.

**The metadata trap, confirmed independently.** `r2_s2l2/metadata.json` reports
`map_steps: 350, n_samples: 1024` — the `KNOWN_ISSUES.md` item-1 argparse-default corruption.
`map.npz`'s own `chisq_hist` is **1500** entries long, matching `RESULTS_v2.md` §2.1's stated
1500 steps at batch 4096. (`r1_s1l1`'s recorded `map_steps: 350` is genuine: its `chisq_hist`
is 350 long.) The metadata is left as it stands, per item 1's own policy.

### 2.1 The SVI guard reference stays empty, and that is a measurement

`verify.py` check 5 reports itself skipped because DESI234's `svi_guard_reference` is empty.
The open question was whether `r1_s1l1` could populate it as a known-diverged run. It cannot:

| run | min eigenvalue | condition number | ELBO sd (last 300) | `check_svi_covariance` |
|---|---|---|---|---|
| `r1_s1l1` | 9.626e-10 | **1.627e+07** | 8.5 | **PASSES** |
| `r2_s2l2` | 1.097e-08 | **7.447e+07** | 23.5 | **PASSES** |

Both sit inside the guard's own healthy band (1e6-6e8), against a diverged reference
(`twosource`) of 8.8e11 — three-plus orders of magnitude away. **`r1_s1l1`'s SVI covariance was
not diverged.** It failed on R-hat, which is a different thing: the 917-nat MAP deficit
propagated through SVI's Gaussian mean into a badly-seeded sampler. Populating
`svi_guard_reference` from it would assert something the measurement contradicts, so check 5
stays skipped and honest.

---

## 3. Preflight

`verify.py --lens DESI234` had never been run on this system. It passes 5/5:

| check | result |
|---|---|
| 1. Data conditioning | PASS — crop (80, 80) max 6.63, psf (27, 27) sum 1.0, worst unmasked corner lower-right 0.241 < 0.30 |
| 2. Parameter count and ordering [1src 1ll] | PASS — 22 parameters, ordering and bijector round trip verified |
| 3. Simulator flux scale [1src 1ll] | PASS — median prior peak 8.04 vs data 6.63 (ratio 1.21), `log_prob` finite |
| 4. Short end-to-end MAP -> SVI -> HMC | PASS — MAP (22,) -> SVI (22,) -> HMC (4, 10, 22), chains first |
| 5. SVI covariance guard | **SKIPPED** — no `svi_guard_reference` runs in the lens config (see §2.1) |

`claude/pipeline/tests/` is green: `test_sersic_bn` 5/5, `test_mask` 6/6, `test_corner_check`
3/3, `test_make_data` 21/21, `test_diagnose_and_metadata` 11/11, `test_masked_chisq` 1/1,
`test_foreground_roundtrip` 2/2.

**One thing check 1 surfaces that blocks the unmasked fit.** `corner_report` checks *unmasked*
pixels, so the 0.241 above is the worst corner *with the interloper masked*. The interloper
puts **0.859 e/s** in the raw upper-left corner, which exceeds this config's
`companion_max = 0.30`. Setting `MASK_CIRCLES = ()` for round 3 therefore fails check 1 until
`companion_max` is raised — exactly as it was for DESI257, which raised its own to 1.05 against
its SW object's 0.957 e/s peak when it unmasked. See §4.

Re-run against the §4 config (unmasked, `companion_max = 0.95`, `supersample = 4`), `verify.py`
passes 5/5 again: check 1 now reports `worst corner upper-left max 0.859 < 0.95` — the
interloper's own flux, now inside the likelihood — and check 3's median prior peak moves 8.04
-> 8.35 with the finer quadrature. Check 5 remains skipped, per §2.1.

---

## 4. Config changes for round 3

Three changes to `config.py`, each with its own reason and its own way of being wrong.

**(a) `supersample = 2 -> 4`** (§1.1). Justified by the certified ladder, free in wall time,
and inside one PSF convention. `verify.py` re-run against the changed value passes 5/5; check
3's median prior peak moves 8.04 -> 8.35 against the data's 6.63 (ratio 1.21 -> 1.26), which
is the render getting slightly brighter at the cusp, as a finer quadrature of a cuspy profile
should.

**This is a confound and it is named here rather than buried.** Round 3's `theta_E` will
differ from round 2's through *two* changes, the mask and the rendering, not one. The two are
separable after the fact if it matters — scoring round 2's own draws under the ss=4 likelihood
isolates the rendering term at no new fit — and §7's systematic budget carries it either way.
The alternative, holding ss=2 for comparability, means knowingly fitting at a 3.1 sigma
quadrature error to protect a comparison; that trade is not worth making.

**(b) `MASK_CIRCLES = ((64.0, 14.0, 12.0),) -> ()`.** The round's question. The two fits must
run as a **pair**: an unconstrained free Sersic sitting on a 110 sigma blob is a real failure
mode, and `r3_fg`'s position-tied component is what distinguishes "the model reproduces this
object" from "some component went and sat on it". Rounds 1-2 stay diagnosable because
`rebuild_model` reads each run's own recorded `mask_circles` back (`KNOWN_ISSUES.md` item 8).
**Log posteriors do not cross this change** — DESI257 measured a 14,347-nat offset on
identical parameters.

**(c) `companion_max = 0.30 -> 0.95`.** Not a loosened standard, a moved goalpost that had to
move: `verify.py`'s `corner_report` checks corners *after* masking, so 0.30 was only ever
being cleared because the interloper was masked out of the upper-left corner. Unmasking
restores its 0.859 e/s there. 0.95 gives ~11% headroom, mirroring DESI257's 1.05 against its
0.957 e/s peak. If `MASK_CIRCLES` is ever restored, 0.30 must be restored with it.

**(d) `FOREGROUND = ForegroundPrior(...)`**, inert unless `run_fit.py --foreground` is passed.
`priors.FOREGROUND_PRIOR_VERSION = 1`: SIS mass and Sersic light at a shared centroid, with
the SIS `theta_E` HalfNormal (support reaching exactly 0) and `Ie` LogNormal with no floor, so
that `theta_E -> 0` and `Ie -> 0` recover the 36-parameter model. The nesting is what makes
the posterior's distance from zero mean something.

Geometry (`analysis/foreground_calibration.py`), each number re-derived independently rather
than accepted:

| field | value | how |
|---|---|---|
| `center_x` | −1.63514" | crop pixel (64.0, 14.0) through the running code's WCS |
| `center_y` | +1.58600" | same |
| separation | 2.2934" | from the deflector centroid; `config.py`'s prose says 2.30" |
| array-space PA | 314.6 deg | `config.py`'s prose says 314 |
| `Ie_median` | 67.0 | prior-draw render, per-seed medians 0.758/0.922/0.854/0.820/0.726 -> median **0.8203** e/s against this object's measured **0.82** e/s peak, within 0.1% |

The row/col mapping — the field most likely to be silently transposed — was checked by
*rendering*, not by arithmetic: a Sersic placed at `center_x = 0.5, center_y = -1.0` peaks at
array (row 24, col 47) on DESI234's own grid, confirming **`center_x` is column-derived and
`center_y` is row-derived**.

Parameter counts confirmed live against the built model: **36** without the foreground, **46**
with it.

---

## 5. The fits

Two fits, differing in exactly one thing — `--foreground` — launched as a pair by
`runs/launch_r3.sh`.

| | `r3_unmasked` | `r3_fg` |
|---|---|---|
| parameters | 36 | 46 |
| foreground SIS + Sersic | no | yes |
| `MASK_CIRCLES` | `()` | `()` |
| `supersample` | 4 | 4 |
| GPUs | 0, 1 | 2, 3 |

Everything else is shared: 2 source / 2 lens-light Sersics, MCLMC, 50 chains, burn-in 5000,
`--map-steps 5000 --n-samples 1024 --n-vi 1024 --svi-steps 12000 --svi-peak-lr 1e-3`.

**Why two GPUs each and not four for one at a time.** `KNOWN_ISSUES.md` item 2 records device
count as the *unattributed second variable* in DESI257's frozen-chain failure. The two round-3
fits must therefore not differ in it, which rules out giving one fit four devices and the other
two. `normalize_hmc_samples` checks the chain-axis layout by construction, and R-hat/ESS on the
result is the second guard.

**Why the batch fell from round 2's 4096.** Not a downgrade — a fixed memory budget spent
differently. See `KNOWN_ISSUES.md` item 9: raising `supersample` 2 -> 4 costs memory as ss², and
the MAP/SVI batch buffer is allocated *per device*, so the ceiling is one card (~61 GiB) no
matter how many GPUs the fit has. The first launch died at
`RESOURCE_EXHAUSTED ... 178.02GiB [jit_run_map]`; a second attempt at 1536 asked 67.07 GiB and
failed too, which is what identified the per-device ceiling — 67 GiB is comfortably under
2 x 80 GB, so the allocation is not being sharded. The per-sample cost is 10.9 MiB at ss=2 and
**43.6 MiB at ss=4** (read off both failures, 178 GiB at 4096 and 67.07 GiB at 1536, which
agree). At ss=4's measured 43.6 MiB/sample,
1024 samples is **43.6 GiB — the same footprint round 2 ran at** (4096 x ss=2 = 44.6 GiB). The
smaller batch is paid for with `--map-steps` 1500 -> 5000, which is what §2's 86.5-nat deficit
called for anyway.

### 5.1 Both fits ran to completion. **Neither converged.**

*(`analysis/r3_diagnostics.py`, `analysis/r3_diagnostics.txt`)*

| | `r3_unmasked` | `r3_fg` | round 2 (masked, ss=2) |
|---|---|---|---|
| max R-hat | **13.94** | **1.92** | 1.008 |
| params with R-hat > 1.01 | **36 / 36** | **46 / 46** | 0 |
| min bulk ESS (of 37,500 draws) | **50.7** | **71.3** | — |
| `chi2_nu` | 1.662 | **1.082** | 1.300 |
| MAP deficit | **151.9 nats** | 27.9 nats | 86.5 nats |
| main `theta_E` | 1.539 +/-0.002" | 1.473 +/-0.006" | 1.539 +0.003/−0.002" |

**No `theta_E` in that table is a measurement.** An R-hat of 13.94 means the 50 chains are not
sampling the same distribution, so their pooled percentiles are a description of where the
chains happened to be, not of a posterior. `r3_unmasked`'s interval is quoted only to show that
a tight-looking interval survives total non-convergence — which is the reason R-hat is checked
before intervals are read, not after.

Two supporting facts, both pointing the same way. `r3_unmasked`'s MAP deficit got **worse**
(151.9 nats) despite 5000 steps against round 2's 1500 — an optimiser falling further behind a
sampler that is finding much better points than it can is the signature of a rugged or
multimodal surface, not of too few steps. And the ordering is consistent throughout: the
36-parameter unmasked model is the badly-behaved one, the 46-parameter model with the
foreground component is markedly better on every line.

**§4(b) predicted a specific failure mode** — two free source Sersics, no foreground component,
and a newly-unmasked 110-sigma object in the field, with the chains splitting on whether to put
a source component on the interloper. §5.4 tests that directly, and it is **not** what happened
in either run. The two runs failed for two *different* reasons, neither of them the predicted
one.

### 5.2 What can and cannot be said already

**Can:** `r3_fg` fits far better than `r3_unmasked` on a shared likelihood — `chi2_nu` 1.082 vs
1.662, about 3700 in chi2 for 10 extra parameters. Ten parameters do not buy that by chance.
The interloper is not reproduced by the same mass model and source components that make the
arc.

**Can:** the nested null is excluded, and by a wide margin. `r3_fg`'s foreground parameters sit
far from the zero that would recover the 36-parameter model:

| parameter | median | 16/84 | distance from 0 | mass below 0.05" |
|---|---|---|---|---|
| `planes/0/mass/2/theta_E` (SIS) | **0.370"** | [0.342, 0.400] | **12.2 sd** | 0.0000 |
| `planes/0/light/2/Ie` (Sersic) | **89.4** | [74.1, 100.6] | **6.2 sd** | — |

(Column names read from `model.z_param_names`, not from guessed indices. Note that these paths
are in gigalens' pre-2026-08-07 scheme; today they read `planes/lens/mass/foreground/theta_E`.
See `KNOWN_ISSUES.md` item 11.) Caveat, **strengthened 2026-08-10**: these are percentiles of
an unconverged chain set, and of a chain set whose mass model is rejected. The original caveat
said only that the widths were not quotable while "the *sign* of the conclusion is safe —
nothing about non-convergence moves a 12-sd separation to zero." That is true and insufficient:
the threat to this detection was never non-convergence but misspecification, and a 12-sd
separation measured under a 7:1-needle deflector is 12 sd from zero *in a model that is
wrong*.

**Cannot:** any value of `theta_E`, from either run, at any precision. See §7.

**RETRACTED (2026-08-10) — the paragraph that stood here.** It read: *"With the foreground SIS
carrying `theta_E = 0.370"`, the main deflector's `theta_E` falls 1.539 -> 1.473", a 0.066"
shift — roughly 20x round 2's 0.003" statistical error,"* and concluded that the interloper's
mass was a real systematic masking had hidden rather than avoided.

It does not survive `r3_fg`'s rejection. The 1.473" end of that shift is the needle's
`theta_E`, not a better estimate of the deflector's: the comparison is between a sane mass
model and a broken one, so the 0.066" measures the breakage, not the interloper. The reasoning
about masking remains correct in principle — masking removes an object's light from the
likelihood and leaves its deflection unmodelled, and that *is* a different error from the one
round 2 guarded against — but round 3 does not size it. **Nothing yet quantifies the
interloper's mass systematic.** Round 4 exists to.

### 5.3 Two secondary findings from the diagnostics

**The `r3_fg` SVI NaN is cosmetic.** The logged `ELBO loss: final nan` had a first non-finite
value at step 7794 of 12000, with 35% of steps non-finite after it; the loss was finite and flat
near −19070 for several hundred steps beforehand, then diverged sharply rather than degrading.
The **saved surrogate arrays are fully finite** — `mean` (46,) and `scale_tril` (46, 46) both
pass `np.isfinite` — so the 50 chains were seeded from a valid Gaussian. The NaN is in the
logged history, not in what was used. It is not the cause of the R-hat failure. (It is still a
real reporting defect: a run whose headline SVI line reads `nan` looks compromised and is not.)

**One boundary-pinned parameter per run**, both at a *prior* edge rather than a data preference:
`r3_unmasked`'s `source/light/src1/e2` at 0.8% of its TruncatedNormal range, and `r3_fg`'s
`source/light/src0/n_sersic` at 1.5% of its Uniform lower bound. Their R-hats should be
discounted accordingly — but they are 2 parameters out of 36 and 46, and every other parameter
fails R-hat too, so pinning is not the explanation for the non-convergence.

**Mask reconstruction verified.** `rebuild_model` printed no `NOTE:` for either run: both
recorded `mask_circles=[]`, both kept-pixel counts matched the recorded `n_masked_pixels`, and
both plots confirm 6400/6400 pixels in the likelihood.

**What would have decided the round.** If `r3_fg`'s foreground `Ie` and SIS `theta_E` both stay away from
zero and the previously-masked disc's residual goes flat, the object is a foreground galaxy and
round 2's judgement stands with the mask replaced by a model. If `r3_unmasked` reproduces it
with no dedicated component, it is a lensed image and `theta_E` moves — DESI257's outcome. If
`r3_fg` drives both to zero *and* the disc residual stays bad, neither reading works, and that
is what gets written here.

Comparisons are between `r3_unmasked` and `r3_fg`, which share a likelihood. **No log posterior
crosses the mask change** — see §4(b).

### 5.4 Why each run failed: frozen chains, and a ridge — not modes

*(`analysis/mode_structure.py`, `analysis/mode_structure.txt`)*

A summary R-hat table cannot tell a *frozen* chain from a chain exploring a *different mode*.
They look identical there and have opposite fixes: frozen is a sampler problem, multimodality
is a model problem. The statistic that separates them is the ratio

  R = (between-chain sd of the per-chain medians) / (median within-chain sd),

read alongside within-cluster R-hat. A frozen chain has essentially no within-chain motion, so R
is large and no partition of the chains rescues it; a chain sampling a distinct mode moves by
about a mode width, so R is order unity and R-hat computed *within* a cluster of like-minded
chains comes back near 1.

| | `r3_unmasked` | `r3_fg` |
|---|---|---|
| R on `theta_E` | **15.4** | 1.59 |
| R on that run's worst-R-hat parameter | **20.9** (`src1/R_sersic`) | — |
| range of per-chain `theta_E` medians | 1.5318 – 1.5421" | 1.4599 – 1.4835" |
| max within-cluster R-hat, k = 2 | **10.6** | 1.83 |
| max within-cluster R-hat, k = 5 | **8.4** | 1.92 |
| min bulk ESS within a cluster of n chains | **≈ n** (24.4 on 24, 26.5 on 26) | 65.1 on 44 |
| diagnosis | **frozen** | **under-mixed along a ridge** |

**`r3_unmasked` is frozen, and clustering proves it.** Each chain's median `theta_E` sits within
a 0.010" band, and each chain wanders 1.3e-4" about its own median — 15x less than the chains
are apart. The decisive line is the last row: split the 50 chains into 2, 3, 4 or 5 clusters by
their per-chain medians, and *within* every cluster the bulk ESS still equals the number of
chains in it and R-hat still exceeds 5. There is no partition under which these chains mix.
That is the definition of frozen: every chain contributes about one effective draw, so R-hat is
measuring 50 point estimates, not 50 distributions. This is `KNOWN_ISSUES.md` item 2's failure
recurring on a different lens — and note that the fix is the sampler, not the model. It also
retires the tight-looking `1.539 ± 0.002"`: that interval is the *scatter of 50 frozen points*,
which is why it can look like round 2's answer while carrying no information.

**`r3_fg` is not frozen and not multimodal.** R = 1.59 on `theta_E`: the chains move by about as
much as they are apart, which is ordinary under-mixing. Clustering buys almost nothing —
splitting into 3 clusters drops max R-hat only 1.92 → 1.63, and `lens/mass/host/theta_E` is the
worst parameter in nearly every cluster at every k. Discrete modes would have shown up as one
cluster with R-hat ≈ 1; instead the difficulty follows `theta_E` into every subset. That is a
**ridge** — a slowly-mixing direction, most likely the `theta_E`–`gamma`–foreground-mass
degeneracy the extra deflector introduces — and its fix is chain length and step-size
adaptation, not another component.

**The bimodality hypothesis for the interloper is refuted.** §4(b) predicted the chains would
split on whether the interloper is present. They do not:

- **No chain visits the nested null.** The per-chain fraction of draws with SIS `theta_E` <
  0.05" is exactly 0.0000 for all 50 chains — not "small on average", zero in every chain.
- **The SIS `theta_E` is continuous across chains**, per-chain medians spanning 0.305 – 0.428"
  with R = 1.16, i.e. within-chain motion as large as the between-chain spread.
- **The foreground `Ie` is, if anything, over-dispersed within chains**: R = 0.50, per-chain
  medians 69.7 – 105.6 against a median within-chain sd of 11.9. On this parameter the 50 chains
  are statistically indistinguishable draws from one distribution.
- Per-chain medians of SIS `theta_E` and `Ie` correlate at only −0.19 across chains — no
  "interloper on / interloper off" axis exists for them to line up along.

So the interloper is present in every chain of `r3_fg`, at a consistent mass and brightness, and
the run's convergence failure has nothing to do with it. **That much stands** — it is a
statement about the chains, which is what was measured.

**The inference drawn from it does not, and is retracted (2026-08-10).** This section
previously concluded that *"`r3_fg` is the right model and an insufficiently long run, while
`r3_unmasked` is a sampler failure on a model the data has already rejected by Δchi² ≈ 3700."*
Every chain agreeing on the SIS shows the detection is not a sampling artefact; it does not
show the model is right. All 50 chains also agree the deflector is a 7:1 needle, and that is
not a credible galaxy. Consistency across chains measures the posterior, not its adequacy —
distinguishing those two is the point of §5.2's physicality check, which came later. The
Δchi² argument falls with it: a likelihood ratio presupposes both models are admissible, and
`r3_fg` is not.

**What this sets up for round 4 (revised).** `r3_fg` needs a **model** change first: constrain
the deflector's ellipticity to a physical range and refit, then ask whether the foreground
component survives. Only if it converges *and* stays physical does the sampling question —
more steps, MCLMC's step-size adaptation along the `theta_E` ridge — become the live one.
`r3_unmasked` needs the investigation `KNOWN_ISSUES.md` item 2 already owes DESI257; its frozen
chains are a genuine sampler failure. It should not be dismissed on Δchi² against `r3_fg`, as
the previous version of this paragraph did: `r3_unmasked` is the one of the two with a sane
mass model, and float64 (`pipeline/data.py:118` pins float32) is the first thing to vary.

---

## 6. Reproducing this

```bash
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python

# Preflight (section 3).
PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0 $PY claude/pipeline/verify.py --lens DESI234

# Stage 0 diagnostics (sections 1-2). Independent; one GPU each.
CUDA_VISIBLE_DEVICES=0 $PY claude/DESI234/analysis/undersampling_check.py
CUDA_VISIBLE_DEVICES=1 $PY claude/DESI234/analysis/map_deficit.py
CUDA_VISIBLE_DEVICES=2 $PY claude/DESI234/analysis/arc_residual_anatomy.py
CUDA_VISIBLE_DEVICES=0 $PY claude/DESI234/analysis/arc_width_test.py
CUDA_VISIBLE_DEVICES=3 $PY claude/DESI234/analysis/undersampling_certify.py

# The two round-3 fits (section 5). Launches both, two GPUs each, and waits.
bash claude/DESI234/runs/launch_r3.sh

# Post-fit diagnostics (sections 5.1-5.4). Both read the finished checkpoints.
PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0 $PY claude/DESI234/analysis/r3_diagnostics.py \
  > claude/DESI234/analysis/r3_diagnostics.txt 2> claude/DESI234/analysis/r3_diagnostics.err.txt
PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0 $PY claude/DESI234/analysis/mode_structure.py \
  > claude/DESI234/analysis/mode_structure.txt 2> claude/DESI234/analysis/mode_structure.err.txt

# Pipeline tests touched this round (CPU, no --lens, no GPU).
PYTHONPATH=claude/pipeline $PY claude/pipeline/tests/test_check_theta_E.py   # 5/5 pass
```

`launch_r3.sh` carries the batch-size arithmetic in its own comments, so re-running it after a
`supersample` change will not silently repeat the OOM that produced it. Logs land in
`runs/r3_unmasked.log` and `runs/r3_fg.log`.

---

## 7. `theta_E`: why round 3 quotes no value, and what the budget looks like when it can

**Round 3 produces no `theta_E` measurement.** Both fits failed R-hat on every parameter
(§5.1), and §5.4 identifies why: `r3_unmasked`'s chains are frozen, `r3_fg`'s are under-mixed
along a `theta_E` ridge. `r3_fg`'s ridge runs along exactly the parameter of interest, which is
the worst possible place for it — `lens/mass/host/theta_E` is the max-R-hat parameter of the
full run and of nearly every chain cluster at every k. Percentiles from these chains are not
posterior quantiles and are not quoted as such anywhere above.

**Round 2's `1.539 +0.003/−0.002"` is the answer this system currently has**, and round 3 does
not displace it. It came from a converged run (max R-hat 1.008) of a model with the
interloper's *mass* absent, since masking removes an object's light from the likelihood and
leaves its deflection unmodelled. That remains a real and unquantified systematic — but it is a
caveat on the number, not a refutation of it, and round 3 supplies no replacement.

*(Retracted 2026-08-10: this paragraph previously read "is likewise not the answer, and round 3
is what shows it", on the strength of a 0.066" shift that came from the rejected `r3_fg`. See
the banner and §5.2.)*

So the budget below is a **statement of what must be accounted for when a converged `r3_fg`-class
run exists**, not an error bar on a number this round measured. Sizes are what has been measured
so far; the sign of each is known, the magnitude of the first is not yet.

| term | size | status |
|---|---|---|
| statistical | ~0.006" (`r3_fg`'s width) | **not yet meaningful** — R-hat 1.92 on this parameter, and the run is rejected |
| **interloper mass** | **unquantified** | ~~0.066", measured (§5.2)~~ — **retracted 2026-08-10**: that figure came from the rejected `r3_fg`. The sign is unknown and the magnitude unmeasured. Round 4 exists to size it |
| background RMS, ±15% patch-to-patch | not propagated | ±15% in σ is ±30% in chi², and it scales the whole likelihood; `README.md` |
| core residual artifact | floor, not removed | §1.1: ss=4 is certified and adopted, and the checkerboard **survives it** (Pearson r = −0.109 against the predicted map). It is real unmodelled structure at the deflector core, not a rendering choice, and nothing in round 3 removes it |
| supersample 2 → 4 | included | a change of round 3 relative to round 2; §1.1's ladder certifies ss=4 against an ss=32 reference |
| arc substructure | not propagated | §1.2: the arc residual is genuine clumpiness, not a displacement, so it does not bias the deflector centroid — but it does inflate chi² and is unmodelled |

The honest bottom line, **revised 2026-08-10**: the interloper's mass is a systematic that
round 2 could not see by construction, and round 3 **did not succeed in sizing it**. The
previous version of this paragraph called the 0.066" term "the round's most durable result" and
argued it held independently of the convergence failures, "because a 0.066" shift driven by a
12.2-sd mass detection is not something non-convergence manufactures." The reasoning about
non-convergence was right and beside the point: what invalidates the number is not that the
chains failed to mix but that the model they were mixing towards is unphysical. The durable
result of this round is the smaller one — `r3_unmasked` reproducing round 2 to 0.0007" on a
sane mass model.

**What a quotable `theta_E` requires:** a converged *and physical* `r3_fg`-class run — §5.4's
reading that the fix is sampling rather than modelling is wrong for `r3_fg`, which needs its
ellipticity constrained first — the ±15% RMS spread propagated rather than noted, and the core
artifact either modelled or explicitly declared a floor on chi².
