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

Open items, needing work: **2** (MCLMC frozen chains), **6** (MAP step count), **9** (a sizing
rule to apply, not a bug), **10** (the rendering grid and the PSF convolution convention).
Everything else is closed — item 5's reporting change landed 2026-08-11.

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

## 5. The excess residual at the deflector core — CLOSED

DESI234's and DESI257's residuals both show an excess-chi-squared region at the deflector core.
Two rounds of work went into explaining it, and the explaining was the mistake.

> **Naming correction, 2026-08-11.** This item used to call the feature "a pixel-scale
> alternating +/-sigma checkerboard ... reaching roughly +/-5 sigma", from the round-2 fits. **The
> alternation is not there.** A checkerboard has *negative* lag-1 autocorrelation between
> neighbouring pixels; measured on four fits — both systems, both rounds, both supersamples — it
> is positive every time (`DESI257/analysis/core_checkerboard_extent.py`):
>
> | run | ss | core lag-1 rho | field lag-1 rho |
> |---|---|---|---|
> | DESI257 `r3_unmasked` | 2 | **+0.198** | +0.713 |
> | DESI234 `r3_unmasked` | 4 | **+0.345** | +0.778 |
> | DESI257 `r2_s2l2` | 2 | **+0.207** | +0.722 |
> | DESI234 `r2_s2l2` | 2 | **+0.357** | +0.757 |
>
> The round-2 runs are included precisely because the old description came from them, and DESI234
> changed supersample between the rounds — so the pair tests both "was the description loose?"
> and "did something between the rounds remove the alternation?". Neither. The core *is*
> measurably rougher than the field (rho falls from ~+0.75 to +0.20-0.35, real excess pixel-scale
> power), but positively correlated: a smooth compact excess, not a sign-flipping pattern.
> **The physical conclusion below is untouched.** Only the name and the round-2 amplitude figure
> should stop being requoted as measured — a description repeated across two systems' results
> files for two rounds, and never checked with the one statistic that defines the word.

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

- ~~`chi2_nu` as currently reported includes the core pixels and is therefore inflated on every
  run of every system.~~ **FIXED 2026-08-11: `diagnose.py` reports both numbers.**
  `residual_figure` returns a `ChiSquares` namedtuple (`full`, `core_masked`, the counts and the
  radius) instead of a bare float, so every call site must now name which chi-squared it means —
  deliberate, since the point of the change is that two confusable numbers exist. Both come from
  the same `masked_chisq` helper; no second formula was written (item 3 is why). `main`'s summary
  and the residual panel's title print both, with the radius and the excluded count, and state
  that the core pixels remain in the likelihood.

  **The radius is 0.520" = 8 px, measured, not chosen**
  (`DESI257/analysis/core_checkerboard_extent.py`): the radius enclosing 95% of the excess
  chi-squared above a *local* background — 7 px on DESI257 `r3_unmasked`, 8 px on DESI234
  `r3_unmasked`, insensitive to that fraction (90% -> 7/6 px, 99% -> 8/8 px). The larger is used
  so neither system is under-masked. It lives in `diagnose.CORE_MASK_RADIUS_ARCSEC` as an
  **instrument-level constant in arcsec** rather than a per-lens config field, because the two
  systems agree to one pixel and all three are HST WFC3/IR F140W at 0.065"/px — the script tests
  that fork explicitly instead of assuming it, and reports the per-lens answer if they disagree.

  | run | full-field | centre-masked | px excluded |
  |---|---|---|---|
  | DESI257 `r3_unmasked` | 1.154 (6400 px) | **1.065** (6198 px) | 202 |
  | DESI234 `r3_unmasked` | 1.662 (6400 px) | **1.609** (6202 px) | 198 |

  **The full-field values are unchanged to four figures** — the check that the change was purely
  additive rather than a rewrite of the arithmetic. Both centre-masked values were reproduced
  independently from the radial profile, through a code path sharing nothing with the
  implementation (1.0646 and 1.6088). Six new cases in
  `tests/test_diagnose_and_metadata.py` (24/24), including the discriminating one — an injected
  core excess, where a bug returning the same number twice would otherwise pass.

  Two criteria were tried and rejected before the one that worked, both recorded in the script's
  docstring: a 3-sigma test on each annulus's mean squared residual is powerless (heavy-tailed
  squares; a 9-px annulus reads 8.56 +/- 4.13, so it rejected an excess visible in the profile),
  and a *far-field* background annulus reads essentially the whole-field `chi2_nu`, which made the
  cumulative excess non-monotone and left no plateau to take a fraction of.
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

**2026-08-11: `supersample` was the second gap, found before it bit.** `data.load` built the
`SimulatorConfig` (and so the rendering grid `ProbModel` optimises against) from
`cfg.supersample` — today's config — exactly the bug `mask_circles` had. DESI234 round 3 had
already raised it 2 → 4 (`DESI234/RESULTS_v3.md` sec. 1/3); DESI257 was about to follow, which
would have silently changed what `diagnose.rebuild_model` reconstructed for its three archived
runs (`r1_s1l1`, `r2_s2l2`, `r3_unmasked`), all fitted at ss=2, with no error raised.

Fixed the same way as the mask: `run_fit.py` now writes `"supersample"` into `metadata.json`;
`data.load(cfg, supersample=...)` takes an optional override (a keyword, not a `cfg` mutation,
so `cfg` stays a faithful record of the config file); `rebuild_model` uses the run's recorded
value when present, falls back to `cfg.supersample` for runs predating the key (with a `NOTE:`
either way), same as `mask_circles`. Adversarially checked: with DESI257's config temporarily
set to ss=4, `diagnose.py claude/DESI257/runs/r3_unmasked` still reports ss=2 from metadata.

**Back-filled the archived runs**, verified from `config.py`'s git history before writing, not
assumed:

- **DESI257** (`r1_s1l1`, `r2_s2l2`, `r3_unmasked`): `supersample=2`. `config.py`'s entire git
  history carries exactly one `supersample=` literal — the file was created with it at 2 and it
  has never changed — so this is unambiguous.
- **DESI238** (the twelve `"api": "scene"` runs; pre-port runs are unaffected since
  `rebuild_model` already refuses those outright): `supersample=2`, same reasoning — one
  `supersample=` literal in the file's whole history.
- **DESI234** `r1_s1l1`, `r2_s2l2`: `supersample=2` — `config.py` was created at commit
  `2efab5a` with `supersample=2`, before either run.
- **DESI234** `r3_fg`, `r3_unmasked`: `supersample=4` — **measured, not inferred.** The
  `git_commit` these runs recorded (`ba9fc2a`, DESI257's round-3 commit — unrelated to
  DESI234) still shows `supersample=2` in `config.py` at that blob: the 2 → 4 edit was not
  committed until `1566e6c`, dated the *next day*, so the git blob alone is ambiguous (an
  uncommitted working-tree edit made before the runs and committed after them — a workflow
  habit visible elsewhere in this repo's history). The first pass through this back-fill
  resolved that ambiguity from prose (`RESULTS_v3.md` secs. 1/3 stating round 3 raised
  `supersample` 2 → 4, corroborated by item 9's matching `n_samples`/`map_steps`) — weaker
  evidence than a measurement, so it was re-checked directly: `rebuild_model` can be pointed
  at either supersample, so each run was rebuilt at ss=2 and ss=4, rendered at its own
  HMC-median point (`diagnose.point_estimate`), and scored with `diagnose.masked_chisq`
  against its recorded keep-mask. Compared against `RESULTS_v3.md`'s own recorded values
  (`chi2_nu` = 1.662 for `r3_unmasked`, table row `chi2_nu` line 420; 1.082 for `r3_fg`, same
  row and confirmed again at line 24):

  | run | ss=2 | ss=4 | recorded |
  |---|---|---|---|
  | `r3_unmasked` | 1.9051 | **1.6621** | 1.662 |
  | `r3_fg` | 1.2557 | **1.0823** | 1.082 |

  ss=4 reproduces both recorded values to four significant figures; ss=2 misses by 0.24–0.24
  in chi2_nu, an order-of-magnitude larger gap than the reproduction error. This confirms
  `supersample=4` directly rather than by inference — the git blob at the recorded commit
  understates it because it lagged an uncommitted edit, not because the run used ss=2.

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

## 10. Every rendering grid tested fails the quadrature tolerance, and the PSF convolution convention is the blocker — OPEN

Measured on DESI257 with `gigalens.jax.analysis.undersampling.diagnose_undersampling` over a
9-sample batch against a `reference_supersample=64` reference, at the library's own 0.2 sigma/px
tolerance (`DESI257/RESULTS_v3.md` §8.1, `DESI257/analysis/undersampling_report.txt`):

| rung | max \|delta\|/sigma | mean `delta_chi2` |
|---|---|---|
| ss=1 (native-kernel PSF) | 5.290 | +720 |
| ss=2 | 3.697 | -216 |
| ss=4 | 1.497 | -86 |
| ss=8 | 0.439 | -34 |

**All four fail, and so does the reference's own self-check** (subgrid self-error 0.072 sigma
against a 0.05 threshold), so these are reported, not certified. The ranking is still usable:
doubling the reference 32 -> 64 cut its self-error 2.7x while ss=2 and ss=4 moved under 5%.

**The cross-cutting lessons, which is why this is a pipeline item and not a DESI257 one:**

- **`supersample` is not a knob that can be turned up until the error goes away.** ss=8 still
  fails, and ss=16 is likely infeasible per item 9's per-device sizing rule.
- **A remedy measured on one lens does not port.** DESI234 was fixed by ss=2 -> 4; on DESI257 ss=4
  fails. **Compactness, not Sersic index, tracks the error**: DESI257 is *less* cuspy by index
  (`n_sersic` 2.52, 1.30 against DESI234's 6.74, 2.45) but its inner lens-light component is ~3x
  more compact (`R_sersic` 0.225" against 0.746"). Measure per system.
- **Crossing the ss=1 -> ss>=2 boundary changes the PSF model, not the quadrature.** `supersample=1`
  convolves the native-resolution kernel on the detector grid; `supersample>=2` convolves a
  subgrid kernel before pooling. The gap is **3.69 sigma on DESI257, 5.02 sigma on DESI234**, and
  is **not reducible by supersampling**. Any flux-scale calibration rendered at ss=1 for a fit
  that runs at ss>=2 inherits it — DESI257's does.
- **The experimental adaptive supersampler cannot sidestep it.**
  `gigalens.jax.experimental.adaptive_supersample` reaches tolerance cheaply — factor-8 opt-in
  clears 0.2 sigma/px at 187k light-evaluation points, 2.2x cheaper than the *failing* uniform
  ss=8 rung — but only in its own `bin_first` convention; it has **no subgrid path**. Measured
  against subgrid it fails at 3.70-4.03 sigma and barely moves as the grid refines, because that
  number is the convention gap again. It also freezes its factor map from the **data** at
  construction, so a burn-in proposal displaced more than `dilate` px from observed flux is
  silently under-resolved — a real constraint on HMC exploration.
  (`DESI257/analysis/adaptive_supersample_check.py`.)
- **Reweighting an existing posterior to a finer grid will not work.** The per-draw `delta_chi2`
  spread is ~17 nats, which guarantees weight collapse. The error tilts the likelihood rather
  than offsetting it, so a refit is required.

**What it costs a headline number, and how to state it honestly.** First-order linear response —
regress the per-draw log-likelihood perturbation on the mass parameters, propagate through the
posterior covariance (`DESI257/analysis/theta_e_systematic.py`) — gives `Delta theta_E = +0.0106"`
= 4.12 posterior sd, 5.3x DESI257's quoted +/-0.002" half-width. But `R^2 = 0.197` and the
regression's overall test is **`F(6,53) = 2.167`, p = 0.061**: the gradient is *not* statistically
distinguishable from zero at 5%, and a "first-order" shift of 4.1 sd is not small in the sense
linear response requires. **So the reportable form is a systematic of order 0.01" to state, not a
correction to apply** — and a statistical interval must not be quoted as a total error bar. The
general rule: **when the grid systematic exceeds the statistical interval, quote fewer digits.**

**The blocker is a modelling decision, not an engineering one.** Reaching a certified grid
affordably means adopting bin-first convolution, which must first be validated against how each
system's PSF was actually measured (bin-first assumes an already pixel-integrated, e.g.
star-stacked, empirical PSF). Until that is settled, DESI257 stays at ss=2 — the most-tested
pathway — with the systematic stated. **Nothing here is a defect in gigalens**; the library
measured all of it, and its `undersampling` module is what made the problem visible.

---

## 11. gigalens renamed its parameter paths, silently reordering every stored checkpoint — FIXED 2026-10-03

**Every checkpoint this project wrote decoded to wrong physical values**, with no error, in
every tool that reads one. DESI234's `r3_fg` read back as `theta_E = 231.93"`.

**The dates are the whole story.** Upstream commit `cc5a078`, *"Put names in parameter keys;
order z columns by derivation, not key text"*, is dated **2026-08-07**. Our editable checkout
of nanolens pulled it on **2026-08-12** (`git reflog`: the 08-12 pull is the first that
contains it; the 2026-09-25 pull, which earlier versions of this note wrongly blamed, merely
also contains it). Every fit on all three systems finished by **2026-08-11**. So the entire
run archive was written under the old order and is read under the new one — not some of it,
all of it.

The flat-`z` column order is *derived* from the scene (`CLAUDE.md`, "the scene API"), never
declared. gigalens used to spell component indices into the parameter paths and sort them, so
`planes/0/light/0/Ie` came first and lens *light* preceded lens *mass*. It now spells the
scene's own keys and orders columns by **derivation** — planes in order; per plane geometry,
mass, light; per component the profile's own declared parameter order — so
`planes/lens/mass/host/theta_E` is column 0 and mass precedes light. Same model, same 36 or 46
floats, different permutation — and an `.npz` of unconstrained samples carries no record of
which order it is in.

**This was not an oversight upstream, and that is worth stating plainly.** `cc5a078`'s message
anticipates the exact hazard and argues the change *reduces* it: under the old scheme column
order was the sorted order of the key text, so **any** key edit permuted `z` — including
reordering a plane's `mass=[a, b]` list, which is a bit-identical physics no-op that "silently
permuted every saved chain, MAP seed and covariance taken against it". Derivation order is the
fix: a rename provably cannot move a column, and a reorder moves only what was reordered.
Positional keys never avoided the hazard; they hid it behind an edit that looks innocent. The
new scheme is the better one. What bit us is the one-time transition across it, with an archive
of checkpoints that record no schema version.

This is the hazard `rebuild_model`'s docstring already warned about, arriving by a route its
guard did not cover: the `api: scene` check compares the *API*, and both sides are the scene API.
Everything else was faithfully reconstructed — lens, prior versions, `supersample`, `mask_circles`
— and the one unversioned thing, the column order, was the one that moved.

**Why it was caught at all:** only because `theta_E = 231.93"` is impossible. The same run's
`gamma` decoded to 1.42 against a recorded 2.95, and `q` to 0.939 against 0.143 — both wrong, both
entirely plausible, neither self-announcing. A reader who looked only at `gamma` would have
accepted it. **The published numbers were never wrong; they were computed before the rename. What
broke was the ability to re-derive them**, which is most of what the diagnostics are for.

**The fix.** `run_fit.save_map_checkpoint` has always stored `model.z_param_names` verbatim in
`map.npz`, so each run is self-describing. `diagnose.column_permutation` resolves the stored names
against the rebuilt model's **by name** — reducing today's key-named paths to the old index-named
scheme and requiring the two sets to match exactly, so a mismatch raises rather than permuting
hopefully. `load_checkpoint(run_dir, name, rm)` applies it; `rm.column_perm` is `None` when the
orders already agree, so new runs pay nothing. SVI's `scale_tril` is permuted as a covariance and
re-factored, not shuffled — a permuted Cholesky factor is not triangular.

Validated against numbers recorded *before* the rename, which nothing in today's code can
retro-fit (`tests/test_column_permutation.py`):

| run | recorded | decoded today |
|---|---|---|
| DESI234 `r3_fg` | `theta_E` 1.47259, `gamma` 2.9514, `q` 0.143 | 1.47259, 2.9514, 0.1436 |
| DESI234 `r3_unmasked` | 1.53863, 2.1300, 0.943 | 1.53863, 2.1300, 0.9440 |
| DESI238 `mclmc_long` | 1.478 ± 0.003, 1.800 | 1.47842, 1.7996 |
| DESI257 `r3_unmasked` | 1.991 [1.989, 1.994], 1.986 | 1.99141, 1.9853 |

All three systems' headline results reproduce. **DESI234's `q = 0.143` is confirmed, not an
artefact**, so round 4's premise stands.

Two further consequences:

- **`diagnose.column`'s `"mass/0/theta_E"` idiom was broken by the same rename** and now raises
  `KeyError` rather than mis-resolving — the good failure, but it broke `theta_E_check`,
  `corner_plot` and `test_foreground_roundtrip` at HEAD. A numeric component index in a suffix is
  now resolved *positionally* (the n-th component declared at that site), which is what the idiom
  always meant. Not by hardcoding the key `host`: nothing makes that name the host's.
- **Reading without `rm` is refused.** `point_estimate(run_dir)` raises when a run needs permuting,
  rather than returning stored-order columns. The failure mode is silence, so the guard matters
  more than the fix.
- **`model.constrained(...)`'s structured output is keyed by name too**, so every
  `params["planes"][0]["mass"][0]` broke with `KeyError: 0` — in `residual_figure`,
  `render_source_plane`, `source_component_params` and `plot_source.py`. **Use
  `model.component_params(params, plane, kind, j)`**, which gigalens documents as "the single
  supported way to reach a component's params" (with `plane_params` and `kind_params` for the
  intermediate levels). Do not index the tree.

  Two traps make that worth obeying rather than reimplementing. The keys are the component's
  *name* with a `str(index)` fallback when unnamed, which only `component_key` reproduces. And
  the tree's order is **not** `model.planes`' order: on `r3_fg`, `params["planes"]` comes out
  `{"source": ..., "lens": ...}`, reversed. A positional read there returns the wrong plane
  **while raising nothing**, since both planes carry `light` — the deflector would be rendered
  as the source. `docs/explanation/scene-design.md:106` warns against assuming that order
  outright.

  **I wrote my own `plane_params`/`nth` before checking, then deleted them.** They worked —
  the swap to the library's accessors reproduced every number byte-for-byte (θ_E 1.47259, SIS
  0.3702, χ²_ν 1.088, `plot_source`'s component-sum residual 4.768e-07) — but working is not
  the standard. CLAUDE.md's rule is that a hand-rolled version of something the library
  already has is unvalidated, and mine was also narrower: it had no unnamed-component
  fallback. **Read `GIGALENS_INDEX.md` and `docs/` before writing the helper, not after.**

**A separate, smaller finding, left open.** Recomputing a stored MAP vector's log posterior today
gives a few nats less than the run recorded (`r3_fg` −11.7, DESI257 `r3_unmasked` −1.3), entirely
in the prior + Jacobian term; the likelihood is unaffected. It is **not** a decode problem — the
unpermuted vector scores −324,530 against a recorded 19,422, five orders of magnitude away, while
the permuted one is within 12. Today's computation was checked and is correct: the `theta_E`
LogNormal and `gamma` TruncatedNormal terms agree with `scipy.stats` to 1e-8, and the bijector's
`forward_log_det_jacobian` agrees with a numerical `slogdet` of its Jacobian to 2e-7. So the
*stored scalars* are stale, written under an older normalisation, and the current numbers are the
trustworthy ones. gigalens added two opt-in reparameterisations after these runs
(`unconstrain="gaussian"`, `e69c769`; `TruncatedDiskNormal`, `27cd141`), but `priors.py` uses
neither, so neither explains it and the source is unattributed. It affects only MAP-vs-HMC nat
comparisons *across* the rename — never a parameter value, never R̂/ESS, never χ²_ν.

**gigalens already solves this for new runs, and we are not using it.** `gigalens.chains`
provides `save_chain`/`load_chain`: the saver stores `z_param_names` beside the samples *and* a
"physical witness" — a few decoded draws — and the loader checks both, so a column reorder
raises and `load_chain(..., reorder=True)` migrates it by matching names. It also catches
something `column_permutation` cannot: a **prior-support change**, which leaves every name
intact while silently changing what the same `z` decodes to.

It does not rescue the existing archive. Our checkpoints are plain `.npz` files holding
`samples` (and `param_names` only in `map.npz`), not `save_chain` files, and `reorder=True`
requires the two name *sets* to match — ours differ by scheme entirely
(`planes/0/light/0/Ie` vs `planes/lens/mass/host/theta_E`), so it would raise. Hence
`column_permutation`, which reduces today's names to the old scheme to make the match
checkable. **That function is a migration for the pre-2026-08-12 archive and should not
outlive it: `run_fit` should write `save_chain` files.** Logged as a next step below.

**The general lesson.** A derived quantity is only reproducible if something records what it was
derived from. Prior *versions* were versioned because the project's rule says a support change
must be; the column order was not, because nobody chose it — it fell out of the library. The
checkpoints survived only because `save_map_checkpoint` happened to store the names. **Record what
you derived from, not just what you derived.**

---

## Next steps, ordered by value

Closed items are not repeated here; see the item bodies. What remains:

1. **Settle the PSF convolution convention** (item 10). It gates every grid decision on every
   system, it is the difference between a 3.7 sigma systematic and none, and the check is cheap:
   determine how each system's PSF was measured, and validate a bin-first render against a
   `lenstronomy` reference (`CLAUDE.md`'s ground-truth rule). Only then is a certified-grid refit
   worth the compute.

2. **Re-fit DESI234 round 4 with the deflector ellipticity constrained** (items 5, 7a).
   `r3_fg` is rejected: `q = 0.143`, `gamma = 2.951`. The question round 4 asks is whether a
   foreground component survives when the deflector is required to be round. Until it is
   answered, nothing about the interloper's mass is established.

3. **Write checkpoints with `gigalens.chains.save_chain`** (item 11). It stores the column
   names and a decoded physical witness beside the samples, so a column reorder *and* a
   prior-support change both raise on load instead of decoding silently to wrong values. Our
   own `column_permutation` covers neither case for new runs and the second case not at all;
   it exists only to migrate the pre-2026-08-12 archive, and should be retired once nothing
   reads that archive. Cheap, and it closes the failure mode item 11 is about rather than
   detecting it after the fact.

4. **Re-run `prior_posterior_tension` over every archived run on all three systems.** New as of
   2026-08-10 and applied only to DESI234. DESI238's converged result and DESI257's rounds have
   never been checked for a parameter sitting far out on an unbounded prior, which is exactly
   the failure it was built to catch.

5. **Isolate the MCLMC freeze** (item 2) by capturing the adapted step size and trajectory
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

6. **Measure `--map-steps` per model variant** (item 6). The check reports the deficit; nothing
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

- **`verify.py` counted a SKIPPED check as a pass — fixed 2026-08-11.** A check that cannot run on
  a lens signals it by *returning* a `"SKIPPED: ..."` detail rather than raising, so it landed with
  `ok=True`, and the summary printed `len(RESULTS) - n_fail` as the pass count.
  `verify.py --lens DESI257` therefore reported **5/5 checks passed** when only four had run — and
  the one that had not was check 5, the SVI covariance guard, which is the last thing that should
  be silently credited. `verify.summarize` / `verify.status_of` now report passes, skips and
  failures as three categories (the accounting was split out of `main` so it is testable without
  running the suite against a real lens); tested in `tests/test_verify_summary.py`, which pins both
  directions — a skip is not a pass, and a pass whose detail merely *mentions* skipping is still a
  pass. Exit status is still driven by failures alone. **A skipped check is honest; the bug was
  reporting absent evidence as evidence.**

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
