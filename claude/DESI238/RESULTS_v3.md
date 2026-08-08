# DESI238 round 3 — the SVI fix, a bulge+disk deflector, and a converged error bar

> **Round 3 of 5. Superseded — current result is [`RESULTS_v5.md`](RESULTS_v5.md):
> `theta_E = 1.478 ± 0.003"`, all 36 parameters at R̂ ≤ 1.003, run `mclmc_long`.
> Do not quote this round's number.**
> This round: `theta_E = 1.480 ± 0.003"` (R̂ 1.001, ESS 94,186, run `bulgedisk_long`) — the
> first error bar to survive its own diagnostic, though only 26/36 parameters reached
> R̂ ≤ 1.01. Establishes the 36-parameter bulge+disk model (ΔBIC ≈ −1762) that rounds 4 and
> 5 both keep, and fixes the round-2 SVI divergence. Superseded on the b_n correction
> (`RESULTS_v4.md`) and then the scene-API port (`RESULTS_v5.md`).

2026-07-31. Three runs on one H100, `--seed 0`, ~8 minutes each except the last (~28).

| run | params | model | chi²ᵥ | worst Rhat | Rhat > 1.01 |
|---|---|---|---|---|---|
| `twosource` (round 2, **broken**) | 29 | 2 source, 1 lens light | 1.302 | 7.17 | 29/29 |
| `twosource_v2` | 29 | 2 source, 1 lens light | 1.277 | 1.544 | 23/29 |
| `bulgedisk` | 36 | 2 source, **2 lens light** | **0.991** | 3.242 | 33/36 |
| `bulgedisk_long` | 36 | as above, **4× draws** | 0.989 | **1.338** | **10/36** |

All used `--max-leapfrog 60` and the new warmup-then-decay SVI schedule.

Three results, in order of how much they matter:

1. **`theta_E = 1.480 ± 0.003"`, Rhat 1.001, ESS 94,186** — the first error bar from a
   multi-component fit that survives its own diagnostic. It supersedes round 1's
   `1.516 ± 0.002"`. ~~The 0.036" gap between them is a model systematic an order of
   magnitude larger than either statistical error.~~ **Corrected 2026-07-31 (see the
   correction under "The Einstein radius" below): the round-1 model is decisively rejected
   by the data, so the gap is a bias that has been removed, not a systematic uncertainty
   that remains. Quote `± 0.003"`.**
2. **The bulge+disk deflector is preferred by ΔBIC ≈ −1762** and fixes the coherent core
   residual, without cannibalising arc flux.
3. **The SVI divergence that broke round 2 is fixed**, and the residual non-convergence it
   was confused with turned out to be nothing more than short chains.

## The SVI divergence is fixed

This was the round-2 blocker, and the fix is unambiguous:

| | `twosource` | `twosource_v2` | `bulgedisk` |
|---|---|---|---|
| SVI covariance condition | 8.8e+11 | **3.4e+07** | **3.0e+07** |
| ELBO sd, last 300 steps | 7,384 | **6.6** | 41.6 |
| worst Rhat | 7.17 | 1.544 | 3.242 |

A 26,000× improvement in conditioning. Both new runs are better conditioned than three of
the four healthy 22-parameter runs from round 1. The guard
(`compat.check_svi_covariance`, hard-fail above 1e10) did not fire on either.

**`theta_E` reproduces at 1.4837 ± 0.0024"** against the broken run's 1.484 ± 0.002",
confirming what was suspected: the point estimate was never affected, only the error bar.
The MAP/SVI optimum does not depend on the momentum distribution.

**The round-2 "slow mixing" diagnosis was wrong, but not entirely wrong.** Fixing the
preconditioner took worst Rhat from 7.17 to 1.544 — most of the failure was the indefinite
mass matrix. What remains (23/29 above 1.01) is now genuinely under-mixed, so a longer
chain is the right next step for the residue, just not for the bulk. See the correction
block in `RESULTS_v2.md`.

## The second deflector component works

`chi²ᵥ = 0.991`, from 1.302. The decisive test was not chi² but whether the coherent
radial residual identified in round 2 dissolved. Normalised residual, azimuthally averaged
about the lens-light centre, in sigma:

    r [px]      0.5    1.5    2.5    3.5    4.5    5.5    6.5    7.5    8.5    9.5   10.5
    1 component -3.92  -0.07   0.61   0.51   0.40   1.17   0.05   0.14  -0.42  -0.66  -0.46
    2 component -1.49   0.74  -0.21  -0.53  -0.39   0.23  -0.35  -0.17  -0.31  -0.37  -0.24

The one-component profile ramps monotonically from +0.6 at r = 2.5 px down to −0.7 at
r = 9.5 px — the S-curve a single Sersic cannot avoid. The two-component profile
alternates sign and stays inside ±0.75. The central pixel improved from −3.9σ to −1.5σ.
Core chi² (r ≤ 8 px) **7.09 → 2.97**.

Recovered geometry is a textbook bulge+disk, and close to the 1D indicator fit that
motivated the priors:

| | R_sersic (") | n_sersic | Ie |
|---|---|---|---|
| bulge (`lens_light0`) | 0.326 | 3.19 | 251.1 |
| disk (`lens_light1`) | 2.730 | 2.25 | 7.8 |

An 8.4× separation in scale radius, from priors that overlap at ~1σ. The data chose this.

## How strongly the data prefer bulge+disk

Two chi²ᵥ values are an assertion, not a measurement. The numbers, at each run's point
estimate over N = 6400 pixels (`/tmp/model_compare.py`; k read from the flattened
unconstrained vector, not assumed):

| | k | chi² total | chi²/N | chi²/(N−k) | log L | BIC |
|---|---|---|---|---|---|---|
| `twosource_v2` | 29 | 8170.5 | 1.2766 | 1.2824 | 18146.6 | −36039.0 |
| `bulgedisk` | 36 | 6340.6 | 0.9907 | 0.9963 | 19058.4 | −37801.4 |

**Δlog L = +911.9 for 7 added parameters; ΔBIC = −1762, ΔAIC = −1810.** BIC charges
ln(6400) ≈ 8.8 per parameter and the fit returns ~130 per parameter, so the penalty is
irrelevant at this margin. This is not a close call, and it is worth saying so with a
number rather than leaving it implicit in 1.302 → 0.991.

Two caveats on that figure. It is evaluated at point estimates, so it is a
maximum-likelihood comparison, not a Bayes factor — a proper marginal likelihood would
penalise the 36-parameter model's larger prior volume more heavily, though not by 1762.
And `ForwardProbModel`'s likelihood does include the −Σ log σ normalisation (it builds
`tfd.Normal(im_sim, err_map)` and takes the full log-density), which is not constant
between models here because `err_map` depends on the model image. I checked whether that
term was doing any of the work: it contributes about −3 of the +912, so the comparison is
carried by chi² after all. Worth confirming rather than assuming, since a
parameter-dependent noise map is exactly the setting where it need not have been.

### The noise scale, and why 0.991 should not be read as a triumph

`background_rms = 0.0075` is one number drawn from six blank patches spanning
0.0070–0.0092. Propagating that range:

| background_rms | `twosource_v2` chi²/N | `bulgedisk` chi²/N | ΔBIC |
|---|---|---|---|
| 0.0070 | 1.348 | 1.049 | −1848 |
| 0.0075 (fiducial) | 1.277 | **0.991** | −1762 |
| 0.0092 | 1.073 | 0.824 | −1526 |

**`bulgedisk`'s chi²ᵥ ranges 0.82–1.05 over the plausible noise scale.** Landing on 0.991
is therefore not evidence that the model is now correct at the noise level — it is
consistent with that, but a ±10% error on the noise scale is ±20% on chi², which is larger
than the distance from 1.0. Quote the improvement, not the absolute value.

The model comparison is untouched by this: ΔBIC stays between −1526 and −1848 across the
whole range, because both models are evaluated under the same noise map and the scale
factor largely cancels. **The preference for bulge+disk is robust; "we have reached
chi²ᵥ = 1" is not a claim this data supports.** This is the argument for building the
per-pixel exposure map from `WHT` — see next steps.

## No arc cannibalisation

This was the main risk of the exercise: a flexible second deflector component contributes
flux near the Einstein radius and could eat arc flux, biasing `theta_E`. It did not
happen — the clump ratios *improved*.

| clump | round 2 | `bulgedisk` |
|---|---|---|
| SW arc | 1.09 | **1.03** |
| E blob | 0.99 | **0.99** |
| W clump | 1.02 | **1.02** |
| N arc | 0.91 | 0.83 |

Three of four within 3%. `theta_E` moved only 1.484 → 1.480, well inside the shift already
seen between round 1 and round 2, so there is no sign of the deflector stealing arc flux.

## What is not fixed

- ~~**Convergence, at both sizes.**~~ **Resolved for `bulgedisk` by `bulgedisk_long`** — see
  above. What was written here (33/36 failing, worst 3.24, read as "the expected signature
  of a bulge–disk degeneracy") was wrong about the cause: 4× the draws took it to 10/36 and
  1.338, so it was short chains, not degeneracy. The claim that `theta_E = 1.516"` from
  round 1 "remains the quotable value" no longer holds; `1.480 ± 0.003"` from
  `bulgedisk_long` is better converged and comes from a better model. `twosource_v2`'s
  23/29 stands uncorrected — it was never re-run long, and there is now little reason to,
  since `bulgedisk` supersedes it.
- **`source1.n_sersic` pinned at its lower bound** (0.606 here, 0.522 and 0.4% of range in
  `bulgedisk_long`, where it is the worst Rhat in the run at 1.338 with ESS 201). This is
  the third round in which a `n_sersic` has pinned at 0.5, and it is now the single
  clearest unresolved defect in the model. The floor is deliberate — gigalens'
  `bn = 1.9992n − 0.3271` is 3% wrong at n = 0.5 and 13.6% wrong at n = 0.3 — so the fix is
  a different profile, not a lower bound. A parameter on a boundary cannot mix, so its bad
  Rhat is a symptom of the pinning and not independent evidence of anything.
- **`lens_light1.n_sersic` sits at 87% of `Uniform(0.5, 2.5)`** (79.6% in `bulgedisk_long`)
  and has the worst *non-boundary* Rhat, 1.096. Not pinned by the 2% rule, but close enough
  to the ceiling to be worth widening next round (a *support* change: bump the prior
  version).
- ~~**ELBO tail sd 41.6 at 36 parameters**~~ — **not a real effect**, see the correction
  under "The convergence residue was chain length". An identically-configured re-run gives
  5.7.

## The convergence residue was chain length

`bulgedisk_long`: identical to `bulgedisk` in every respect — same seed, same priors, same
MAP and SVI settings, verified by an identical SVI starting loss — except 3000 draws after
1000 burnin instead of 750 after 250. One variable, so the comparison is clean.

| | `bulgedisk` | `bulgedisk_long` |
|---|---|---|
| draws × burnin | 750 × 250 | 3000 × 1000 |
| worst Rhat | 3.242 | **1.338** |
| params with Rhat > 1.01 | 33/36 | **10/36** |
| chi²ᵥ | 0.991 | 0.989 |
| SVI cov condition | 3.0e+07 | 3.8e+07 |

**The residue was under-mixing, not a posterior ridge.** Rhat fell toward 1 rather than
plateauing near 3, which is the outcome that says the geometry is samplable and the chains
were simply too short. No reparametrization is needed.

**`theta_E = 1.480 ± 0.003"` is now quotable** — Rhat 1.001, ESS 94,186. This is the first
error bar from a multi-component fit that survives its own diagnostic, and it supersedes
round 1's `1.516 ± 0.002"` as the best available value. The two do not agree within their
quoted uncertainties; the shift is a *model* difference (a second source component and a
second deflector component), not a sampling difference, and the round-3 model is the better
one by ΔBIC ≈ −1762. ~~Treat the 0.036" gap as the honest systematic floor on `theta_E` from
light-profile choice — it is an order of magnitude larger than either statistical error bar,
which is the more important number for any downstream use.~~

> **Correction, 2026-07-31.** The struck sentence is wrong, and the reasoning behind it was
> backwards. Treating a model-to-model spread as a systematic uncertainty is only valid when
> the models are *competitive* — when the data cannot decide between them, so both remain
> live descriptions and the spread measures a real ambiguity. That is not the situation
> here. Round 1's single-Sérsic-source, single-Sérsic-deflector model is rejected far more
> decisively than round 2 was: at the point estimates, chi² total ≈ 11,757 (k = 22) against
> `bulgedisk`'s 6,340.6 (k = 36), giving **Δlog L ≈ +2708 for 14 added parameters and
> ΔBIC ≈ −5294** (the BIC penalty is 123, i.e. irrelevant at this margin).
>
> A model the data rejects at ΔBIC ≈ −5294 is not a competing estimate of `theta_E`; it is
> a wrong one. Its value does not belong in an error budget. **The correct statement is that
> round 1 is superseded, and the 0.036" is the size of a bias that has been removed — not an
> uncertainty that remains.** The quotable uncertainty on `theta_E` is `± 0.003"` (Rhat
> 1.001, ESS 94,186), and inflating it to cover a rejected model would be reporting a range
> the data has already ruled out.
>
> Two caveats that do survive. First, all the Δlog L / ΔBIC figures here are evaluated at
> point estimates, so they are maximum-likelihood comparisons, not Bayes factors; a proper
> marginal likelihood would penalise the 36-parameter model's larger prior volume more
> heavily, though nowhere near enough to close gaps of this size. Second, `± 0.003"` is the
> statistical error *within the adopted model family*. It does not cover systematics this
> comparison never probed — the noise model (§ "What chi² does and does not establish"), the
> single-scalar exposure time, or the EPL mass parametrisation itself. Those remain open;
> they are simply not measured by the round-1/round-3 difference.

What still fails Rhat is confined and interpretable: the two deflector components' shape
parameters (`lens_light1.n_sersic` 1.096, `R_sersic` 1.042, `e2` 1.040) and
`source1.n_sersic` (1.338, ESS 201). The last is pinned at the 0.5 floor at 0.4% of its
range — a parameter on a boundary cannot mix, so its Rhat is expected to fail and its value
is a bound, not a measurement. **All eight lens-mass parameters have Rhat ≤ 1.002.** The
non-convergence never touched the lensing.

### Two corrections to the sections above

- **The "ELBO tail sd 41.6 at 36 parameters" concern was over-read.** `bulgedisk_long`'s
  SVI stage is configured identically to `bulgedisk`'s and gives 5.7 against 41.6. A 7×
  spread at fixed settings is stochasticity in the ELBO estimator, so the 6.6-vs-41.6
  comparison between `twosource_v2` and `bulgedisk` does not establish a trend with
  dimension. Two samples of a noisy statistic are not a trend.
- **The progress bar labelled "ELBO" reports the loss**, which is why it descends. Cosmetic,
  but it invites exactly the misreading that the SVI stage is diverging when it is not.

Practical note: at 3000 draws × 50 chains × 36 parameters the run held 62.7 GB of an 80 GB
H100 and HMC took ~20 minutes against ~2. Memory, not compute, is the binding constraint on
a further 4×; that would need thinning.

## Next steps

Ordered by what gates a quotable result. **Items 1 and 3 are done** — see "The convergence
residue was chain length" and "How strongly the data prefer bulge+disk" above.

1. ~~**Settle convergence by brute force before reparametrizing.**~~ **Done.** 4× draws
   took worst Rhat 3.242 → 1.338 and the failing count 33/36 → 10/36. Chain length, not a
   ridge; no reparametrization needed. Remaining work here is narrow: widen
   `lens_light1.n_sersic` (now at 79.6% of `Uniform(0.5, 2.5)`, still the worst
   non-boundary Rhat at 1.096), which is a *support* change and needs a prior-version bump.
   Deliberately not folded into the long run, since it would have confounded the one thing
   that run was measuring.
2. **Re-run the falsification test with a genuinely exclusionary prior** (`theta_E`
   truncated below ~1.7"). Outstanding since round 1, and it is the actual science
   question — whether the E blob is a counter-image. Deferring it was defensible while the
   source model was inadequate, because a failed A/B could be blamed on the model. At
   chi²ᵥ ≈ 1 that excuse is gone.
3. **Quantify the bulge+disk preference instead of asserting it.** Δchi² ≈ 1800 for 7 added
   parameters is overwhelming by any information criterion, but the number belongs in this
   file explicitly rather than implicit in two chi²ᵥ values.
4. **Treat `chi²ᵥ = 0.991` with suspicion, not satisfaction.** The blank-patch
   `background_rms` spans 0.0070–0.0092 — about ±10%, hence ±20% in chi². "We reached
   unity" is therefore soft at roughly the level of the claim itself. The *relative*
   improvement is robust (identical noise model in both runs); the absolute statement is
   not. Building the per-pixel exposure map from the `WHT` extension is now *more* worth
   doing than before, not less: at chi²ᵥ = 1.3 the noise model was a second-order worry, at
   0.991 it is the leading systematic.

One cheap habit alongside these: have `run_fit.py` record the SVI condition number in
`metadata.json`. The 1e10 threshold is interpolated between 5.9e+08 and 8.8e+11 with
nothing observed near it; logging the number on every run calibrates it for free.

## Reproduce

```bash
export PYTHONPATH=gigalens/src:claude/DESI238/model NCCL_NVLS_ENABLE=0
GIGALENS_DEVICE=0 python claude/DESI238/model/run_fit.py \
    --label twosource_v2 --seed 0 --lens-light-n-max 10 \
    --n-source 2 --n-lens-light 1 --max-leapfrog 60
GIGALENS_DEVICE=0 python claude/DESI238/model/run_fit.py \
    --label bulgedisk --seed 0 --lens-light-n-max 10 \
    --n-source 2 --n-lens-light 2 --max-leapfrog 60
GIGALENS_DEVICE=0 python claude/DESI238/model/run_fit.py \
    --label bulgedisk_long --seed 0 --lens-light-n-max 10 \
    --n-source 2 --n-lens-light 2 --max-leapfrog 60 \
    --num-results 3000 --burnin 1000
```

The model-comparison numbers come from `/tmp/model_compare.py` (scratch, not checked in);
everything else is `diagnose.py <run_dir>`.
