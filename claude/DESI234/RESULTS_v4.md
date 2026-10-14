# DESI234 round 4 — the foreground model with the deflector ellipticity bounded

**Fitted 2026-10-05.** Two runs: `runs/r4_fg_q` (46 parameters, `mass_prior_version=2`) and
`runs/r4_fg_q2` (the same model with a substantially better surrogate). Launch scripts:
`runs/launch_r4.sh`, `runs/launch_r4_svi.sh`, `runs/launch_r4_surrogate.sh`.
Sections 1–7 describe `r4_fg_q`; **section 8 describes `r4_fg_q2` and is where the round
ends.**

> **Status: neither fit converged, and nothing in either is quotable.** In both, 46 of 46
> parameters sit above the 1.01 gate (worst R̂ 2.125 and 2.330) and the Vehtari ESS floor
> **fails** (minimum bulk-ESS 63 and 56, against a floor of 400 total and 50 per split
> half). Below that floor R̂ is uninterpretable in *both* directions, so this round neither
> confirms nor refutes anything about the posterior.
>
> **The surrogate was tried and is not the cause (§8).** Tripling MAP and SVI closed the
> MAP-vs-HMC deficit from 193.75 nats to 69.60 and produced *no* convergence improvement —
> worst R̂ and minimum ESS both moved slightly the wrong way. Chain length was already
> measured sublinear twice. What remains is the model itself.
>
> **Round 2's `theta_E = 1.539 +0.003/-0.002"` remains the system's number.** Round 4 does
> not touch it.
>
> **What round 4 did establish, and it is not what it set out to establish.** The round was
> designed on the premise that `r3_fg`'s 7:1 needle was the fit escaping through an
> unbounded prior, and that bounding the ellipticity would let a physical deflector emerge.
> That premise is now in doubt. With `q >= 0.2` enforced, the deflector went to the new wall
> and stayed there: `q = 0.295`, with `e1 = -0.4692` against a bound at `-0.4714`, and
> **96.3% of all draws within 0.01 of that bound.** The bound did not reveal a physical
> solution; it relocated the pathology from `q = 0.143` to `q = 0.295`. Two independent
> surrogates reproduce this to within 0.007 in `q` (§8).

## 1. What this round changed against `r3_fg`

| | `r3_fg` (round 3) | `r4_fg_q` (round 4) |
|---|---|---|
| host EPL `e1`/`e2` prior | `Normal(0, 0.15)`, **unbounded** | `TruncatedNormal(0, 0.15, ±0.4714)` (`mass_prior_version=2`) |
| burnin | 5,000 | 25,000 |
| draws kept | 750 | 4,000 |
| MAP batch (`--n-samples`) | 1024 | 512 |
| MAP steps | 5,000 | 8,000 |
| chains returned | 50 | 48 |

Everything else is held at `r3_fg`'s value: 2 source components, 2 lens-light components,
unmasked, `supersample = 4`, MCLMC, `--n-vi 1024`, `--svi-steps 12000`,
`--svi-peak-lr 1e-3`, seed 0.

Three of those differences are not modelling choices and are recorded so they are not later
mistaken for any:

- **`--n-samples 1024 -> 512`** is forced by hardware. Round 3 ran on 80 GB A100s; this
  allocation has 40 GB cards. Measured before launch: 1024 dies `RESOURCE_EXHAUSTED` asking
  46.37 GiB, 768 dies asking 34.19 GiB, 640 and 512 run. `--map-steps` rose 5000 -> 8000 to
  partly pay for the smaller batch.
- **50 chains -> 48** is MCLMC sharding: 50 chains over 4 devices rounds down to 12 per
  device. `r3_fg` used 2 devices and kept all 50. This is benign for R̂ (48 chains is far
  above the floor's minimum of 4) but it is a difference, so it is stated.
- **The first launch attempt failed** in the SVI covariance guard at condition 9.958e+12
  against the 1e10 limit. The guard was **not** raised. The cause was `--n-vi 512`, which
  the first attempt had lowered along with the MAP batch; `r3_fg` passed the same guard at
  condition 2.397e+07 with `--n-vi 1024` and the identical learning rate and step count.
  Restoring `--n-vi 1024` — after measuring that SVI's own buffer fits on these cards, which
  MAP's does not — gave condition **2.639e+07**, in the healthy band. The two batch sizes
  were never required to match.

## 2. The ellipticity bound itself

Verified before launch, not assumed. The bound is `|e_i| <= 0.4714`, the inscribed square of
gigalens' own soft `q >= 0.2` plausibility floor (`priors.MASS_V2_ELLIP_ABS_MAX`,
`gigalens.physicality.axis_ratio_constraint`):

| | v1 (rounds 1–3) | v2 (round 4) |
|---|---|---|
| min `q` over 20,000 prior draws | −0.626 | 0.2445 |
| fraction of prior draws with `q < 0.2` | 79.8% | 0 |

Prior mass removed at `ellip_sigma = 0.15`, over 4e6 draws: **3.359e-3** for the box and
**5.150e-5** for the true `q >= 0.2` disc — agreeing with `priors.py`'s 3.35e-3 and 5.15e-5
to the three significant figures that file quotes. `r3_fg`'s `c = 0.7498` is outside the bound; round 2's `q = 0.900` and
`r3_unmasked`'s `q = 0.943` are nowhere near it. The bijector round-trips to 0.0 max |dz|
over 4,000 draws at 46 parameters, and the v1 and v2 flat-`z` column orders are identical,
so the version is the only thing that changed.

## 3. Convergence — the reason nothing below is a measurement

```
min ESS over all parameters (bulk and tail) 63   [lens/mass/host/center_x]
  total          >= 400      63    FAIL
  per split half >= 50      0.7    FAIL   (48 chains -> 96 split halves)
  chains         >= 4        48    ok
46/46 parameters have Rhat > 1.01 (worst 2.125)
```

R̂ here is the **rank-normalized folded** statistic gated at 1.01, not the classic PSRF
gated near 1.2 in Foundry I. Worst offenders: `lens/mass/host/center_x` (2.125),
`lens/mass/host/gamma` (2.079), `source/light/src1/center_y` (1.899),
`lens/mass/shear/gamma1` (1.828), `lens/mass/host/center_y` (1.774).

**The ESS-vs-draws slope is 0.08 — sublinear.** `diagnose.py`'s verdict is that more draws
will not fix this. This run already spent 5× `r3_fg`'s burnin and 5.3× its draws, so chain
length is a lever that has been pulled and measured not to move. `rank_plots.png` shows the
chains splitting into two groups, some piling at rank 0 and others at rank 1, on every one
of the six worst parameters — multimodality or a ridge the sampler crosses rarely, not slow
diffusion within one mode.

**The surrogate is under-converged — and that lever has since been pulled, and was not the
cause.** MAP finished **193.750 nats below** the best scored HMC draw (18973.945 against
19167.695), and its χ² tail slope over the last 800 of 8000 steps was −0.00002/step — still
descending when it ran out of steps. ELBO tail sd 52.3, where DESI238 converged at 0.1. The
SVI *covariance* is healthy (condition 2.639e+07), so this was an under-converged optimum
rather than a broken one. `r4_fg_q2` fixed most of it and changed nothing: see §8.

## 4. The finding: the bound did not fix the deflector

This is the part of the round worth keeping, and it is a negative result.

| run | prior | `q` | `gamma` | converged |
|---|---|---|---|---|
| `r2_s2l2` | v1 (masked) | 0.900 | 2.199 | yes, R̂ ≤ 1.008 |
| `r3_unmasked` | v1 | 0.943 | 2.130 | no (frozen, R̂ 13.94) |
| `r3_fg` | v1, unbounded | **0.143** | **2.951** | no (R̂ 1.92) |
| `r4_fg_q` | **v2, `q >= 0.2`** | **0.295** | **2.898** | no (R̂ 2.125, ESS floor failed) |

`e1 = -0.4692` against the bound at `-0.4714`: `diagnose.py` reports it **PINNED at lower
bound**, at 0.2% of its range. 96.3% of draws lie within 0.01 of the bound and 24.2% within
0.001. `gamma = 2.8977` sits at 94.9% of its `[1, 3]` range and 3.65 prior sigma above
centre, with a posterior 0.072× the prior's width — the same signature `r3_fg` showed at
2.951, barely moved.

**The pinning is not an artifact of the bad MAP start.** This is the one test that
distinguishes "the sampler never left a poor starting region" from "the likelihood wants the
excluded region", and it was run before concluding anything. Scoring 3,000 thinned draws by
log posterior:

| | fraction of draws with `e1` within 0.01 of the bound | median `q` | median `gamma` |
|---|---|---|---|
| top 10% by log posterior | 0.960 | 0.3009 | 2.9162 |
| all draws | 0.963 | 0.2951 | 2.8978 |

The best-fitting draws are pressed against the wall just as hard as the typical ones. A
better starting point moves a sampler to the mode; it does not move a mode off a boundary.

**JUDGEMENT (unmeasured):** the foreground SIS and the host EPL are competing to supply the
same deflection, and the host flattens to a needle because the SIS is taking mass that
belongs to it — so the pathology is a degeneracy between the two mass components rather than
a property of either. *Falsified by:* a fit with the foreground SIS `theta_E` fixed at its
round-3 value (or at zero) while the host EPL stays free under v2 — if the host relaxes to
`q > 0.5`, the degeneracy is the cause; if it still pins, it is not.

Supporting the reading, though not establishing it: the lens-plane critical curve has swollen
to **1.8222"**, against the host's own `theta_E = 1.4582"`. `diagnose.py` attributes the
+0.3640" (25.0%) excess to the second mass component. In `residuals.png` the critical curve
visibly engulfs the interloper, so the two deflectors are acting as one extended mass rather
than as a lens plus a perturber. The foreground SIS fits `theta_E = 0.4752 +0.046/-0.042"`
(R̂ 1.609) — larger than `r3_fg`'s withdrawn 0.370" — but **that number is not a detection
and must not be quoted**: it is a statistic computed on a non-converged fit whose host mass
model is itself unphysical, which is precisely the error `RESULTS_v3.md` retracted on
2026-08-10.

## 5. Residuals and χ²ᵥ, weighted last

Full-field `chi2_nu = 1.199` over all 6400 pixels; core-masked (display only, radius 0.520",
198 px excluded) `1.117`. Both are **inflated relative to a published value**, which would
mask the core by convention.

χ²ᵥ is quoted here as an improvement on nothing, because it cannot be: the noise scale
carries ±15% patch-to-patch spread (`README.md`), which is ±30% on χ² — several times the
distance from unity. `r3_fg` had the *best* χ²ᵥ on this system (1.082) and a 7:1 needle. A
χ²ᵥ near 1 is not evidence the model is right.

`residuals.png`, read rather than summarised: the arc and counter-image are reproduced and
the interloper is fitted rather than left as a hole. What remains is a striped ±4σ dipole at
the deflector core — the PSF mismatch `RESULTS_v2.md` already records, unchanged — plus
coherent positive residuals on the southern arc near (+1, −1.7)" and (+0.8, −2.0)",
reaching about +4σ, which are the unmodelled arc knots round 2 also saw. Both source
components sit inside the caustic in `residuals.png`'s source-plane panel, and
`src1` at `R_sersic = 0.0611"` is compact rather than a smooth pedestal — so neither is the
"large component outside the caustic buying χ²" failure mode. Note `src1`'s `R_sersic` is
itself 3.13 prior sigma low with a posterior 0.088× the prior width, which is the third
tension flag and is unexplained.

## 6. What is and is not settled

**Settled by this round:**

- Bounding the host ellipticity at gigalens' `q >= 0.2` floor does **not** rescue the
  foreground model. The deflector pins against whatever bound it is given, and does so at
  the highest-posterior draws, not merely in the tails — reproduced across two surrogates
  differing by 124 nats (§8).
- The non-convergence is not chain length. The ESS slope is 0.08 after 5× the burnin and
  5.3× the draws of round 3, and 0.04 on the second run.
- The non-convergence is not surrogate quality either (§8). Closing two thirds of the MAP
  deficit improved R̂ not at all.
- `--n-vi` and `--n-samples` are independent budgets. Lowering both together for memory cost
  one failed launch.

**Not settled, and explicitly still open:**

- **Round 4's own question — does a foreground mass component survive when the deflector is
  required to be round? — is NOT answered.** The deflector is still not round; it is merely
  less flat. The question needs a fit in which the host is physical for a reason other than
  a prior wall.
- Whether the host–SIS degeneracy is the cause (the tagged judgement in §4, with its
  falsifier).
- `r3_unmasked`'s frozen chains. float64 remains untried (`pipeline/data.py:118` pins
  float32); this round did not touch it.

## 7. What was run, in order

| attempt | script | outcome |
|---|---|---|
| 1 | `runs/launch_r4.sh` | MAP succeeded (χ²ᵥ 1.287, best lp 1.897e+04, `map.npz` saved); **died** in the SVI guard at condition 9.958e+12 |
| 2 | `runs/launch_r4_svi.sh` | `r4_fg_q`: resumed `--stages svi,hmc` from that MAP with `--n-vi 1024`; ran to completion, **did not converge** |
| 3 | `runs/launch_r4_surrogate.sh` | `r4_fg_q2`: `--map-steps 24000 --n-samples 640 --svi-steps 24000`, sampling held fixed; ran to completion, **did not converge, and was slightly worse** |

## 8. Attempt 3 — the surrogate lever, pulled and exhausted

Run `runs/r4_fg_q2`, fitted 2026-10-05. MAP 8000 → 24000 steps, batch 512 → 640, SVI
12000 → 24000 steps. Burnin, draws, model and seed held at attempt 2's values, so the
surrogate is the only variable.

**The surrogate did improve, measurably.** MAP's χ² fell 1.287 → 1.217 and its best log
posterior rose 1.897e+04 → 1.911e+04, closing the MAP-vs-HMC deficit from **193.75 nats to
69.60**. The SVI covariance stayed healthy (condition 8.411e+07).

**Convergence did not follow it. It got slightly worse:**

| | `r4_fg_q` | `r4_fg_q2` |
|---|---|---|
| worst R̂ | 2.125 | **2.330** |
| min ESS (bulk/tail) | 63 | **56** |
| ESS-vs-draws slope | 0.08 | **0.04** |
| MAP deficit (nats) | 193.75 | 69.60 |
| full-field χ²ᵥ | 1.199 | 1.276 |

Both runs fail the ESS floor, so neither R̂ is interpretable and the 2.125 → 2.330 difference
should not be read as a trend — the honest statement is that **a materially better surrogate
produced no convergence improvement at all.** Guide 04 §4.3 ranks the surrogate second and
warns instinct ranks it too low; here it was tried, and it is not the cause. MAP is *still*
descending at 24000 steps (tail slope −0.00001/step), but the 124-nat improvement already
bought changed nothing, so more MAP steps are not the answer either.

**The mass model is unmoved, which is the point.** The deflector pins against the bound in
both runs, and in both the pinning survives at the highest-posterior draws:

| | `r4_fg_q` | `r4_fg_q2` |
|---|---|---|
| `e1` (bound −0.4714) | −0.4692 | −0.4685 |
| median `q` | 0.2951 | 0.3021 |
| median `q`, top 10% by lp | 0.3009 | 0.3263 |
| fraction pinned within 0.01, all draws | 0.963 | 0.921 |
| fraction pinned within 0.01, top 10% | 0.960 | 0.880 |
| `gamma` | 2.8978 | 2.9173 |
| host `theta_E` | 1.4583 | 1.4620 |
| foreground SIS `theta_E` | 0.4752 | 0.4316 |

Two independent surrogates, differing by 124 nats in starting quality, land on the same
unphysical deflector to within 0.007 in `q`. §4's conclusion is strengthened, not weakened:
**the bound is not what is holding the deflector flat — the likelihood is.**

The slight softening of the pinned fraction (0.963 → 0.921, and 0.960 → 0.880 at the top
decile) is in the direction a better surrogate would predict and is worth noting, but it is
far too small to change the reading: at q ≈ 0.30 the deflector is a 3.3:1 ellipsoid, still
well outside anything typical of a galaxy-scale lens and still against its wall.

**What this closes.** Of guide 04 §4.3's ordered list, items 2 (surrogate), 3 (kernel —
MCLMC is already in use) and 4 (longer chains, measured sublinear twice) are now spent on
this model. **Item 1, improve the model, is what remains**, and §4's tagged judgement names
the specific fit: hold the foreground SIS `theta_E` fixed and see whether the host EPL
relaxes off the bound. Until that runs, round 4's question stays unanswered and round 2's
`theta_E` stays the system's number.
