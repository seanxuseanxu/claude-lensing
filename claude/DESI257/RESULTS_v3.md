# DESI257, round 3: the SW object refitted as the fourth image

**Date:** 2026-08-09. **Status: complete.** One diagnostic was abandoned rather than
measured — the 20,000-step MAP point of section 3's sweep, which does not bear on the
posterior below; section 3 says why. This file was written as the round was run, so what is
here is measured and what is missing is said to be missing.

> **AMENDMENT, 2026-08-11 — read section 8 before quoting any number from this file.**
> Nothing below is retracted and no central value has moved. Three things were measured after
> this file was written, and two of them change how its numbers must be quoted:
>
> 1. **The grid is undersampled.** `supersample = 2` fails the library's quadrature tolerance
>    by 18x, and the resulting systematic on `theta_E` is of **order 0.01"** — roughly 5x the
>    `[1.989, 1.994]"` interval quoted throughout. That interval is **statistical only**.
>    **Quote `theta_E = 1.99"`.** The grid was left at ss=2 deliberately; section 8.1 gives the
>    measurements that rejected each alternative.
> 2. **The E arc's `+0.33 sigma` coherent excess in section 5.1 is source substructure**, not a
>    displaced model — tested and reported in section 8.3. Section 5.1's reading of it stands.
> 3. **"Checkerboard" is the wrong name** for the core residual excess this file refers to in
>    section 5.1 (via `KNOWN_ISSUES.md` item 5). Measured, the excess does not alternate in
>    sign. Section 8.5. The physical conclusion is unchanged; only the description.
>
> The convergence result — R-hat <= 1.001 on all 36 parameters, all 50 chains — was re-checked
> against three diagnostics that post-date this file and passes all three, with one new flag on
> the source size. Section 8.2.

**Headline.** Unmasking the SW object closes the bimodality. All 36 parameters reach
R-hat <= 1.001 on all 50 chains with no chain selection, `theta_E = 1.991 [1.989, 1.994]"`,
and `gamma`, host ellipticity and `|shear|` relax simultaneously from round 2's three
extremes to an isothermal profile, `q ~ 0.75` and an ordinary line-of-sight shear. The model
puts 0.9688 e/s where the data has 0.9572 at the SW object's peak, with no dedicated
component for it. The object is the fourth image; the mask is what produced round 2's exotic
mass model.

Round 3 changes **one variable** against round 2 (`r2_s2l2`): the 12 px mask over the SW
object is removed, and those 403 pixels enter the likelihood. Same model otherwise — EPL +
external shear, 2 Sersic source components, bulge+disk deflector light, 36 free parameters.

Why: round 2's mask rested on an argument that is a physics error. See `README.md`, "The SW
object", for the retraction, and `RESULTS_v2.md`'s banner for what that costs round 2.

---

## 1. What had to be fixed before the fit was worth running

**Round 2's MAP never reached a mode**, and since SVI starts from MAP and MCLMC is seeded
from SVI, every chain inherited that. Measured through `ProbModel.log_prob`
(`diagnose.map_convergence_check`, added for this):

| | log posterior |
|---|---|
| `r2_s2l2/map.npz`, stored `best_lp` | 16603.621 |
| the same point, recomputed | 16603.617 |
| `r2_s2l2` MCLMC draws, max of 3000 | **17285.414** |
| `r2_s2l2` MCLMC draws, median | 17198.242 |

The MAP is the highest-density point by definition, so finishing **681.8 nats below the best
draw of the chain it seeded** is an optimiser failure. `chisq_hist`'s tail slope over the last
150 of 1500 steps is **-0.00018/step** — still descending when it stopped. The consequence is
measured in `RESULTS_v2.md` §3.4(a): 48 of 50 chains started inside one mode.

So round 3's MAP step count is chosen by measurement, not inherited.
`claude/pipeline/KNOWN_ISSUES.md` items 6, 7 and 8 record the three pipeline defects this
turned up, all now fixed and tested.

## 2. The reference floor, and a warning about comparing log posteriors across the mask

Round 3's MAP has to clear the best draw of the posterior it seeds — but that posterior does
not exist yet. Round 2's draws are physical parameter vectors under the same 36-parameter
model and the same priors, so scoring them through the **unmasked** `ProbModel` gives a valid
floor in advance (`analysis/unmasked_reference.py`, 4000 thinned draws):

| likelihood | max over r2 draws | median |
|---|---|---|
| unmasked (6400 px) | **9393.810** | 2851.383 |
| masked (5997 px) | 17285.371 | 17198.129 |

**The two columns are not comparable, and this is the trap to avoid.** They are different
likelihoods over different pixel sets; the median offset on identical parameters is
**-14346.7** nats. A round-2 log posterior and a round-3 log posterior may never be compared
directly. Only the unmasked column is a floor for round 3.

That offset is itself informative, though it is not yet a result: round 2's parameters,
scored on the pixels round 2 never saw, lose ~14,347 nats over 403 pixels. The masked model
does put light at the SW position (`RESULTS_v2.md` §5.2) but does not put it there
*correctly*, which is what a fit that was never shown those pixels should be expected to do.

## 3. MAP step-count sweep (unmasked)

`--n-samples 4096`, two A100s, one self-contained optax schedule per step count (the
polynomial schedule's `transition_steps` is the step count, so a longer run is not an
extension of a shorter one).

| `--map-steps` | wall time | best log posterior |
|---|---|---|
| 1500 | 501.9 s | 16510 |
| 5000 | 1640.4 s | 18040 |
| 20000 | not measured | not measured |

**1500 steps was nowhere near converged**: 5000 steps buys 1530 more nats, so round 2's
choice of 1500 was not a marginal shortfall.

**The 5000-step point is the one used for the fit, and the plateau test is its own
`chisq_hist` trace rather than the gap to the next sweep point** — a comparison of two
endpoint `best_lp` values across two independently-scheduled runs conflates "the optimiser
has stopped improving" with "a slower decay schedule reaches further", and only the first is
the question:

| | round 2, 1500 steps | round 3, 5000 steps |
|---|---|---|
| final chi-squared | 1.639 | **1.163** |
| `chisq_hist` slope over the last 500 steps | -1.8e-4 /step | **-2.8e-6 /step** |

The trace is 64x flatter and level to within 1.4e-3 in chi-squared over its final 500 steps.

**The 20,000-step point has no number, and the honest entry is "not measured".** It ran to
completion on separate GPUs while the fit was written up, but it wrote its checkpoint and its
stdout into a session scratch directory under `/tmp`, and the node cleared that directory
before either was read. Nothing of that run survives. It was not re-run: it was never a gate
on the fit — the plateau judgement is `map_5000.npz`'s own `chisq_hist` slope above, and the
binding test is the post-hoc one below — so a second 110-minute GPU run would have bought a
confirmation the fit does not depend on. The lesson is a process one, and it is the reason
`runs/mapsweep/` holds the surviving checkpoints inside the repository: **do not write a
result you intend to keep into `/tmp`.**

Already well clear of the 9393.8 floor at 1500 steps. The floor is a weak one — it is the
best of round 2's draws under a likelihood round 2 was not fitted to — so the binding test
remains the one stated in the verification plan: `map best_lp >= max(hmc log posterior)` on
round 3's *own* sample, asserted by `diagnose.map_convergence_check` after the fit.

## 4. The fit

`runs/r3_unmasked/`, 36 parameters, unmasked (6400 px). The MAP stage was **not** re-run:
`map_5000.npz` from the sweep was copied in as `map.npz` and `run_fit.py --stages svi` loaded
it from disk, so the fit starts from exactly the checkpoint section 3 characterises.

### 4.1 SVI (`--svi-steps 6000 --n-vi 1000`, two A100s, 532.0 s)

Round 2 ran 1500 SVI steps and its ELBO was still moving; that is the second half of the
seeding problem section 1 measures, since MCLMC's start distribution *is* this surrogate.

| | round 2 (`r2_s2l2`) | round 3 (`r3_unmasked`) |
|---|---|---|
| SVI steps | 1500 | 6000 |
| ELBO sd over last 300 steps | **1597.9** | **0.4** |
| final ELBO loss | — | -17905.9 |
| covariance min eigenvalue | — | +1.080e-8 |
| covariance condition | — | 4.099e6 (healthy band 1e6-6e8, guard limit 1e10) |

An ELBO sd of 0.4 against a healthy-run reference near 10 is settled by any reading, and the
covariance is positive-definite and well conditioned, so `svi_health.check_svi_covariance`
passes rather than being tolerated. This is the surrogate MCLMC is seeded from.

Note this is **two variables** against round 2, not one: the mask is gone *and* the
optimiser settings changed. That is deliberate and it is not a confound for the physical
question — a converged optimiser is a precondition for the comparison being about the data at
all, not a competing explanation of it. Round 2's own numbers are what establish that its
optimiser had not converged.

### 4.2 MCLMC — the bimodality closed

50 chains, burnin 5,000, 10,000 kept draws, two A100s, 173.3 s. `hmc.npz` is
`(50, 10000, 36)`.

**All 36 parameters reach R-hat <= 1.001 on all 50 chains with no chain selection**, min
bulk-ESS 70,858, min tail-ESS 87,145. `label_switching_check` finds 0 crossings in 50 chains;
the two source components are separated by 9.86 sigma in `R_sersic`. Round 2 could only reach
R-hat 1.003 after selecting 17 of its 50 chains, and the selection was itself an artefact of
seeding (`RESULTS_v2.md` §3.4a).

Posterior medians with 16-84th percentiles, computed on the draws (not as functions of
component medians), against round 2's over the same thinning:

| | round 2 (masked) | round 3 (unmasked) |
|---|---|---|
| `theta_E` | 1.983 [1.978, 1.998] | **1.991 [1.989, 1.994]** |
| `gamma` | 2.537 [2.477, 2.825] | **1.986 [1.965, 2.007]** |
| \|shear\| | 0.148 [0.136, 0.194] | **0.044 [0.040, 0.048]** |
| \|e_host\| | 0.380 [0.353, 0.621] | **0.143 [0.137, 0.149]** |
| chi2_nu | 1.032-1.054 (5997 px) | 1.154 (6400 px) |

The three simultaneous contortions §3.4 of `RESULTS_v2.md` identified — `gamma` pressed toward
its ceiling, `q ~ 0.22`, and `|shear|` sitting exactly on the plausibility flag `priors.py`
raises at build time — **all relaxed at once** when the 403 masked pixels were restored, to an
essentially isothermal profile, `q ~ 0.75`, and an ordinary line-of-sight shear. That is the
predicted signature of a model no longer being asked to suppress an image that wants to exist.

**The chains are not frozen, and this was checked rather than assumed.** Every one of the 50
chains changes state at every one of its 9,999 transitions and visits 9,891-9,915 distinct
values of its first coordinate. The contrast is with `runs/r2_s2l2/hmc_80k_FROZEN_do_not_use.npz`,
which is now characterised rather than merely named: **50 of 50 chains there returned their
starting state, bit-identical, 80,000 times, with no non-finite entries anywhere.** That is a
sampler taking zero-length steps — an MCLMC step-size controller collapse — not a divergence.
The cause remains unattributed; what is new is that the failure has a signature. Note this
run does not isolate it: it differs from the frozen run in burnin (5,000 vs 2,000) *and* in
draw count (10,000 vs 80,000), so it shows burnin 5,000 at 10,000 draws is healthy, not that
burnin was the cause.

### 4.3 The MAP check still fails, by 18 nats

`diagnose.map_convergence_check`: `map.npz` scores 18035.266 (stored 18035.264), the best of
3000 thinned draws scores 18053.334. **MAP is 18.07 nats below the best draw, so it is still
not exactly at the mode** — down from round 2's 681.8 nats, which is the difference between an
optimiser that stopped short and one that never arrived. The seeding pathology is fixed in
practice (one mode, R-hat 1.001, no chain selection); the check is still red and is reported
red.

## 5. What the fourth-image hypothesis predicts, recorded before the fit

Stated in advance so the comparison is not made after seeing the answer:

- the SW object is reproduced by the **existing** source components, with no new component
  and no new mass, at good chi-squared over the newly-unmasked 403 pixels;
- `gamma`, host ellipticity and `|shear|` relax toward plausible values, away from round 2's
  mode B (`gamma = 2.836`, `|e| = 0.645`, `|shear| = 0.195`);
- the bimodality weakens or closes.

**The first check is visual and comes before any convergence question:** `residuals.png`
panel 4 over the previously-masked disc. A fourth image reproduced by the existing source
components leaves noise there. A foreground galaxy leaves a large coherent positive residual.

If the chi-squared over those pixels is poor, or the mass parameters go *more* extreme, the
object is a foreground galaxy and the mask was right. Either outcome is a result.

### 5.1 Outcome: the SW object is reproduced, and the mass parameters relaxed

`analysis/sw_region_chisq.py`, at the HMC median, over the identical 403 pixels rounds 1-2
dropped:

| | value |
|---|---|
| chi2_nu inside the previously-masked disc | 2.049 (403 px) |
| **mean normalised residual inside** | **-0.0014 sigma** |
| disc pixels beyond +3 sigma / -3 sigma | 4.7% / 1.5% |
| data peak inside disc | 0.9572 e/s |
| **model peak inside disc** | **0.9688 e/s** |

**The model puts 0.9688 e/s where the data has 0.9572 — 1.2% agreement on the peak brightness
of a 131-sigma object it has no dedicated component for**, with a mean residual of zero to
0.0014 sigma and near-symmetric 3-sigma tails. The prediction was a large coherent *positive*
residual if the object were a foreground galaxy; there is none. Combined with §4.2's
simultaneous relaxation of `gamma`, `|e|` and `|shear|`, the fourth-image reading is confirmed
and the mask is what produced round 2's exotic mass model.

**On the chi2_nu of 2.049: it is not a misfit, and reading it as one was an error of
comparison.** 2.049 looks large against the field average of 1.154, but the field average is
dominated by blank sky. The right control is the same statistic on the other bright compact
structure in the same image, at the same parameters and the same error map:

| region | chi2_nu | n px | mean residual | peak data |
|---|---|---|---|---|
| **SW object** (prev. masked) | **2.049** | 403 | **-0.0014** | 0.957 |
| E arc | 1.959 | 441 | +0.3259 | 0.636 |
| deflector core | 2.163 | 451 | -0.0195 | 8.468 |
| N blob | 0.950 | 441 | -0.1658 | 1.324 |
| whole image | 1.154 | 6400 | — | — |

The SW object sits between the E arc and the deflector core and has the **smallest** mean
residual of the four; the E arc carries a genuine coherent +0.33 sigma excess that the SW
object does not. chi2_nu near 2 is what bright compact structure costs in this field — source
structure the two Sersics cannot capture, plus the PSF (`KNOWN_ISSUES.md` item 5 is the core
checkerboard, and the core is the worst region here). Nothing distinguishes the SW object from
the images that were never in question.

### 5.2 Two things the record got wrong, corrected by this fit

**`theta_E` was never diagnostic between the two readings.** `README.md` and the round-3 plan
argued that the interloper reading predicts ~1.70" and the fourth-image reading ~2.1-2.2",
and treated the masked fit's 1.98-2.00" as evidence for the latter. Unmasked, with the object
fully in the likelihood, `theta_E` is **1.991 [1.989, 1.994]** — it moved by 0.008", less than
its own round-2 interval. The 2.1-2.2" prediction is simply not borne out. `theta_E`'s
robustness to a 403-pixel change in the data is reassuring for the number itself, but it is
not evidence for either reading and must stop being cited as such.

**The demagnification retraction needs restating.** `README.md`'s retraction argued that a
demagnified image requires a radial critical curve, which requires `gamma < 2`, and that both
round-2 modes had `gamma > 2`. Round 3 has **`gamma = 1.986 [1.965, 2.007]`, with
P(`gamma` < 2) = 0.75**, and `LensSolver` accordingly finds a second critical curve at
0.0278". So the premise as stated is false for this fit. The conclusion survives on different
grounds, and this is the form that belongs in the record: **a radially demagnified image forms
near the radial critical curve — here at ~0.03" from the centre, buried in the deflector core
— not at 2.58".** The brightness-ordering argument cannot explain an object 2.6" out
regardless of `gamma`.

## 6. The interloper comparison (round 3b) — built, and deliberately not run

**Status: pipeline support is complete and tested (`--foreground`, 46 params,
`tests/test_foreground_roundtrip.py`); the run is not on the critical path.** §5.1's controls
are why. Round 3b was justified on the grounds that the SW disc's chi2_nu was anomalous and
needed a competing explanation; measured against the other images, it is not anomalous. There
is no outcome in doubt for the run to resolve.

What it would still buy is narrower than the plan claimed: not a discrimination between two
readings, but an **upper bound** — the data constraining any foreground deflector at that
position to `theta_E` below some value. That is a real measurement and worth having if a
referee asks for it. It is not a test. And by the funnel argument below it is exactly the run
whose posterior piles up against a boundary and converges worst, so it is expensive for what
it returns.

The design, kept here because the support is in the code: unmasked, plus a foreground SIS
deflector and its own Sersic light at the SW position — the interloper hypothesis taken seriously as a deflector rather than merely
hidden, since an interloper deflects whether or not its light is in the likelihood.

The parameterisation is deliberately **nesting**: with the SIS `theta_E` going to zero and
the extra light doing the work it reduces to "the SW object is a foreground galaxy of
negligible mass"; with the extra light's `Ie` going to zero it reduces to round 3's
fourth-image model. So the comparison is a nested one over the extra parameters, mask-aware
`chi2_nu` via `diagnose.masked_chisq` over the identical pixel set, and the fitted SIS
`theta_E` is itself the discriminating measurement.

**A convergence caveat recorded before the run, so it is not read as a failure after it.**
The nesting requires supports that reach zero — `theta_E` is `HalfNormal(0.5)`, and the
interloper light's `Ie` has no floor — and that creates a funnel. As `theta_E -> 0` the SIS's
`center_x`/`center_y` stop entering the likelihood at all and are constrained only by their
`Normal(., 0.05)` priors; as `Ie -> 0` the same happens to that Sersic's shape parameters.
Under the unconstraining bijector `theta_E = 0` sits at minus infinity, so **if the data
genuinely prefer no interloper, the chains will descend a narrowing funnel and R-hat on those
10 parameters will look bad — and that would be the model behaving correctly, not failing.**
The discriminating statistics for round 3b are therefore the mask-aware `chi2_nu` over the
identical pixel set and the bulk of the `theta_E` posterior; the convergence table on the
foreground block is diagnostic of geometry, not of the answer. The 36 parameters round 3b
shares with round 3 carry no such excuse and must still converge.

## 7. Reproducing what is here

```bash
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python
CUDA_VISIBLE_DEVICES=0 $PY claude/DESI257/analysis/unmasked_reference.py
CUDA_VISIBLE_DEVICES=0 $PY claude/pipeline/diagnose.py claude/DESI257/runs/r2_s2l2
```

`diagnose.py` on `r2_s2l2` will print `NOTE: run's recorded mask_circles ... differs from
DESI257's current config (None); using the recorded value` and reconstruct 5997 kept pixels.
That is correct and is the point of `KNOWN_ISSUES.md` item 8: the config is now unmasked, the
run was fitted masked, and the run's own record wins.

---

## 8. Amendment, 2026-08-11: what this number is, and what it is not

Everything in this section was measured after sections 1–7 were written. It changes no central
value and retracts nothing. What it changes is the precision this round's headline can honestly
carry, and it closes three questions that sections 4–5 left open.

### 8.1 The rendering grid is undersampled, and ss=2 is still the right choice

`gigalens.jax.analysis.undersampling.diagnose_undersampling`, over a 9-sample batch (MAP, the
HMC median this file's figures were rendered at, and 7 draws spread across chains and steps,
since quadrature error is parameter-dependent), against a `reference_supersample=64` reference.
Tolerance is the library's own: 0.2 sigma/px.

| rung | max \|delta\|/sigma | mean `delta_chi2` | verdict |
|---|---|---|---|
| ss=1 (native-kernel PSF) | 5.290 | +720 | FAIL |
| **ss=2 (this fit)** | **3.697** | **-216** | FAIL |
| ss=4 | 1.497 | -86 | FAIL |
| ss=8 | 0.439 | -34 (spread -41.5..-24.7) | FAIL |

**Every rung fails, including ss=8.** The reference did not self-certify either (subgrid
self-error 0.072 sigma against a 0.05 sigma threshold), so these numbers are reported, not
certified — but the *ranking* is defensible: doubling the reference from 32 to 64 cut its own
self-error 2.7x while ss=2 and ss=4 moved under 5%, and stability under a materially different
reference is what distinguishes signal from reference noise. ss=8, sitting only ~6x above the
reference's own floor, is the rung to trust least.

**The grid was left at ss=2 anyway, and each alternative was measured rather than argued
against:**

- **ss=1 is worse on both axes.** 5.290 sigma/px, and it crosses the PSF-convolution convention
  boundary: `supersample = 1` convolves the native-resolution kernel on the detector grid,
  `supersample >= 2` convolves a subgrid kernel before pooling. On this system those two
  conventions differ by **3.69 sigma**, and that gap is *not* reducible by supersampling — it is
  a different PSF model, not a finer integral. (This also bears on `README.md`'s flux-scale
  calibration, which was rendered at ss=1 while every fit ran at ss=2.)
- **ss=4 is not a fix.** It fails at 1.497 sigma/px. A naive port of DESI234's ss=2 -> 4 remedy
  would not have worked here, and the reason is measurable: DESI257 is *less* cuspy than DESI234
  by Sersic index (`n_sersic` 2.52, 1.30 against 6.74, 2.45) but its inner lens-light component
  is ~3x more compact (`R_sersic` 0.225" against 0.746"). **Compactness tracks the quadrature
  error, not Sersic index** — which refutes the hypothesis this measurement was started to test.
- **ss=8 and ss=16 are uniform-grid brute force.** ss=8 still fails; ss=16 needs ~6.5 GB per
  buffer per device at `n_vi = 1000` on an 80x80 cutout and is likely infeasible
  (`KNOWN_ISSUES.md` item 9's per-device sizing rule).
- **The experimental adaptive supersampler reaches tolerance cheaply, but not in this fit's PSF
  convention.** `analysis/adaptive_supersample_check.py`: factor-8 opt-in clears 0.2 sigma/px at
  187k light-evaluation points — 2.2x cheaper than the *failing* uniform ss=8 rung — but only in
  the module's `bin_first` convention. It has no subgrid path. Measured against subgrid, every
  setting fails at 3.70–4.03 sigma and barely moves as the grid is refined, because that number
  is essentially the 3.69 sigma convention gap again. Adopting it therefore bundles a modelling
  decision — switching the whole model's PSF convolution — that needs its own validation against
  how DESI257's PSF was actually measured. That is a prerequisite, not a detail.

**Importance-reweighting this posterior to a finer likelihood will not work.** The per-draw
`delta_chi2` spread at ss=8 is ~17 nats, which guarantees weight collapse. That same parameter
dependence is why the error is not a constant offset: it tilts the likelihood.

**What the tilt costs `theta_E`.** `analysis/theta_e_systematic.py` estimates it to first order
(linear response): regress the per-draw log-likelihood perturbation `delta = delta_chi2/2` on the
six host-mass parameters over 60 draws spread across all 50 chains, then propagate through the
posterior covariance, `Delta_theta = Sigma @ g`. Both rungs render under the same subgrid
convention, so this isolates quadrature error from the convention gap.

| | |
|---|---|
| `Delta theta_E` | **+0.0106" = +4.12 posterior sd** (sd = 0.00257") |
| `Delta gamma` | -0.0090 = -0.43 sd |
| ratio to the quoted +/-0.002" half-width | **5.3x** |
| regression `R^2` | 0.197 (adjusted 0.106) |
| overall significance | **`F(6,53) = 2.167`, p = 0.061** |
| `theta_E` coefficient | `t = +1.92` |
| split-half (30 v 30) | +0.0176" v +0.0127", same sign |
| sanity check on mean `delta` | -108.4 nats against -110.6 already measured (2.0%) |

**This does not license moving the central value, and the honest reason is the F-test.** At
p = 0.061 the fitted gradient is not statistically distinguishable from zero at 5%, and a
"first-order" shift of 4.1 posterior sd is not small in the sense linear response requires — two
independent reasons the same conclusion holds. The split-half sign agreement is weak
corroboration: two halves agreeing on a sign is a coin flip a priori. So:

> `theta_E = 1.991 [1.989, 1.994]"` is a **statistical** interval. There is in addition a grid
> systematic of **order 0.01"**, marginal in significance, sign suggestively positive. **Quote
> `theta_E = 1.99"`; the third decimal is not defensible.** Widening the interval to a specific
> number, or shifting the centre, would both overstate what was measured.

Settling this properly needs a refit at a certified grid, which means first settling the PSF
convention. That is round 4's business, not a correction to this round.

### 8.2 The convergence result re-checked against diagnostics that post-date it

This run was diagnosed 2026-08-09. `prior_posterior_tension` landed 08-10; the Vehtari ESS
floor, rank plots and the ESS-vs-draws slope landed 08-11 and regenerated `r2_s2l2`'s figures
while leaving this run alone. Re-run in full (56.8 s, CPU):

- **`chi2_nu` = 1.154 over 6400/6400 px — reproduces exactly**, confirming the `supersample`
  provenance fix (`KNOWN_ISSUES.md` item 8) changed no arithmetic.
- **R-hat <= 1.001 on all 36 parameters** (worst 1.001). Unchanged.
- **ESS floor passes all three Vehtari conditions**: min bulk-ESS 70,858 total, 708.6 per split
  half, 50 chains, against floors of 400 / 50 / 4. Min tail-ESS 87,145. ESS-vs-draws slope ~0.99,
  i.e. linear growth — additional draws would still buy information.
- **No label switching.** Source `R_sersic` components separate by 9.86 sigma
  (0.0520 +/- 0.0013 against 0.2480 +/- 0.0198) with 0 of 50 chains crossing.
- **Nothing pinned on a prior bound.** Closest is `bulge/e2` = -0.2481 at 8.6% of [-0.3, 0.3],
  about 21 sigma from the bound.
- **SVI healthy**: min eigenvalue 1.08e-08, condition 4.10e+06, ELBO tail sd 0.4.
- **The MAP deficit stands at 18.066 nats** (MAP 18035.264 against the best of 3000 thinned
  draws, 18053.330), tail slope -0.00003/step. Section 4.3's finding, unchanged.
- **`verify.py --lens DESI257`: 4 passed, 1 skipped** — see `README.md`'s status list.

**One new flag, and it is worth taking seriously: `prior_posterior_tension` fires on
`source/light/src0/R_sersic`.** The median is **0.0520"**, **3.40 prior sigma below** the prior
centre, with a posterior sd 0.4% of the prior's (`ln(0.052/0.4)/0.6 = -3.40`, confirmed by hand).
This is the check that caught DESI234's `r3_fg` after everything else passed it.

Two facts pull against reading it as misspecification, and one pulls for it. Against: the prior
is `LogNormal(log 0.4, 0.6)` whose scales `README.md` explicitly flags as **unmeasured**,
inherited from DESI238 — tension against an arbitrary centre is weak evidence. And the data
constrains the parameter tightly (sd 0.0013 on 0.0520, 2.5%), so this is not a degenerate
direction going soft. For: 0.0520" is **0.8 px in the source plane** with `Ie = 218` — a
near-point knot. The failure mode to rule out is that `src0` is absorbing PSF mismatch rather
than describing the source.

**This flag and section 8.3's result are probably one finding**, and section 8.4 explains why
that constrains the order of round 4's work.

### 8.3 The arc and blob residual dipoles are substructure, not a displaced model

Section 5.1 records a coherent **+0.33 sigma** mean residual on the E arc. Read by eye, the
residual dipoles at three locations looked like a rigid sub-pixel misregistration — if so, a
single small shift would remove most of the local chi-squared, and no new source freedom would
be warranted. `analysis/arc_residual_anatomy.py` tests that directly: a rigid shift gives
`r ~ a_row * G_row + a_col * G_col` with `G` the error-normalised model gradients, so an OLS of
the normalised residual on those gradients returns the implied shift, and the fraction of local
chi-squared it removes is the verdict (>=50% displacement, 20–50% both, else substructure).

| region | px | peak | chi2_nu before | after | frac explained | implied shift |
|---|---|---|---|---|---|---|
| E arc (r 1.76–1.92", PA 40–110 deg) | 84 | 26.8 sigma | 5.407 | 4.620 | **14.6%** | 0.30 px = 0.020" |
| N blob (r 1.40–1.70", PA 320–20 deg) | 118 | 36.8 sigma | 1.713 | 1.710 | **0.2%** | 0.015 px = 0.001" |
| SW object (r 2.40–2.76", PA 195–240 deg) | 169 | 34.2 sigma | 2.970 | 2.738 | **7.8%** | 0.12 px = 0.008" |

**All three are far below the 50% displacement threshold, and the three fitted shift vectors
disagree** (cosine similarity -0.38 between E arc and N blob, -0.72 between N blob and SW; the
+0.92 between E arc and SW is coincidence between two weak fits). No single rigid displacement
explains all three, which is what an astrometric or PSF-centring error would have produced.

**Verdict: substructure** — source-plane structure the two-component Sersic source cannot
reproduce. This is a **null result for the displacement hypothesis**, which was the reading this
test was built to confirm. Section 5.1's interpretation of the E arc excess as "source structure
the two Sersics cannot capture" is upheld by direct test rather than by inference.

Related, and recorded as an observation rather than a defect claim: `theta_E = 1.991"` sits
outside all four measured clump radii (N blob 1.55", E arc 1.80/1.86/1.90"), and the tangential
critical curve at 2.014" is 0.1–0.2" beyond the E arc. With `|e_mass| = 0.143` the curve's radius
varies enough to cover the arc, but the N blob is well inside it.

### 8.4 Sequencing for round 4: the grid must be fixed before the source gains freedom

Sections 8.1–8.3 together imply an order of work, and it cuts against the obvious next move.

The tension flag on `src0/R_sersic` (8.2) and the substructure verdict (8.3) are most likely one
finding. The residuals say there is compact source structure two smooth Sersics cannot reach;
`src0` has collapsed to a 0.052" near-point knot — 0.8 px in the source plane — which is what a
component straining to supply that structure looks like. The natural response is to give the
source more freedom: a third component, or a pixelated source.

**Doing that first would be a mistake**, because a flexible source will absorb the ss=2
quadrature error (8.1) just as readily as it absorbs real substructure, and the two are not
distinguishable after the fact. The quadrature error at ss=2 is 3.7 sigma/px and localised on
exactly the bright compact structure a freer source model would be reaching for. A numerical
artifact would be quietly reinterpreted as astrophysics, and the resulting "detection" of source
substructure would be unfalsifiable from the fit alone.

So the order is: **settle the PSF convention, certify a grid, refit, and only then add source
freedom.** `analysis/arc_residual_anatomy.py`'s own docstring warns of this trap; 8.1's
measurement is what turns the warning into a constraint.

This is also why round 3b — the foreground/interloper run, built and tested at 46 parameters —
stays not-run (section 6). It resolves no outcome in doubt, and DESI234's `r3_fg` is a live
example of a foreground component producing an unphysical mass model.

### 8.5 "Checkerboard" is the wrong name for the core residual excess

Section 5.1 cites `KNOWN_ISSUES.md` item 5 as "the core checkerboard", and item 5 describes a
"pixel-scale alternating +/- sigma checkerboard". A checkerboard has **negative** lag-1
autocorrelation between neighbouring pixels. Measured on four fits — both systems, both rounds,
both supersamples — it is positive in every case (`analysis/core_checkerboard_extent.py`):

| run | supersample | core lag-1 rho | field lag-1 rho |
|---|---|---|---|
| DESI257 `r3_unmasked` | 2 | **+0.198** | +0.713 |
| DESI234 `r3_unmasked` | 4 | **+0.345** | +0.778 |
| DESI257 `r2_s2l2` | 2 | **+0.207** | +0.722 |
| DESI234 `r2_s2l2` | 2 | **+0.357** | +0.757 |

The round-2 runs are included because item 5's description came from them, and DESI234 changed
supersample between the rounds — so the pair tests both "was the description loose?" and "did
something between the rounds remove the alternation?". Neither. The core *is* measurably rougher
than the field (rho falls from ~+0.75 to +0.20–0.35, so there is real excess pixel-scale power
there), but it is positively correlated: a smooth compact excess, not a sign-flipping pattern.

**Item 5's physical conclusion is untouched** — a PSF artifact of Sersic deflector modelling,
masked when reporting and not when modelling, which came from field experience rather than from
this statistic. Only the name, and the round-2 amplitude description, should stop being requoted
as measured.

### 8.6 A centre-masked `chi2_nu`, so the number means what a referee would expect

`KNOWN_ISSUES.md` item 5's last open piece. The published convention is to mask the deflector
core when computing chi-squared but **not** when modelling, so every `chi2_nu` in this repository
— including this file's — has been inflated by core pixels a referee's number would exclude.
`diagnose.py` now reports both. **The core pixels remain in the likelihood; no fit changed.**

Radius 0.520" = 8 px, fixed by measurement (`analysis/core_checkerboard_extent.py`, section 8.5):
the radius enclosing 95% of the excess chi-squared above a local background, 7 px on this system
and 8 px on DESI234, insensitive to that fraction (90% -> 7/6 px, 99% -> 8/8 px). The larger is
used so neither system is under-masked. It is ~3.7x the F140W PSF FWHM, and an instrument-level
constant in arcsec is justified because the two systems agree to one pixel — the script tests
that fork rather than assuming it.

| run | full-field `chi2_nu` | centre-masked `chi2_nu` | px excluded |
|---|---|---|---|
| **DESI257 `r3_unmasked`** | **1.154** (6400 px) | **1.065** (6198 px) | 202 |
| DESI234 `r3_unmasked` | 1.662 (6400 px) | 1.609 (6202 px) | 198 |

**The full-field values are unchanged to four figures**, which is the check that the change was
purely additive — had they moved, the arithmetic would have been altered rather than extended.
Both centre-masked values were also reproduced independently from section 8.5's radial profile,
through a separate code path sharing nothing with the implementation: 1.0646 and 1.6088 against
1.065 and 1.609, with pixel counts matching exactly.

So this round's fit quality, on the footing a published number would use, is
**`chi2_nu` = 1.065** — the core artifact was worth 0.089, about 8%. Section 5.1's regional table
is unaffected: those are regional statistics quoted as such, and the deflector-core row (2.163)
is precisely the region this reporting change now excludes from the field average.
