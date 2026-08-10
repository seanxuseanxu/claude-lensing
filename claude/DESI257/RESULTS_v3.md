# DESI257, round 3: the SW object refitted as the fourth image

**Date:** 2026-08-09. **Status: complete.** One diagnostic was abandoned rather than
measured — the 20,000-step MAP point of section 3's sweep, which does not bear on the
posterior below; section 3 says why. This file was written as the round was run, so what is
here is measured and what is missing is said to be missing.

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
