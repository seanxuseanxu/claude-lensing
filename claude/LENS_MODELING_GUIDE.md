# A Guide to Bayesian Strong-Lens Modeling for Language Models

Written by Claude, 2026-07-31, after fitting one HST lens system (DESI238) through three
rounds of a MAP → SVI → HMC pipeline; revised 2026-08-08 after two further rounds and a move
onto a new interface, reviewing the session records for what actually went wrong. Examples are
drawn from that work; the advice is meant to transfer to any lens and any package.

It is about **modelling** — the data, the priors, the inference, the reporting, and the person
you are doing it for. It is deliberately not about software engineering, and nothing in it
should depend on which package you use.

The guide is organized around one claim: **in lens modeling, almost every error you will
make is silent.** It does not raise. It does not produce a NaN, an obviously wrong
magnitude, or a plot that looks broken. It produces a plausible number. Every hour you
spend building redundant checks buys back a round of fitting you would otherwise have to
throw away.

---

## 0. What kind of problem this is

A strong-lens fit is a high-dimensional (20–100 parameter), non-linear inverse problem in
which:

- The likelihood is a per-pixel Gaussian whose **variance you have to supply**, and getting
  it wrong rescales your entire posterior without changing its shape.
- The model is a **sum of degenerate components** — mass and light both live at the same
  place, source structure and lens structure can each explain the same residual, and two
  identical source components can swap labels.
- The parameters have **hard physical boundaries** (Sérsic index, ellipticity, radii > 0)
  that the sampler will happily press against.
- The thing you actually want to report (θ_E, the mass slope, a time delay) is often
  **better constrained than the model is identified**, so a broken fit can still give you
  the right answer, and a converged fit can still give you a biased one.

**The target is convergence in every parameter.** That is the goal, and in practice it is
reachable — by improving the model, getting a better variational surrogate, sampling
longer, or switching sampler. Do not treat partial convergence as an acceptable resting
state; treat it as unfinished work with a known remedy list (§4.3).

But the two things are separable, and that matters for how you report. A result can be
**preliminarily reportable** while some parameters remain unconverged, *provided* the
quantity you are quoting is one of the converged ones and you say which are not. For three
rounds DESI238 sat exactly there: all eight lens-mass parameters at R̂ ≤ 1.002 while several
light-profile shape parameters failed, so θ_E was quotable and the morphology was not.

**It was eventually reachable, and the route was not more sampling.** Round 5 got all 36
parameters to R̂ ≤ 1.003 by spending twenty minutes on a better variational surrogate and
switching sampler (§4.3). Chain length had already been increased fourfold and had stopped
helping. Treat partial convergence as unfinished work with a known remedy list, and work the
list in the right order — the surrogate before the chain length.

One caveat that §4.4 develops: an R̂ that only just clears the threshold is a sign the budget
is too small, not a finish line.

---

## 1. Before anything: measure the data, do not read about it

Every number that enters the likelihood should be derived from the files in front of you
and recorded with its derivation. Documentation, headers, and the human's memory are all
starting points, not sources.

**Pixel scale.** Take it from the WCS: `sqrt(|det CD|) * 3600` arcsec/pixel. In DESI238
this gave 0.065"/pix. A cutout usually carries no WCS of its own, so this comes from its
provenance — *verify the provenance*. I confirmed the DESI238 cutout was bit-for-bit
identical to `SCI[1010:1130, 1165:1285]` of the parent drizzle rather than assuming it.

**The PSF.** Check three things, in this order:
1. **Normalization.** The DESI238 PSF summed to 1.1445. Passing it unnormalized inflates
   every rendered image by 14% — which the fit absorbs into the light amplitudes, so
   nothing looks wrong.
2. **Centering.** Odd-sized and centroid within a fraction of a pixel of the middle, or
   you have introduced an astrometric offset between model and data.
3. **Scale.** Whether it is oversampled relative to the science grid. This is frequently
   *not derivable from the file* — the DESI238 PSF had no WCS or scale keywords. When a
   number cannot be derived, ask the human and record the answer as sourced from them.

**The noise.** This is the single most consequential number and the easiest to get wrong.

> `sigma_clipped_stats` on a cutout centered on a galaxy does not measure the sky. It
> measures the galaxy. In DESI238 it returned `background_rms ≈ 0.0315`; the true value,
> measured from six widely separated blank 60×60 patches in the parent frame, was
> **0.0075** — a factor of 4. At 0.0315 the arcs are down-weighted into irrelevance and
> the fit quietly becomes a fit to the deflector alone.

Measure the background from blank sky, in several separated patches, and **record the
spread across patches** (0.0070–0.0092 here). You will need that spread later, in §6.

**Units.** Know whether the image is counts or a rate. A rate image (electrons/s) has
Poisson variance `rate / exp_time`, which is what a `sqrt(bg_rms² + max(model,0)/exp_time)`
noise model expects. Passing a scalar exposure time to a drizzled mosaic is correct only to
first order: `NCOMBINE = 3` means coverage varies across the frame, and the weight (`WHT`)
extension is the honest way to build a per-pixel exposure map.

**Interlopers: mask, crop, or model — and modeling is often the best of the three.** You
have three ways to handle a companion or foreground galaxy in the frame, and they are not
equally good:

1. **Model it.** Add a light profile (and, if it is at the lens redshift and massive
   enough, a mass profile) for the interloper. This is the only option that uses the
   information rather than discarding it, and the only one available when the contaminant
   overlaps the arcs. A nearby satellite can perturb the deflection field measurably, so
   for precision work on θ_E or a time delay this is not optional.
2. **Mask it.** Clean, but throws away pixels. Check that your package supports per-pixel
   masking before planning around it; not all do.
3. **Crop it out.** The crudest option, and it works only when the contaminant is well
   separated from the arcs.

For DESI238 I used (3): the companion sat in a corner far from the lensed structure, and a
crop that centres the deflector to within 0.1 px excludes it. I added an automated check that
the corner maximum stays below a threshold, so the companion cannot creep back in when
someone adjusts the crop. That was pragmatic, not the best available physics. **If an
interloper is close enough to affect the deflection, cropping is not a solution to it — it is
a decision to accept an unmodelled systematic, and should be reported as one.**

### Write these down as a preflight test, not as prose

Everything above should live in a check that runs in seconds: array shapes, finiteness, PSF
sum, PSF centring, corner contamination. Prose in a README describes what you believed; a
test enforces it. Mine took ~5 minutes against fits that took hours, and every check in it
exists because something once went wrong.

---

## 2. Building the model: start small, escalate on evidence

The temptation is to write down the most flexible model you can afford and let the data
sort it out. Resist it. A model you cannot diagnose is a model you cannot defend, and
extra components in a degenerate problem do not sit idle — they eat each other's flux.

**Escalate one axis at a time, and let the residual choose the axis.** The DESI238
sequence, which I would repeat:

| Round | Model | Params | Motivated by | Result |
|---|---|---|---|---|
| 1 | EPL + shear, 1 Sérsic deflector, 1 Sérsic source | 22 | baseline | χ²ᵥ 1.837; N arc underfit 2× |
| 2 | + 2nd source component, freed `n_sersic` ceiling | 29 | one clump reproduced at 0.55 of its flux | χ²ᵥ 1.302; all four clumps within 9% |
| 3 | + 2nd deflector component (bulge+disk) | 36 | coherent 1–3% radial S-curve in the core residual | χ²ᵥ **0.991**; core χ² 7.09 → 2.97 |

Each step was justified by a **specific, localized, structured** feature of the residual —
not by a global χ² that was merely "too high." A coherent radial S-curve about the lens
centre (over-bright at r < 1 px, too faint at 1.5–5.5 px, too bright at 6.5–11.5 px) is a
shape a single Sérsic *cannot* produce, because a Sérsic is a one-shape family. That is an
argument for a second component. "χ² is 1.3" is not.

**Rule out the boring explanation first.** My initial reading of that core residual blamed
a sub-pixel centroid offset. A ±0.6 px shift scan disproved it: the fitted position was
optimal to within 0.15 px with χ² rising steeply in every direction. Only then was
"amplitude/shape mismatch" earned. (The centroid explanation did still hold for the *arcs*
— two features, two causes, and conflating them cost a round.)

### Priors on exchangeable versus physically distinct components

This distinction matters more than it looks:

- **Two source components: give them identical priors.** They are exchangeable —
  you have no prior knowledge of which clump is which. Let the data assign the morphology.
  Make the amplitude prior wide enough that an unwanted component can *switch itself off*
  by going faint (`Ie ~ LogNormal(log 20, 1.2)`, spanning 1.2–326 at 1–99%). Accept that
  you now have a label-switching symmetry and check for it (§5).
- **Bulge and disk: give them different priors.** They are physically distinct populations
  — compact/high-n and extended/low-n. Identical priors would add a label-switching mode
  for nothing. In DESI238 I derived the split from a 1D indicator fit to the inner-arcsecond
  profile (compact Rₑ = 0.216", n = 0.72; extended Rₑ = 1.150", n = 1.13) and set
  `R_sersic ~ LogNormal(log 0.25, 0.5)` and `LogNormal(log 1.2, 0.5)` — priors that
  overlap at about 1σ, so the separation the fit found (0.33" vs 2.73", 8.4×) is a
  statement from the data, not from me.

### Derive amplitude priors by rendering, not by reasoning

Surface-brightness normalizations depend on your simulator's internal conventions in ways
that are very hard to predict. Mine multiplied every render by the pixel area — a factor of
0.004225 at 0.065"/pix — so a unit-amplitude Sérsic peaked at 0.083 against data peaking at
7.2. My first-guess amplitude priors, copied from a demo built on a synthetic image with a
different flux scale, were **50× too faint**. The fix is mechanical and takes two minutes:

> Draw ~64 samples from your prior, render them, and compare the median peak to the data
> peak. Require the ratio to land within a factor of ~3. Then keep that as a permanent
> preflight check.

Calibrated values were `LENS_IE = 80`, `SOURCE_IE = 20`, giving median peak 5.26 against
data 7.18.

### Know where your profile family stops being valid

Analytic profiles are usually implemented with an approximation somewhere — the Sérsic
constant *b(n)* is the common one, and closed forms for it degrade badly at low index. In my
case the error reached −13.6% at n = 0.3 while being negligible above n ≈ 1. **Measure that
error against the defining relation rather than trusting the range the profile is advertised
for, and keep the measurement as a test.** Then set the prior's floor where the profile is
still trustworthy, and treat that floor as a physical statement rather than a formality.

**A boundary placed for a known reason becomes a diagnostic.** When a parameter pins against
one, the question is whether the bound is too tight or the family is wrong, and there is a
cheap experiment that distinguishes them: **move the bound once, deliberately.**

- A parameter that *settles at an interior mode* once you widen it was genuinely prior-limited.
- A parameter that *follows the bound down* is telling you the data want something outside
  the family. The fix is a different profile, not another bound.

Mine followed. Lowering the source index floor from 0.5 to 0.3 moved the pinning to 0.3, and
it pinned again in a later round under a different likelihood, a different parameter ordering
and four independent samplers. That is a profile-family verdict, not a prior artefact.

**Then ask whether the component is resolved at all.** The same component's half-light radius
came out at 0.069" against a 0.065" pixel — below the resolution of the data. A shape
parameter fitted to an unresolved component is not a morphology measurement in the first
place. Report it as a bound, describe the component as point-like, and check that it is not
correlated with your headline parameter before you stop worrying about it (here the
correlation with θ_E was r = 0.018, so the headline was safe).

---

## 3. Parameter-space conventions: the highest-density region for silent bugs

Nearly every inference package works internally in an **unconstrained** space, with a
bijector mapping to physical values. For a `Uniform(low, high)` that is
`low + (high − low)·sigmoid(z)`. Three consequences, all of which bit me:

**(a) If you checkpoint unconstrained samples, the prior's *support* becomes part of your
data format.** Widening the DESI238 source `n_sersic` from `Uniform(0.5, 4)` to
`Uniform(0.5, 6)` made an *already-finished* run report `source.n_sersic = 0.6146` where
the fit had actually said `0.5729`. Reduced χ² moved 1.837 → 1.841 from the same cause.
**No error was raised. Nothing about the samples' shape or finiteness changed — only their
meaning.**

There are two fixes, and **the first is better**:

> **Fix 1 — store physical values.** Nothing forces a saved posterior to be in unconstrained
> space. If it holds constrained parameters, the samples mean the same thing forever, no
> matter what you later do to the prior. Keep the unconstrained array alongside if you need
> it to restart a stage, but let the physical values be the artefact you read. I did not do
> this and should have; the versioning below patches a problem a different storage choice
> would have prevented outright.

> **Fix 2 — if you must store unconstrained samples, version the priors.** Record a prior
> version with every run, and when you change a support, **keep the old definition verbatim**
> rather than editing it in place, so your diagnostics can reconstruct the exact prior a run
> was fitted under. Location-scale changes (a Normal's σ) are safe; support changes are not.

Fix 2 is worth having even under Fix 1, because the prior is part of a run's provenance
regardless of how you store the samples. But it should be a record, not a load-bearing
decoder ring.

**(b) Constrain in exactly one place.** One function that everything calls. Every inlined
transformation elsewhere is a future bug. Correspondingly: the optimizer, the variational
stage and the sampler all return unconstrained values, so anything you interpret physically
without passing it through that function is nonsense that will look like a number.

**(c) Never assume the parameter ordering — derive it, then probe it.** Take the names from
the prior structure, then verify the mapping empirically: perturb one coordinate of the
unconstrained vector at a time, map it to physical space, and assert that exactly the
parameter you expected moved. Do it for every index and every model variant.

This matters most right after a model or library change, when the ordering can shift without
anything raising. A transposed or off-by-one parameter vector produces a fit that converges
to a physically absurd but numerically finite answer, and you will spend a day blaming the
data. Component identity is the dangerous case: bulge parameters read as disk, or one
source's flux attributed to the other, gives you a converged fit describing the wrong system.

---

## 4. Running the three stages, and how each one fails

The MAP → SVI → HMC chain is common (in spirit) to most modern lens
pipelines: a multi-start optimizer to find the basin, a variational fit to shape a
preconditioner, then gradient-based MCMC for the posterior. Each stage consumes the last,
which means **each stage can be poisoned by the last, silently.**

### 4.1 MAP: never use bare `argmax`

Multi-start MAP runs ~1024 optimizations in one batch and picks the best by log-probability.
`jnp.argmax` (and `np.argmax`) **propagates NaN**: one diverged start out of 1024 and you
select *that* index.

Observed in DESI238: 1/1024 starts diverged at 22 parameters, 4/1024 at 29. The reported
"best reduced χ²" was 3907.6 when the true best was 2.006. The MAP progress bar read
χ² = 2.003 the whole time, so **the failure was invisible until SVI died 1500 steps later
with a NaN ELBO.**

Select the best start with a NaN-aware maximum, count the non-finite starts, refuse to
continue if all of them diverged, and check that the *chosen* start has finite parameters
before handing it downstream. Each of those is one line and each catches a distinct failure.

Two generalizations worth internalizing:

- **This surfaces exactly when you relax priors.** Wider or less informative priors make a
  divergent start more likely. It did not fire in my first three runs purely by luck, and
  fired the moment I widened a bound.
- **Report the count of excluded starts.** A run where 400/1024 diverged is telling you
  something about your priors even if the survivor is fine.

### 4.2 SVI: a diverged variational fit poisons HMC and looks like slow mixing

This is the most expensive lesson in this document.

The sampler's mass matrix is built by inverting the variational covariance, often in single
precision. If that covariance is badly conditioned, **the inverse is not an inverse**, every
chain samples against an effectively indefinite mass matrix, and *every* R̂ fails at once. The
natural reading — "the chains need to be longer" — is wrong and costs you a full round of
compute.

Calibration numbers from DESI238:

| params | covariance condition | max &#124;inv(C)·C − I&#124; | R̂ outcome |
|---|---|---|---|
| 22 | 9.4e+05 | 0.07 | fine |
| 22 | 5.9e+08 | 0.98 | fine (worst healthy) |
| 29 | 8.8e+11 | 55.3 | 29/29 fail, worst 7.17 |

At 8.8e+11 the single-precision inverse carried five negative eigenvalues reaching −3.2e+07.

**A positive-definiteness test does not catch this.** The pathological covariance *is* PD
(min eigenvalue +5.7e-13). An audit of mine that formed `L @ L.T` in float32 reported
−1.76e-12 and blamed a negative eigenvalue — that was rounding noise in a matrix spanning
twelve orders of magnitude, not a property of the fit. **The condition number is the
discriminant.** And compute the spectrum by squaring the singular values of the Cholesky
factor, never by eigendecomposing `L @ L.T`: the product halves your precision exactly
where the matrix has none to spare.

Compute the condition number in double precision and hard-fail above a threshold — mine was
1e10, against healthy runs at 9.4e5–5.9e8 and the diverged one at 8.8e11.

Set the threshold with an order-of-magnitude margin on **both** sides of the measured gap,
and then write a test asserting the guard fires on your known-bad run and passes on your
known-good ones. A threshold with no measured population behind it is decoration.

Design decisions worth copying:
- **Hard-fail, do not repair.** No eigenvalue floor, no override flag. A repaired
  covariance produces downstream numbers that look fine and are not.
- **Save the checkpoint before the health check**, so a diverged run stays inspectable.
- **The root cause was the learning-rate schedule.** Mine ramped *upward* (init −1e-6 →
  end −3e-3), putting the largest steps at the end of the run. Survivable at 22 parameters,
  not at 29. Warmup-then-decay (peak −2e-3 at 10% of steps, decaying to −2e-4) fixed it:
  conditioning 8.8e+11 → 3.4e+07, ELBO tail sd 7,384 → 6.6, worst R̂ 7.17 → 1.544. **If a
  higher-dimensional fit trips the guard, lower the peak learning rate; do not raise the
  threshold.**
- **Read the SVI health line — condition number and the sd of the last ~300 ELBO steps —
  *before* you look at any R̂.** Healthy runs here settle to ELBO sd ~10.

One more caution about reading ELBO traces: check what your progress bar is actually
printing. Mine was labelled "ELBO" and reported the *loss*, so its descending was correct
behavior and I briefly read it as divergence. And I once inferred a trend ("tail sd grows
with dimension: 41.6 vs 6.6") from two runs that turned out to be estimator noise — an
identically configured re-run gave 5.7. **Two points are not a trend.**

### 4.3 Elevated R̂: work the remedy list, do not diagnose it as fate

When R̂ is elevated and SVI is healthy, the first move is cheap and diagnostic:

> **Re-run with 4× the draws and change nothing else.** If R̂ falls, it was chain length.

In DESI238 (`bulgedisk` → `bulgedisk_long`, 750 → 3000 draws, burn-in 250 → 1000): worst R̂
3.242 → **1.338**, failing count 33/36 → **10/36**, χ²ᵥ unchanged at 0.989. It fell. It was
chain length, and the "expected signature of a bulge–disk degeneracy" I had written down
was wrong. Confirm the runs really are otherwise identical — I checked that the SVI
starting loss matched.

If R̂ does not fall, **that is not a verdict that the posterior is unsamplable.** In this
project's experience the remedy list is short and it works:

1. **Improve the model.** Persistent non-mixing is often a modeling problem wearing a
   sampling costume — a parameter pinned at a prior boundary cannot mix by construction
   (§5), and a missing component leaves a degeneracy the sampler has to crawl along.
2. **Get a better variational surrogate.** Lower the SVI peak learning rate, run it longer,
   check the conditioning improved (§4.2). A mediocre-but-passing preconditioner still
   costs you mixing.
3. **Sample longer**, and be honest that 4× may not have been enough.
4. **Change sampler** — NUTS instead of fixed-length HMC, or a different adaptation scheme.

Reparametrization is available but is the last thing to reach for, not the diagnosis you
jump to. I have not encountered a genuinely plateaued R̂ in this work: every elevated one
either fell with more draws or traced back to a boundary-pinned parameter. Treat "the
geometry is intractable" as a claim requiring evidence that you have exhausted the list
above, not as an explanation for why a fit did not finish.

**Remedy 2 is where the leverage is at high dimension, and I had it ranked too low.** DESI238
at 36 parameters sat at worst R̂ 1.338 after a 4× chain-length increase had already been
spent. Remedy 3 was not the answer. Twenty-one minutes of MAP+SVI on four GPUs — 4,096
multi-start particles, 1,500 MAP steps, then 12,000 SVI steps at n_vi 2,048 with peak LR 1e-3
— took the covariance condition to 2.3e+06 and the **ELBO tail sd to 0.1**, against the ~10
that §4.2 calls healthy at 22 parameters. Every sampler tried afterwards cleared or nearly
cleared R̂ < 1.01 from that one surrogate, and the worst R̂ went 1.338 → 1.003.

> The generalization: **the preconditioner is a fixed cost that every subsequent draw pays
> interest on.** Chain length buys you samples from whatever geometry the surrogate hands the
> sampler; a better surrogate changes that geometry. As dimension rises the second is the
> better purchase, and it is far cheaper than it looks — 21 minutes bought what no amount of
> extra HMC had.

Note also that "passing the conditioning guard" is a floor, not a target. Both surrogates
passed. The one with ELBO tail sd 0.1 converged the fit and the one at ~10 did not. Read the
tail sd as a quality number, not just a pass/fail.

**Remedy 4 was worth more than expected: change the kernel, not just its length.** With the
surrogate held fixed, MCLMC was ~9× faster per draw than fixed-length HMC on this problem, so
at *two thirds* the wall time it delivered 8× the draws and 1.7× the minimum ESS. MAMS landed
in between. If your package offers modern gradient-based kernels (MCLMC, MAMS, NUTS), try one
early on a short run — it is a cheap experiment and the speedup was worth more here than any
tuning I did.

> **Hold the surrogate fixed when you compare samplers.** Copy the MAP and SVI checkpoints
> into each run directory rather than re-running those stages. Otherwise you are comparing
> sampler-plus-surrogate pairs and cannot attribute the difference. This is what makes a
> sampler comparison controlled rather than confounded, and it costs one `cp`.

### 4.4 A marginal R̂ is a budget problem — buy draws, not seeds

A 32-chain × 3,000-draw HMC run gave worst R̂ 1.003 across all 36 parameters. Re-running it
four more times changing **only the seed** gave 1.019, 1.009, 1.093, 1.019 — two of five clear
R̂ < 1.01, with minimum bulk-ESS ranging 2,735–10,928.

The useful reading of that is **not** "add a multi-seed protocol." It is that a threshold met
by a small margin is telling you the budget is too small, and the remedy is the one already in
§4.3: sample longer. The 4×-longer, 2×-wider run reached worst R̂ 1.003 at min ESS 18,692,
where seed-to-seed scatter of that size is no longer in play. Spend the compute on draws.

Where seed variation still earns its keep is as a *cheap diagnostic of marginality*: if two
short runs straddle your threshold, that settles whether you are converged or lucky, and it is
faster than arguing about it. Use it to decide you need a bigger budget, not as a deliverable.

Also: **a fit can be reportable before every parameter converges** (§0), but you
have to say which parameters. After the long run, all eight lens-mass parameters sat at
R̂ ≤ 1.002 while the two light components' shape parameters and `source1.n_sersic` still
failed. Since the quantity being reported was θ_E, that is a reportable fit — *stated that
way*, not glossed as "converged."

Run many chains from independent draws of the variational posterior (50 here) sharing one
seed. Independent starting points are what make cross-chain R̂ meaningful.

---

## 5. Diagnostics you should run every time, in this order

Order matters, because an early failure invalidates everything after it.

**1. SVI health.** Condition number and ELBO tail sd. Ahead of any convergence table
(§4.2). If this fails, nothing downstream means anything.

**2. Prior-boundary pinning.** For every bounded parameter, compute the posterior median's
fractional position within its support and flag anything within 2% of either end. Check only
bounded priors — a tail value of an unbounded prior is a different phenomenon.
**Automate this.** In DESI238 two parameters pinned in round 1 and I noticed one by eye;
the other (`source.n_sersic` at its *lower* bound 0.5) went unrecorded for a full round.
A Sérsic index has pinned at that 0.5 floor in all three rounds. Pinning matters twice
over: it usually means your prior is fighting the data, and HMC mixes badly along a
boundary — that pinning drove 18/22 parameters past R̂ > 1.01 in round 1.

Note the diagnostic value of a *deliberate* boundary: because the 0.5 floor is there for a
known numerical reason (§2), a parameter pinned against it is telling you the profile
family is wrong, not the bound.

**3. Convergence.** R̂ and ESS per parameter, with the failing set named, not just counted.

**4. Multimodality and label switching.** Check both, but calibrate your expectations:
**genuine persistent multimodality is rare, and what looks like it usually dissolves once
the fit actually converges.** Treat an apparently multimodal posterior as evidence that
something upstream is wrong — a poisoned preconditioner, too-short chains, a boundary-pinned
parameter — before treating it as a property of the physics. Cheap checks: per-chain means
of the key parameter (θ_E was unimodal over 1.479–1.488 across 50 chains); for exchangeable
components, count sign crossings of the ordering statistic (0 crossings in 50 chains — no
label switching).

Two lessons from how I used these. First, I ruled both out correctly in round 2 and then
drew the *wrong* conclusion from having done so, because I had not considered the SVI
hypothesis — **eliminating two explanations does not confirm a third.** Second, because
label switching so often fails to materialize, prefer a *detect-always, correct-only-if-
detected* design over unconditional post-hoc relabeling. Relabeling that fires when there
was nothing to fix is a silent corruption of exactly the kind this guide is about.

**5. Residual structure, not residual magnitude.** A χ² tells you there is a problem; only
structure tells you what to add. Look at azimuthally binned residuals about each component
centre, and at the residual in physically meaningful regions (core, arcs) separately —
global χ²ᵥ 1.302 concealed a core χ² of 7.09.

**5b. Agreement across independent kernels.** If you have run more than one sampler off the
same surrogate (§4.3), check they land in the same place. DESI238's mass parameters agreed
across HMC, MAMS and MCLMC to ≤ 0.00004" on θ_E — far inside any single run's posterior width.
This is the check that the *sampling* is not the thing setting your answer, and it is nearly
free once you have the runs. It does not validate the surrogate, which is common to all of
them; only a re-run of MAP+SVI does that.

**6. A physical cross-check of the headline quantity.** Compare θ_E against the observed
image geometry — I kept the four measured clump radii (1.06, 1.25, 1.64, 2.11 arcsec) in
the priors module specifically so the diagnostic script could assert the θ_E posterior sat
sensibly among them. An inference number that has never been checked against the picture is
not yet a measurement.

**7. Closed-form invariants on any geometry you derive.** This is the cheapest check in the
list and it caught the worst bug of the port.

I wrote a finite-difference critical-curve finder, ran it on a fitted EPL with γ = 2.53, got
*two* critical curves, and confidently wrote up the inner one as a real feature caused by the
divergent deflection at the origin for γ > 2. The human's correction was one line:

> *"gamma greater than 2 should only have one critical curve."*

That is checkable in closed form. The radial eigenvalue of a circular EPL is
`1 − (2−γ)(θ/θ_E)^(1−γ)`, which vanishes only where `(θ/θ_E)^(1−γ) = 1/(2−γ)`. For γ ≥ 2 the
right-hand side is non-positive and there is **no radial critical curve at all**. So the inner
curve was not a physical consequence of γ > 2 — it was **manufactured by my own contour finder**
tripping over the `α ~ θ^(2−γ)` singularity at the origin.

Note how well it hid. The same code recovered θ_E to 2e-4 arcsec, so every quantitative check
I had passed. Only a *qualitative* invariant — how many curves there should be — exposed it.

> Write down the invariants your derived quantities must satisfy — how many critical curves
> a given mass slope admits, whether a caustic lies inside its critical curve, whether a
> magnification has the right sign, whether the enclosed mass rises monotonically — and assert
> them. They cost nothing, they need no reference implementation, and they catch the class of
> bug that gets the headline number right and the geometry wrong. Make the check fail loudly
> rather than print a caveat next to a figure nobody will re-examine.

### On residual maps and σ maps

A raw normalized-residual map is easy to over-read. In DESI238 several clumps showed 6σ+
peak pixels while matching to within 10% in *integrated* flux — small (~1 px) astrometric
offsets create large point-wise residuals. **Aperture photometry on an azimuthal-median-
subtracted image is the more reliable check**, and it is the one that revealed the N arc
was genuinely underfit 2× (ratio 0.55) rather than merely offset.

### Check for cannibalisation whenever you add a component

Adding a flexible light component near the Einstein radius creates the risk that it steals
arc flux and biases the mass model. Do not assume a χ² improvement is safe. When I added
the second deflector component I compared clump-flux ratios before and after — three of
four within 3% (1.03 / 0.99 / 1.02 / 0.83) — and confirmed θ_E moved only 1.484 → 1.480,
well inside the shift already seen between rounds. Then I could say "no cannibalisation."

---

## 6. Reporting: the part most likely to be wrong even when the fit is right

**Quote R̂ and ESS alongside every parameter value.** A θ_E without a convergence
diagnostic is not a measurement. Apply this even when it produces an uncomfortable
sentence: round 1's θ_E was reported to three significant figures *with* the note that the
joint chain had not fully converged, rather than dropping the caveat because the number
looked precise.

**A spread between models is only a systematic if the models are competitive.** This is
where I got the DESI238 write-up wrong, and the error is an easy one to make because the
cautious-sounding move is the incorrect one.

The headline moved from 1.516 ± 0.002" (round 1) to **1.480 ± 0.003"** (round 3), and I
reported the 0.036" gap as "the honest systematic floor on θ_E from light-profile choice,"
reasoning that a model-to-model spread an order of magnitude larger than either error bar
must dominate. That reasoning is wrong. **Averaging over models, or inflating an error bar
to span them, is only meaningful when the models have comparable evidence.** Round 3 beats
round 2 by Δlog L = +911.9 for 7 parameters (ΔBIC ≈ −1762, robust across the whole
background_rms range), and round 1's single-Sérsic source is worse still. A model that the
data rejects this decisively contributes nothing to a systematic budget — its θ_E is not a
competing estimate, it is a wrong one. The correct statement is that round 1 is superseded,
full stop; the 0.036" is the size of a *bias that has been removed*, not an uncertainty that
remains.

> **The test.** Before quoting a model-to-model spread as a systematic, compute the
> evidence gap. If one model is decisively preferred, quote its error bar and say the
> others are excluded. Reserve the spread-as-systematic treatment for models that are
> genuinely hard to choose between — where the evidences are close and the difference is a
> real ambiguity in what the data supports. Inflating an error bar to cover a rejected
> model is not conservatism; it is reporting a number the data has already ruled out.

The caveat that survives: my ΔBIC is evaluated at point estimates, so it is a
maximum-likelihood comparison, not a Bayes factor. A proper marginal likelihood would
penalize the 36-parameter model's larger prior volume more heavily — though nowhere near
enough to close a 1762 gap. Say which one you computed.

**χ²ᵥ ≈ 1 is not evidence your noise model is right.** Propagate the uncertainty you
measured in §1. Over the blank-patch range 0.0070–0.0092, DESI238's χ²ᵥ = 0.989 spans
0.82–1.05 — ±10% on the noise scale is ±20% on χ², larger than the distance from unity.
**Quote the improvement (1.302 → 0.991), not the absolute value.** Conversely, this makes
the per-pixel exposure map more worth building, not less.

**Model comparison: report Δlog L, ΔBIC, and the parameter count together**, and check the
verdict survives your nuisance uncertainties. Bulge+disk over single-Sérsic deflector was
Δlog L = +912 for 7 parameters, ΔBIC ≈ −1762, robust across the whole background_rms range.
A ΔBIC that flips when you move the noise scale within its measured error is not a result.

**Supersede explicitly; never silently overwrite.** Each round of DESI238 got its own
results file (`RESULTS.md`, `RESULTS_v2.md`, `RESULTS_v3.md`), and corrections were added
*inline and dated* to the superseded documents — including struck-through text where a
conclusion was withdrawn — rather than by editing the old claim away. When a number should
no longer be used, say so in the place someone would go looking for it:
*"Do not quote θ_E = 1.484 ± 0.002" from that run."*

**Report partial validations as partial.** When I productionized an ad hoc photometry
script, it reproduced the original clump ratios to within a few percent but left a 5–11%
discrepancy in absolute flux for three clumps, from an unresolved difference in the binning
convention. That went in the writeup as an unresolved discrepancy. Rounding a partial check
up to "confirmed" is the single easiest way to lose a reader's trust permanently.

---

## 7. Tools, records, and provenance

**Do not start patching a package until you have checked whether it is simply out of date.**
I wrote and carried a compatibility layer for three rounds against an incompatibility that
had already been fixed upstream — entirely avoidable work, and every line of it was surface
area for the silent failures this guide is about. "The version in front of me is broken" and
"this package is broken" are different claims, and only the second justifies patching. Ask
the human whether the copy you have is pinned deliberately or just old; they may not know it
is stale either.

**Use the library's implementation of anything numerical, and go looking for it first.** I
hand-rolled a critical-curve and caustic finder — finite differences, a Jacobian determinant,
a heuristic for the plot window. The human's redirect was one line: *"[the package] contains
some analysis functions. You can use these instead of your homegrown ones."* The library's
version was tested against its own ray trace. Mine was not merely redundant: it is what
invented the phantom critical curve of §5. The whole problem was self-inflicted.

> A hand-rolled version of something the library already has is not just wasted effort — it is
> **unvalidated**. The library's version has tests, usually against an independent reference
> implementation. Yours has your confidence.

This matters more in lens modelling than in most fields, because the quantities involved —
deflection fields, critical curves, caustics, magnifications — are easy to compute *almost*
right and very hard to eyeball.

**If the likelihood changes, re-fit — and say in advance what that puts at risk.** I moved
this work onto a new interface that altered the noise convention: the variance went from
being recomputed each step from the *simulated* image to being derived once from the
*observed* one. That is a change to the likelihood, not a refactor. Two things made it
defensible:

- **Refuse to reinterpret old checkpoints rather than converting them.** Stored samples that
  predate the change decode to plausible wrong numbers. A converter would be guesswork
  presented as data; refusing to read them sends you to the version that can.
- **Re-fit, and treat the agreement as the result.** θ_E came back 1.480 → 1.478 against a
  ±0.003" error bar, from a different normalisation, a different parameter ordering and a
  different sampler. Stated beforehand as the thing at risk, that agreement is evidence about
  the physics. Stated afterwards, it is just a number that happened to match.

**Record enough provenance to reconstruct the exact model a run used**: every hyperparameter,
the seed, the code version, the prior versions, the model configuration. Your diagnostics
should rebuild the model a run was fitted with rather than assuming today's defaults —
otherwise re-reading an old run silently reports it under a model it never used.

**Test that your safety checks fire.** The preflight suite in §1 should include an assertion
that each guard *rejects* a known-bad case and accepts the known-good ones, with an
order-of-magnitude margin either side. This is the item people skip. A safety check nobody has
seen fail is a safety check you do not know works.

**Put durable knowledge where the human can read and correct it.** If you keep private notes
of any kind, do not let them become a second copy of the project's own documentation. I wrote
seven private entries summarising a modelling round and was told to delete them: every one
restated what the results file already said. Findings, caveats and procedure belong in the
versioned write-up, where they are dated, reviewable, and cannot drift out of step with the
code they describe. **The test: if a human would ever want to read it or correct it, it goes
in the repository.**

**Keep the measurement scripts that produced your priors.** My clump photometry is still in
the repository, so the θ_E prior centre traces back to a measurement rather than to an
assertion in a planning document. A prior whose provenance you cannot reconstruct is an
assumption you can no longer defend.

**Write down coordinate conventions where they will be read.** My clump position angles are
**array-space** angles, measured from the +row axis toward +col — *not* sky PA. Converting
them through the WCS the standard way puts every marker on empty sky. This cost real time,
and it is exactly the kind of thing that is obvious to whoever wrote it and invisible to
everyone else, including to you three weeks later. Position angle, ellipticity sign, and the
handedness of the coordinate system are the three that recur.

---

## 8. Working with the human

You are working with a person who knows the physics better than you do and who cannot check
every number you produce. Your obligation is to make your work **cheap for them to verify**.

**Ask about what the data cannot tell you.** Some numbers are simply not in the files. The
DESI238 PSF carried no scale keywords; whether it was oversampled relative to the drizzled
grid was unknowable from the data and decisive for the fit. My data audit said so in those
terms — *"Whether it's at 0.065"/pix or oversampled is unknowable from the file... I flagged
it as needing external knowledge, not as a settled value"* — and the answer came back
"the PSF scale is the same." It is now recorded as *"confirmed by the repository owner, not
derivable from the data."* Asking cost a minute; guessing would have cost a round. Good
candidates: PSF scale and provenance, whether a cutout's parent frame is what you think,
which quantity they actually want, detector/reduction quirks, and how much compute you have.

**When a question goes unanswered, flag the assumption you made instead of it.** My question
about the lens-light `n_sersic` ceiling was never answered across two rounds. I chose
`Uniform(2, 10)` and said so explicitly in the writeup. That is the correct fallback:
proceed, but leave the choice visible and attributed to you rather than letting it pass as
though the data had determined it.

**Never invent a fact about the person.** I put "Sean Jackson" on a report's author line,
inferred from the `seanjx` in the filesystem paths. That is a guess dressed as a fact, in
the one field where being wrong is most personal. The correct move was to leave it blank or
ask. The same applies to affiliation, pronouns, seniority, and what they already know —
none of it is derivable from a username or a directory tree.

**State your assumptions in the same breath as your result.** Not in an appendix. If the
number depends on `background_rms = 0.0075`, say so where you say the number.

**Distinguish what you verified from what you inferred, every time.** "Bit-for-bit
identical to `SCI[1010:1130, 1165:1285]` — verified, not inferred" is worth the extra four
words. So is "TF code changed but unverified, since TensorFlow is not installed in this
environment." A reader who can trust your epistemic labels can skip re-checking everything;
one who cannot must re-check everything, and will.

**Retract loudly.** When you find you were wrong, correct the claim where it was made, date
the correction, and say plainly which conclusion is withdrawn. I got several things wrong
in this project — the sub-pixel-centroid explanation for the core residual, the "slow
mixing" diagnosis, the "posterior ridge" reading (written up as "the expected signature of
a bulge–disk degeneracy," and actually short chains), a two-point trend that was noise, an
author name I invented. Each correction made the record more useful. Silently improving a
number in a later document makes it less.

**Do not accept a validation you have not read.** A delegate reported "max abs diff 0.0, as
expected since both call the same code path." That clause makes the check circular — it
cannot fail, so it is not evidence. I caught it by reading the check rather than the summary.
Treat any check whose own description explains why it must pass as unrun. This applies to
your own checks and to anything a delegate hands you.

**If you delegate, verify a delegate's report of *state* against the artifacts.** I was told a
fit was still running when its process had already died. Whoever or whatever reports to you —
a collaborator, a job scheduler, another agent — a report about the state of a computation is
a claim about it, not an observation of it. The output files and their timestamps are the
truth, and checking them costs seconds.

**Trust the human's physical intuition against your defaults.** I proposed unconditional
post-hoc relabeling to handle label switching between the two source components. The reply
was skeptical: *"Ideally MAP and SVI will have separated the parameters for the two
components enough such that the chains won't be switching between the two."* That was
right — 0 sign crossings across 50 chains, components separated by 24σ — and the design
became detect-always, correct-only-if-detected, which is strictly better than either.
The textbook default was the worse answer here, and the person who knows the system said so.

**Do not present a fit as finished when it is convergent-but-partial.** Say which
parameters failed, and why it does or does not matter for the quantity being reported.

**Manage their compute, and say what you are spending it on.** Multi-start MAP over 1024
samples plus 1500 SVI steps plus 50 HMC chains is not free, and neither is your own token
budget. Prefer: cheap preflight before expensive fits; single-variable ablations that
answer one question each over grand redesigns that answer none cleanly; delegating
exploratory reading and routine implementation where you can; and telling the human *before*
you launch something long, not after.

**Check the stated goal against the results you already hold before buying more compute.**
The target was R̂ < 1.01 in every parameter. A 32-chain trial met it — worst 1.003 — and my
next action was to launch a 128-chain run "for margin" that nobody had asked for. The human
stopped it with four words: *"Wait, hasn't it already converged?"* Escalating past a
satisfied criterion is not caution, it is spending someone else's allocation on a goal you
invented. Re-read the success condition, compare it to the numbers in hand, and if it is met,
say so and stop.

**When the deciding check is already running, say so instead of guessing its answer.** Asked
*"Surely seeds don't matter that much...?"*, I said mostly they don't — reasoning from two
runs, while a five-seed sweep that would settle it was in flight. It came back 2 of 5 passing
and I had to retract within the same exchange. The honest answer was available and costs
nothing: *"I don't know yet — the replicate seeds land in about twelve minutes and will tell
us."* A skeptical question from the human is a prompt to check, not a prompt to reassure, and
an answer that the pending evidence then contradicts costs more trust than the delay would
have. This is the mirror image of *trust the human's physical intuition* above: there the
human's doubt was right about the physics, here it was right about the rigor. Both times the
doubt was the signal.

**Do not confuse the machine's inventory with your allocation.** `nvidia-smi` showed eight
GPUs with six idle, so I planned parallel fitting arms. The correction was one sentence:
*"You only have 1 GPU, actually."* A scheduler's view of what is yours is not visible from
inside the node. Ask before designing around resources you have only observed, not been
granted.

**Take the stated scope, and stop there.** Given *"Let's do 3 first, and then 1,"* the right
behavior is to do 3, then 1, and then report — not to continue into 2 and 4 because they
were on a list you wrote. Proposing next steps is useful; executing unrequested ones spends
someone else's budget on your judgment of what matters.

**Let the human hold credential-sensitive steps.** When a push failed on an SSH key, I
offered three remediation paths rather than guessing at credentials, and the answer was
*"I'll push it myself."* Outward-facing and hard-to-reverse actions are theirs to authorize
each time; approval for one is not approval for the next.

**Elicit their modeling philosophy; it is not derivable from the data.** How informative a
prior should be is a judgement call about what you are willing to assume, and the person
whose result it is gets to make it. Mine was stated plainly: *"I think we should generally
set less informative priors. We should allow for both the possibility that the source has
two separated components and the possibility that the source is a compact clump situated in
a more extended host."* That single instruction is why the two source components got
identical wide priors (§2) — so the data, not I, chose the morphology. Ask for this early.
It changes the model, not just the presentation.

**Treat their formatting requests as substantive.** "Use an asinh scale for the observed,
model, and source-plane panels" is not decoration — a linear stretch hides exactly the
low-surface-brightness arc structure the fit is about. I made the change and re-ran all
three numeric validations to confirm nothing else moved. Do both parts.

**Write for the reader you will be in three weeks.** The corrections in these results files
are dated because I needed to know which of two contradictory statements was later. Convert
relative dates to absolute. Name files and functions. Quote the actual number, not "the
value improved."

**Push back when the analysis asks for it.** The round-1 falsification test I designed did
not work: its prior's 99th percentile (2.07") already contained the hypothesis it was
supposed to exclude, so it converged to the same basin and tested nothing. Saying "this
test as designed does not discriminate the two hypotheses, and here is what would" is more
useful than reporting agreement as confirmation. **A test that cannot fail is not evidence.**

---

## 9. The short version

**The data (§1)**

1. Measure every likelihood input from the files; record the derivation; test it in preflight.
2. Background RMS from blank sky in several separated patches — and keep the spread.
3. Normalize the PSF. Check its centering. Ask about its scale; it is often not in the file.
4. Interlopers: model them if you can, mask them if you must, crop only when they are far
   from the arcs — and call a crop what it is, an accepted unmodelled systematic.

**The model (§2–§3)**

5. Start simple; add a component only when a *structured* residual demands it; rule out the
   boring explanation first.
6. Calibrate amplitude priors by rendering prior draws, never by reasoning.
7. Identical priors for exchangeable components; distinct priors for physically distinct ones.
8. Know where your profile family stops being valid, and put the prior's floor there.
9. Store parameters in physical space. If you store unconstrained samples, the prior's
   *support* becomes part of your file format — version it and never edit one in place.
10. Never assume the parameter ordering. Derive it, then probe it one coordinate at a time.

**Running the fit (§4)**

11. NaN-aware selection, never bare `argmax`, over multi-start optimizers.
12. Check the variational covariance's *conditioning* — not its positive-definiteness —
    before reading any R̂. Passing the guard is a floor; read the ELBO tail sd as quality.
13. Elevated R̂: 4× the draws, change nothing else. If it does not fall, work the remedy list
    before blaming the geometry — and at high dimension reach for a **better surrogate and a
    different kernel before more draws**. Hold the surrogate fixed when comparing samplers.
14. An R̂ that only just clears the threshold means the budget is too small. Buy draws.

**Diagnostics (§5)**

15. Automate prior-boundary pinning detection at the 2% level. A pinned parameter cannot mix.
    Move a suspect bound *once*: a parameter that follows it wants a different profile family;
    one that settles at an interior mode was genuinely prior-limited. And ask whether the
    component is resolved at all before reading its shape parameter as a morphology.
16. Structure in the residual, not magnitude, tells you what to add.
17. Assert closed-form invariants on any geometry you derive — how many critical curves a
    given slope admits, signs, monotonicity. A phantom critical curve survived every
    quantitative check I had; only counting them exposed it.
18. Persistent multimodality is rare; suspect your sampler before your posterior. Detect
    label switching always, correct only if detected.
19. Check for flux cannibalisation whenever you add a component near the arcs.

**Reporting (§6–§7)**

20. Report R̂ with every number. Convergence in all parameters is the goal and is reachable;
    a result is reportable before you get there if the quoted parameter is converged and you
    name the ones that are not.
21. A model-to-model spread is a systematic only if the evidences are comparable. A
    decisively rejected model is superseded, not a source of uncertainty.
22. χ²ᵥ ≈ 1 is not evidence the noise model is right — propagate the noise-scale uncertainty.
23. Supersede explicitly and date your corrections. Report partial validations as partial.
24. Test that your safety checks fire. Keep the scripts that produced your priors. Write down
    your coordinate conventions where they will be read.

**The human (§8)**

25. Ask what the data cannot tell you — PSF scale, prior philosophy, your compute allocation.
    Label verified versus inferred. Never invent a fact about the person.
26. A check whose description explains why it must pass has not been run. Read it.
27. Retract loudly, in the place the claim was made, with a date.
28. Before buying more compute, check whether the stated goal is already met.
29. If the deciding check is already running, say so rather than guessing its answer.
30. Durable knowledge goes where the human can read and correct it.

The unifying reason for all of it: **these failures are silent.** None of the real errors in
this project — the contaminated background estimate, the array-space PA convention, the
NaN-argmax selection, the prior-version drift, the indefinite mass matrix, the float32
eigenvalue artifact, the mislabeled progress bar, the non-discriminating falsification prior,
the phantom critical curve, the dict that impersonated a list — raised an exception or
produced an obviously wrong magnitude. Not one was caught by code review or by reasoning a
priori. Each was caught by comparing independent runs, re-deriving a quantity by a second
method, running a deliberate single-variable experiment, asserting an invariant, or chasing a
downstream anomaly backwards.

Worth noting which of them a *human* caught, and how cheaply. "The background is contaminated
by galaxy light." "Gamma greater than 2 should only have one critical curve." "Hasn't it
already converged?" Each was one sentence, none required reading my code, and each cost me a
round of work to have missed. §8 is not an afterthought to the technical sections; it is where
most of the real errors in this project were actually found.

Build the redundant check. The absence of an error message is not evidence of correctness.
