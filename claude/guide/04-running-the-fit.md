[Guide index](INDEX.md) · previous: [03-parameter-conventions.md](03-parameter-conventions.md) · next: [05-diagnostics.md](05-diagnostics.md)

---

# 4. Running the three stages, and how each one fails

The MAP → SVI → HMC chain is common (in spirit) to most modern lens
pipelines: a multi-start optimizer to find the basin, a variational fit to shape a
preconditioner, then gradient-based MCMC for the posterior. Each stage consumes the last,
which means **each stage can be poisoned by the last, silently.**

## 4.1 MAP: never use bare `argmax`

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

## 4.2 SVI: a diverged variational fit poisons HMC and looks like slow mixing

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

## 4.3 Elevated R̂: work the remedy list, do not diagnose it as fate

When R̂ is elevated and SVI is healthy, the first move is cheap and diagnostic:

> **Re-run with 4× the draws and change nothing else.** If R̂ falls, it was chain length.

In DESI238 (`bulgedisk` → `bulgedisk_long`, 750 → 3000 draws, burn-in 250 → 1000): worst R̂
3.242 → **1.338**, failing count 33/36 → **10/36**, χ²ᵥ unchanged at 0.989. It fell. It was
chain length, and the "expected signature of a bulge–disk degeneracy" I had written down
was wrong. Confirm the runs really are otherwise identical — I checked that the SVI
starting loss matched.

**There is a cheaper test that predicts the answer before you spend the 4×.** Plot ESS against
the number of draws used, on the run you already have. ESS should grow roughly linearly; if it
grows sublinearly or falls, "simply running longer chains will likely not solve the convergence
issues" (`papers/r-hat_convergence/NOTES.md` §5, §7). That distinguishes *buy more draws* from
*fix the model or the surrogate* using samples already on disk. Rank plots — histograms of the
pooled ranks, one panel per chain, uniform if mixed — are recommended there over trace plots for
the same diagnostic, and they catch a misbehaving chain at R̂ values that would pass a loose
gate. Both are now in `diagnose.py` and run on every invocation — `ess_vs_draws.png` with the
fitted slope printed and read out in words, and `rank_plots.png`. On the largest run to date
(2M draws) they add ~30 s to a 172 s diagnosis, against the hours a 4× run costs.

They earn that on the runs already in this repository. DESI234's `r3_unmasked`, the
frozen-chain failure, gives ESS 51 flat across the whole range — slope −0.02, no growth at
all — and rank histograms that are a single spike per chain: the chains never moved, and no
number of extra draws would have helped. The converged `r2_s2l2` gives slopes 0.98–1.26 and
broadly uniform ranks. The two cases are distinguishable at a glance, from draws already
on disk.

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

## 4.4 A marginal R̂ is a budget problem — buy draws, not seeds

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

[Guide index](INDEX.md) · previous: [03-parameter-conventions.md](03-parameter-conventions.md) · next: [05-diagnostics.md](05-diagnostics.md)
