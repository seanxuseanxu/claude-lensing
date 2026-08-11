---
name: elevated-rhat
description: Decide what to actually do about an elevated R̂ or a failed convergence table, in the order that has worked on this project. Use when a run comes back with R̂ above 1.01, before proposing a longer run, a reparametrization, or a claim that the posterior geometry is intractable.
---

# Elevated R̂: work the list, do not diagnose it as fate

`claude/guide/04-running-the-fit.md` §4.3–§4.4. Every elevated R̂ in this project either fell with the list below or traced
back to a boundary-pinned parameter. **"The geometry is intractable" is a claim requiring
evidence that you have exhausted this list**, not an explanation for an unfinished fit.

## Step 0 — is the surrogate healthy?

If `svi_health` failed or the ELBO tail sd is large, stop. This is not a sampling problem and
no amount of sampling fixes it. A badly conditioned variational covariance inverts to an
effectively indefinite mass matrix, **every** R̂ fails at once, and the natural reading —
"the chains need to be longer" — costs a full round of compute. Lower the SVI peak learning
rate; never raise the guard threshold.

## Step 1 — the free test, before spending anything

`diagnose.py` already wrote both, from the run you have. Read them before proposing anything.

- **`ess_vs_draws.png`**, and the `=== ESS vs draws ===` block above it. ESS should grow
  roughly linearly. The fitted slope is printed per parameter and the verdict is stated in
  words: at ~1 a longer run buys what it costs; sublinear means more draws will not fix it
  (Vehtari §5, §7) and sends you to step 3 without spending anything. `r3_unmasked` reads
  −0.02 flat; the converged `r2_s2l2` reads 0.98–1.26.
- **`rank_plots.png`** — pooled ranks, one panel per chain, uniform if mixed. This names
  *which* chain disagrees, which R̂ cannot. On a >10-chain run it draws the ten least
  uniform, chosen by chi-square against uniform, so the bad chain survives the cut.

Also read the `=== ESS floor ===` block. If it fails, R̂ is not interpretable in either
direction and the only move is more draws — none of the reasoning below applies yet.

## Step 2 — if ESS grows linearly: 4× the draws, change nothing else

Confirm the runs really are otherwise identical (check the SVI starting loss matches). If R̂
falls, it was chain length, and whatever degeneracy you had written down as the cause was
wrong — DESI238's "expected signature of a bulge–disk degeneracy" was short chains.

## Step 3 — if it does not fall, in this order

1. **Improve the model.** Non-mixing is often a modelling problem in a sampling costume. A
   parameter pinned at a prior boundary cannot mix by construction; a missing component
   leaves a degeneracy the sampler must crawl along.
2. **Get a better surrogate.** *This is where the leverage is at high dimension, and it is
   ranked too low by instinct.* DESI238 at 36 parameters sat at R̂ 1.338 after a 4× chain
   increase had already been spent. Twenty-one minutes of MAP+SVI — 4096 particles, 1500 MAP
   steps, 12,000 SVI steps at n_vi 2048, peak LR 1e-3 — took the condition to 2.3e+06 and the
   ELBO tail sd to 0.1, and worst R̂ went 1.338 → 1.003. The preconditioner is a fixed cost
   every subsequent draw pays interest on.
3. **Change the kernel, not just its length.** MCLMC was ~9× faster per draw than fixed-length
   HMC here: 8× the draws in two thirds the wall time at 1.7× the minimum ESS. Try one early
   on a short run.
4. **Sample longer**, and be honest that 4× may not have been enough.

Reparametrization is last, not the diagnosis you jump to.

**When comparing samplers, hold the surrogate fixed** — copy `map.npz` and `svi.npz` into
each run directory rather than re-running those stages. Otherwise you compare
sampler-plus-surrogate pairs and can attribute nothing. It costs one `cp`.

## If R̂ only just clears the threshold

That is a budget problem, not a finish line (§4.4). Five seeds of a 32×3000 run gave 1.003,
1.019, 1.009, 1.093, 1.019 — two of five passing. The remedy is draws, not a multi-seed
protocol. Seed variation earns its keep only as a cheap test of *marginality*: if two short
runs straddle the threshold, that settles whether you are converged or lucky.

## Before you launch anything

Re-read the success condition and compare it to the numbers in hand. A satisfied criterion
plus a bigger run "for margin" is spending someone else's allocation on a goal you invented
(§8). Say what you are spending and on what, *before* launching, and background the fit so
completion is reported rather than polled.
