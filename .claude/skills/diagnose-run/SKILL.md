---
name: diagnose-run
description: Read out a finished MAP/SVI/HMC run in the order §5 of the lens modelling guide requires, including the steps that have no automated check — opening the residual and source-plane images, and reading the mass model against physical typicality. Use after any run_fit.py invocation completes, before quoting any number from it.
---

# Diagnose a finished run

Argument: a run directory, e.g. `claude/DESI234/runs/r3_fg`.

**Order matters, because an early failure invalidates everything after it**
(`claude/guide/05-diagnostics.md`). Do
not skip ahead to R̂ because it is the number you want; a poisoned surrogate makes every
convergence statistic below it meaningless.

## 1. Run the tooling, once

```bash
export PYTHONPATH=claude/pipeline
python claude/pipeline/diagnose.py <run_dir>
```

That covers steps 1, 2, 2b, 3, 4 and part of 5 below and writes the figures. Read its output
top to bottom rather than grepping it for R̂.

## 2. Then read it in this order

1. **SVI health** — condition number and ELBO tail sd. If the guard failed, stop: nothing
   downstream means anything, and the fix is a lower SVI peak learning rate, never a raised
   threshold (§4.2). Passing is a floor, not a target — tail sd ~0.1 converged DESI238 and
   ~10 did not.
2. **Prior-boundary pinning**, then **prior–posterior tension**. Pinning means the prior set
   the value; tension means the data set it against the prior's advice, and tension is the
   commoner sign of misspecification. Tension covers unbounded priors, which pinning cannot
   see — this is exactly what `r3_fg`'s `|e| = 0.749` deflector escaped through.
3. **The ESS floor, then convergence, in that order.** The `=== ESS floor ===` block checks
   ≥50 per split half, ≥400 total, ≥4 chains. If it fails, stop reading the R̂ table: below
   the floor a *passing* R̂ means nothing either, and the run is undetermined rather than
   converged. If it passes, name the failing R̂ parameters, do not just count them. State
   which R̂ — this pipeline computes the rank-normalized folded statistic gated at 1.01, not
   the classic PSRF.
4. **Multimodality and label switching.** Suspect the sampler before the posterior.
5. **Residual structure, not magnitude.** Weight χ²ᵥ last: it is the diagnostic least able to
   fail informatively, and `r3_fg` had the best χ²ᵥ on its system with a 7:1-needle deflector.

## 3. Open the images. This step is not optional and has no automated substitute

Read these files with the Read tool — they render:

- `<run_dir>/residuals.png`
- `<run_dir>/lensing_overview.png`
- `<run_dir>/rank_plots.png` and `<run_dir>/ess_vs_draws.png` — only when something failed
  above. They diagnose *why* a fit is not mixing and which chain is responsible; on a clean
  run the printed slope and the floor block already say everything they show.

Reasoning about residual structure from summary statistics while these sit unopened is a
failure this project sustained for several rounds (§5). Two things to look for that no
statistic reports:

- A source component with `R_eff` several times the caustic scale, centred outside the
  caustic, is a **smooth pedestal buying χ²**, not a lensed galaxy.
- Coherent, *localized* structure is what justifies adding a component. A merely high global
  χ² does not.

## 4. Read the mass model against physical typicality

No automated check can do this (§5 step 2b). Prompts, not gates:

| quantity | typical | worth arguing for |
|---|---|---|
| axis ratio `q` | `q ≳ 0.6` | `q < 0.5` |
| slope `gamma` | 2.0–2.2 | `> 2.5` or `< 1.8` |
| external shear | few × 0.01 | `> 0.2` implausible |
| interloper carrying mass | rare | a detection needs an alternative ruled out |

The band is a prompt because Foundry I publishes `gamma = 1.372 ± 0.023` — outside it, and
correct. An atypical value must be *noticed and argued for*, not gated out.

## 5. Cross-check the headline against the picture

Compare θ_E to the measured clump radii kept in the lens's `config.py`. R̂ compares chains to
each other and never to the truth; four chains can agree on a mean that is 3 SD wrong with
clean R̂ and ESS (Vehtari Appendix C). This step and step 4 are the only defences against that.

## Report

State, in this form: which parameters failed R̂ and what that costs; the χ²ᵥ *with* the note
that it includes core pixels and is inflated relative to a published one; every number you
quote alongside its R̂. If the fit is convergent-but-partial, say so — do not present it as
finished (§6).
