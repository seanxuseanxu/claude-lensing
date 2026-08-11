# Papers index

Five sources live under `papers/`, each as LaTeX with figures. Every one has a `NOTES.md`
beside it: a reading note written against the source, with section and equation numbers
cited so any claim can be checked back to the paper rather than taken on trust.

Read the note first. Go to the `.tex` when the note says the paper is silent, or when you
need a number the note did not carry.

| Dir | Paper | Entry point | Note | What it settles |
|---|---|---|---|---|
| `gigalens_paper/` | Gu et al. 2022, *GIGA-Lens: Fast Bayesian Inference for Strong Gravitational Lens Modeling* (arXiv:2202.07663) | `ms_SL-model.tex` | [NOTES.md](gigalens_paper/NOTES.md) | The MAP→SVI→HMC pipeline we run, with its actual hyperparameters, likelihood and priors |
| `strong_lens_foundry_I/` | *DESI Strong Lens Foundry I: HST Observations and Modeling with GIGA-Lens* (submitted, ApJ) | `main.tex` | [NOTES.md](strong_lens_foundry_I/NOTES.md) | Where our three lenses come from; the survey, the data, the conventions |
| `gravitational_lensing_lecture_notes/` | Narayan & Bartelmann 1996, *Lectures on Gravitational Lensing* (SISSA; astro-ph/9606001) | `SISSA.tex` | [NOTES.md](gravitational_lensing_lecture_notes/NOTES.md) | The lensing theory itself: caustics, image configurations, the fifth image |
| `mclmc/` | Robnik & Seljak, *Fluctuation without dissipation: Microcanonical Langevin Monte Carlo* (AABI 2024) | `main.tex` | [NOTES.md](mclmc/NOTES.md) | The sampler behind `blackjax.mclmc`; what it proves and what it does not |
| `r-hat_convergence/` | Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021, *Rank-normalization, folding, and localization: An improved R̂* (Bayesian Analysis) | `rhat.tex` | [NOTES.md](r-hat_convergence/NOTES.md) | The convergence statistic we gate on, and the failure modes it cannot see |

## What to read for a given question

**"Is this fit converged?"** → `r-hat_convergence/NOTES.md` §4 for the threshold, §1 for what
R̂ is blind to, §7 for what to do once it fails. Note §1 item 5: chains can agree with each
other and all be wrong together. R̂ compares chains to each other, never to the truth.

**"Why did MCLMC freeze / is my step size wrong?"** → `mclmc/NOTES.md` §4. The one load-bearing
claim: drift and diffusion each separately preserve the target, so ε is the only bias source —
suspect the step size, not the decoherence parameter.

**"What should this parameter's prior be?"** → `gigalens_paper/NOTES.md` §3 (the 22-parameter
table, simulation vs. fitting prior) and `strong_lens_foundry_I/NOTES.md` §3 (the 41-parameter
version on real HST data). The Foundry priors are the closer analogue to our runs.

**"Is this image configuration physical? Should there be a fifth image?"** →
`gravitational_lensing_lecture_notes/NOTES.md` PART C.

**"What is normal for θ_E, γ, or an error bar?"** → `strong_lens_foundry_I/NOTES.md` §4 for the
one fully modelled system, `gigalens_paper/NOTES.md` §6 for the 100-system recovery test.

## Findings from this reading worth carrying forward

These are the places where the papers bear directly on decisions already made in this repo.
Each is stated with its source so it can be argued with.

1. **Our R̂ convention is the modern one, and 1.01 is the right gate.** `diagnose_convergence`
   uses ArviZ `method="rank"`, which is already the max of the rank-normalized and folded
   terms (`nanolens/src/gigalens/jax/analysis/diagnostics.py` module docstring). That is
   exactly the statistic Vehtari et al. recommend, and 1.01 is exactly their threshold
   (`r-hat_convergence/NOTES.md` §4). **Foundry I uses a different convention** — TFP's R̂,
   "approximately the square" of the modern one, gated at 1.2 (`strong_lens_foundry_I/NOTES.md`
   §4, footnote `fn:rhat`). Their 1.2 ≈ our 1.1. Do not compare their numbers to ours without
   that conversion.

2. **R̂ < 1.01 alone is not sufficient.** The paper couples it to ESS: at least 50 per split
   half, ≥400 total on ≥4 chains, before R̂ is trustworthy at all (`r-hat_convergence/NOTES.md`
   §4). We gate on R̂ and report ESS; the ESS floor is not currently enforced.

3. **The GIGA-Lens paper's published validation is simulation-only.** The HST demonstration
   section of `results.tex` is commented out and not in the compiled paper, and the masking
   discussion lives only there (`gigalens_paper/NOTES.md` §2, §6). So the paper prescribes no
   masking convention — our centre-mask practice is inherited from elsewhere, not from it.

4. **Foundry I says almost nothing about our three lenses.** DESI-234.4783+14.7232,
   DESI-238.5690+04.7276 and DESI-257.4348+31.9046 appear only as one-line rows in the
   51-system photometry table — no θ_E, no redshift, no morphology
   (`strong_lens_foundry_I/NOTES.md` §2). The single exception: DESI-238 is called a
   galaxy-scale lens "likely with two lensed sources, which are quadruply (fainter) and doubly
   (brighter) lensed" in the Fig. 5 caption. There is no published number to check our fits
   against. The template to follow is DESI-165.4754-06.0423, the paper's one worked system.

5. **Foundry I masks an ambiguous object rather than modelling it — and justifies that
   empirically.** They tested a mass model for the object embedded in arc A, found "negligible
   difference for the main lens parameters," and left it out, flagging its nature as unresolved
   (`strong_lens_foundry_I/NOTES.md` §5). That is the same question DESI234's interloper poses,
   answered by measurement rather than by argument. They also give a group member a *light-only*
   model with no mass component. No formal model selection — no Bayes factor, no BIC — is used
   anywhere in that paper.

6. **A prior-boundary sensitivity test is standard practice, and they published a null-ish
   result from one.** Foundry I re-ran with two alternative γ priors specifically to check
   whether γ was pinning against a boundary, reported that χ² improves when the boundary is
   raised, and did not claim to know why γ wants to be low
   (`strong_lens_foundry_I/NOTES.md` §5). This is the published form of the check
   `diagnose.prior_posterior_tension` automates.

7. **MCLMC's convergence-rate guarantee does not formally cover our posteriors.** Geometric
   ergodicity is proved only for globally m-convex, M-smooth targets with periodic boundary
   conditions (`mclmc/NOTES.md` §4). Plain ergodicity holds for any nonzero noise, but says
   nothing about how long it takes. The paper is silent on float32 vs float64, on
   preconditioning, and on any runtime bias diagnostic beyond the tuning-phase energy-error
   target. Note that gigalens goes beyond the paper here: `run_fit.py:656-658` seeds MCLMC's
   initial inverse mass matrix from the SVI surrogate, a preconditioning step the paper never
   describes. So the implementation is not the algorithm as published, and the paper's
   guarantees are about the published one.

8. **Our HMC configuration is far from the GIGA-Lens paper's.** Theirs: 50 chains, 250 burn-in,
   750 sampling, ε=0.3, L=5, mass matrix from the SVI covariance, dual-averaging active for the
   first 80% of burn-in only (`gigalens_paper/NOTES.md` §1). The stated *primary* purpose of the
   SVI stage is the mass matrix; chain initialization is secondary.

Generated 2026-08-10 from the five sources then present in `papers/`. Regenerate a note by
re-reading its `.tex`; the notes carry no numbers that are not in their source.
