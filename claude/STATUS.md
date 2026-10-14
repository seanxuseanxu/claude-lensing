# Status of the three modelled systems

**As of 2026-10-05.** This is a snapshot, taken from each system's README banner and its
latest `RESULTS_vN.md`. Where it disagrees with those files, they win. The most recent fit is
DESI234's round 4 (`r4_fg_q`, 2026-10-05); before that, nothing had run since 2026-08-11.

## DESI238: converged, the most secure of the three

**Round 5** (`DESI238/RESULTS_v5.md`, 2026-08-08), run `runs/mclmc_long`:

- **θ_E = 1.478 ± 0.003″**; γ = 1.800 ± 0.021
- All 36 parameters at R̂ ≤ 1.003; bulk-ESS about 79,000; χ²_ν = 0.968
- This is the first fit under the scene API. It reproduces round 4's pre-port 1.480″ to
  within 0.002″, even though the likelihood normalisation, the parameter ordering, the
  sampler and the code all changed.

Caveats (`RESULTS_v5.md` §8–9):

- The error is statistical only.
- The four-image interpretation is an input, not something the fit tested.
- The N arc is underfit by about 16%.
- A striped residual remains at the deflector core.
- The noise is understated near the core, where only part of the exposure stack covers it
  (coverage drops to about half depth).
- The headline run was not repeated with a second seed.

## DESI257: converged, but quote only 1.99″

**Round 3** (`DESI257/RESULTS_v3.md`, run `r3_unmasked`, fitted 2026-08-09):

- **θ_E = 1.991 [1.989, 1.994]″ (statistical)**; γ = 1.986 ± 0.021;
  |shear| = 0.044 ± 0.004
- R̂ ≤ 1.001 on all 36 parameters and all 50 chains, with no chains dropped; minimum
  bulk-ESS 70,858
- χ²_ν = 1.065 with the core masked (1.154 over all pixels)
- Unmasking the SW object removed round 2's bimodality, so round 2 is retracted. The
  repository owner independently confirmed on 2026-08-10 that the SW object is a
  counter-image.

The largest caveat is the pixel grid. The fit renders at `supersample = 2`, which misses the
library's quadrature tolerance by a factor of 18 (`analysis/undersampling_report.txt`). That
costs θ_E a systematic of **about 0.01″** (`analysis/theta_e_systematic.txt`), roughly five
times the statistical half-width. **Quote θ_E = 1.99″; the third decimal cannot be
defended.** Two smaller caveats:

- The MAP finishes 18.1 nats below the best HMC draw.
- The PSF has not been rebuilt, and the core fits worst (χ²_ν = 2.16 there).

## DESI234: no converged result on the full model, through four rounds

**Round 3 failed** (`DESI234/RESULTS_v3.md`, revised 2026-08-10):

- **`r3_fg`** (46 parameters, with a foreground mass) is **rejected as unphysical**. The
  deflector came out with axis ratio q = 0.143 and γ = 2.951, about 4 prior σ out. It also
  did not converge (max R̂ 1.92). The earlier claim that the interloper carries mass and
  moves θ_E by 0.066″ is retracted.
- **`r3_unmasked`** (36 parameters) has frozen chains (max R̂ 13.94), but its mass model is
  sane (q = 0.943, γ = 2.130) and it lands at θ_E = 1.5386″.

**The number that stands is round 2's θ_E = 1.539 +0.003/−0.002″** (`r2_s2l2`,
R̂ ≤ 1.008). It was fitted with the interloper masked, so the interloper's mass is not
modelled; that systematic has not been quantified. `r3_unmasked` agrees with it to 0.0007″.

**Round 4 ran on 2026-10-05 and did not converge** (`DESI234/RESULTS_v4.md`, run
`r4_fg_q`, 46 parameters at `mass_prior_version=2`):

- 46/46 parameters above the 1.01 gate, worst R̂ 2.125, and the Vehtari **ESS floor failed**
  (min bulk-ESS 63 against 400 total / 50 per split half). R̂ is uninterpretable either way.
- Bounding the host ellipticity at gigalens' soft `q ≥ 0.2` floor moved the deflector from
  q = 0.143 to **q = 0.295 and pinned it against the new bound** — 96.3% of draws within
  0.01 of it, and 96.0% among the top 10% by log posterior, so this is not an artifact of
  the fit's poor start. The likelihood prefers the excluded region.
- **Round 4's question is therefore unanswered.** The fitted foreground SIS
  θ_E = 0.4752″ (R̂ 1.609) must not be quoted.
- The ESS-vs-draws slope is 0.08 — sublinear — after 5× round 3's burnin, so chain length is
  not the lever.

**Round 4b (`r4_fg_q2`) ruled out the surrogate too.** Tripling MAP and SVI closed the
MAP-vs-HMC deficit from 193.75 to 69.60 nats and gave no convergence improvement: worst R̂
2.125 → 2.330, min ESS 63 → 56, slope 0.08 → 0.04. The deflector pinned at the same place
(q = 0.302 against 0.295, γ = 2.917 against 2.898), so two surrogates 124 nats apart agree
on the same unphysical mass model. Of the guide's ordered remedies, only "improve the model"
is left; `RESULTS_v4.md` §4 names the fit that would test the host–SIS degeneracy.

Still open and untouched by round 4: float64 for `r3_unmasked`'s frozen chains
(`pipeline/data.py:118` pins float32).

## Open problems found while compiling this

1. ~~**DESI234's README contradicts itself.**~~ **Fixed 2026-10-03.** The results table, the
   file head, and the round-2 checklist entry all still carried the pre-2026-08-10 reading —
   round 2 "superseded", the retracted 0.066″ shift, "no θ_E is currently quotable". So did
   the banner at the top of `RESULTS_v2.md`, which stated the retracted claim as current.
   All four now agree with the README banner: round 2's θ_E stands, qualified by the
   unmodelled interloper mass.
2. **DESI238's grid accuracy has never been checked.** `DESI238/config.py:209` renders at
   `supersample = 2`, the setting that failed by a factor of 18 on DESI257. No grid-accuracy
   (quadrature) check turned up in DESI238's markdown or config. Until one is run, its
   ±0.003″ may carry a systematic like DESI257's.
