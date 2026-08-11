# Notes on Vehtari, Gelman, Simpson, Carpenter & Bürkner, "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC" (Bayesian Analysis)

Audience: an agent that quotes max R-hat as a pass/fail gate on strong-lens MCMC fits.
All section/equation refs are to `rhat.tex`.

## 1. Classic split-R-hat: definition and blind spots

**Setup** (§"Split-R-hat", the section labeled `\label{SplitRhat}`): N draws/chain, M chains
(chains already split in half, so M = 2 × number of sampled chains), S = MN total draws.

Between-chain variance:
B = N/(M-1) Σ_m (θ̄^(.m) − θ̄^(..))²

Within-chain variance:
W = (1/M) Σ_m s_m², s_m² = 1/(N-1) Σ_n (θ^(nm) − θ̄^(.m))²

Pooled variance estimate:
var-hat⁺(θ|y) = (N-1)/N · W + (1/N) · B

R-hat:
R-hat = sqrt( var-hat⁺(θ|y) / W )

This is **split**-R-hat (following BDA3): applied to chains split in half so non-stationary
drift within a chain still shows up as between-"chain" variance. Text explicitly: "Without
splitting, R-hat would get fooled by non-stationary chains as in Figure 1b."

**Failure modes the paper states explicitly** (§"Example where traditional R-hat fails",
and the itemized list right after Figure 2, around line 300-305):

1. **Chains with different variance/scale but the same location (mean).** "If the chains
   have different variances but the same mean parameters, traditional R-hat ≈ 1." Demonstrated
   with one AR(1) chain manually set to 1/3 the variance of the other three — R-hat stays
   well under 1.1 in 1000 replications (Fig. 2 top-left).
2. **Infinite-variance (heavy-tailed) targets.** "If the chains have infinite variance,
   traditional R-hat ≈ 1 even if one of the chains has a different location parameter to the
   others." Demonstrated with Cauchy-distributed chains, one shifted 2 units — R-hat
   incorrectly returns ≈1, because R-hat "is defined based on second-moment statistics" and
   the infinite variance destroys its effectiveness. The paper adds this "can also lead to
   numerical instability for thick-tailed distributions even when the variance is technically
   finite," and that it is "typically hard to assess empirically if a chain has large but
   finite variance or infinite variance."
3. **Only diagnoses the mean, says nothing about the tails.** "A related problem is that
   R-hat is typically computed only for the posterior mean. While this provides an estimate
   for the convergence in the bulk of the distribution, it says little about the convergence
   in the tails" — relevant for posterior intervals and rare-event inference.
4. (Appendix A, "Scaling one chain") Directly confirms #1 numerically: split-R-hat "is not
   able to detect scale differences between chains" even down to one chain having std dev
   1/4 of the others (at threshold 1.1) — folded-split-R-hat is needed to catch this (see §3).
5. (Appendix C, thinned centered eight-schools) A distinct failure mode: **all chains
   converge together to the same *wrong* stationary distribution** (all four chains stick
   similarly in the narrow part of a funnel). Split-R-hat and ESS "miss the problems" here —
   posterior mean off by >3 SD — because R-hat only compares chains to each other, not to the
   true target. Thinning removed the autocorrelation information that would have revealed
   "sticking." This is a fundamental limit, not fixed by rank-normalization or folding.

## 2. Rank-normalization (§"Rank normalization helps R-hat when there are heavy tails")

**Motivation:** split-R-hat and S_eff are "well defined only if the marginal posteriors have
finite mean and variance." Rank-normalization removes that dependency.

**Transformation** (right after "Rank normalization proceeds as follows"):
1. Replace each draw θ^(nm) by its rank r^(nm) within the **pooled** draws from all chains
   (average ranks used for ties, to conserve the number of unique values of discrete
   quantities).
2. Transform ranks to normal scores (z-scores) via the inverse normal CDF with a Blom (1958)
   fractional offset:
   z^(nm) = Φ⁻¹( (r^(nm) − 3/8) / (S + 1/4) )    [Eq. numbered, unlabeled, ~line 824]

Rank-normalized split-R-hat and S_eff are then computed by the *same* formulas as §1/ESS
section, substituting z^(nm) for θ^(nm).

**Why (three stated benefits, verbatim reasons):**
(1) for continuous variables the normality assumptions used in computing R-hat and S_eff
"are fulfilled (via the transformation)";
(2) values are "practically the same as before for nearly normally distributed variables"
— so it doesn't change the diagnostic where the original R-hat already worked;
(3) rank-normalized R-hat and S_eff "are invariant to monotone transformations (e.g. we get
the same diagnostic values when examining a variable or logarithm of a variable)."

**What it fixes:** it makes R-hat well-defined even when the marginal has infinite mean or
variance (the Cauchy failure mode in §1, item 2), because the statistic now operates on
normal scores, not raw moments.

**What it does NOT fix by itself:** rank-normalization alone still compares only *location*
information after normalization — it does not address the same-location/different-scale
blind spot (§1, item 1). That requires folding (§3). Also stated: "due to the rank
normalization, bulk-ESS is no longer directly applicable to estimate the Monte Carlo standard
error of the posterior mean" — MCSE for the actual mean needs separate treatment (§6/mcse
section covers quantile MCSE instead).

Named "bulk effective sample size" (bulk-ESS / bulk-S_eff): ESS computed on the
rank-normalized draws. "Useful for diagnosing problems due to trends or different locations
of the chains," well-defined even for infinite-mean/variance distributions.

## 3. Folding (§"Folding reveals problems with variance and tail exploration")

**Stated gap:** "Both original and rank normalized split-R-hat can be fooled if the chains
have the same location but different scales. This can happen if one or more chains is stuck
near the middle of the distribution."

**Transformation** (Eq. `\zeta`, ~line 862):
ζ^(mn) = |θ^(nm) − median(θ)|
i.e. absolute deviation of each draw from the (pooled) median. Rank-normalized split-R-hat
computed on the ζ values is called **folded-split-R-hat**. Text: "This measures convergence
in the tails rather than in the bulk of the distribution."

**Recommended statistic:** report the max of rank-normalized split-R-hat and rank-normalized
folded-split-R-hat, for each parameter — "To obtain a single conservative R-hat estimate, we
propose to report the maximum of rank normalized split-R-hat and rank normalized
folded-split-R-hat."

**Empirical confirmation, Appendix A "Scaling one chain":** with one chain's SD scaled to
0.75× the others (4 chains, 250 draws/chain), plain split-R-hat is *not* able to detect the
scale difference at all across the tested range; folded-split-R-hat does detect it, with
folded-split-R-hat = 1.06 for the 0.75× case, and quantified thresholds: at cutoff 1.01,
folded-split-R-hat detects a chain with scale < 3/4 of the others' SD; at cutoff 1.1, it
detects scale < 1/4 of the others' SD. The paper notes even at folded-split-R-hat = 1.06 the
numeric value alone could be missed as "fine" under a 1.1 threshold — rank plots (§8/diag.
visualizations) "clearly show that the first chain behaves differently," i.e. the number
alone is not sufficient even when it does flag something.

So: max(rank-normalized split-R-hat, rank-normalized folded-split-R-hat) is the paper's
actual recommended diagnostic — plain (even rank-normalized) split-R-hat is knowingly
insensitive to pure-scale disagreement between chains.

## 4. Recommended threshold and its justification (§"Recommendations for practice")

**Threshold: only use the sample if R-hat < 1.01** — computed on rank-normalized, folded
statistics (max of the two, per §3). Stated directly: "we propose modifications to R-hat
based on rank-normalizing and folding the posterior draws, only using the sample if
R-hat < 1.01."

**Why not 1.1 (Gelman & Rubin 1992's number):**
- "This threshold is much tighter than the one recommended by Gelman+Rubin:1992, reflecting
  lessons learnt over more than 25 years of use, as well as the simulation results in
  Appendix A."
- Gelman & Rubin (1992) derived R-hat assuming within-chain variance rises monotonically
  while between-chain variance falls monotonically toward a stable ratio of 1; a factor of
  1.1 meant little more precision was to be gained from longer runs.
- But (citing Brooks & Gelman 1998) MCMC dynamics don't guarantee that: "the between-chain
  variance can decrease before it increases, if the initial part of the simulation pulls all
  the chains to the center of the distribution, only for them to be redispersed with further
  simulation." Consequence: "R-hat cannot in general be interpreted as a potential scale
  reduction factor," and empirically (also noted by Vats & Knudson 2018) "R-hat can dip below
  1.1 well before convergence in some examples" — the paper reports this is "much more rare"
  under the 1.01 threshold.
- Appendix A quantifies the 1.01 vs 1.1 gap directly for two failure modes: for a trend
  common to all chains, threshold 1.01 detects trends accounting for ≥2% of total marginal
  variance, vs ≥30% for threshold 1.1; for one chain shifted in mean, 1.01 detects shifts
  ≥1/3 of the marginal SD, vs shifts ≥1 full marginal SD for threshold 1.1. I.e., 1.1 misses
  much larger, more consequential departures than 1.01 does.

**Implication for chains/draws — explicit minimum-ESS coupling to R-hat's validity:**
"We only recommend relying on the R-hat estimate to make decisions about the quality of the
chain if each of the split chains has an average ESS estimate of at least 50. In our minimum
recommended setup of four parallel chains, the total ESS should be at least 400 before we
expect R-hat to be useful." So R-hat < 1.01 is not sufficient on its own — it must be paired
with ESS ≥ 400 total (≥ 50 per split-chain-half) to be trusted at all; values of R-hat and ESS
"require reliable estimates of variances and autocorrelations," which in turn require "enough
independent replicates."

**Minimum number of chains:** "we recommend running at least four chains by default" —
justified because multiple chains "are more likely to reveal multimodality and poor
adaptation or mixing," and because chains parallelize with "no increase in computation time."
The paper explicitly does not address massive parallelization (~1000+ chains): "further
research is needed" there.

## 5. Localization: quantile-R-hat and local efficiency (§"Localizing convergence
diagnostics: assessing the quality of quantiles, the median absolute deviation, and
small-interval probabilities")

**Motivating fact:** convergence "is not uniform across the parameter space" — bulk (mean,
median) can converge while tails (extreme quantiles) have not. Standard R-hat/bulk-ESS say
nothing about this by construction (they operate on the rank-normalized full sample, not a
specific quantile).

**Quantile efficiency via indicator functions:** any cumulative probability Pr(θ ≤ θ_α) is
written as an expectation of an indicator function I(θ ≤ θ_α), estimated by the sample mean
Ī_α = (1/S) Σ_s I(θ^(s) ≤ θ_α). Rank-normalizing and computing R-hat/ESS *on the indicator
function's results* gives an efficiency estimate at any specific θ_α — this is how
quantile-specific R-hat/ESS is constructed; there is no separate closed-form "quantile-R-hat"
equation distinct from applying the same machinery to I(·).

**Tail-ESS:** defined as min(ESS at the 5% quantile, ESS at the 95% quantile) — "we propose
to compute the minimum of the effective sample sizes of the 5% and 95% quantiles, which we
will call tail effective sample size." Useful for diagnosing "problems due to different
scales of the chains."

**Median / MAD efficiency:** median = 50% quantile, so its efficiency follows the same
indicator-function recipe. MAD (median absolute deviation) efficiency is obtained the same
way but applied to the folded values ζ (Eq. `\zeta`, §3): Pr(ζ ≤ ζ_0.5) ≈ (1/S) Σ I(ζ^(s) ≤
ζ_0.5).

**Small-interval / local efficiency measures:** for a chosen resolution δ = 1/k, compute
efficiency of Ī_{α,δ} = Pr(Q̂_α < θ ≤ Q̂_{α+δ}) for α stepping through (0, δ, ..., 1−δ). Each
bin holds S/k draws; this is described as "more localized than efficiency measure for
quantiles," useful for building intuition about *which* posterior functionals are hard to
estimate (bulk vs. tail) and for practical equivalence testing. The paper also notes a
possible multivariate extension (small probability boxes/spheres, k-means-defined
hyper-spheres in high dimensions) but does not develop it beyond a suggestion.

**MCSE for quantiles (§mcse):** given S_eff for the indicator function at α, an MCSE interval
for θ̂_α is obtained via quantiles of a Beta(S_eff·α+1, S_eff·(1−α)+1) distribution,
propagated through the empirical order statistics (steps enumerated ~lines 1008-1043). This
gives extra, quantile-specific uncertainty information that a single global R-hat/ESS number
cannot.

**Diagnostic plots enabled by localization** (§"Diagnostic visualizations"): rank plots
(histograms of pooled ranks per chain — should look uniform if mixed), quantile/small-interval
efficiency plots (efficiency vs. quantile position), and efficiency-vs-iteration-count plots
(ESS should grow ~linearly with more draws; sublinear growth or decrease signals a problem
that more draws alone won't fix).

## 6. Bulk-ESS vs tail-ESS: definitions, minimum, and R-hat link

**Bulk-ESS (bulk-S_eff):** ESS computed on rank-normalized draws (§2). Diagnoses trend/
location problems; well-defined for infinite-mean/variance distributions.

**Tail-ESS:** min(ESS(5% quantile), ESS(95% quantile)) (§5). Diagnoses scale/tail-exploration
problems specifically.

**Underlying ESS formula** (§"The effective sample size"), for a single chain:
N_eff = N / (1 + 2 Σ_{t≥1} ρ_t)   (autocorrelation sum)
Multi-chain combined autocorrelation at lag t:
ρ̂_t = 1 − ( W − (1/M)Σ_m s_m² ρ̂_{t,m} ) / var-hat⁺   (Eq. `rhohat`)
with stated limits: if ρ̂_{t,m}=0 ∀m, ρ̂_t = 1 − R-hat⁻²; if chains mix well (R-hat≈1) with
nonzero within-chain autocorrelation, ρ̂_t ≈ average of the ρ̂_{t,m}; if R-hat ≫ 1,
ρ̂_t ≈ 1 − R-hat⁻² — i.e., **poor mixing (high R-hat) is baked directly into a lower ESS
estimate**, this is the paper's explicit R-hat/ESS link, beyond the well-known general
citation to Vats & Knudson (2018) for single-chain R-hat/ESS relations.
Truncated sum via Geyer's (1992) initial monotone sequence estimator gives
S_eff = NM / τ̂, τ̂ = 1 + 2 Σ_{t=1}^{2k+1} ρ̂_t (paired/monotone-reduced).

**Recommended minimum:** rank-normalized ESS > 400 (both bulk and tail), "a number we chose
based on practical experience and simulations... as typically sufficient to get a stable
estimate of the Monte Carlo standard error." With the minimum-recommended 4 chains, this is
100 per chain on average, but the paper's harder constraint is stated per split-half: "each
of the split chains has an average ESS estimate of at least 50" (i.e. per one of the M=2×(#
chains) split halves) before R-hat itself is trustworthy (§4/recommendations, restated in §
"The effective sample size" closing paragraph).

**Multimodality diagnosis via ESS:** "If the chains are not mixing at all (e.g., if the
posterior is multimodal and the chains are stuck in different modes), then our S_eff is close
to the number of distinct modes that are found" — stated as a usable diagnostic in its own
right, not just a warning sign.

**Super-efficiency:** S_eff > S is possible for antithetic samplers (e.g., dynamic HMC in
Stan) on near-Gaussian, low-dependence parameters; the paper caps the ESS estimate at
S·log10(S) in their software to guard against spurious large values from short antithetic
chains.

## 7. Guidance when R-hat fails: diagnosis, not just detection

The paper gives no separate prescriptive checklist ("do X, then Y") for what to do after a
bad R-hat; instead it works this by worked example (§"Examples") and licenses the following
concrete diagnostic moves, each tied to a specific tool defined above:

- **Local efficiency plots** (small-interval, §5) to locate *where* in the distribution
  (which quantile region) sampling is inefficient — e.g., in the Cauchy example, efficiency
  "is low in the tails, which is clearly caused by slow mixing in long tails."
- **Quantile-efficiency plots** (§5) to see if particular posterior intervals are
  untrustworthy even if bulk statistics look fine.
- **Rank plots** (§8/diagnostic visualizations) as the recommended *replacement* for trace
  plots: "If rank plots of all chains look similar, this indicates good mixing... As compared
  to trace plots, rank plots don't tend to squeeze to a fuzzy mess when used with long
  chains." Demonstrated to expose problems even when a numeric R-hat looks passable, e.g. the
  scaled-chain case where folded-split-R-hat = 1.06 (would pass a 1.1 gate) but rank plots
  "clearly show that the first chain behaves differently," and the shifted-chain case at
  split-R-hat = 1.05 similarly.
- **Efficiency-vs-iteration-count plots** (§5/§8) to check whether ESS grows ~linearly with
  more draws; if it grows sublinearly or decreases, "this indicates that simply running
  longer chains will likely not solve the convergence issues" — i.e. distinguishes "just run
  it longer" from "reparameterize/rethink the model."
- **Reparameterization as a fix, demonstrated not merely suggested:** the Cauchy
  nominal-vs-scale-mixture parameterization and the eight-schools centered-vs-non-centered
  parameterization examples show R-hat/ESS failing under one parameterization and passing
  under an equivalent reparameterization of the same model. E.g., centered eight schools:
  bulk-ESS(τ)=67, tail-ESS(τ)=82 (both ≪ 400) with divergent transitions concentrated at small
  τ; non-centered: all R-hat < 1.01, all ESS > 400, no divergences.
- **Cross-check against sampler-specific diagnostics when available:** for HMC, "divergent
  transitions" and "iterations exceeding max treedepth" are used alongside R-hat/ESS as
  corroborating evidence of poor mixing (noted as an advantage of R-hat/ESS that they, unlike
  divergences, work "with all MCMC algorithms not only with HMC").
- **Watch for thinning discarding diagnostic information** (Appendix C): thinning can make
  R-hat/ESS look fine while the chains have jointly converged to a *biased* stationary
  distribution (posterior mean off by >3 SD) because "sticking" behavior visible in
  unthinned autocorrelations is destroyed by thinning. The paper's practical recommendation:
  don't thin — "we don't recommend this approach, as it throws away useful information...
  and the dependent simulation draws are not a problem for estimating the Monte Carlo
  error." If you must thin (memory constraints), be aware R-hat/ESS become less reliable as
  convergence checks in that regime; rank plots on the unthinned run were still able to show
  the problem in their example.
- The paper is explicit that no diagnostic is complete: "We do not intend with this example
  to claim that our new R-hat is perfect — of course, it can be defeated too."

## 8. What the paper licenses you to say about max R-hat = 1.92 vs 13.94

**Preliminary caveat the paper requires before either number means anything:** confirm which
statistic is being reported. The paper's own recommended "R-hat" is the max of rank-normalized
split-R-hat and rank-normalized folded-split-R-hat (§3), gated at < 1.01 (§4), and is only
"useful" once each split-chain half has average ESS ≥ 50 (total ESS ≥ 400 with the
recommended 4 chains) (§4). If the reported number is classic (non-rank-normalized,
non-folded) split-R-hat, it inherits the specific blind spots of §1 and cannot be interpreted
against the paper's 1.01 threshold with the same confidence — the threshold and its
sensitivity numbers (Appendix A) were derived using the *rank-normalized* statistic.

**R-hat = 1.92:** far above both 1.01 and the old, already-superseded 1.1 rule of thumb. Per
Appendix A's sensitivity figures (§4), 1.1 alone already flags shifts ≥ 1 full marginal SD or
trends accounting for ≥30% of marginal variance — 1.92 is well past even that coarse
threshold. The paper licenses saying: this indicates a serious, easily-detectable mixing
failure (chains disagree substantially in location and/or have a strong shared trend); it
does **not** by itself tell you whether the cause is multimodality, a trend, non-stationarity,
or something else — that requires the diagnostic follow-up in §7 (rank plots, quantile/local
efficiency plots, ESS-vs-iteration plots, or reparameterization).

**R-hat = 13.94:** an extreme value not directly discussed by any of the paper's numerical
examples (which range up to ~1.1–1.3 in the illustrative failure cases). The paper does not
give worked guidance for values this large specifically. What can be said from stated
principles: it is unambiguously far outside "R-hat declines to 1 as N→∞ for an ergodic
process" (§1 definition) and indicates the chains are not sampling anything resembling a
common distribution — plausibly severe multimodality with chains stuck in different, well-
separated modes (the paper's own multimodality diagnostic, §6, notes S_eff in that regime
approaches the number of distinct modes found — worth checking bulk-ESS/tail-ESS alongside
this R-hat, since a very low ESS close to the chain count would corroborate mode-stuck
chains) or a comparably severe non-stationarity/trend problem. Beyond that, the paper is
silent on what specifically a value this large implies; do not extrapolate its Appendix A
sensitivity curves (fit for R-hat in the ~1.0–1.3 range) out to 13.94.

**In both cases**, the paper's stance is procedural, not just numerical: neither value alone
tells you *why* the chains failed to mix. The paper's answer to "what to say" is "run the
follow-up diagnostics" (§7) before drawing conclusions about the model or reparameterizing.
