[Guide index](INDEX.md) · previous: [04-running-the-fit.md](04-running-the-fit.md) · next: [06-reporting.md](06-reporting.md)

---

# 5. Diagnostics you should run every time, in this order

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

**2b. Prior–posterior tension, and a physical sanity read.** Pinning and tension are different
questions, and running only the first leaves the larger hole. **Pinning means the prior set the
value; tension means the data set it, against the prior's advice.** The second is the commoner
signature of model misspecification — one parameter absorbing an error made somewhere else —
and by construction the pinning test cannot see it, because it inspects only bounded priors.

DESI234's `r3_fg` is the case that proves it. Deflector ellipticity has an unbounded
`Normal(0, 0.15)` prior on `e1`, `e2`, so no check looked at it; it fitted `|e| = 0.749`,
an **axis ratio q = 0.143**, a 7:1 needle. Alongside it, `gamma = 2.951 ± 0.005` — near-singular,
3.95 prior sigma out, posterior 50× narrower than the prior. Every diagnostic that existed
passed or was misread: `chi2_nu = 1.082` was the best on the system, and R̂ = 1.92 was written up
as a sampling problem. The round's physical conclusions were published and later retracted in
full (`DESI234/RESULTS_v3.md`, correction banner).

`diagnose.prior_posterior_tension` measures the prior quantile of the posterior median as an
equivalent Gaussian sigma, which works for bounded, unbounded and skewed priors alike, and prints
the posterior/prior sd ratio beside it. **A narrow posterior far from its prior is the signature
to learn.**

Then read the mass model against what is physically typical, which no automated check can do for
you:

| quantity | typical, galaxy scale | worry above / below |
|---|---|---|
| axis ratio `q` | round; `q ≳ 0.6` | `q < 0.5` needs an argument |
| slope `gamma` | near-isothermal, 2.0–2.2 | `> 2.5` or `< 1.8` |
| external shear `|γ|` | low, few × 0.01 | `> 0.2` is implausible (`gigalens.jax.physicality`) |
| mass-carrying interlopers | **rare** | a detection needs an alternative ruled out |

Published reference points for the scale of these numbers, both from systems fitted with this
same pipeline: Foundry I's one fully modelled lens gives `theta_E = 2.6463 (+0.0017/-0.0016)"`
and `gamma = 1.372 (+0.023/-0.022)` (`papers/strong_lens_foundry_I/NOTES.md` §4); the GIGA-Lens
paper's 100-system recovery test finds a typical `gamma` posterior width of ±0.05, widening to
±0.1 for fold configurations (`papers/gigalens_paper/NOTES.md` §6, §7).

**That published `gamma = 1.37` sits outside the band in the table above, which is the point.**
It is a refereed measurement, and the authors discuss it as low against NFW expectations,
consistent with a handful of BELLS/SL2S/LSD systems at `gamma ~ 1.4`, and unresolved. A table of
typical values that would have flagged it is doing its job; a gate that would have rejected it
would have been wrong.

Keep these as prompts, not gates. `gigalens.jax.physicality` states its bands as *soft*
plausibility warnings for the same reason, and `pipeline/priors.py` surfaces them without
blocking: an atypical system is a thing that exists, and a hard cut would forbid the interesting
case along with the broken one. The value is that the atypical case has to be noticed and argued
for, rather than passing silently into a results table.

**3. Convergence.** R̂ and ESS per parameter, with the failing set named, not just counted.
Four things about R̂ that are not optional, all from Vehtari et al. 2021
(`papers/r-hat_convergence/NOTES.md`):

- **"R̂" names two different statistics, and the thresholds differ by roughly a square.** The
  modern one is the max of rank-normalized split-R̂ and rank-normalized *folded* split-R̂, gated
  at **1.01** (§4 there). TFP's `potential_scale_reduction` is the classic PSRF, which the
  Foundry paper's own footnote calls "approximately the square" of it — which is why that paper
  gates at 1.2 and means roughly our 1.1 (`papers/strong_lens_foundry_I/NOTES.md` §4). This
  repository is on the modern statistic: `diagnose_convergence` calls ArviZ `method="rank"`.
  **A threshold quoted without its definition is not a number.** Before comparing your R̂ to a
  published one, establish which statistic each is.
- **R̂ is only trustworthy above an ESS floor**: ≥50 per split half, ≥400 total, on ≥4 chains
  (§4 there). Below that, R̂ is uninformative *in both directions* — a passing value means
  nothing either. `diagnose.py` now checks all three and prints a pass/fail line per condition
  before the R̂ verdict it qualifies, and says so in the summary when R̂ passes but the floor
  does not — that combination is an *undetermined* result, not a converged one. The per-split-
  half condition is the binding one: split-R̂ halves every chain, so an aggregate 400
  concentrated in two good chains still fails.
- **Plain split-R̂ is blind to chains that differ in scale but not location.** That is precisely
  what folding is for, and it is why the recommended statistic is the max of the two rather than
  the bulk term alone (§3 there). A chain stuck in the middle of the distribution while the
  others explore it is invisible to the bulk term.
- **The deepest limit: R̂ compares chains to each other, never to the truth.** Vehtari's
  Appendix C exhibits four chains agreeing on a stationary distribution whose mean is off by
  more than 3 SD, with R̂ and ESS both clean. There is no threshold that catches this. It is the
  same failure `r3_fg` would have had if it had converged — a clean statistic on a 7:1-needle
  mass model — and the only defences against it are the ones in §5 step 2b and step 6: physical
  typicality, and a cross-check of the headline quantity against the picture.

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

Two things about χ²ᵥ specifically, both learned the hard way:

- **It is the diagnostic least able to fail informatively, so weight it last.** A single scalar
  near 1 is nearly uninformative about whether the parameters are measured. `r3_fg` above had
  the best χ²ᵥ on its system *and* a 7:1 needle for a deflector; it bought the fit with an
  unphysical mass distribution. Rank diagnostics by how specifically they can fail, and an
  aggregate goodness-of-fit sits at the bottom of that ordering.
- **The core pixels are excluded by convention when reporting it.** A pixel-scale residual
  pattern at the deflector core is an ordinary artifact of Sérsic deflector modelling, generally
  taken to be PSF-related. The field masks the centre when *computing* χ², and does **not** mask
  it when modelling — the mass model still needs those pixels for the centroid and inner slope.
  Every χ²ᵥ in this repository includes them and is therefore inflated relative to a published
  one. See `pipeline/KNOWN_ISSUES.md` item 5.

**Open the residual map.** It is rendered into every run directory as `residuals.png`, with
`lensing_overview.png` giving the source plane and critical curves beside it. Reasoning about
residual structure from summary statistics while those files sit unopened is a failure mode I
sustained for several rounds. The source-plane panel in particular catches a class of error no
statistic reports: a source component with an effective radius several times the caustic scale,
centred well outside the caustic, is not being lensed at all — it is a smooth pedestal buying
χ², and it should be justified or removed. `r3_fg`'s `src0` was exactly that: `R_sersic = 1.71"`,
centred 2.41" outside the caustic, `n_sersic` pinned at its floor.

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

## On residual maps and σ maps

A raw normalized-residual map is easy to over-read. In DESI238 several clumps showed 6σ+
peak pixels while matching to within 10% in *integrated* flux — small (~1 px) astrometric
offsets create large point-wise residuals. **Aperture photometry on an azimuthal-median-
subtracted image is the more reliable check**, and it is the one that revealed the N arc
was genuinely underfit 2× (ratio 0.55) rather than merely offset.

## Check for cannibalisation whenever you add a component

Adding a flexible light component near the Einstein radius creates the risk that it steals
arc flux and biases the mass model. Do not assume a χ² improvement is safe. When I added
the second deflector component I compared clump-flux ratios before and after — three of
four within 3% (1.03 / 0.99 / 1.02 / 0.83) — and confirmed θ_E moved only 1.484 → 1.480,
well inside the shift already seen between rounds. Then I could say "no cannibalisation."

---

[Guide index](INDEX.md) · previous: [04-running-the-fit.md](04-running-the-fit.md) · next: [06-reporting.md](06-reporting.md)
