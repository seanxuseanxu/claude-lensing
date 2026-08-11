# Reading note: Robnik & Seljak, "Fluctuation without dissipation: Microcanonical Langevin Monte Carlo" (AABI 2024)

Source: `papers/mclmc/main.tex`. All section/equation numbers below refer to this file.
Audience: someone running `blackjax`'s MCLMC on ~20-40 parameter strong-lens posteriors and
who needs to know when/why it can fail.

## 1. Core idea

Standard HMC/Langevin build a Markov process whose **phase-space** distribution converges to
the canonical ensemble $\rho(\Z) \propto e^{-H(\Z)}$ (Sec 1, ~line 105-106). To get the desired
marginal $\rho(\x)\propto e^{-S(\x)}$, damping and diffusion must be tied together by the
fluctuation-dissipation theorem.

MCLMC's move: only the **configuration-space marginal** needs to be canonical, not the full
phase-space density. This opens a wider family of valid ODE/SDEs than the one characterized by
Ma et al. 2015 (cited as `Ma15`, Eq. in Sec 1 lines 118-126), which assumed phase-space-canonical
equilibrium. MCLMC instead targets the **microcanonical ensemble**: energy is conserved, velocity
is confined to unit norm (isokinetic constraint), and the stationary density is
$\rho_\infty \propto e^{-S(\x)}\sqrt{g(\Th)}$ (Eq. "stationary distribution", ~line 229) — uniform
on the velocity sphere, exactly $e^{-S(\x)}$ marginalized over $\x$.

Because energy/speed is fixed rather than resampled from a Boltzmann distribution, there is no
damping term paired with the noise term — "fluctuation without dissipation." Key structural
result (Sec 3, "Discretization", Theorem 3, line 306-312): **the deterministic drift and the
stochastic diffusion each separately preserve $\rho_\infty$**, for *any* $\eta$ (noise strength).
This is unlike standard Langevin, where drift and diffusion only preserve the stationary
distribution jointly (line 314-316). Practical consequence claimed by the authors: in a
Euler-Maruyama discretization, **the only source of bias is discretizing the deterministic ODE**
(Eq. \eqref{eq: ODE}) — the stochastic/noise update is exact and unbiased at any step size,
because Brownian motion on the sphere can be sampled exactly (line 299) or replaced by any
discrete process with the uniform distribution as its stationary distribution (line 318, 457).

MCHMC vs MCLMC (the paper is explicit about this distinction, keep it straight):
- **MCHMC** (Robnik et al., cited as `MCHMC`) = the earlier, *discrete-time* algorithm: integrate
  the deterministic isokinetic ODE and periodically/randomly reorient the momentum direction
  ("bounce"). It is the discrete algorithm actually implemented and tuned in practice; MCLMC's
  tuning scheme (Sec on Applications, line 457-458) explicitly borrows MCHMC's ESS-based $\eta$
  tuning and MCHMC's energy-error step-size targeting.
- **MCLMC** = this paper's *continuous-time* SDE analog (Eq. \eqref{eq:SDE}, line 257-260), where
  the momentum reorientation becomes continuous stochastic diffusion on the sphere. The paper's
  ergodicity and geometric-ergodicity proofs (Sec 4-5) are proved for the continuous-time SDE, not
  directly for a discretized version at finite $\epsilon$.
- The practical/numerical results in Sec 6 (`phi4`, Brownian Motion, Item Response Theory) are run
  with a discretized algorithm that uses the Minimal Norm integrator for the deterministic part
  and a specific discrete angular-update formula for the noise (line 457) — i.e., this is
  effectively MCHMC machinery with the continuous-time justification MCLMC provides. blackjax's
  `mclmc` sampler is this discrete algorithm.

## 2. Update equations, integrator, and tunable parameters

**Continuous-time SDE** (Eq. \eqref{eq:SDE}, line 257-260):
$$d\x = \U\,dt, \qquad d\U = P(\U)\boldsymbol f(\x)\,dt + \eta\,P(\U)\,d\boldsymbol W$$
where $P(\U) = I - \U\U^T$ projects orthogonal to the current velocity (keeps $|\U|=1$), and
$\boldsymbol f(\x) \equiv -\nabla S(\x)/(d-1)$ is the **rescaled force** — note the $1/(d-1)$
factor, not the naive $-\nabla S$ (Sec 2, footnote at line 172-173). This rescaling is what makes
the sampler converge to $e^{-S}$ exactly without importance weights; the original MCHMC/ESH papers
used a $1/d$ factor and required reweighting. **This factor matters for a $d\approx 30$ lens
posterior**: verify blackjax uses the $(d-1)$ convention, not a stale $1/d$ one, if ever hand
implementing or debugging normalization.

**Practical discrete update** actually used in the numerical experiments (line 457-458), not a
literal Euler-Maruyama of the SDE:
- Deterministic step: Minimal Norm integrator (cited `MinimalNorm`, `MCHMC`) applied to Eq.
  \eqref{eq: ODE} — this is the leapfrog-like integrator for the isokinetic ODE, analogous to a
  higher-order symplectic integrator for HMC.
- Stochastic step: $\U_{(n+1)\epsilon} = (\U_{n\epsilon} + \nu\boldsymbol r)/|\U_{n\epsilon} +
  \nu\boldsymbol r|$, with $\boldsymbol r\sim\mathcal N(0,I)$ — a partial random reorientation of
  the velocity followed by renormalization, replacing the exact Brownian-motion-on-sphere update.
  Justified by the remark (line 318) that *any* discrete process on the sphere with the uniform
  distribution as its stationary law preserves $\rho_\infty$ exactly (no bias from this choice of
  discretization, by the same separate-preservation argument as Theorem 3).

**Tunable parameters, named exactly:**
1. **$\epsilon$ (step size)** — controls the deterministic ODE integration error. This is the sole
   source of discretization bias claimed by the theory (line 316). Tuned by targeting a
   **predefined energy-error variance per dimension** (line 458), same criterion as MCHMC.
2. **$\eta$ (continuous-time) / $\nu$ (discrete-time analog)** — strength of the stochastic
   perturbation to the velocity direction. "Introduces one additional free parameter... analogous
   to the bounce rate in MCHMC" (line 150). Controls how fast the velocity direction decorrelates
   — i.e., this is the paper's version of what other MCHMC/MCLMC writeups call the **decoherence
   length / momentum decoherence scale L**. Tuned by maximizing effective sample size (ESS) via a
   short prerun (line 150, 457), *not* by the energy-error criterion.
   Note: the paper does **not** use the symbol `L` at all — no explicit "decoherence length"
   parameter name appears in this text; $\eta$/$\nu$ is the single noise-strength knob. If
   blackjax's API exposes an `L` parameter, expect it to correspond to this $\eta$/$\nu$ role
   (or to a related trajectory-length quantity), not to a separate mechanism described here.
3. No other free parameters are named. Mass matrix / preconditioning is **not discussed anywhere**
   in this paper — the paper does not address how to handle strongly anisotropic or correlated
   targets via a mass matrix (see Sec 4 "Failure modes" below and Sec 6 of this note).

## 3. Tuning/adaptation scheme and diagnostics

The paper's own tuning description is brief (all of it is in the "Applications" preamble, line
457-459) and defers the actual mechanics to the earlier MCHMC paper (cited `MCHMC`) rather than
deriving them here:

- **$\eta$ (noise strength)**: "We tune the parameter $\eta$ by estimating the effective sample
  size (ESS)" (line 457), citing MCHMC for the method. No formula for the ESS estimator is given
  in *this* paper.
- **$\epsilon$ (step size)**: "tune the step size by targeting a predefined energy error variance
  per dimension, as in [MCHMC]" (line 458). The paper does **not** state the numeric target value
  for that energy-error-variance-per-dimension criterion in this document — it only names the
  criterion and cites MCHMC for it. If you need the actual target number, it is not recoverable
  from this file; check the MCHMC paper directly.
- Tuning cost is empirically reported for the $\phi^4$ experiment: "the tuning of the step size and
  $\eta$ is done at each $\lambda$ level separately and is included in the sampling cost. It
  amounts to around 10% of the sampling time" (line 459).
- For the Inference Gym experiments (Brownian Motion, Item Response Theory), no separate tuning
  cost breakdown is given; efficiency is measured purely in gradient evaluations to reach
  $b^2<0.01$ (line 557).
- The paper reports NUTS/HMC step size is tuned by dual averaging targeting an 0.8 acceptance rate
  (line 521) for comparison, but that is describing the **baseline**, not MCLMC.

**No diagnostic beyond ESS and the energy-error-variance target is described.** There is no
mention of $\hat R$, divergence counts, or trace-based convergence diagnostics anywhere in this
paper — it is a methods/theory paper with a benchmarking section, not a diagnostics paper.

## 4. Failure modes — the important section

This paper is almost entirely a positive-result/existence-proof paper (ergodicity, geometric
ergodicity, bias-free noise term). It is **not** a paper about failure modes, and it says
comparatively little about them. What it does say:

- **The predecessor deterministic dynamics is not ergodic in general.** "The deterministic
  dynamics of [isokinetic0,ESH] is not generically ergodic, even if additional variables are
  introduced... and even if it were, the convergence to equilibrium can be slow [MCHMC]" (line
  147). This is exactly why the stochastic term (MCLMC's contribution) is needed at all — pure
  deterministic isokinetic/MCHMC dynamics without noise can fail to explore the full space or can
  mix arbitrarily slowly. **Implication: a run with $\eta$ (or blackjax's noise/decoherence
  parameter) set to zero or too small is not guaranteed ergodic and could silently fail to
  explore multimodal or non-generic posteriors.**
- **Geometric ergodicity is proved only under strong regularity conditions** (Sec 5, line 384-388):
  target $S(\x)$ must be **$M$-smooth and $m$-convex**, i.e. $mI \prec \partial_{ij}S(\x) \prec MI$
  globally, *and* periodic boundary conditions are assumed at large $\x$ so that $\nabla S(\x)$ is
  globally bounded by $g_{\max}$. **A strong-lens posterior is not globally log-convex** (multiple
  local minima from image-position degeneracies, mass-sheet-like degeneracies, occasional
  multimodality from image misidentification) and gradients are not bounded on $\mathbb R^d$ (no
  periodic boundary condition) — so the paper's own convergence-rate guarantee (Theorem,
  "Geometric ergodicity", line 438-443) **does not formally apply** to our use case. The paper
  proves *ergodicity* unconditionally (Theorem, Sec 4, any nonzero $\eta$ — line 377-382), but
  *geometric* (exponentially fast) ergodicity is the conditional, stronger claim.
- **Ergodicity itself requires $\eta \neq 0$.** Stated in the abstract: "We prove that MCLMC is
  ergodic for any nonzero amount of stochasticity" (abstract, line 90) and Theorem in Sec 4 (line
  377-382, via Hörmander's condition, Lemma 1-3). $\eta=0$ recovers the deterministic dynamics
  which is explicitly flagged as generically non-ergodic (line 147) — **so $\eta$ must never be
  tuned to zero**, and if an ESS-based tuner drives $\eta\to0$ on a well-conditioned unimodal
  posterior, ergodicity for that run is no longer covered by the theorem.
- **Bias mechanism, precisely stated:** because drift and diffusion separately preserve
  $\rho_\infty$ (Theorem 3, line 306-312), "the integration scheme for the deterministic dynamics
  is the only bias source" (line 316). This means: **any residual bias in an MCLMC run should be
  attributable to step size $\epsilon$ being too large for the curvature of $S(\x)$ along the
  trajectory**, not to the noise term or to $\eta$. This is a specific, checkable claim: if you
  see bias, first suspect $\epsilon$/the deterministic integrator, not the momentum-decoherence
  parameter.
- The paper does **not** discuss heavy-tailed targets, funnel/hierarchical-model pathologies (even
  though it *runs* a hierarchical model — Item Response Theory — it reports only aggregate
  gradient-count efficiency, not failure diagnostics or funnel behavior), non-convex multimodal
  targets beyond the one line about non-ergodicity of the deterministic limit, or what happens
  when $M$-smoothness/$m$-convexity fail in practice (e.g., what "geometric ergodicity" degrades
  to, or how to detect non-geometric slow mixing at runtime).
- No statement anywhere about what a "wrong step size" *looks like* diagnostically (e.g., no
  discussion of energy-error blow-up thresholds, no discussion of divergent trajectories/NaNs).
  The energy-error-variance tuning target (Sec 6, line 458) is presented as a tuning objective,
  not as a runtime health check the paper tells you to monitor during sampling.

**Bottom line for us:** this paper establishes *that* MCLMC converges and is asymptotically
unbiased under conditions our posteriors likely violate (log-convexity, bounded gradient), and it
identifies $\epsilon$ (not $\eta$) as the only theoretical bias source when those conditions hold.
It gives essentially no guidance on diagnosing failure on curved/non-convex, non-periodic targets
like ours — that guidance must come from elsewhere (e.g. the MCHMC paper, or blackjax's own
tuning diagnostics), not from this document.

## 5. Numerical precision (float32/float64, energy-error accumulation)

**The paper does not address this.** There is no mention of float32 vs float64, floating-point
precision, or numerical energy-error accumulation over long trajectories anywhere in the text. The
only "energy error" discussion is the tuning target for $\epsilon$ (Sec 6, line 458), which is
about the *discretization* energy error (a modeling/algorithmic quantity), not about floating-point
round-off accumulation. If float32 vs float64 matters for a 20-40 dim lens fit with long MCLMC
trajectories, that has to be established independently (e.g. empirically, or from blackjax /
JAX-specific documentation), not from this paper.

## 6. Stated limits on dimension or class of target distributions

- **Dimension:** No explicit maximum or minimum dimension is stated as a limit of applicability.
  The paper's own force normalization derivation explicitly **breaks in $d=1$**: "In $d=1$ this
  reweighting does not work and the original Hamiltonian formulation should be used" (footnote,
  line 172-173) — i.e., the $(d-1)$-normalized force is undefined/degenerate at $d=1$ and MCLMC as
  formulated here requires $d\geq 2$. This is irrelevant for a 20-40 parameter lens fit but worth
  knowing as a hard edge case.
- Scaling claims are empirical, not theoretical guarantees: ESS/gradient for HMC/NUTS is observed
  to scale as $d^{-1/4}$ (citing `NealHandbook`, line 530), while MCLMC's ESS is reported as "almost
  independent of $\bar\lambda$ and $L$" in the $\phi^4$ experiments up to $d=64^2=4096$ (line
  536-538), with an extrapolated (not proven) 2-3 order-of-magnitude advantage claimed for
  $d=10^8$ lattice QCD (line 538). This is an empirical/heuristic scaling argument, not a proven
  dimension-independence result — the paper's actual theorems (ergodicity, geometric ergodicity)
  make no quantitative dimension-dependence claim.
- **Class of targets:** the *proved* results (ergodicity, Sec 4) require only that $S(\x)$ be
  smooth enough for the SDE/Hörmander machinery to apply (implicitly $C^\infty$ or at least
  several-times differentiable — not stated precisely). The *geometric*-ergodicity result (Sec 5)
  additionally requires global $m$-convexity, $M$-smoothness, and periodic boundary conditions (see
  Sec 4 of this note above) — the paper is explicit these are assumptions of that section only
  ("we will assume that the target $S(\x)$ is $M$-smooth and $m$-convex," line 387), not a general
  property of MCLMC. No statement is made about non-smooth targets (e.g. targets with kinks,
  truncated priors with hard boundaries), which is relevant if any lens-model priors are hard
  bounds rather than smooth reparameterizations.

## Open questions this paper does NOT answer

- What is the actual numeric target for the energy-error-variance-per-dimension step-size tuning
  criterion? (Only cited to MCHMC, not given here.)
- What does the ESS estimator used to tune $\eta$/$\nu$ actually compute, and how expensive/robust
  is it in $\gtrsim$30 dimensions with correlated parameters?
- How does MCLMC behave when $m$-convexity fails locally (saddle points, near-degenerate
  directions — exactly what strong-lens mass-sheet/source-position degeneracies produce)? No
  guidance, empirical or theoretical.
- Is there a mass-matrix / preconditioning analog for MCLMC to handle strongly anisotropic
  posteriors (very different parameter scales, e.g., Einstein radius in arcsec vs. flux ratios)?
  Not discussed at all.
- What diagnostic should be monitored *during* a production run to detect that $\epsilon$ is too
  large (i.e., detect bias in real time), beyond the tuning-phase energy-error target? Not given.
- How sensitive are the ergodicity/bias-free guarantees to the discrete stochastic update actually
  used (the normalize-after-perturbation formula, line 457) versus the idealized exact
  Brownian-motion-on-sphere update the theorems are proved for? The paper asserts equivalence in
  principle (line 318) but does not analyze the chosen discrete update's own convergence rate.
- Any guidance on multimodal targets (relevant for lens systems with image-identification
  ambiguity)? Not addressed — ergodicity is proven, but nothing about mixing time or bias under
  multimodality.
- float32 vs float64 recommendation, or any statement on numerical round-off effects on long
  trajectories? Not addressed at all (Sec 5 above).
