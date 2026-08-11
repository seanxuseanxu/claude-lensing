# GIGA-Lens paper notes (arXiv:2202.07663)

Source: `ms_SL-model.tex` + `modeling.tex`, `results.tex`, `conclusion.tex`, `appendix.tex`, `abstr.tex`, `figures/summary_compact.tex`. All numbers below are the paper's own, for the "reference system" (§3, Fig. ref-sys) and the 100-system simulation test (§4/results.tex), on 4x A100 GPUs unless noted.

## 1. Three-stage inference scheme (§3.4 "Pipeline Summary", Table `tab:pipeline` in `figures/summary_compact.tex`; per-stage detail in §3.2–3.4 i.e. `sec:map`, `sec:vi`, `sec:hmc`)

### Stage 1 — MAP (§3.2, `sec:map`)
- Optimizes: negative unnormalized log posterior `f(Θ̃) = -(log p̃(I_obs|Θ̃) + log p̃(Θ̃))` over the unconstrained (bijected) parameter space (Eq. `obj-fun`).
- Method: multi-start gradient descent (not lenstronomy's PSO — paper argues PSO has "little to no guarantees" of finding even a local optimum, citing Sengupta et al. 2018).
- Particles: `n_MAP = 300`, each initialized by sampling from the prior: `Θ̃_i^(1) ~ p̃(Θ̃)`.
- Iterations: `K_MAP = 300` gradient-descent steps. The best of the 300 samples after 300 iterations is taken as `Θ̃*_MAP`.
- Optimizer: Adam. Learning-rate schedule: linear decay from `α=10⁻²` to `α=10⁻³` over the 300 iterations.
- Stopping criterion: fixed iteration count (300), not a convergence test. Paper notes ~200 iterations is "rapid improvement," then approximately geometric convergence; a supplementary 800-iteration run (Fig. `ref-loss`) is shown only "for demonstration," confirming 300 already suffices — at 300, "the best performing trajectory always ends up at the global minimum."
- Success rate stated: "typically ~5% (and at least 1%, for systems with a low signal-to-noise ratio)" of the 300 starts converge to the global optimum.
- Runtime: 17 sec (4x A100) for the reference system; total wall time incl. overhead listed as 50 min in the table's rightmost column (this is likely compile/setup-dominated — the paper doesn't explain the 17s vs 50min discrepancy explicitly; report both as given).

### Stage 2 — VI (§3.3, `sec:vi`)
- Fits a multivariate normal surrogate `q̃(Θ̃; μ̃, Σ̃)` to the true posterior in unconstrained space by minimizing KL divergence via the ELBO (Eqs. `vi`, `elbo`, `elbo-grad`).
- Covariance is parameterized via a Cholesky bijector (unconstrained vector → lower-triangular L with exponentiated diagonal → Σ=LLᵀ), keeping Σ optimized in an unconstrained space too.
- Particles/samples per gradient step: `n_VI = 500` (Monte Carlo samples used to estimate the ELBO gradient, Eq. `elbo-grad` — this is "stochastic VI").
- Iterations: `K_VI = 1000`.
- Initialization: `μ̃^(1) = Θ̃*_MAP` (the MAP estimate); `Σ̃^(1) = 10⁻⁶ I` (deliberately an underestimate — "easier for VI to approximate the true covariance starting from an underestimate").
- Learning-rate schedule: quadratic increase from `α=0` to `α=10⁻³` over 500 iterations (then presumably held, though the text doesn't state the schedule for iterations 500–1000 explicitly beyond "increased quadratically ... over 500 iterations"). Starting at 0 is deliberate, to let Adam's moment estimates adjust and to avoid instability from the crude Σ̃^(1).
- Stopping criterion: fixed iteration count (1000), not a convergence test.
- Runtime: 52 sec (4x A100).
- Framework detail: TensorFlow implementation uses TFP's VI methods; JAX implementation computes Eq. `elbo-grad` directly.

### Stage 3 — HMC (§3.4, `sec:hmc-result`)
- Samples the posterior `p̃(Θ̃|I_obs)` in unconstrained space, then maps back to physical parameters via the bijector g.
- Chains: `n_HMC = 50`, each initialized by sampling from the VI surrogate posterior `N(μ̃*_VI, Σ̃*_VI)`.
- `n_burn = 250` burn-in steps, then `n_sample = 750` sampling steps per chain.
- HMC hyperparameters (Table `tab:pipeline`): step size `ε = 0.3`, leapfrog steps `L = 5`, mass matrix `M̃ = (Σ̃*_VI)⁻¹` — i.e., "preconditioned" HMC, the VI covariance inverse sets the mass matrix. This is stated as the primary purpose of the VI step (see item 4 below).
- Step-size/trajectory-length autotuning: dual-averaging adaptation (Hoffman & Gelman 2014; Hoffman et al. 2021) as implemented in TFP, run during the first 80% of the burn-in phase only, then fixed for the remainder — "their adaptation generally prevents chains from reaching the stationary distribution." Target acceptance probability = 0.75 (paper cites Betancourt 2018's recommended range 0.6–0.8).
- Stopping criterion: fixed step counts (250 burn-in + 750 sampling), not an active convergence check; paper reports diagnostics post hoc.
- Runtime: 36 sec (4x A100) for the reference system.
- Reference-system result: ESS > 26000 and R̂ < 1.01 for every one of the 22 parameters, achieved in "just under 36 seconds."

### Total pipeline
- Table `tab:pipeline` total: **105 sec on 4x A100 GPUs**, vs. **6 min on a single A100** (~3.5x slower single-GPU; source of the 3.5x factor is the emcee-vs-HMC comparison in §3.4.2). Per-stage: MAP 17s, VI 52s, HMC 36s (sums to 105s).
- Correction to the raw `.tex`: the `deluxetable` column spec (`figures/summary_compact.tex`) is `l m{3.1cm} r l m{2.8cm} c h` — the last column type `h` is a **hidden** (non-printing) deluxetable column. The "50 min." / "70 min." / (blank) values sit in that hidden column and do NOT appear in the compiled PDF; they look like leftover/draft numbers. Only "17 sec.", "52 sec.", "36 sec." (the visible `c` column) and the "105 sec." total are the paper's real reported runtimes. Don't cite 50min/70min as pipeline runtime.

## 2. Likelihood, noise model, variance map (§2.2 `sec:prob-model`, Eq. `model-def`)

Exact form (Eq. `model-def`):
```
log L(Θ; I_obs) = -1/2 Σ_{x,y} [ (I_obs(x,y) - I_model(x,y;Θ))² / σ_tot²(x,y;Θ)  +  log(2π σ_tot²(x,y;Θ)) ]
σ_tot²(x,y;Θ) = σ_bkg²  +  I_model(x,y;Θ) / (G · t_exp)
```
- Gaussian per-pixel likelihood; sum over all pixels, assumes **independent** per-pixel noise ("The form of this likelihood assumes independent per-pixel noise" — no masking/covariance beyond that unless the user supplies a modified σ_tot²).
- First term = χ² (Eq. after `model-def`); second term is the normalization (not dropped — it is Θ-dependent since σ_tot² depends on Θ).
- Noise model = Gaussian background (σ_bkg, user-specified) + Poisson shot noise term `I_model/(G·t_exp)`, derived from: expected electron counts = `t_exp·I_model·G`, Poisson variance = same, converted back to counts/sec² by dividing by `(G·t_exp)²`.
- **Key design choice**: variance is built from `I_model` (the forward-model image), *not* `I_obs`. Paper explicitly contrasts this with the common alternative `σ_tot² = σ_bkg² + I_obs/(G·t_exp)` (used e.g. by lenstronomy), which is Θ-independent and cheaper (enables closed-form linear light-profile solves) but "may induce biases at low signal-to-noise ratios" (cites Horne 1986) and requires ad hoc handling of negative pixels (lenstena zeroes them, which the paper calls "not the most rigorous way"). GIGA-Lens "opt[s] to use the more rigorous definition" (Θ-dependent, via I_model) "since... this incurs little additional computational cost" in their pipeline. For the paper's own 100-simulated-system test, the two definitions were checked and found to give "virtually identical" results.
- Inputs required: preprocessed I_obs (counts/sec, sky already subtracted — sky brightness I_sky could in principle be a model parameter but "In this work, we assume the sky brightness has been subtracted"), σ_bkg, exposure time t_exp, gain G (e⁻/count).
- Masking convention: **the paper does not address masking** in the main text's likelihood definition (the one masking discussion — center disc + interloper square masks — is in the *commented-out* HST section of `results.tex`, `sec:result-ray-obs`, which is not part of the compiled paper).
- Supersampling: I_model is evaluated on a grid supersampled by integer factor `k_super` (in this work k_super=2) relative to the image pixel grid, then downsampled by averaging back to the image grid before comparison to I_obs (§2.1 `sec:phys-model`, last paragraph).

## 3. Lens/light models and priors (§2.1 `sec:phys-model`, Eq. `sim-dist`)

Models used for the paper's simulated test set (100 systems) and the reference system:
- **Lens mass**: EPL (elliptical power law, Tessore & Metcalf 2015; equivalent to PEMD, Barkana 1998) + external shear. Convergence: `κ(x_lens,y_lens) = (3-γ_epl)/2 · (θ_E / sqrt(q x_lens² + y_lens²/q))^(γ_epl - 1)`. Shear deflection: `α_ext(x,y) = (γ_ext,1 x + γ_ext,2 y,  γ_ext,2 x - γ_ext,1 y)`.
- **Lens light** and **source light**: elliptical Sérsic, `I(x,y) = I_0 exp(-b_n((sqrt(q x_light² + y_light²/q)/R_eff)^(1/n) - 1))`, `b_n = 1.9992n - 0.3271`. Lens-light center is *not* tied to the lens-mass center.
- Ellipticity reparameterization used throughout: `(ε_1,ε_2) = (1-q)/(1+q) · (cos 2φ, sin 2φ)`.
- 22 free parameters total per system.

Prior table (Eq. `sim-dist`) — the paper gives **two** distributions per parameter: the "simulation distribution" (used to generate the 100 mock systems) and the (broader, less informative) "prior distribution" (used for modeling/recovery). Reproduced verbatim below as `simulation / prior`:

| Parameter | Simulation dist. | Prior (used for fitting) | Description |
|---|---|---|---|
| θ_E | exp(N(ln 1.25, 0.25)) | exp(N(ln 1.25, 0.4)) | Einstein radius (") |
| γ_epl | TN(2, 0.25; 1, 3) | TN(2, 0.5; 1, 3) | Mass slope |
| ε_epl,1, ε_epl,2 | N(0, 0.1) | N(0, 0.2) | Lens mass eccentricities |
| x_epl, y_epl | N(0, 0.03) | N(0, 0.06) | Lens mass center (") |
| γ_ext,1, γ_ext,2 | N(0, 0.05) | N(0, 0.1) | External shear components |
| R_l | exp(N(ln 1.6, 0.15)) | exp(N(ln 1.6, 0.25)) | Lens Sérsic radius (") |
| n_l | U(2, 6) | U(0.5, 8) | Lens Sérsic index |
| ε_l,1, ε_l,2 | TN(0, 0.05; -0.15, 0.15) | TN(0, 0.1; -0.15, 0.15) | Lens light eccentricities |
| x_l, y_l | N(0, 0.01) | N(0, 0.02) | Lens light center (") |
| I_l | exp(N(ln 300, 0.3)) | exp(N(ln 300, 0.5)) | Lens half-light intensity |
| R_s | exp(N(ln 0.25, 0.15)) | exp(N(ln 0.25, 0.25)) | Source Sérsic radius (") |
| n_s | U(0.5, 4) | U(0.5, 8) | Source Sérsic index |
| ε_s,1, ε_s,2 | TN(0, 0.15; -0.5, 0.5) | TN(0, 0.3; -0.5, 0.5) | Source light eccentricities |
| x_s, y_s | N(0, 0.25) | N(0, 0.5) | Source light center (") |
| I_s | exp(N(ln 150, 0.5)) | exp(N(ln 150, 0.9)) | Source half-light intensity |

Notation: U(a,b) = uniform on [a,b]; N(μ,σ) = Gaussian; TN(μ,σ;lo,hi) = truncated Gaussian.

Other stated facts about the priors:
- 5 parameters use a log-normal (exp(N(...))) form: θ_E, R_l, I_l, R_s, I_s.
- Ellipticities and centers deliberately use (bivariate) Normal rather than independent-uniform-per-component, specifically to preserve rotational symmetry — independent uniform ε priors would bias the induced distribution on position angle φ and axis ratio q (Fig. `prior-choice`).
- The prior is constructed so the source-lens offset scale matches θ_E: "we avoid this [vanishing gradient from extreme lens/source misalignment] by setting our prior in a way such that √((x_s-x_epl)² + (y_s-y_epl)²) ~ θ_E" (§2.2 footnote).
- Light-amplitude priors chosen so simulated arcs have SNR ~100 (range 30–200), matched to the lenstronomy starting-guide notebook (Birrer 2021).
- Pixel scale for the simulated data: 0.065"/pixel; cutout 5.2"×5.2" (80×80 pixels). Noise: σ_bkg=0.2, t_exp=100 sec, G=1 (assuming HST-like observations) (results.tex, Fig. `lens-sample` caption).

## 4. Why VI sits between MAP and HMC (§3.3 `sec:vi`, §3.4 `sec:hmc-result`)

This is stated as load-bearing and explicit, not incidental:

- **Primary, stated purpose: set the HMC mass matrix.** "This was the main purpose of the previous section, §3.3: we set M̃ = (Σ̃*_VI)⁻¹, the inverse of the inferred covariance matrix from VI" (§3.4). This is "preconditioned HMC" — using the VI covariance estimate as the momentum-distribution mass matrix "significantly improve[s] the sampling efficiency" (citing Brooks et al. 2011, Ch. 5). The abstract of §3.3 frames it the same way: "This covariance estimate plays an important auxiliary role for Monte Carlo sampling: it sets a scale for each of the parameters that is used to define a proposal distribution in the sampling step."
- **Secondary, stated purpose: initialize the HMC chains.** "Note that a secondary use of the VI step is that each of the n_HMC chains are initialized ... by sampling from the VI posterior, N(μ̃*_VI, Σ̃*_VI)" (§3.4, footnote-like remark right after the mass-matrix statement). So: both mass matrix AND initialization, with the mass-matrix role emphasized as primary ("main purpose").
- The paper frames VI itself as approximating "the posterior mode and evaluating the Hessian of the log posterior density around the mode" (footnote, §3.3) — i.e., a cheap local-Gaussian approximation, not meant to replace HMC.
- Explicitly flagged as a design choice that might become unnecessary: "There have been proposals (e.g., Sountsov et al. 2021) to adapt the mass matrix M̃ on the fly during burn-in steps, which may render the VI step irrelevant. For now, these methods are not yet well-tested ... hence VI remains a necessary step of our modeling pipeline" (§3.4). Conclusion.tex repeats this: eliminating VI via online mass-matrix adaptation "may offer up to a factor of two speed gain," and alternatively a more expressive VI fit could eventually eliminate HMC itself.
- Caveat on quality: the VI Gaussian surrogate is only approximate — "the VI posterior does not always exactly agree with the true posterior (i.e., HMC samples)," and Fig. `ref-corner` shows a "banana-shaped" cross-section despite Gaussian-looking marginals. Still: "we find that the VI covariance matrix is almost always a sufficiently good estimate of the true covariance" (§3.3, closing paragraph). Appendix archetypes (Fig. `archtypes`) show case-by-case VI-vs-true mismatches: VI underestimates posterior scale for a fold system, overestimates θ_E scale for a cusp system, matches well for cross and double systems.

## 5. Convergence diagnostics (§3.4.1 `sec:hmc-result`, §4 results.tex, Table `tab:err`)

- Diagnostics reported: effective sample size (**ESS**) and potential scale reduction factor (**R̂**, Gelman & Rubin 1992/PSRF).
- Threshold cited from the literature (not necessarily enforced as a hard gate in the pipeline, just quoted as the accepted rule of thumb): "an appropriate condition is R̂ < 1.1" (Gelman 1992, as cited in §3.4.1). Target HMC acceptance probability = 0.75 (see §1 above).
- Chain count: n_HMC = 50 chains (rationale given: parallelizes better than one long chain, and multiple chains are needed to compute R̂ at all).
- Burn-in: n_burn = 250 steps (with dual-averaging step-size/trajectory-length adaptation active only for the first 80% of burn-in, i.e., first 200 of the 250 steps).
- Sampling: n_sample = 750 steps per chain after burn-in.
- Reference system result: for every one of the 22 parameters, ESS > 26000 and R̂ < 1.01 (Fig. `demo-posterior-samples` caption), achieved in "just under 36 seconds" of HMC.
- 100-simulated-system result (Table `tab:err`): across all 100 systems and 22 parameters, the *worst-case* max R̂ over any parameter/system was **1.017**, and the *worst-case* min ESS was **26822** — "an order of magnitude higher than the typical value with emcee." Per-parameter mean R̂ is ~1.000–1.001 for every parameter; per-parameter mean ESS ~35400–35800.
- No explicit statement of a formal automated stopping rule tied to R̂/ESS thresholds — stage lengths (250/750) are fixed hyperparameters tuned once on the reference system and then left unchanged for the full 100-system test (explicitly stated in §3.4.3 `sec:pipeline-sum`).

## 6. Validation (§4 results.tex, §3.4.2 `sec:hmcvemcee`, Appendix)

- **Simulated recovery test**: 100 lensing systems simulated with lenstronomy, drawn from the "simulation distribution" (narrower than the fitting prior, same eq. `sim-dist`), each with 22 ground-truth parameters known. Pipeline run with the *same* fixed hyperparameters as the reference system (no per-system tuning).
- **Accuracy claims** (Table `tab:err`, "Mean error" and bias-normalized `μ_z` columns): for every one of the 22 parameters, the mean error across 100 systems is small and the scaled bias `μ_z = (E[Θ]-Θ_true)/sqrt(Var[Θ])`, averaged over the 100 systems, is "consistent with zero bias" for all parameters. Selected values: θ_E mean error -0.00026 (μ_z = -0.04 ± 0.09); γ_epl mean error 0.01608 (μ_z = 0.12 ± 0.08); other lens/light parameters similarly small (see full table reproduced in the file's §1 companion pull, or `tab:err` directly — units are the same as the prior table above, e.g. θ_E in arcsec).
- **Comparison to lenstronomy / emcee** (§3.4.2): HMC vs. lenstronomy's emcee sampler, both applied to the same reference system, same prior (well, lenstronomy uses uniform priors — paper states this "has virtually no effect on the sampling results"), same PSF and k_super=2. Both run to central-value agreement, but:
  - Sampling rate: HMC ~40 ESS/iteration (~300 ESS/sec on 1x A100) vs. emcee ~0.2 ESS/iteration (~0.04 ESS/sec on CPU) — roughly 3 orders of magnitude in ESS/sec.
  - emcee chains "devolve to random walks," show high inter-chain variance and much higher R̂ (specifically stated to exceed the recommended 1.1).
  - Autocorrelation: HMC autocorrelation ~negligible within 10 iterations; emcee's characteristic autocorrelation lag ~300 iterations.
  - Citations for gradient-free samplers' general weakness in higher dimensions: Betancourt 2018, Huijser et al. 2017 (the latter: affine-invariant ensemble samplers can "misleadingly appear to converge even when it has not" in 50–100+ dimensions).
- **Archetype recovery** (Appendix, Fig. `archtypes`): four hand-picked systems (fold, cusp, cross, double) from the 100 — posterior mean agrees with ground truth within uncertainty for all four. Fold system flagged as harder: γ_epl posterior std ±0.1 vs. the more typical ±0.05 for other systems, "worth keeping in mind when doing density profile slope studies."
- Paper does **not** report a validation run against real (non-simulated) HST data in the compiled text — the one section that would do this (`results.tex`, `sec:result-ray-obs`, an HST demonstration system) is entirely commented out in the LaTeX source and not part of the published paper.

## 7. Stated limitations / failure modes (verbatim where noted)

- Multimodality caveat (conclusion.tex): "although we did not find significant multimodality in the posterior for the model we consider in this work (i.e., all local modes have vanishing posterior density compared to the global mode), it is unclear whether this will still be the case for more complex lens and source models."
- MAP is not guaranteed on every run — only "typically ~5%" (at least ~1% for low-SNR systems) of the 300 multi-start samples reach the global optimum; the method's robustness rests on n_MAP being "large enough" and the prior being "broad," not on any single start succeeding (§3.2, §3.4.3).
- Variance-map choice: using I_obs instead of I_model for σ_tot² (the cheaper, common alternative) "may induce biases at low signal-to-noise ratios" (Horne 1986) — paper avoids this by using I_model, but flags it as a real bias risk in the alternative formulation that other pipelines (e.g. lenstronomy) use.
- VI is only a local Gaussian approximation and known to sometimes mismatch the true posterior shape (banana-shaped cross-sections, per-archetype under/over-estimation of scale) — see §4 above.
- Gradient can vanish for "extreme misalignment of the lens and source" — mitigated only by prior construction (tying source-lens offset to θ_E), not by any inference-time fix.
- Fold-type systems: weaker constraint on γ_epl (±0.1 vs. typical ±0.05).
- Sérsic index / effective radius degeneracy: "as n_l and n_s increase, their uncertainties increase as well... because the light becomes more compact at higher Sérsic indices, resulting in higher degeneracy between the Sérsic indices and the half-light radii" (Fig. `sim-parity` caption).
- Framework tested only on one lens model choice (EPL + shear, Sérsic lens/source light) and only on simulated data in the published results — real-HST application is deferred to "an upcoming publication (Gu et al. in prep.)" using 51 systems from Huang et al. 2020/2021 observed under HST program ID 15867.
- TensorFlow implementation lacks the JAX implementation's multi-GPU distributed support ("only the JAX implementation... supports distributed computing over multiple GPUs, due to the lack of support for distributed computing on TensorFlow").

## 8. Timing / hardware

- GPU: NVIDIA A100 (obtained via NERSC Perlmutter early access), and one head-to-head comparison figure uses a "modern CPU" for emcee.
- Headline pipeline timing (reference system, all confirmed in the *visible* deluxetable columns — see §1 correction above):
  - 4x A100: MAP 17s + VI 52s + HMC 36s = **105 sec total**.
  - 1x A100: **~6 min total** (stated as ~3.5x the 4-GPU time).
- emcee-vs-HMC comparison (§3.4.2): HMC on 1x A100 achieves ~300 ESS/sec; emcee on (unspecified) CPU achieves ~0.04 ESS/sec.
- Scaling claim (conclusion.tex): using shapelets instead of Sérsic for the source light model costs "only ~10% more computation time."
- Future-hardware claim (conclusion.tex, not a measured result): moving from 4 to 8 A100 GPUs is expected to roughly halve execution time, "bringing the total time to below 1 minute" — stated as a plan/expectation, not a demonstrated number.
- Motivating scale claim (abstract, conclusion): pipeline is pitched as enabling modeling of O(10^5) lenses expected from LSST/Euclid/Roman "on a reasonable time scale," given the ~105 sec/system, multi-GPU figure.

