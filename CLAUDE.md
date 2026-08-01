# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Three unrelated top-level directories, tracked in one git repo:

- `gigalens/` — a vendored copy of the GIGA-Lens package (upstream: https://github.com/giga-lens/gigalens). Source lives under `gigalens/src/gigalens/`. This is the code you will actually modify.
- `papers/` — LaTeX sources for two papers: `gigalens_paper/` (the GIGA-Lens method paper, arXiv:2202.07663; entry point `ms_SL-model.tex`) and `strong_lens_foundry_I/` (entry point `main.tex`). Both use the AASTeX 6.31 class, which is vendored alongside.
- `claude/DESI238/` — observational data for one lens system (see "The DESI238 dataset" below) plus `claude/DESI238/model/`, a working MAP→SVI→HMC fitting pipeline for it (see "The DESI238 model pipeline" below).

## Environment

The active conda env is `gigajax6.0` (Python 3.14). Two things about it that will bite you:

- **`gigalens` is not pip-installed.** It resolves as an implicit namespace package pointing at the repo directory, so `import gigalens.jax` fails from the repo root. Run anything that imports it with `PYTHONPATH=gigalens/src` (or from `gigalens/` with `PYTHONPATH=src`).
- **TensorFlow is not installed.** Only the JAX half of the package is usable here. `tensorflow_probability` is present as the JAX substrate (`tensorflow_probability.substrates.jax`). Do not assume `gigalens.tf.*` can be imported or tested; if you change TF code, say plainly that it is unverified.

Installed versions diverge sharply from the pins in `gigalens/setup.py` (which asks for `jax==0.2.24`, `lenstronomy==1.9.3`, `tensorflow>=2.6`). Actual: jax 0.11.0, lenstronomy 1.14.2, numpy 2.4.6, optax 0.2.8, astropy 8.0.1. Treat `setup.py` and `tox.ini` (`envlist = py37`) as upstream artifacts, not as a description of this environment.

## Tests

`pytest` is **not** installed in `gigajax6.0`. Once it is:

```bash
cd gigalens && PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python -m pytest tests/test_profiles.py::test_epl -q   # single test
```

`tests/test_profiles.py` parametrizes over both backends in one loop body and will fail on the TF entries in this env. `tests/tf/test_model.py` imports `tensorflow` at module level. `tests/conftest.py` fixtures (`default_prior`, `default_physmodel`, `default_data`) are TF-only.

The profile tests are the load-bearing ones: each asserts numerical parity against the corresponding `lenstronomy` profile, typically `rtol=1e-5, atol=1e-4`. **Any new or modified profile should be validated the same way** — a lenstronomy reference is the accepted ground truth in this codebase.

## Architecture

The package is a dual-backend mirror. `gigalens/src/gigalens/` holds framework-agnostic abstract base classes; `gigalens/tf/` and `gigalens/jax/` hold parallel concrete implementations with identical class names and near-identical logic. A change to one backend almost always needs the same change to the other.

| Abstraction | Base (`gigalens/`) | Implementations |
|---|---|---|
| Profile interfaces | `profile.py` — `Parameterized`, `LightProfile.light()`, `MassProfile.deriv()` | `{tf,jax}/profiles/{light,mass}/` |
| System description | `model.py` — `PhysicalModel`, `ProbabilisticModel.log_prob()` | `{tf,jax}/model.py` |
| Image rendering | `simulator.py` — `SimulatorConfig`, `LensSimulatorInterface` | `{tf,jax}/simulator.py` |
| Inference driver | `inference.py` — `ModellingSequenceInterface` | `{tf,jax}/inference.py` |

### The three-stage inference sequence

`ModellingSequence` implements the pipeline described in §2.3–2.5 of the method paper, and the stages are meant to be chained — each consumes the previous one's output:

1. **`MAP`** — multi-start Adam over ~500 samples drawn from the prior, all optimized in parallel as one batch. Returns unconstrained parameters for every start; you pick the best by `log_prob`.
2. **`SVI`** — fits a full-covariance Gaussian ansatz by minimizing the ELBO, started from the best MAP sample. Returns a `tfd.MultivariateNormalTriL` over *unconstrained* space.
3. **`HMC`** — preconditioned HMC whose momentum distribution is built from the SVI covariance, with dual-averaging step-size adaptation and gradient-based trajectory-length adaptation.

Note the argument-order inconsistency between backends: TF is `MAP(optimizer, start, ...)` and `SVI(optimizer, start, ...)`; JAX is `MAP(optimizer, start, ...)` but `SVI(start, optimizer, ...)`.

### Parameter-space conventions

These are the two conventions most likely to cause silent errors:

- **Everything internal is in unconstrained space.** Priors are `tfd.JointDistributionSequential` nesting `JointDistributionNamed` dicts, structured as `[lens, lens_light, source_light]`. `ProbabilisticModel.bij` composes `pack_bij` (reshape a flat `(bs, d)` tensor into that nested dict structure — determinant 1) with `unconstraining_bij` (the prior's default event-space bijector). `bij.inverse` flattens and unconstrains; `bij.forward` restores. MAP/SVI/HMC all return unconstrained values; apply `bij.forward` before interpreting anything physically.
- **`log_prob` returns a 2-tuple**, `(log_posterior, reduced_chi_squared)`, both shaped `(bs,)`. Callers index `[0]` for the density; the second element is only used to drive progress bars.

`ForwardProbModel` estimates the noise map from the *simulated* image each step (so the variance depends on parameters). `BackwardProbModel` fixes the noise map from the *observed* image once and solves the linear light amplitudes by weighted least squares inside the simulator (`lstsq_simulate`). Pick `BackwardProbModel` when using `use_lstsq=True` profiles.

### Simulator mechanics

`LensSimulator` precomputes a coordinate grid at construction and is **bound to a fixed batch size `bs`** — every stage of the inference sequence constructs its own instance because `n_samples`, `n_vi`, and `n_hmc` differ. Ray tracing is a simple sum of deflections over `phys_model.lenses`; lens light is evaluated on the image plane, source light on the deflected plane. Supersampling renders at `supersample × num_pix` and average-pools down; the PSF is upsampled to match via `lenstronomy.Util.kernel_util.subgrid_kernel`. Output is multiplied by `conversion_factor = det(transform_pix2angle)` so surface brightness integrates correctly.

`LightProfile.depth` is the number of linear components a profile contributes (1 for Sersic, `(n_max+1)(n_max+2)/2` for Shapelets); the least-squares path stacks them along the channel axis and uses a depthwise convolution, so `depth` must be right or the design matrix is malformed.

The JAX inference code `pmap`s across devices and silently truncates `n_samples` to a multiple of `jax.device_count()`. On this machine JAX sees a single CPU device (no CUDA), so expect the package to be slow and to exercise none of the multi-GPU paths it was designed for.

### Adding a profile

Subclass `MassProfile` (implement `deriv`) or `LightProfile` (implement `light`), set the class-level `_name` and `_params`, add the module to `gigalens/{tf,jax}/profiles/{mass,light}/__init__.py`, mirror it in the other backend, and add a lenstronomy-parity test. Parameters arrive as keyword arguments broadcastable against the `(H, W, bs)` coordinate grids.

## The DESI238 dataset

One lens system, HST WFC3/IR in F140W, `EXPTIME = 1197.699` s, `NCOMBINE = 3`.

| File | Contents |
|---|---|
| `final_94_drz.fits` | DrizzlePac product. HDU1 `SCI` (2258×1997, float32, `BUNIT = ELECTRONS/S`, full WCS), HDU2 `WHT`, HDU3 `CTX`, HDU4 `HDRTAB`. HDU0 has no data. |
| `cutout238.npy` | 120×120, dtype `>f4`. **Bit-for-bit identical to `SCI[1010:1130, 1165:1285]`** — verified, not inferred. |
| `psf238.fits` | Single HDU, 27×27, dtype `>f8`. |

Values I verified directly, suitable for a `SimulatorConfig` / `ForwardProbModel`:

- `delta_pix = 0.065` — from the SCI CD matrix, `sqrt(|det CD|) × 3600`. The cutout carries no WCS of its own, so this comes from its provenance as a crop.
- `num_pix = 120`.
- `exp_time = 1197.699`.
- `background_rms ≈ 0.0315` from `astropy.stats.sigma_clipped_stats(cutout, sigma=3.0)` — **this number is wrong for likelihood use, kept here only as a documented dead end.** The clip is dominated by galaxy light (the cutout is mostly deflector, not sky), not clean background. The corrected value, measured from six widely separated blank patches in the parent drizzle frame, is **`background_rms = 0.0075`** (range 0.0070–0.0092 across patches) — this is what `claude/DESI238/model/data.py` actually uses. Using 0.0315 down-weights the arcs into irrelevance.

Three traps:

1. **Both arrays are big-endian, and JAX refuses them.** `jnp.asarray(np.load(...))` raises `TypeError: Error interpreting argument ... as a JAX value`. Cast first: `.astype(np.float32)`.
2. **The PSF is not normalized** — it sums to 1.1445. Divide by its sum before passing it as `kernel` unless you have a reason not to. It is odd-sized and centered (flux centroid within 0.12 px of pixel 13,13), so no recentering is needed.
3. **The PSF carries no WCS or scale keywords**, so its scale cannot be read off the file. It is at the same 0.065"/pix as the science image — confirmed by the repository owner, not derivable from the data. It is therefore not oversampled relative to the drizzled grid, and `subgrid_kernel` will handle any `supersample > 1` correctly on its own.

On units: the data is a rate (electrons/s), not counts, which is what `ForwardProbModel`'s `sqrt(background_rms² + max(model,0)/exp_time)` expects — the Poisson variance of a rate image scales as `rate / exp_time`. Passing `exp_time = EXPTIME` is correct to first order. It is only first order, though: `NCOMBINE = 3` means coverage varies across the frame, so a single scalar exposure time understates the noise wherever fewer exposures overlap. The `WHT` extension is the proper way to build a per-pixel exposure map if that matters.

## The DESI238 model pipeline

`claude/DESI238/model/` fits this system through `gigalens.jax`'s MAP→SVI→HMC sequence.
Run `python claude/DESI238/model/verify.py` before any real fit — it's a 5-minute
pre-flight check, not the fit itself. See `claude/DESI238/model/README.md` for the JAX-0.11
compatibility shims this needed (upstream `gigalens.jax.inference` does not run as-is
under JAX 0.11.0; `compat.py` patches it without touching `gigalens/src`) and
`MODELING_PLAN.md` for how the priors were derived from this specific image's flux scale.

Two rounds of production fits have run. `RESULTS.md` covers round 1 (single Sersic source,
22 params); `RESULTS_v2.md` covers round 2 (2026-07-30), an ablation that freed the
lens-light `n_sersic` ceiling and added a second source component (29 params). Both carry
inline corrections dated 2026-07-31 — read those before the sections they correct.

`RESULTS_v3.md` covers round 3 (2026-07-31): `twosource_v2` (29 params, a trustworthy
re-run of the broken round-2 fit) and `bulgedisk` (36 params, `--n-lens-light 2`). The
second deflector component was motivated by a coherent 1–3% radial S-curve in the round-2
core residual that a single Sersic cannot produce. Unlike the source components, the two
lens-light components get *different* priors: bulge and disk are physically distinct, and
identical priors would add a label-switching mode for nothing.

- **The SVI fix worked.** Covariance conditioning 8.8e+11 → 3.4e+07, ELBO tail sd 7,384 →
  6.6, worst Rhat 7.17 → 1.544. `theta_E` reproduced at 1.4837 ± 0.0024, confirming the
  point estimate was never affected — only the error bar.
- **The bulge+disk deflector worked.** chi²ᵥ 1.302 → **0.991**, core chi² (r ≤ 8 px)
  7.09 → 2.97, and the coherent radial S-curve flattened into sign-alternating scatter
  within ±0.75σ. Recovered geometry is a textbook bulge+disk (R_sersic 0.33" and 2.73",
  an 8.4× separation from priors overlapping at ~1σ). **No arc cannibalisation** — the
  clump ratios improved to 1.03 / 0.99 / 1.02 / 0.83 and `theta_E` moved only 1.484 →
  1.480.
- **Convergence was chain length, and is now fixed.** `bulgedisk_long` re-ran `bulgedisk`
  changing only the HMC length (3000 draws after 1000 burnin, 4×): worst Rhat 3.242 →
  **1.338**, failing count 33/36 → **10/36**, chi²ᵥ unchanged at 0.989. Rhat fell instead
  of plateauing, so the residue was under-mixing, not the posterior ridge it was read as —
  no reparametrization needed. What still fails is the two components' shape parameters
  plus `source1.n_sersic`, which is pinned at the 0.5 floor and therefore cannot mix by
  construction. All eight lens-mass parameters sit at Rhat ≤ 1.002.
- **`theta_E = 1.480 ± 0.003"` (Rhat 1.001, ESS 94,186) is the quotable value**, from
  `bulgedisk_long`. It supersedes round 1's 1.516 ± 0.002". **The 0.036" gap between them
  is not a systematic** (corrected 2026-07-31): round 1 is rejected at ΔBIC ≈ −5294, so the
  gap is a bias that was removed, not an uncertainty that remains. Quote ± 0.003". A
  model-to-model spread is only a systematic when the evidences are comparable.
- **Bulge+disk over single-Sersic deflector: Δlog L = +912 for 7 params, ΔBIC ≈ −1762**,
  robust across the whole background_rms range. **But chi²ᵥ = 0.99 is not evidence the
  noise model is right**: over the blank-patch range 0.0070–0.0092 it spans 0.82–1.05, so
  ±10% on the noise scale is ±20% on chi². Quote the improvement, not the absolute value.
  This makes the `WHT` per-pixel exposure map more worth building than before, not less.

- **Clump position angles in `MODELING_PLAN.md`'s table are array-space angles** (from the
  +row axis toward +col, in the 80×80 crop), **not sky PA.** Converting them through the
  image's WCS the standard way (North through East) places the markers on empty sky.
- **Round 1's theta_E = 1.516 ± 0.002" is superseded** by 1.480 ± 0.003" above, though it
  was itself solid: reproduced across independent seeds and a differently-centered
  falsification prior, with no multimodality. Round 2's 1.484 ± 0.002" was **not** usable —
  29/29 parameters failed Rhat (worst 7.17) because its SVI stage diverged and HMC ran on
  the resulting ill-conditioned preconditioner (see the SVI trap below). Label switching, a
  dead component, and multimodality were each ruled out; the "slow mixing" conclusion drawn
  from that was wrong.
- **Two source-model deficiencies from round 1 are fixed.** A second Sersic source brings
  all four clumps within 9% (the N arc was underfit 2× at 0.55) and, more importantly,
  makes the model produce *detached clumps* rather than a smooth modulated ring. Reduced
  chi² 1.837 → 1.302.
- **Parameters keep pinning at prior boundaries**, so check for this routinely —
  `diagnose.py` flags any Uniform-prior parameter within 2% of a bound. `lens_light.n_sersic`
  pinned at its ceiling (fixed by `Uniform(2, 10)`, lands at 7.43); `source.n_sersic` pinned
  at its *lower* bound 0.5 and went unnoticed for a full round. A `n_sersic` has pinned at
  the 0.5 floor in all three rounds (`source1.n_sersic` = 0.606 in round 3). That floor is
  deliberate — gigalens' `bn = 1.9992n − 0.3271` is 3% wrong at n = 0.5 and 13.6% wrong at
  n = 0.3 — so the fix is a different profile, not a lower bound.

Three pipeline traps, each latent for a full round before firing:

- **A diverged SVI silently poisons HMC, and the symptom looks like slow mixing.**
  `compat.hmc_single_device` builds the momentum distribution from
  `inv(q_z.covariance())` in float32. If the SVI covariance is badly conditioned, that
  inverse is not an inverse — at the 8.8e+11 seen in round 2, `max|inv(C)·C − I| = 55` and
  the spectrum carries negative eigenvalues — so every chain samples against an
  effectively indefinite mass matrix and *every* Rhat fails at once. Note that the
  covariance is still positive definite, so a PD test does not catch it; **conditioning is
  the discriminant.** `compat.check_svi_covariance` now hard-fails above 1e10 (healthy runs
  here sit at 1e6–6e8) and `diagnose.py` prints an SVI health line ahead of the convergence
  table. The root cause was an SVI learning-rate schedule that ramped *upward*, putting the
  largest steps at the end of the run; it now warms up and decays.
- **Never use bare `argmax` to pick the best MAP start.** `jnp.argmax` propagates NaN, so
  one diverged start out of 1024 silently selects garbage and SVI then yields NaN for
  every step. Use `np.nanargmax`. Fixed in `run_fit.py`, including the `--stages svi,hmc`
  resume path, which had to recompute the index rather than trust the checkpoint.
- **Changing a prior's *support* silently rewrites old runs' physical values**, because
  checkpoints store unconstrained samples and the bijector depends on the bounds. Bump
  `priors.SOURCE_PRIOR_VERSION` (or `lens_light_prior_version_for`) and keep the old prior
  function verbatim rather than editing one in place; `metadata.json` records both
  versions. Location-scale changes (a Normal's sigma) are safe — support changes are not.

An upstream hazard, documented but not fixed, since `gigalens/src/**` is off limits:
`Parameterized.__init__` does `self.params = self._params`, binding the *class* list, so
constructing any `use_lstsq=True` profile appends `"amp"` to the class attribute and
retroactively corrupts every already-built instance. Harmless for these runs (all
`use_lstsq=False`) but it will bite the shapelets escalation path.

## Working notes

- The demo notebooks (`gigalens/jax-demo.ipynb`, `tf-demo.ipynb`, `shapelets-demo.ipynb`) are the best worked examples of a full prior → MAP → SVI → HMC run, including the `optax.polynomial_schedule` learning-rate choices that actually converge. Note that optax schedules there use *negative* learning rates composed with `scale_by_adam` (i.e. descent by scaling), which is easy to misread.
- `gigalens/docs/` is generated HTML committed for GitHub Pages; edit `gigalens/docsrc/source/` and run `make github` from `docsrc/` instead.
