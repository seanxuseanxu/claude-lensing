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
lens-light `n_sersic` ceiling and added a second source component (29 params).

- **Clump position angles in `MODELING_PLAN.md`'s table are array-space angles** (from the
  +row axis toward +col, in the 80×80 crop), **not sky PA.** Converting them through the
  image's WCS the standard way (North through East) places the markers on empty sky.
- **theta_E = 1.516 ± 0.002" is the quotable value**, from round 1 — reproduced across
  independent seeds and a differently-centered falsification prior, with no multimodality.
  The two-source run gives 1.484 ± 0.002" but **that error bar is not usable**: 29/29
  parameters fail Rhat < 1.01 (worst 7.17). Checked and it is slow mixing, not
  multimodality and not label switching. The 1.515 → 1.484 shift stays uninterpretable
  until that run gets a longer HMC.
- **Two source-model deficiencies from round 1 are fixed.** A second Sersic source brings
  all four clumps within 9% (the N arc was underfit 2× at 0.55) and, more importantly,
  makes the model produce *detached clumps* rather than a smooth modulated ring. Reduced
  chi² 1.837 → 1.302.
- **Three parameters have pinned at prior boundaries across these runs**, so check for this
  routinely — `diagnose.py` now flags any Uniform-prior parameter within 2% of a bound.
  `lens_light.n_sersic` pinned at its ceiling (fixed by `Uniform(2, 10)`, lands at 7.43);
  `source.n_sersic` pinned at its *lower* bound 0.5 and went unnoticed for a full round
  (relieved by the second component, not by widening).

Two pipeline traps, both latent for a full round before firing:

- **Never use bare `argmax` to pick the best MAP start.** `jnp.argmax` propagates NaN, so
  one diverged start out of 1024 silently selects garbage and SVI then yields NaN for
  every step. Use `np.nanargmax`. Fixed in `run_fit.py`, including the `--stages svi,hmc`
  resume path, which had to recompute the index rather than trust the checkpoint.
- **Changing a prior's *support* silently rewrites old runs' physical values**, because
  checkpoints store unconstrained samples and the bijector depends on the bounds. Bump
  `priors.SOURCE_PRIOR_VERSION` and keep the old prior function verbatim rather than
  editing one in place; `metadata.json` records the version. Location-scale changes (a
  Normal's sigma) are safe — support changes are not.

An upstream hazard, documented but not fixed, since `gigalens/src/**` is off limits:
`Parameterized.__init__` does `self.params = self._params`, binding the *class* list, so
constructing any `use_lstsq=True` profile appends `"amp"` to the class attribute and
retroactively corrupts every already-built instance. Harmless for these runs (all
`use_lstsq=False`) but it will bite the shapelets escalation path.

## Working notes

- The demo notebooks (`gigalens/jax-demo.ipynb`, `tf-demo.ipynb`, `shapelets-demo.ipynb`) are the best worked examples of a full prior → MAP → SVI → HMC run, including the `optax.polynomial_schedule` learning-rate choices that actually converge. Note that optax schedules there use *negative* learning rates composed with `scale_by_adam` (i.e. descent by scaling), which is easy to misread.
- `gigalens/docs/` is generated HTML committed for GitHub Pages; edit `gigalens/docsrc/source/` and run `make github` from `docsrc/` instead.
