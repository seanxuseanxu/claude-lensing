# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Three unrelated top-level directories, tracked in one git repo:

- `gigalens/` — a vendored copy of the GIGA-Lens package (upstream: https://github.com/giga-lens/gigalens). Source lives under `gigalens/src/gigalens/`. This is the code you will actually modify.
- `papers/` — LaTeX sources for two papers: `gigalens_paper/` (the GIGA-Lens method paper, arXiv:2202.07663; entry point `ms_SL-model.tex`) and `strong_lens_foundry_I/` (entry point `main.tex`). Both use the AASTeX 6.31 class, which is vendored alongside.
- `claude/` — the lens-modelling work. `claude/pipeline/` is a lens-independent MAP→SVI→HMC fitting pipeline (see "The modelling pipeline" below); `claude/<LENS>/` holds one system's data, its `config.py`, its `runs/`, and its own README. `LENS_MODELING_GUIDE.md` and `WORKING_WITH_AGENTS.md` are lessons learned, using DESI238 as the worked case.

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

## The modelling pipeline

`claude/pipeline/` fits a lens through `gigalens.jax`'s MAP→SVI→HMC sequence. It is
lens-independent: nothing in it names a specific system. Each lens gets a directory
`claude/<LENS>/` holding its data, its `config.py`, its `runs/`, and its own README.

```bash
export PYTHONPATH=gigalens/src:claude/pipeline
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 python claude/pipeline/verify.py --lens DESI238
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py --lens DESI238 --label baseline
python claude/pipeline/diagnose.py claude/DESI238/runs/baseline
```

`verify.py` is a ~5-minute pre-flight check, not the fit; run it before spending a
GPU-hour. `run_fit.py` requires `--lens` and writes to `claude/<LENS>/runs/<label>/`. The
read-only tools (`diagnose`, `dump_params`, `plot_overview`, `plot_corner`, `plot_source`,
`clumps`) take a run directory and infer the lens from its `metadata.json`.

**One GPU per process.** Upstream's pmap path is broken under JAX 0.11 (sharding mismatch
at `gigalens/jax/inference.py:55`) and `compat.require_single_device` refuses to start with
more than one device visible. `compat.py` carries the JAX-0.11 shims without touching
`gigalens/src`. Run several lenses concurrently as separate processes.

See `claude/pipeline/README.md` for the file-by-file breakdown and for how to add a lens.

### The config seam

A prior is built from a **version**, which fixes the functional form and in particular each
parameter's *support*, and a **`LensConfig`**, which supplies location and scale within that
form. The division is load-bearing:

- **Supports live in `pipeline/priors.py`, keyed by version.** A config cannot reach them.
- **Locations and scales live in `claude/<LENS>/config.py`.**

This is what makes the versioning invariant hold by construction rather than by
documentation. See the trap below for why it matters.

### Reading a finished run

`diagnose.rebuild_model(run_dir)` reconstructs the exact model a run was fitted with from
its `metadata.json`, and every read-only tool calls it. Its fallbacks are the values that
runs *predating each key* actually used — never today's defaults. Do not reimplement it.

### Traps, each latent for a full round before firing

- **A diverged SVI silently poisons HMC, and the symptom looks like slow mixing.**
  `compat.hmc_single_device` builds the momentum distribution from `inv(q_z.covariance())`
  in float32. If the SVI covariance is badly conditioned, that inverse is not an inverse —
  the spectrum carries negative eigenvalues — so every chain samples against an effectively
  indefinite mass matrix and *every* Rhat fails at once. The covariance is still positive
  definite, so a PD test does not catch it; **conditioning is the discriminant.**
  `compat.check_svi_covariance` hard-fails above 1e10 and `diagnose.py` prints an SVI health
  line ahead of the convergence table. The original cause was an SVI learning-rate schedule
  that ramped *upward*; it now warms up and decays.
- **Never use bare `argmax` to pick the best MAP start.** `jnp.argmax` propagates NaN, so one
  diverged start out of a thousand silently selects garbage and SVI then yields NaN for
  every step. Use `np.nanargmax`. `run_fit.py` does, including on the `--stages svi,hmc`
  resume path, which has to recompute the index rather than trust the checkpoint.
- **Changing a prior's *support* silently rewrites old runs' physical values**, because
  checkpoints store unconstrained samples and the bijector depends on the bounds. Add a
  version in `priors.py` and keep the old builder verbatim rather than editing one in place;
  `metadata.json` records the versions. Location-scale changes are safe — support changes
  are not. This is why a config may not set a bound.
- **Parameters pin at prior boundaries**, so check routinely — `diagnose.py` flags any
  Uniform-prior parameter within 2% of a bound. Note that gigalens' Sersic constant
  `bn = 1.9992n − 0.3271` is 3% wrong at n = 0.5 and 13.6% at n = 0.3, so a source
  `n_sersic` pinned at a 0.5 floor calls for a different profile, not a lower bound.
- **A model-to-model spread is only a systematic when the evidences are comparable.**
  Inflating an error bar to span a model the data rejects reports a range the data excluded.

An upstream hazard, documented but not fixed, since `gigalens/src/**` is off limits:
`Parameterized.__init__` does `self.params = self._params`, binding the *class* list, so
constructing any `use_lstsq=True` profile appends `"amp"` to the class attribute and
retroactively corrupts every already-built instance. Harmless while everything is
`use_lstsq=False`, but it will bite the shapelets escalation path.

## Modelled systems

Each has its own README with the dataset, the conditioning decisions, and the results.

- `claude/DESI238/` — HST WFC3/IR F140W. Three rounds fitted; `theta_E = 1.480 ± 0.003"`.
  See `claude/DESI238/README.md`.
- `claude/DESI257/`, `claude/DESI234/` — no data yet.
## Working notes

- The demo notebooks (`gigalens/jax-demo.ipynb`, `tf-demo.ipynb`, `shapelets-demo.ipynb`) are the best worked examples of a full prior → MAP → SVI → HMC run, including the `optax.polynomial_schedule` learning-rate choices that actually converge. Note that optax schedules there use *negative* learning rates composed with `scale_by_adam` (i.e. descent by scaling), which is easy to misread.
- `gigalens/docs/` is generated HTML committed for GitHub Pages; edit `gigalens/docsrc/source/` and run `make github` from `docsrc/` instead.
