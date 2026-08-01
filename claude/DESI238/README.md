# DESI238 lens modelling

Fits the DESI238 strong lens with the vendored GIGA-Lens JAX backend, through the
MAP → SVI → HMC sequence. `gigalens/src/**` is not modified; the JAX-0.11 fixes live in
`compat.py`.

| File | Role |
|---|---|
| `compat.py` | JAX 0.11 shims, single-device guard, pmap-free HMC |
| `data.py` | load / crop / normalise; returns image, PSF, `SimulatorConfig` |
| `priors.py` | priors + `PhysicalModel`; derives the flat parameter ordering |
| `run_fit.py` | MAP → SVI → HMC with per-stage checkpointing |
| `diagnose.py` | R̂ / ESS, residual maps, corner plot, label-switching and boundary-pinning checks |
| `clumps.py` | azimuthal-median clump photometry (array-space PA, not WCS) |
| `verify.py` | pre-flight checks — run this first |

## Running

All commands from the repo root. `PYTHONPATH` must include both `gigalens/src` (the
package is not pip-installed) and this directory.

```bash
export PYTHONPATH=gigalens/src:claude/DESI238/model
export NCCL_NVLS_ENABLE=0

# Pre-flight. ~5 minutes. Do this before spending a GPU-hour.
GIGALENS_DEVICE=0 python claude/DESI238/model/verify.py

# Production fit.
GIGALENS_DEVICE=0 python claude/DESI238/model/run_fit.py --label baseline --seed 0

# Diagnostics.
python claude/DESI238/model/diagnose.py claude/DESI238/model/runs/baseline
```

`--stages` restricts which stages run (`--stages map`, `--stages svi,hmc`, …). Later
stages load the earlier stages' checkpoints from `--out`, so a crashed HMC can be retried
without redoing MAP and SVI.

### Model variant flags

`--n-source` (default 1) sets the number of Sersic source components; `--lens-light-n-max`
(default 10.0) sets the `lens_light.n_sersic` prior ceiling. Both are recorded in
`metadata.json` alongside `source_prior_version`, and `diagnose.py` reads all three back so
it reconstructs the right model. The two-component run in `RESULTS_v2.md` was:

```bash
GIGALENS_DEVICE=1 python claude/DESI238/model/run_fit.py \
    --label twosource --seed 0 --lens-light-n-max 10 --n-source 2
```

With `--n-source 2` both components get **identical, deliberately wide** priors, so the
data picks the morphology. They are exchangeable in principle; `diagnose.py` measures
whether any chain actually switched labels rather than assuming, and leaves the samples
untransformed when none did.

### Multiple GPUs

Use them for *independent* runs, not for splitting one batch — the model is too small to
saturate one H200, and the multi-device path is broken (see below). `GIGALENS_DEVICE=n`
picks the n-th GPU listed in `CUDA_VISIBLE_DEVICES`, so with `CUDA_VISIBLE_DEVICES=1,2`:

```bash
GIGALENS_DEVICE=0 python claude/DESI238/model/run_fit.py --label seed0 --seed 0 &
GIGALENS_DEVICE=1 python claude/DESI238/model/run_fit.py --label seed1 --seed 1 &
wait
```

Two seeds is the multimodality check: a `theta_E` prior spanning 1.0–2.2" can support
distinct solutions fitting different subsets of the clumps, and one batch on one device
cannot detect that. If the two runs disagree on `theta_E` beyond their quoted
uncertainties, report the disagreement rather than picking the better-looking one.

The falsification run — fit the inner three clumps only, and ask whether the E blob is a
counter-image or an interloper:

```bash
GIGALENS_DEVICE=1 python claude/DESI238/model/run_fit.py \
    --label inner3 --theta-e-center 1.3 --theta-e-width 0.2
```

## Four JAX 0.11 breakages

`gigalens.jax.inference` was written against JAX 0.2.24. All four below were reproduced by
traceback in this environment, not inferred:

1. `jax.tree_map` removed → `AttributeError` in `SVI`. Patched in `compat.py`.
2. `float(jnp.nanmin(x, keepdims=True))` → `TypeError` in `MAP`, on step 1. Patched.
3. `jnp.array(jnp.split(...))` under `pmap` → sharding `ValueError` in `MAP`. **Not
   patched** — the multi-device path is unused. `require_single_device()` refuses to start
   rather than dying mid-run.
4. `HMC`'s `run_chain` is `@pmap`-wrapped while `q_z` is built outside it, so Auto-sharded
   arrays meet a Manual mesh → context-mesh `ValueError`. This fires **even with one
   device**, so the guard in (3) is no defence. Worked around by
   `compat.hmc_single_device`, which reproduces the identical kernel stack without `pmap`.

Because of (4), HMC output is `(num_results, n_hmc, n_params)` — draws on axis 0, chains on
axis 1 — not upstream's 4-D pmap shape.

## Conventions that cause silent errors

- **Everything in the `.npz` files is unconstrained.** MAP, SVI and HMC all return the
  internal parameterisation. Apply `prob_model.bij.forward` before interpreting anything
  physically. `diagnose.py` does this in one place (`to_constrained`).
- **Flat parameter order is alphabetical within each block**, not declaration order —
  `tfb.pack_sequence_as` sorts the keys. The lens block comes out as `center_x, center_y,
  e1, e2, gamma, theta_E`. Use `priors.param_names(prior)`; never hardcode. `verify.py`
  cross-checks it against a perturbation probe, because a mislabelling here corrupts every
  plot without raising an error.
- **`log_prob` returns a 2-tuple** `(log_posterior, reduced_chi_squared)`. Index `[0]`.
- **`LensSimulator.simulate` squeezes the batch axis at `bs=1`** — it returns `(80, 80)`,
  not `(1, 80, 80)`.
- **optax learning rates are negative**, composed with `scale_by_adam` to descend. Easy to
  misread as a bug and "fix".
- **`SVI(start, optimizer, ...)`** in JAX — the reverse of the TF backend.

## Data choices that differ from `CLAUDE.md`

- **`background_rms = 0.0075`, not 0.0315.** The 0.0315 in `CLAUDE.md` is
  `sigma_clipped_stats` over the whole cutout, which is mostly galaxy — the clip never
  finds sky. Measured instead from six widely separated 60×60 blank patches in the parent
  drizzle frame: mean −0.0020 to +0.0014, std 0.0070–0.0092. Using 0.0315 would down-weight
  the arcs into irrelevance. **`CLAUDE.md` should be corrected.**
- **No sky subtraction.** The crop's outer ring reads ~0.035 e/s, but that is the
  deflector's own Sersic wings, not sky: the radial profile falls smoothly 0.057 (2.3") →
  0.0054 (5.9") → 0.0007 (24"), and true blank sky is consistent with zero. Subtracting
  0.035 would remove real galaxy light and bias the lens-light `R_sersic` and `n_sersic`
  low.
- **Crop is `cutout[18:98, 18:98]`.** Puts the deflector centroid (57.48, 57.64) at
  (39.48, 39.64) against an array centre of (39.5, 39.5), and excludes the companion galaxy
  at ~(11, 11). The exclusion matters: `grep -rn "mask" gigalens/src` returns nothing, so
  the likelihood runs over every pixel unconditionally and an in-frame companion would be
  fit as part of the lens system.
- **`Ie` priors are order 100, not order 1.** `LensSimulator` multiplies by
  `conversion_factor = delta_pix² = 0.004225`, so a unit-`Ie` Sersic pair peaks at 0.083
  e/s against a data peak of 7.22. `verify.py` check 3 guards this.

## Known approximation

`exp_time` is a scalar, but `NCOMBINE = 3` and the WHT map shows coverage falling to ~half
depth (WHT 150 vs a median 308) within 2.5 px of the deflector. The noise is therefore
understated exactly where it matters most. `ForwardProbModel` accepts only a scalar; the
fix, if it proves to matter, is a per-pixel exposure map from the WHT extension, which
needs a `ForwardProbModel` subclass. State this in any writeup.

## Reporting

Quote R̂ alongside any parameter value — a `theta_E` without a convergence diagnostic is
not a measurement. Under the four-image interpretation, expect `theta_E` between the inner
clumps and the E blob, roughly 1.3–1.9". A posterior pinned at either prior edge means the
two radial scales are pulling the model apart; `diagnose.py` warns on this. The decisive
plot is the normalised residual: coherent arc-shaped structure above ~3σ means the model is
wrong regardless of R̂.
