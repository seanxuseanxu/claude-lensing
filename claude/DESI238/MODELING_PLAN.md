# Modelling DESI238 with GIGA-Lens (JAX)

## Context

Goal: fit a strong-lens model to DESI238 (HST WFC3/IR F140W, `claude/DESI238/`) using the
vendored GIGA-Lens JAX backend, producing a posterior over lens mass and light parameters
via the package's MAP → SVI → HMC sequence.

The cutout is a genuine lens. Subtracting an azimuthal median reveals a bright
early-type deflector plus counter-image structure: two arcs (NE and SW), a detached
blob to the E, and a fourth fainter clump — a plausible quad or a fold pair plus a
companion image. Measured from the residual:

| Clump | r (arcsec) | PA (deg) | residual flux (e/s) |
|---|---|---|---|
| SW arc | 1.06 | 224 | 18.8 |
| E blob | 2.11 | 88 | 24.0 |
| W clump | 1.64 | 278 | 12.5 |
| N arc | 1.25 | 339 | 8.7 |

**Decision (user): all four clumps are to be treated as counter-images.** The model must
reproduce the E blob at 2.11" as well as the three inner clumps, so the `theta_E` prior is
widened to span the full 1.06–2.11" range rather than centred on the inner three. See
"Consequences of the four-image choice" below — this is the main risk in the plan and
needs an explicit check.

Deflector centroid is at cutout pixel **(row 57.48, col 57.64)** — offset from the array
centre (59.5, 59.5) by (-0.131", -0.121"). Light second moments give
`e1 = -0.017, e2 = +0.036` (nearly round, |e| = 0.04).

**Four findings from a read-only dry run that shape this plan.** These were established by
running throwaway `python -c` snippets; no repo file was modified.

1. **`gigalens.jax.inference.ModellingSequence` does not run under JAX 0.11.** Three
   independent breakages, all confirmed by traceback (see Step 1).
2. **`background_rms` in CLAUDE.md (0.0315) is wrong for this purpose.** It is measured
   over the whole cutout and is dominated by galaxy light. Blank sky in the parent
   drizzle frame gives **0.0072**, consistent across six widely separated patches
   (0.00694–0.00876). Using 0.0315 would down-weight the arcs into irrelevance.
3. **There is a companion galaxy** at cutout pixel ~(11, 11), 66 px from the deflector.
   `grep -rn "mask" gigalens/src` returns **nothing** — the likelihood runs over every
   pixel unconditionally. The companion must be removed by cropping, not masked.
4. **Four GPUs buy little for this problem size.** Single-GPU timings at 80×80,
   supersample=2: bs=256 → 0.018 s/step, bs=512 → 0.033, bs=1024 → 0.064. The model is
   too small to saturate one H200. Use the GPUs for *independent* runs, not for
   splitting one batch.

Note: GPUs 4–6 are in use by other users. Only **0–3** are free, and
`CUDA_VISIBLE_DEVICES` is already pinned to those four.

## Approach

Work in a new directory `claude/DESI238/model/`. Do not modify `gigalens/src` — apply the
JAX-compat fixes as a local shim so the vendored copy stays clean against upstream.

### Step 1 — `compat.py`: make the JAX backend runnable

Three confirmed failures, with the exact fix for each:

| Location | Symptom | Fix |
|---|---|---|
| `jax/inference.py:98` | `AttributeError: module 'jax' has no attribute 'tree_map'` | `jax.tree_map = jax.tree_util.tree_map` |
| `jax/inference.py:68` | `TypeError: Only scalar arrays can be converted to Python scalars; got arr.ndim=1` | drop `keepdims=True` in the `float(jnp.nanmin(...))` progress string |
| `jax/inference.py:55` | `ValueError: Sharding passed to jit does not match ... P(None, axis)` vs `P(axis)` | `jnp.stack(jnp.split(p, D, axis=0))` instead of `jnp.array(jnp.split(...))`; also unwrap the `@jit`-decorated `log_prob` (`prob.log_prob.__wrapped__`) before `pmap`, since nesting jit inside pmap is what triggers the mismatch |

Verified: with the shim, `pmap` runs clean at 0.010 s/step for bs=512 across 4 devices.

Given finding 4, **default to the single-device path** (plain `jit`, one large batch) and
keep the `pmap` fix available but unused. Simpler, and the timings show it is not the
bottleneck.

Also set `NCCL_NVLS_ENABLE=0` — the node emits NVLS/Fabric Manager errors otherwise
(harmless here, but they flood stderr).

### Step 2 — `data.py`: load and condition the data

- **Crop to 80×80**: `cutout[17:97, 18:98]`. This recentres on the true deflector
  centroid and puts the companion outside the frame. Verified: companion excluded, all
  four clumps retained, crop max 7.22 e/s.
- **Cast to native endianness** — both arrays are big-endian and JAX rejects them:
  `np.ascontiguousarray(arr.astype(np.float32))`.
- **Normalise the PSF** — it sums to 1.1445. Divide by its sum. It is 27×27, odd, and
  already centred (centroid within 0.12 px of centre), so no recentring.
- **Sky subtraction**: subtract a constant ~0.0072–0.012 measured from the crop's outer
  region. Record the value used; the fit is sensitive to it.
- Config: `delta_pix=0.065`, `num_pix=80`, `supersample=2`, `exp_time=1197.699`,
  `background_rms=0.0072`.

**Caveat to state in the writeup, not to fix**: `NCOMBINE=3` and the WHT map show
coverage falling to ~half depth (WHT 150 vs median 308) within 2.5 px of the deflector.
A scalar `exp_time` therefore understates the noise in the very region that matters most.
`ForwardProbModel` accepts only a scalar, so this is a known first-order approximation.
If it turns out to matter, the fix is a per-pixel exposure map built from WHT, which
needs a subclass of `ForwardProbModel`.

### Step 3 — `priors.py`: priors matched to *this* image

The demo notebook's priors are tuned to a synthetic image with a completely different
flux scale (`Ie ~ 500`, sky 0.2). DESI238 peaks at 7.2 e/s. Copying them would place the
prior mass in the wrong region entirely. Set instead, from the measurements above:

- **EPL**: `theta_E ~ LogNormal(log 1.5, 0.35)` — centred between the inner clumps and the
  E blob, with the width widened to put real prior mass across 1.0–2.2" as the
  four-image interpretation requires. `gamma ~ TruncatedNormal(2, 0.25, 1, 3)`;
  `e1, e2 ~ Normal(0, 0.15)`; `center_x, center_y ~ Normal(0, 0.05)` (arcsec, relative to
  the crop centre, which is now the deflector).
- **Shear**: `gamma1, gamma2 ~ Normal(0, 0.1)` — widened from the demo's 0.05. Producing a
  four-image configuration whose images span 1.06–2.11" in radius requires substantial
  angular structure; a tight shear prior would fight the data.
- **Lens light** (SersicEllipse): `R_sersic ~ LogNormal(log 0.8, 0.3)`;
  `n_sersic ~ Uniform(2, 6)`; `e1, e2 ~ TruncatedNormal(0, 0.1, -0.3, 0.3)` (light is
  nearly round); `Ie ~ LogNormal(log 2.0, 0.5)` — **rescaled to the e/s flux level**.
- **Source light** (SersicEllipse): `R_sersic ~ LogNormal(log 0.25, 0.2)`;
  `n_sersic ~ Uniform(0.5, 4)`; `center_x, center_y ~ Normal(0, 0.3)`;
  `Ie ~ LogNormal(log 0.5, 0.5)`.

Structure must be `[lens, lens_light, source_light]` as nested
`JointDistributionSequential[JointDistributionNamed]`. This gives **22 free parameters**
(confirmed by running `bij.inverse` on a prior sample).

Model: `PhysicalModel([epl.EPL(50), shear.Shear()], [SersicEllipse(use_lstsq=False)],
[SersicEllipse(use_lstsq=False)])` with `ForwardProbModel`.

#### Consequences of the four-image choice

Requiring one source to produce images at both 1.06" and 2.11" is a strong demand on a
single elliptical deflector. Three things follow, and the plan should confront them
rather than paper over them:

- **A single Sersic source may not suffice.** A smooth elliptical potential produces
  quad images at comparable radii; a 2:1 radial spread usually needs either high
  ellipticity, strong external shear, or an extended/multi-component source. Hence the
  widened `e1, e2` and shear priors.
- **Escalation path if residuals at the E blob stay large.** In order: (a) add a second
  Sersic source component, (b) swap the source for `Shapelets(n_max=6, use_lstsq=True)`
  with `BackwardProbModel` — the shapelets demo is the worked example, and
  `use_lstsq=True` requires `BackwardProbModel`, (c) add a second mass component (`SIS`)
  at the E blob's position if it is a satellite deflector rather than an image.
- **Falsification test.** Fit the inner three clumps *only* as a comparison run (narrow
  `theta_E ~ LogNormal(log 1.3, 0.2)`), then compare reduced chi-squared and the E-blob
  residual between the two. If the four-image model reproduces the blob at comparable
  chi-squared elsewhere, the interpretation is supported. If it can only fit the blob by
  degrading the inner arcs, that is evidence the blob is an interloper — report it.
  This is cheap: MAP+SVI is seconds-to-minutes here, and the 4 GPUs make the two runs
  concurrent.

### Step 4 — `run_fit.py`: the three-stage sequence

Reuse `ModellingSequence` (`gigalens/src/gigalens/jax/inference.py`) with the shim applied.

1. **MAP** — `n_samples=1024`, `num_steps=350`,
   `optax.chain(scale_by_adam(), scale_by_schedule(polynomial_schedule(init_value=-1e-2,
   end_value=-1e-2/3, power=0.5, transition_steps=350)))`. Note the **negative** learning
   rates: these compose with `scale_by_adam` to descend. At 0.064 s/step this is ~25 s.
   Pick the best start by `prob_model.log_prob(LensSimulator(..., bs=1024), map)[0]` and
   `argmax`.
2. **SVI** — `SVI(best, opt, n_vi=1000, num_steps=1500)`. **Argument order differs from
   the TF backend**: JAX is `SVI(start, optimizer, ...)`. Schedule
   `polynomial_schedule(init_value=-1e-6, end_value=-3e-3, power=2, transition_steps=300)`.
3. **HMC** — `HMC(qz, num_burnin_steps=250, num_results=750, n_hmc=50)`. Returns shape
   `(dev_cnt, n_hmc//dev_cnt, num_results, n_params)`.

Persist MAP/SVI/HMC outputs to `.npz` after each stage so a failure in HMC does not cost
the earlier stages.

**Use the four GPUs here**, one job per GPU via `CUDA_VISIBLE_DEVICES`:

| GPU | Job |
|---|---|
| 0, 1 | four-image model, two different seeds (tests multimodality) |
| 2 | inner-three-clump comparison run (the falsification test above) |
| 3 | escalation variant — two-component or shapelets source |

Multimodality is a real risk here: a wide `theta_E` prior spanning 1.0–2.2" can support
distinct solutions that fit different subsets of the clumps. Independent seeds are the
way to detect that; one batch on one device cannot.

### Step 5 — `diagnose.py`: validation

- **Convergence**: `tfp.mcmc.potential_scale_reduction` (R̂) and
  `effective_sample_size(x, cross_chain_dims=1)`, mapped over `bij.forward(samples)`.
  Report R̂ per parameter; flag anything > 1.01.
- **Residuals**: `(observed - model) / err_map`, plotted on a symmetric scale. This is
  the decisive plot — if the arcs are not reproduced, the model is wrong regardless of R̂.
- **Reduced chi-squared**: the second element of the `log_prob` tuple.
- **Cross-check `theta_E`** against the measured clump radii. Under the four-image
  interpretation, expect the posterior to land between the inner clumps and the E blob,
  roughly 1.3–1.9". A value pinned at either end of the prior means the model is being
  pulled apart by the two radial scales — report that rather than quoting the number.
- **Predicted image positions**: compare the model's image configuration against the four
  measured clump positions (r, PA in the Context table). Reproducing radii but not
  position angles is a specific, diagnosable failure.
- Corner plot of the constrained posterior via `bij.forward`.

**Remember `bij.forward`.** MAP, SVI and HMC all return *unconstrained* values. Anything
interpreted physically must be transformed first.

## Files

All new, under `claude/DESI238/model/`:

| File | Role |
|---|---|
| `compat.py` | JAX 0.11 shims for `gigalens.jax.inference` |
| `data.py` | load/crop/normalise; returns image, PSF, `SimulatorConfig` |
| `priors.py` | prior construction + `PhysicalModel` |
| `run_fit.py` | MAP → SVI → HMC, checkpointing each stage |
| `diagnose.py` | R̂/ESS, residual maps, corner plot |

`gigalens/src/**` is **not** modified.

## Verification

Run with `PYTHONPATH=gigalens/src` and `NCCL_NVLS_ENABLE=0` from the repo root.

1. **Shim** — `MAP(n_samples=64, num_steps=25)` completes without `AttributeError`,
   `TypeError`, or the sharding `ValueError`.
2. **Data** — assert crop shape `(80, 80)`, dtype little-endian float32, PSF sums to
   1.0 ± 1e-6, companion absent (max in a 20×20 corner < 0.1 e/s).
3. **Simulator sanity** — `LensSimulator(..., bs=1).simulate(prior.sample(1))` returns
   `(80, 80)`, finite, with a peak within ~1 dex of the data's 7.2 e/s. If it is orders
   of magnitude off, the `Ie` priors are still mis-scaled — fix before fitting.
4. **Fit quality** — reduced chi-squared near 1. The dry run reached chi2 ≈ 168 after
   only 30 untuned steps, so a converged fit should be far lower; a value stuck in the
   hundreds means the priors or the sky level are wrong.
5. **Residuals** — no coherent arc-shaped structure left above ~3 sigma. **Including at
   the E blob**, which under the four-image interpretation the model is required to
   reproduce. A large residual there is the signal to escalate (Step 3).
6. **Convergence** — R̂ < 1.01 for all 22 parameters; ESS in the hundreds.
7. **Multimodality** — the independent-seed chains agree on `theta_E` to within their
   quoted uncertainties. If they do not, report it rather than picking the best one.
8. **Falsification** — compare four-image vs inner-three fits on reduced chi-squared and
   on the E-blob residual specifically. Report both outcomes honestly; "the blob is an
   interloper" is a legitimate and useful result, not a failure of the run.

## Reporting

State plainly in the writeup: the sky level adopted and how it was measured (this
supersedes the 0.0315 in CLAUDE.md and that file should be corrected); the scalar
`exp_time` approximation and the WHT coverage variation it ignores; and whether the
four-image interpretation held up. Quote reduced chi-squared and R̂ alongside any
parameter values — a `theta_E` without a convergence diagnostic is not a measurement.
