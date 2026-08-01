# The lens-modelling pipeline

Lens-independent code. Nothing in this directory may name a specific system: everything
that distinguishes one lens from another lives in `claude/<LENS>/config.py`.

## Running it

```bash
export PYTHONPATH=gigalens/src:claude/pipeline

# Pre-flight, ~5 minutes. Do this before spending a GPU-hour.
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 python claude/pipeline/verify.py --lens DESI238

# Fit. Results land in claude/DESI238/runs/<label>/.
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py \
    --lens DESI238 --label baseline --n-source 2 --n-lens-light 2

# Diagnose and plot. These infer the lens from the run's metadata.json, so --lens is
# optional; pass it only if the run directory was moved out of claude/<LENS>/runs/.
python claude/pipeline/diagnose.py      claude/DESI238/runs/baseline
python claude/pipeline/dump_params.py   claude/DESI238/runs/baseline
python claude/pipeline/plot_overview.py claude/DESI238/runs/baseline
python claude/pipeline/plot_corner.py   claude/DESI238/runs/baseline
python claude/pipeline/plot_source.py   claude/DESI238/runs/baseline
python claude/pipeline/clumps.py        claude/DESI238/runs/baseline
```

**One GPU per process.** `gigalens.jax.inference`'s pmap path is broken under JAX 0.11
(sharding mismatch at `inference.py:55`), so `compat.require_single_device` refuses to
start with more than one device visible. Run several lenses as separate processes with
different `CUDA_VISIBLE_DEVICES`.

## Files

| File | Role |
|---|---|
| `lenses.py` | `LensConfig` / `SersicPrior` / `MassPrior`, and `load(name)`. The seam. |
| `config.py` (per lens) | Every number specific to one system. Not in this directory. |
| `data.py` | `load(cfg)` -> `(image, psf, sim_config)`. Endian cast, PSF normalisation. |
| `priors.py` | `build_prior(cfg, ...)`, `build_phys_model`, `param_names`. Owns all supports. |
| `run_fit.py` | MAP -> SVI -> HMC. The only writer. |
| `verify.py` | Six pre-flight checks, each pinned to a way the fit fails silently. |
| `diagnose.py` | Convergence, residuals, pinning, label switching. Owns `rebuild_model`. |
| `dump_params.py`, `plot_*.py`, `clumps.py` | Read-only analysis of a finished run. |
| `compat.py` | JAX 0.11 shims for upstream `gigalens.jax.inference`. Touches nothing else. |

## The config seam

The division between `config.py` and `priors.py` is not stylistic, and getting it wrong
corrupts old results silently.

Checkpoints under `runs/` store **unconstrained** samples. A stored value becomes a
physical one through the prior's bijector, and for a `Uniform` that bijector is
`low + (high - low) * sigmoid(z)` — purely a function of the *support*. Change a bound and
every previous run reports different numbers, with nothing raised. Therefore:

- **Supports live in `priors.py`, keyed by a version.** A config cannot reach them.
- **Locations and scales live in the config.** A `LogNormal`'s median, a `Normal`'s sigma.
  These move prior mass without touching the bijector, so old checkpoints still decode
  correctly.

If a lens needs a different bound, add a prior version — do not edit one in place, and do
not add a config field for it. Keep the superseded builder verbatim. `run_fit.py` records
both versions in `metadata.json`; `diagnose.rebuild_model` reads them back.

## Reading a finished run

Every read-only tool calls `diagnose.rebuild_model(run_dir)`, which reconstructs the exact
model a run was fitted with from its `metadata.json`. Its fallbacks are the values that
runs *predating each key* actually used — never today's defaults. Do not reimplement it;
six copies of that logic is what this refactor removed.

## Adding a lens

1. `mkdir claude/<LENS>/` and put the cutout and PSF in it.
2. Write `claude/<LENS>/config.py` defining `CONFIG = LensConfig(...)`. Copy DESI238's as a
   template, but **measure every number again**. In particular:
   - `background_rms` from blank patches in the *parent* frame. `sigma_clipped_stats` over
     a cutout that is mostly galaxy clips galaxy light, not sky, and comes out several
     times too large — which down-weights the arcs into irrelevance.
   - `Ie_median` from a render, not from another lens. `LensSimulator` multiplies by
     `delta_pix**2`, so `Ie` is of order 10–100, not 1, and a value copied from a system
     with a different flux scale can be 50× wrong.
   - `crop` chosen to centre the deflector and exclude companions. gigalens has no pixel
     masking, so anything in frame is fitted as part of the system.
3. Run `verify.py --lens <LENS>`. Check 1 and check 6 will report themselves *skipped*
   until the config supplies `companion_max` and `svi_guard_reference`. A skipped check is
   honest; do not weaken a check to make it pass.
4. Fit.
