# The lens-modelling pipeline

Lens-independent code. Nothing in this directory may name a specific system: everything
that distinguishes one lens from another lives in `claude/<LENS>/config.py`.

**`KNOWN_ISSUES.md`** in this directory tracks pipeline-level bugs, failure modes and
cross-lens findings — nine numbered items, of which four are still open. Per-lens results and
the measurements behind each item stay in `claude/<LENS>/README.md` / `RESULTS_*.md`; read
`KNOWN_ISSUES.md` for the cross-cutting inference and for anything that follows the tooling or
the dataset rather than one fit. **Its item numbers are permanent** — roughly thirty places in
`claude/` cite them as "item N", so an item is condensed when it closes, never renumbered or
deleted.

It fits through the GIGA-Lens **scene API** -- `LensModel` / `ImageData` / `ProbModel` and
the free functions `MAP` / `SVI` / `HMC`. gigalens is a pip-installed (editable) package,
so there is no `PYTHONPATH=gigalens/src` any more.

## Running it

```bash
export PYTHONPATH=claude/pipeline

# Pre-flight, ~75s for the variant you're about to fit (measured on DESI238, A100).
# Add --all-variants (~210s) after a change to the pipeline itself, not to one lens.
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/verify.py --lens DESI238 \
    --n-source 2 --n-lens-light 2

# Fit. Results land in claude/DESI238/runs/<label>/.
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py \
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

**Multiple GPUs are allowed.** `MAP`, `SVI` and the sampling stage (`HMC`, or `MAMS`/`MCLMC`
via `--sampler`) shard across `jax.device_count()` with `shard_map`, and each floors its
batch (`n_samples`, `n_vi`, `n_hmc`) to a device multiple.

## Spending the compute where it pays

Measured on DESI238 at 36 parameters (`claude/DESI238/RESULTS_v5.md`), and the ordering is
the opposite of the obvious one:

1. **Spend on the surrogate first.** MAP+SVI is a fixed cost that every subsequent draw pays
   interest on. 21 min on four GPUs — `--n-samples 4096 --map-steps 1500 --n-vi 2048
   --svi-steps 12000 --svi-peak-lr 1e-3` — took the SVI covariance condition to 2.3e+06 and
   the ELBO tail sd to 0.1, and the worst R̂ went 1.338 → 1.003. A fourfold chain-length
   increase had already been spent and had stopped helping.
2. **`--sampler mclmc`.** ~9× faster per draw than the default HMC here: 8× the draws in two
   thirds the wall time, at 1.7× the minimum ESS. The default stays `hmc` only so that old
   invocations reproduce.
3. **Then buy draws**, and buy enough that R̂ clears the threshold with room. 64 chains ×
   12,000 converged; 32 × 3,000 is a marginal budget, where 3 of 5 seeds fail R̂ < 1.01.

**Passing `svi_health`'s guard is a floor, not a target.** Two surrogates both passed and only
the one at ELBO tail sd 0.1 converged the fit. Read the tail sd as a quality number.

**To compare samplers, copy `map.npz` and `svi.npz` into each run directory** rather than
re-running those stages. Otherwise you compare sampler-plus-surrogate pairs and cannot
attribute the difference.

## Files

| File | Role |
|---|---|
| `lenses.py` | `LensConfig` / `SersicPrior` / `MassPrior`, and `load(name)`. The seam. |
| `config.py` (per lens) | Every number specific to one system. Not in this directory. |
| `data.py` | `load(cfg)` -> `(image, psf, sim_config)`. Endian cast, PSF normalisation, precision. `build_mask(cfg, shape)` -> keep-mask from `cfg.mask_circles`, or `None`. `build_prob_model(...)` -> the `ImageData`/`ProbModel` pair, built in one place so the fit, the pre-flight and the diagnosis cannot drift apart in what likelihood they mean. |
| `priors.py` | `build_model(cfg, ...)` -> `LensModel`. Owns all supports. |
| `run_fit.py` | MAP -> SVI -> sampling stage (`--sampler {hmc,mams,mclmc}`). The only writer. |
| `verify.py` | Five pre-flight checks, each pinned to a way the fit fails silently. Runs against one model variant (`--n-source`/`--n-lens-light`) by default; `--all-variants` sweeps all four. |
| `diagnose.py` | Convergence, residuals, pinning, label switching, the ESS floor, and four figures (`residuals`, `svi_loss`, `corner`, plus `ess_vs_draws` and `rank_plots`). Owns `rebuild_model`. Prints a per-stage timing line; on a 2M-draw run, ArviZ's rank-R̂/ESS is ~46% of the wall time -- see its module docstring. |
| `svi_health.py` | The SVI-covariance conditioning guard. The one thing the library lacks. |
| `dump_params.py`, `plot_*.py`, `clumps.py` | Read-only analysis of a finished run. |
| `tests/` | Standalone regression tests (no pytest in this env) -- Sersic `b_n`, `data.build_mask`, `verify.corner_report`, the masked-chi-squared formula, `make_cutout.py`/`make_psf.py`, the ESS floor and the two convergence plots. See `tests/README.md`. |

## The config seam

The division between `config.py` and `priors.py` is not stylistic, and getting it wrong
corrupts old results silently.

Checkpoints under `runs/` store **unconstrained** samples. A stored value becomes a
physical one through the model's bijector, and for a `Uniform` that bijector is
`low + (high - low) * sigmoid(z)` -- purely a function of the *support*. Change a bound and
every previous run reports different numbers, with nothing raised. Therefore:

- **Supports live in `priors.py`, keyed by a version.** A config cannot reach them.
- **Locations and scales live in the config.** A `LogNormal`'s median, a `Normal`'s sigma.
  These move prior mass without touching the bijector, so old checkpoints still decode
  correctly.

If a lens needs a different bound, add a prior version -- do not edit one in place, and do
not add a config field for it. Keep the superseded builder verbatim. `run_fit.py` records
both versions in `metadata.json`; `diagnose.rebuild_model` reads them back.

## Reading a finished run

Every read-only tool calls `diagnose.rebuild_model(run_dir)`, which reconstructs the exact
model a run was fitted with from its `metadata.json`. Do not reimplement it; six copies of
that logic is what the earlier refactor removed.

It **refuses any run whose `metadata.json` lacks `"api": "scene"`**. The port to the scene
API changed two things at once -- the flat-`z` column order (path-keyed and sorted, so lens
*light* now precedes lens *mass*) and the noise convention -- so a pre-port checkpoint
decodes here to plausible, wrong numbers. Read those runs at the commit that preceded the
port, or re-fit them.

### Traps
- **Changing a prior's *support* silently rewrites old runs' physical values**, because
  checkpoints store unconstrained samples and the bijector depends on the bounds. Add a
  version in `priors.py` and keep the old builder verbatim rather than editing one in place;
  `metadata.json` records the versions. Location-scale changes are safe — support changes
  are not. This is why a config may not set a bound.
  
## Things the API will do to you

- **`model.bijector.forward` takes a flat `(..., num_free_params)` array.** The retired
  `list(z.T)` list-of-columns form raises `TypeError` -- which is the good outcome; it used
  to be accepted and silently misinterpreted.
- **`z_param_names` are paths** (`planes/0/mass/0/theta_E`), not the old `lens.theta_E`.
  `z_labels` is the parallel display list (`lens/mass/host/theta_E`). Names given to
  `Plane`s and `Component`s are display-only and deliberately never enter the path, so
  renaming one cannot permute a saved chain.
- **HMC's return shape is not what it saves, and the samplers disagree with each other.**
  `HMC` returns `(n_steps, n_devices, n_chains_per_device, n_params)`; `MAMS` and `MCLMC`
  return `(n_chains, num_results, n_params)`, chains first. Both were determined empirically
  with tiny runs, not from the docs. `hmc.npz` stores `(n_chains, n_steps, n_params)`
  whatever the sampler, which is what `diagnose_convergence` and ArviZ require, and the file
  is named `hmc.npz` regardless because the analysis tools key on that name.
  `run_fit.normalize_hmc_samples` is the one place that conversion happens, and it checks
  each layout against the shape it must have rather than reshaping hopefully — a silent
  chain/step swap produces a plausible split-R̂ that says an unconverged run converged.
- **Reusing a `tfd.Distribution` object at two sites raises**, because under this API
  identity means sharing. Build a fresh distribution per site (which `priors.py` does), or
  say `shared(...)` and mean it.
- **R-hat numbers are not comparable with pre-port ones.** Convergence now comes from
  `gigalens.jax.analysis.diagnose_convergence` -- ArviZ's rank-normalized split R-hat and
  bulk/tail ESS, computed in unconstrained space. The classic PSRF this pipeline used to
  call over-reports badly on the heavy-tailed z-marginals a prior bound induces.
- **Suspected: `diagnose_convergence` mutates its input in place.** The DESI238 round-5 pass
  reports that it corrupts one column of the array returned by `diagnose.to_constrained`,
  making `dump_params.py`'s printed *mean/sd* inconsistent with its own percentiles.
  `diagnose.py`'s percentile table is computed before the call and is unaffected. **Prefer
  median/percentile values over `dump_params.py`'s mean/sd** until this is settled. Not
  independently reproduced, and it is in the read-only `nanolens` package — an upstream
  report, not something to patch from here.
- **The model warns about implausible priors at construction.** DESI238's deliberately wide
  `shear_width=0.1` puts ~14% of its prior mass above |shear| = 0.2 and trips a plausibility
  warning. That is a prompt, not a bound; the width is a documented modelling choice.

## Adding a lens

1. `mkdir claude/<LENS>/` and put the cutout and PSF in it.
2. Write `claude/<LENS>/config.py` defining `CONFIG = LensConfig(...)`. Copy DESI238's as a
   template, but **measure every number again**. In particular:
   - `background_rms` from blank patches in the *parent* frame. `sigma_clipped_stats` over
     a cutout that is mostly galaxy clips galaxy light, not sky, and comes out several
     times too large -- which down-weights the arcs into irrelevance.
   - `Ie_median` from a render, not from another lens. The simulator multiplies by
     `delta_pix**2`, so `Ie` is of order 10-100, not 1, and a value copied from a system
     with a different flux scale can be 50x wrong.
   - `crop` chosen to centre the deflector and exclude companions. If no crop can exclude a
     companion without also clipping the arcs, `mask_circles` names discs (in crop pixel
     coordinates) to exclude from the likelihood via `ImageData`'s `mask=`. Masking is a
     per-lens judgement -- `mask_circles=None` (the default) leaves everything in frame
     fitted as part of the system, unchanged from before.
3. Run `verify.py --lens <LENS>`. Check 1 and check 5 will report themselves *skipped*
   until the config supplies `companion_max` and `svi_guard_reference`. A skipped check is
   honest; do not weaken a check to make it pass.
4. Fit.
