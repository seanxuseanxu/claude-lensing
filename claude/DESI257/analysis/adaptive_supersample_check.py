"""Validate gigalens' EXPERIMENTAL adaptive supersampler on DESI257 -- a MEASUREMENT only.

Does NOT wire the adaptive grid into the pipeline and does NOT run any fit.

BACKGROUND (measured, `undersampling_report.txt`, do not re-derive): DESI257's r3_unmasked
is fitted at uniform supersample=2 (config.py:369), and every affordable uniform rung FAILS
the library's 0.2 sigma/px max|delta|/sigma tolerance against a self-certified-FAILING
ss_ref=64 reference (ss=1: 5.290, ss=2: 3.697, ss=4: 1.497, ss=8: 0.439 -- all FAIL). Uniform
ss=16 would be the first candidate rung and costs 64x the ss=2 forward memory (memory ~
supersample**2, KNOWN_ISSUES.md item 9). The error is driven by a spatially COMPACT inner
lens-light component (R_sersic = 0.225" at 0.065"/px) -- exactly what the library's adaptive
per-pixel refinement (`gigalens.jax.experimental.adaptive_supersample`, EXPERIMENTAL) is for.

This script:
  1. Rebuilds the exact r3_unmasked model (`diagnose.rebuild_model`, its own recorded
     supersample=2) and swaps in an AdaptiveSceneSimulator (`sim_config.supersample=1`,
     the factor map IS the quadrature spec -- module requirement).
  2. Uses the SAME 9-sample batch as `undersampling_check.py` (MAP, HMC-median, 7 spread
     HMC draws) so the numbers sit directly next to the uniform ladder.
  3. Runs the library's own `compare_to_reference` at `reference_supersample=64` -- the
     same reference depth as `undersampling_recheck.py` -- in BOTH `psf_mode`s ("bin_first",
     which isolates the factor map's quadrature error, and "subgrid", which additionally
     carries the bin-first-vs-subgrid PSF convention gap the module docstring warns is NOT
     reducible by refining the grid). delta_chi2 is computed with the module's own
     `_masked_chi2` formula (its docstring, line 33: "delta_chi2 = chi2(quadrature) -
     chi2(reference)"), on the images `compare_to_reference` returns -- no comparison is
     hand-rolled.
  4. Sweeps 3 `snr_levels` settings (the library default; the default with factor-8 opted
     in per its own suggested opt-in; a more aggressive setting pushing more of the compact
     core to factor 8), reporting max|delta|/sigma, delta_chi2, AdaptiveGrid.n_points (the
     memory-cost driver) and its uniform-supersample equivalent, and wall-clock, per setting.
  5. Produces `plot_factor_map` for every setting -- the check on whether the grid refines
     the deflector core / bright images or blank sky.

Reference self-certification: NOT independently re-measured here. `undersampling_recheck.py`
already measured the ss_ref=64 reference's own self-check (vs ss_ref=32) via the identical
SceneSimulator construction `compare_to_reference` uses internally: native (bin-first)
0.018 sigma, subgrid 0.072 sigma, against the library's tolerance/_REFERENCE_MARGIN =
0.2/4.0 = 0.05 sigma pass margin -- NOT CONVERGED (subgrid). The numbers below inherit that:
they are reported, not certified.

Outputs (this directory):
  - `adaptive_supersample_report.txt` -- full per-setting numbers.
  - `adaptive_factor_map_<setting>.png` -- `plot_factor_map` per setting.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
        python claude/DESI257/analysis/adaptive_supersample_check.py
"""

import dataclasses
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import diagnose  # noqa: E402
from gigalens.jax.experimental.adaptive_supersample import (  # noqa: E402
    AdaptiveGrid, AdaptiveSceneSimulator, compare_to_reference, plot_factor_map,
    DEFAULT_SNR_LEVELS,
)
from gigalens.jax.analysis.undersampling import _masked_chi2  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r3_unmasked"
OUT_DIR = Path(__file__).resolve().parent
REF_SUPERSAMPLE = 64  # same depth as undersampling_recheck.py, for direct comparability

# Reference self-check already measured (undersampling_recheck.py / undersampling_report.txt)
# via the identical SceneSimulator construction compare_to_reference uses internally.
REFERENCE_SELF_ERROR = {"native": 0.01774030092126682, "subgrid": 0.07181432006694986}
REFERENCE_MARGIN_THRESHOLD = 0.2 / 4.0  # tolerance / _REFERENCE_MARGIN, from the library
REFERENCE_CONVERGED = all(e <= REFERENCE_MARGIN_THRESHOLD for e in REFERENCE_SELF_ERROR.values())

SETTINGS = [
    dict(name="library_default", snr_levels=DEFAULT_SNR_LEVELS, smooth=1, dilate=2,
         desc="DEFAULT_SNR_LEVELS as shipped (calibrated on the bundled demo system, not DESI257)"),
    dict(name="factor8_opt_in", snr_levels=((14.0, 8.0),) + DEFAULT_SNR_LEVELS, smooth=1, dilate=2,
         desc="default + factor-8 opted in above SNR 14, per the module's own suggested opt-in"),
    dict(name="aggressive_core",
         snr_levels=((5.0, 8.0), (3.0, 4.0), (1.5, 2.0), (0.5, 1.0), (-np.inf, 0.5)),
         smooth=1, dilate=3,
         desc="lower thresholds pushing more of the compact core to factor 8, wider dilate"),
]


def spread_hmc_draws(samples: np.ndarray, n: int) -> np.ndarray:
    """Same construction as undersampling_check.py -- n draws spread across chains/steps."""
    n_chains, n_steps, _ = samples.shape
    chains = np.linspace(0, n_chains - 1, n, dtype=int)
    steps = np.linspace(0, n_steps - 1, n, dtype=int)
    return np.stack([samples[c, s] for c, s in zip(chains, steps)], axis=0)


def main():
    rm = diagnose.rebuild_model(RUN)
    cfg = rm.cfg

    map_ck = diagnose.load_checkpoint(RUN, "map")
    z_map = np.asarray(map_ck["best_z"]).reshape(-1)

    hmc_ck = diagnose.load_checkpoint(RUN, "hmc")
    hmc_samples = hmc_ck["samples"]
    z_hmc_median, _ = diagnose.point_estimate(RUN, hmc_samples=hmc_samples)
    z_draws = spread_hmc_draws(hmc_samples, 7)

    z_batch = np.vstack([z_map[None, :], z_hmc_median[None, :], z_draws])
    HMC_MEDIAN_IDX = 1
    print(f"z batch: {z_batch.shape} (MAP, HMC-median, 7 spread HMC draws)")

    import jax.numpy as jnp
    params = rm.model.constrained(jnp.asarray(z_batch))

    image = np.asarray(rm.image)
    error_map = np.asarray(rm.prob.error_map)
    mask_all = np.ones_like(image, dtype=bool)
    H, W = image.shape

    cfg1 = dataclasses.replace(rm.sim_config, supersample=1)
    sees = rm.simulator._sees_ref

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("DESI257 r3_unmasked -- EXPERIMENTAL adaptive supersampler validation")
    emit("=" * 78)
    emit()
    emit(f"num_pix={cfg.num_pix} ({H}x{W}), configured supersample={rm.sim_config.supersample}, "
         f"reference_supersample={REF_SUPERSAMPLE}")
    emit()
    emit("Reference self-certification (reused from undersampling_recheck.py -- same "
         "SceneSimulator construction compare_to_reference uses internally, NOT "
         "re-measured here):")
    emit(f"  reference_self_error = {REFERENCE_SELF_ERROR}  "
         f"(threshold tolerance/_REFERENCE_MARGIN = {REFERENCE_MARGIN_THRESHOLD})")
    emit(f"  reference_converged = {REFERENCE_CONVERGED}  "
         "-- ALL numbers below are reported, NOT CERTIFIED.")
    emit()
    emit("Uniform ladder for comparison (ss_ref=64, from undersampling_report.txt):")
    emit("  ss=1  max|d|/sigma 5.290  delta_chi2 mean +720.5  -> FAIL")
    emit("  ss=2  max|d|/sigma 3.697  delta_chi2 mean -221.1  -> FAIL  (configured rung)")
    emit("  ss=4  max|d|/sigma 1.497  delta_chi2 mean  -65.0  -> FAIL")
    emit("  ss=8  max|d|/sigma 0.439  delta_chi2 mean  -34.3  -> FAIL  (0.2 sigma/px tolerance)")
    emit()

    uniform_equivalent = lambda n_points: float(np.sqrt(n_points / (H * W)))

    results = []
    for setting in SETTINGS:
        name = setting["name"]
        emit("-" * 78)
        emit(f"SETTING: {name} -- {setting['desc']}")
        emit(f"  snr_levels={setting['snr_levels']}, smooth={setting['smooth']}, "
             f"dilate={setting['dilate']}")
        emit("-" * 78)

        t0 = time.perf_counter()
        grid = AdaptiveGrid.from_snr(image, error_map, snr_levels=setting["snr_levels"],
                                      smooth=setting["smooth"], dilate=setting["dilate"])
        t1 = time.perf_counter()
        emit(f"  grid: {grid!r}")
        emit(f"  [timing] grid construction: {t1 - t0:.3f}s")

        sim = AdaptiveSceneSimulator(rm.model, cfg1, grid, sees=sees)

        # Time the adaptive render alone (the memory/compute-cost number), separately
        # from the reference builds inside compare_to_reference.
        t0 = time.perf_counter()
        adaptive_img = np.asarray(sim.simulate(params))
        t1 = time.perf_counter()
        adaptive_render_wall = t1 - t0
        emit(f"  [timing] adaptive render (9-sample batch, incl. jit compile): "
             f"{adaptive_render_wall:.2f}s")

        n_points = grid.n_points
        n_points_native = H * W
        ueq = uniform_equivalent(n_points)
        emit(f"  n_points = {n_points}  ({n_points / n_points_native:.3f} per native pixel; "
             f"native supersample=1 uniform cost is {n_points_native}); "
             f"equivalent uniform supersample = {ueq:.2f} "
             f"(uniform ss=2 costs {n_points_native * 4}, ss=8 costs {n_points_native * 64}, "
             f"ss=16 costs {n_points_native * 256})")

        setting_result = dict(name=name, grid=grid, n_points=n_points, ueq=ueq,
                               adaptive_render_wall=adaptive_render_wall)

        for psf_mode in ("bin_first", "subgrid"):
            t0 = time.perf_counter()
            rep = compare_to_reference(sim, params, reference_supersample=REF_SUPERSAMPLE,
                                        error_map=error_map, psf_mode=psf_mode)
            t1 = time.perf_counter()
            dchi2 = (_masked_chi2(rep["adaptive"], image, error_map, mask_all)
                     - _masked_chi2(rep["reference"], image, error_map, mask_all))
            max_ds = rep["max_abs_delta_over_sigma"]
            argmax = np.unravel_index(np.argmax(np.abs(rep["delta_over_sigma"])),
                                       rep["delta_over_sigma"].shape)
            passes = max_ds <= 0.2
            emit(f"  [{psf_mode:>9}] max|delta|/sigma = {max_ds:.3f} at sample/pix {argmax}, "
                 f"delta_chi2 mean {np.mean(dchi2):+.2f}  min {dchi2.min():+.2f}  "
                 f"max {dchi2.max():+.2f}  -> {'PASS' if passes else 'FAIL'} "
                 f"(0.2 sigma/px tolerance)  [timing: {t1 - t0:.2f}s incl. reference render]")
            setting_result[psf_mode] = dict(max_abs_delta_over_sigma=max_ds,
                                             delta_chi2=dchi2, passes=passes,
                                             wall=t1 - t0)
        results.append(setting_result)

        fig, ax = plot_factor_map(grid, title=f"DESI257 r3_unmasked -- {name}")
        png = OUT_DIR / f"adaptive_factor_map_{name}.png"
        fig.savefig(png, dpi=150)
        emit(f"  saved factor map -> {png.name}")
        emit()

    emit("=" * 78)
    emit("SUMMARY")
    emit("=" * 78)
    emit(f"{'setting':<18}{'n_points':>10}{'uniform_eq':>12}{'bin_first max|d|/s':>20}"
         f"{'subgrid max|d|/s':>18}{'bin_first dchi2':>17}{'render_s':>10}")
    for r in results:
        emit(f"{r['name']:<18}{r['n_points']:>10}{r['ueq']:>12.2f}"
             f"{r['bin_first']['max_abs_delta_over_sigma']:>20.3f}"
             f"{r['subgrid']['max_abs_delta_over_sigma']:>18.3f}"
             f"{np.mean(r['bin_first']['delta_chi2']):>17.1f}"
             f"{r['adaptive_render_wall']:>10.2f}")
    emit()
    emit("Module-stated limitations (adaptive_supersample.py docstring, NOT re-derived here):")
    emit("  - EXPERIMENTAL: 'moves faster than the core scene API; treat its interface as "
         "less stable' (how-to admonition).")
    emit("  - PSF convention: bin-first (scatter to native grid, convolve with the native "
         "kernel) -- NOT the subgrid_kernel supersampled convolution the stock ss>=2 "
         "pipeline uses. The two conventions can differ by several sigma at a cuspy peak "
         "(module docstring: ~4 sigma on the bundled demo); this project's own uniform "
         "ladder independently measured a 3.75 sigma convention gap on DESI257 at ss_ref=64. "
         "Reported above as the 'subgrid' psf_mode delta, which conflates this gap with the "
         "factor map's own quadrature error and is NOT reducible by refining the grid.")
    emit("  - The factor map is frozen from the DATA at construction, so it is only accurate "
         "for models whose flux stays within `dilate` native pixels of where the data shows "
         "flux. Early burn-in proposals displaced further than that, or model features "
         "(e.g. a predicted faint counter-image) the data does not show at all, are rendered "
         "at the coarse tier and their likelihood/gradient carries the corresponding "
         "quadrature error -- compare_to_reference only certifies the exact params tested.")
    emit("  - DEFAULT_SNR_LEVELS are 'STARTING POINTS calibrated on the bundled demo system'; "
         "the module explicitly says to re-derive per-system with compare_to_reference, which "
         "is what the setting sweep above does.")
    emit("  - The map is static (frozen, not model-derived) specifically to keep the "
         "likelihood's shape fixed under jit and smooth in the parameters -- a real "
         "constraint on any HMC use, not just a performance choice.")

    (OUT_DIR / "adaptive_supersample_report.txt").write_text("\n".join(lines) + "\n")
    emit()
    emit(f"Wrote {OUT_DIR / 'adaptive_supersample_report.txt'}")


if __name__ == "__main__":
    main()
