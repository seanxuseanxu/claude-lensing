"""Certify the DESI234 r2_s2l2 undersampling ladder at a reference the module's own
self-check accepts, and settle two open questions from the first pass
(`undersampling_check.py` / `undersampling_report.txt`, NOT modified here):

(A) Is the ss=8 rung's 8.943 sigma (== the ss=16 reference's own subgrid self-error,
    4 sig figs) actually measuring ss=8's quadrature error, or is it just re-measuring
    reference noise because ss=8 is uncomfortably close to a not-yet-converged ss=16
    reference? Answered by DROPPING ss=8 and running the ladder (1, 2, 4) against a
    finer reference (ss_ref=32) that is checked for convergence itself.

(B) Is the ss=16 reference's self-error (10.6 sigma native / 8.9 sigma subgrid)
    dominated by one outlier draw in the 9-sample batch? The batch's delta_chi2 at
    ss=8 ran -9.5 to +1665 on a mean of +435 -- a heavy-tailed spread consistent with
    one bad draw. `diagnose_undersampling`'s reference_self_error is a SCALAR (max over
    the whole batch and all pixels), so isolating a per-sample number means calling it
    once per sample (bs=1 each) -- that is what this script does, in addition to the
    one bs=9 certified run.

Method, entirely through the library (`gigalens.jax.analysis.undersampling`), nothing
hand-rolled. Reuses `undersampling_check.py`'s exact batch construction (MAP, HMC-median,
7 HMC draws spread across chains/steps -- see that file's `spread_hmc_draws`) so the two
runs are on the same 9 parameter points and directly comparable.

Outputs (this directory, does not touch the first run's files):
  - `undersampling_certified.txt` -- this script's report.

    CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=claude/pipeline \
        python claude/DESI234/analysis/undersampling_certify.py
"""

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import jax  # noqa: E402
import diagnose  # noqa: E402
from gigalens.jax.analysis import diagnose_undersampling  # noqa: E402
from gigalens.jax.scene_simulator import SceneSimulator  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r2_s2l2"
OUT_DIR = Path(__file__).resolve().parent
LADDER = (1, 2, 4)
REF_SUPERSAMPLE = 32


def spread_hmc_draws(samples: np.ndarray, n: int) -> np.ndarray:
    """Identical to undersampling_check.py's version -- reproduced here (not imported)
    so this script has no import-time dependency on that file, per the task's
    instruction not to touch it."""
    n_chains, n_steps, _ = samples.shape
    chains = np.linspace(0, n_chains - 1, n, dtype=int)
    steps = np.linspace(0, n_steps - 1, n, dtype=int)
    return np.stack([samples[c, s] for c, s in zip(chains, steps)], axis=0)


def build_batch(rm, run_dir):
    """Same [MAP, HMC-median, 7 spread HMC draws] batch as undersampling_check.py."""
    map_ck = diagnose.load_checkpoint(run_dir, "map")
    z_map = np.asarray(map_ck["best_z"]).reshape(-1)

    hmc_ck = diagnose.load_checkpoint(run_dir, "hmc")
    hmc_samples = hmc_ck["samples"]
    z_hmc_median, _ = diagnose.point_estimate(run_dir, hmc_samples=hmc_samples)
    z_draws = spread_hmc_draws(hmc_samples, 7)

    z_batch = np.vstack([z_map[None, :], z_hmc_median[None, :], z_draws])
    labels = ["MAP", "HMC-median"] + [f"draw{i}" for i in range(7)]
    return z_batch, labels


def time_render(rm, supersample, n_reps=5):
    """Wall-clock a single forward render at a uniform `supersample`, JIT-warmed first.

    Uses the run's own simulator config (lstsq-mode amplitude solve happens inside
    the ProbModel/likelihood term, not here -- this times the raw SceneSimulator
    render, which is the part whose cost scales with supersample^2 and is what a
    fit's per-step wall time is dominated by).
    """
    cfg = dataclasses.replace(rm.sim_config, supersample=supersample)
    sim = SceneSimulator(rm.model, cfg, sees=rm.simulator._sees_ref)
    z = jax.numpy.asarray(rm.z_batch_for_timing)  # (bs, D) -- set by caller

    def render():
        params = rm.model.to_params(rm.model.bijector.forward(z))
        img = sim.simulate(params)
        return np.asarray(img)

    render()  # warm-up / JIT compile, not timed
    t0 = time.perf_counter()
    for _ in range(n_reps):
        render()
    jax.block_until_ready(render())
    t1 = time.perf_counter()
    return (t1 - t0) / n_reps


def main():
    rm = diagnose.rebuild_model(RUN)
    z_batch, labels = build_batch(rm, RUN)
    print(f"z batch: {z_batch.shape} ({', '.join(labels)})")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"DESI234 {RUN.name} -- undersampling CERTIFICATION "
         f"(ladder={LADDER}, reference_supersample={REF_SUPERSAMPLE})")
    emit("=" * 78)
    emit()
    emit("Same 9-sample batch as undersampling_check.py: "
         f"{', '.join(labels)}. This run does NOT touch that script's outputs.")
    emit()

    # ------------------------------------------------------------------ #
    # 1. Full 9-sample batch, ladder=(1,2,4), reference_supersample=32.  #
    # ------------------------------------------------------------------ #
    emit("-" * 78)
    emit("STEP 1: full 9-sample batch, ss_ref=32, ladder (1, 2, 4) -- ss=8 dropped")
    emit("-" * 78)
    t0 = time.perf_counter()
    try:
        reports = diagnose_undersampling(
            rm.prob, z_batch,
            supersample_ladder=LADDER,
            reference_supersample=REF_SUPERSAMPLE,
            check_reference=True,
        )
        full_batch_ok = True
        kept_labels = labels
    except Exception as e:  # OOM or similar
        emit(f"FULL 9-SAMPLE BATCH FAILED: {type(e).__name__}: {e}")
        emit("Falling back to a reduced batch (keeping MAP + HMC-median + 3 spread "
             "draws) at the same ss_ref=32 -- ss_ref is not reduced.")
        keep_idx = [0, 1, 3, 5, 7]  # MAP, HMC-median, and 3 of the 7 spread draws
        z_batch = z_batch[keep_idx]
        kept_labels = [labels[i] for i in keep_idx]
        emit(f"kept samples: {kept_labels}")
        reports = diagnose_undersampling(
            rm.prob, z_batch,
            supersample_ladder=LADDER,
            reference_supersample=REF_SUPERSAMPLE,
            check_reference=True,
        )
        full_batch_ok = False
    t1 = time.perf_counter()
    emit(f"[timing] step 1: {t1 - t0:.1f}s")
    assert len(reports) == 1, f"expected one imaging dataset, got {len(reports)}"
    report = reports[0]
    emit()
    emit(report.summary())
    emit()
    emit(f"reference_self_error (batch max, {'all 9' if full_batch_ok else 'reduced'} "
         f"samples): {report.reference_self_error}")
    emit(f"reference_converged: {report.reference_converged}")
    emit(f"recommended_supersample: {report.recommended_supersample}")
    emit()
    emit("Per-rung delta_chi2 (mean, spread over the batch) and max|d|/sigma:")
    for r in report.rungs:
        dchi = r.delta_chi2
        emit(f"  {r.label:<16} mean {np.mean(dchi):+10.2f}   "
             f"min {dchi.min():+10.2f}   max {dchi.max():+10.2f}   "
             f"max|d|/sigma {r.max_abs_delta_over_sigma:.3f}   "
             f"%pix>tol {100*r.frac_above_tolerance:.2f}%   -> "
             f"{'PASS' if r.passes else 'FAIL'}")

    emit()
    emit("Per-sample max|d|/sigma, each rung (from the batched run's per_sample_max):")
    header = "  sample        " + "".join(f"{f'ss={r.supersample}':>12}"
                                           for r in report.rungs if r.supersample is not None)
    emit(header)
    uniform_rungs = [r for r in report.rungs if r.supersample is not None]
    for i, lab in enumerate(kept_labels):
        row = f"  {lab:<12}  " + "".join(f"{r.per_sample_max[i]:12.3f}" for r in uniform_rungs)
        emit(row)

    # ------------------------------------------------------------------ #
    # 2. Per-sample reference self-error: is one draw the outlier?       #
    # ------------------------------------------------------------------ #
    emit()
    emit("-" * 78)
    emit("STEP 2: per-sample reference_self_error (bs=1 calls, same ladder/ss_ref) --")
    emit("        isolates which draw (if any) dominates the batched self-check above.")
    emit("-" * 78)
    per_sample_self_err = []
    t0 = time.perf_counter()
    for i, lab in enumerate(kept_labels):
        z_one = z_batch[i:i+1]
        rep_one = diagnose_undersampling(
            rm.prob, z_one,
            supersample_ladder=LADDER,
            reference_supersample=REF_SUPERSAMPLE,
            check_reference=True,
        )[0]
        per_sample_self_err.append(rep_one.reference_self_error)
        emit(f"  {lab:<12}  self_error = {rep_one.reference_self_error}   "
             f"converged={rep_one.reference_converged}")
    t1 = time.perf_counter()
    emit(f"[timing] step 2 ({len(kept_labels)} single-sample calls): {t1 - t0:.1f}s")

    # Identify the outlier by subgrid self-error (the convention every ladder rung
    # here uses, ss in {1,2,4} all render subgrid PSF except ss=1 which is native --
    # report both conventions' max across samples).
    subgrid_errs = [e.get("subgrid", float("nan")) for e in per_sample_self_err]
    native_errs = [e.get("native", float("nan")) for e in per_sample_self_err]
    worst_subgrid_i = int(np.nanargmax(subgrid_errs))
    worst_native_i = int(np.nanargmax(native_errs))
    emit()
    emit(f"Worst sample by subgrid self-error: {kept_labels[worst_subgrid_i]} "
         f"({subgrid_errs[worst_subgrid_i]:.4f} sigma); "
         f"by native self-error: {kept_labels[worst_native_i]} "
         f"({native_errs[worst_native_i]:.4f} sigma).")
    other_subgrid = [e for j, e in enumerate(subgrid_errs) if j != worst_subgrid_i]
    if other_subgrid and subgrid_errs[worst_subgrid_i] > 3 * np.nanmax(other_subgrid):
        emit(f"-> ONE OUTLIER DRAW ({kept_labels[worst_subgrid_i]}) dominates the "
             f"subgrid reference self-error: {subgrid_errs[worst_subgrid_i]:.4f} sigma "
             f"vs next-worst {np.nanmax(other_subgrid):.4f} sigma "
             f"(>3x gap).")
        outlier_present = True
        outlier_label = kept_labels[worst_subgrid_i]
        outlier_idx = worst_subgrid_i
    else:
        emit("-> No single draw dominates by a >3x margin; the self-error is spread "
             "across the batch, not concentrated in one outlier.")
        outlier_present = False
        outlier_label = None
        outlier_idx = None

    # ------------------------------------------------------------------ #
    # 2b. Ladder with and without the outlier, if one was found.         #
    # ------------------------------------------------------------------ #
    if outlier_present:
        emit()
        emit(f"Re-running the batch ladder WITHOUT {outlier_label} to see how much it "
             "moves the certified numbers:")
        keep = [j for j in range(len(kept_labels)) if j != outlier_idx]
        z_no_outlier = z_batch[keep]
        rep_no = diagnose_undersampling(
            rm.prob, z_no_outlier,
            supersample_ladder=LADDER,
            reference_supersample=REF_SUPERSAMPLE,
            check_reference=True,
        )[0]
        emit(rep_no.summary())
        emit(f"reference_self_error (without {outlier_label}): "
             f"{rep_no.reference_self_error}")
        emit(f"reference_converged (without {outlier_label}): {rep_no.reference_converged}")

    # ------------------------------------------------------------------ #
    # 3. Answer (A): does dropping ss=8 and using ss_ref=32 show the      #
    #    ss=8 rung was measuring reference noise, not its own error?      #
    # ------------------------------------------------------------------ #
    emit()
    emit("-" * 78)
    emit("QUESTION (A): was ss=8's 8.943 sigma just reference noise (ss=16 not "
         "converged), not ss=8's real quadrature error?")
    emit("-" * 78)
    ss4 = next(r for r in report.rungs if r.supersample == 4)
    emit(f"At ss_ref=32 (this run), ss=4's max|d|/sigma = {ss4.max_abs_delta_over_sigma:.3f} "
         f"(previous run at ss_ref=16 had ss=4 -> 0.988, ss=8 -> 8.943 sigma == the ss=16 "
         "reference's own subgrid self-error to 4 figures).")
    emit("Interpretation: an old ss=8 rung sits only one doubling below a reference that "
         "was itself not converged, so the ss=8-vs-ss=16 'quadrature error' and the "
         "ss=16-vs-ss=8 self-check were measuring the SAME pair of renders under a "
         "different label -- not independent quantities. Confirmed if the ss=4 number "
         "here (measured against a converged-or-more-converged ss=32 reference) stays "
         "close to the old ss=4-vs-ss=16 number (0.988), since ss=4 was two doublings "
         "below the old reference and therefore not subject to the same coincidence.")

    # ------------------------------------------------------------------ #
    # 4. PSF-convention-preservation note.                               #
    # ------------------------------------------------------------------ #
    emit()
    emit("-" * 78)
    emit("PSF CONVENTION: ss=2 -> ss=4 vs ss=2 -> ss=1")
    emit("-" * 78)
    conv = {r.supersample: r.convention for r in report.rungs if r.supersample is not None}
    emit(f"Rung conventions this run: {conv}")
    emit("Per the module docstring (gigalens.jax.analysis.undersampling): supersample=1 "
         "convolves the NATIVE-resolution kernel on the detector grid ('native'); "
         "supersample>=2 convolves a subgrid_kernel at supersampled resolution before "
         "pooling ('subgrid'). ss=2 and ss=4 are BOTH 'subgrid' -- raising supersample "
         "2 -> 4 changes only the quadrature (image is refined, integrated more finely) "
         "and does NOT change the PSF model. Going 2 -> 1 would cross the convention "
         "boundary and change the PSF model itself by convention_gap_max "
         f"({report.convention_gap_max if report.convention_gap_max is not None else 'n/a'} "
         "sigma in the first, ss_ref=16 run) -- not reducible by supersampling, a different "
         "kind of change than refining the quadrature.")

    # ------------------------------------------------------------------ #
    # 5. Wall-clock cost ratio: ss=4 vs ss=2 render.                     #
    # ------------------------------------------------------------------ #
    emit()
    emit("-" * 78)
    emit("WALL-CLOCK COST: single render, ss=4 vs ss=2 (same batch size as fit "
         "amplitude solves use internally is irrelevant here -- this times the raw "
         "SceneSimulator.simulate forward render, whose cost dominates fit wall time)")
    emit("-" * 78)
    rm.z_batch_for_timing = z_batch[:1]  # single sample
    t_ss2 = time_render(rm, 2, n_reps=12)
    t_ss4 = time_render(rm, 4, n_reps=12)
    ratio = t_ss4 / t_ss2
    emit(f"  bs=1 (single sample):")
    emit(f"    ss=2 render: {t_ss2*1e3:.2f} ms/call (mean of 12, post-JIT-warmup)")
    emit(f"    ss=4 render: {t_ss4*1e3:.2f} ms/call (mean of 12, post-JIT-warmup)")
    emit(f"    ratio ss=4/ss=2: {ratio:.2f}x  (naive quadrature-cost scaling predicts "
         "(4/2)^2 = 4.0x)")

    # Also at bs=50 -- this run's own HMC used 50 chains (hmc.npz samples shape
    # (50, 40000, 36)), so a per-step render during a real fit is batched at this
    # size; tiny-image kernel-launch overhead (seen at bs=1 above) may not hold here.
    n_chains_repr = 50
    rm.z_batch_for_timing = np.repeat(z_batch[:1], n_chains_repr, axis=0)
    t_ss2_b = time_render(rm, 2, n_reps=8)
    t_ss4_b = time_render(rm, 4, n_reps=8)
    ratio_b = t_ss4_b / t_ss2_b
    emit(f"  bs={n_chains_repr} (this run's HMC chain count -- representative of a real fit step):")
    emit(f"    ss=2 render: {t_ss2_b*1e3:.2f} ms/call (mean of 8, post-JIT-warmup)")
    emit(f"    ss=4 render: {t_ss4_b*1e3:.2f} ms/call (mean of 8, post-JIT-warmup)")
    emit(f"    ratio ss=4/ss=2: {ratio_b:.2f}x  (naive quadrature-cost scaling predicts "
         "(4/2)^2 = 4.0x)")

    (OUT_DIR / "undersampling_certified.txt").write_text("\n".join(lines) + "\n")
    emit()
    emit(f"Wrote {OUT_DIR / 'undersampling_certified.txt'}")


if __name__ == "__main__":
    main()
