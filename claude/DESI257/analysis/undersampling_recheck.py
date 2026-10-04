"""Follow-up to `undersampling_check.py`, requested after that script's ss_ref=32 reference
failed to self-certify (subgrid self-check 0.191 sigma, vs the library's 0.05 sigma
tolerance/margin threshold) and its ss=4 rung also failed the 0.2 sigma/px tolerance
(max|d|/sigma = 1.427). Two questions:

  (1) Does `reference_supersample=64` self-certify? ss^2 quadrature-cost scaling puts this at
      ~4x the ss=32 batched-ladder cost (~28.6s), i.e. a couple of minutes -- cheap enough to
      just try. If 64 ALSO fails to certify, that is reported and this script does NOT escalate
      further to 128: at that point the open question becomes whether the library's certify
      margin (tolerance/_REFERENCE_MARGIN = 0.05 sigma) is reachable on this system at all, a
      different question than DESI257's round-4 supersample choice.

  (2) Does ss=8 reach the 0.2 sigma/px tolerance? Ladder extended to (1, 2, 4, 8) against the
      ss_ref=64 reference (or ss_ref=32 if 64 fails -- either way ss=8 is at least one doubling
      below whichever reference is used, same margin ss=4 had against ss_ref=32).

Same 9-sample batch as `undersampling_check.py` (MAP, HMC-median, 7 spread HMC draws) -- exact
same construction, reproduced from the same checkpoints so this is a strict re-run at a
different ladder/reference, not a different measurement. Everything numerical stays inside
`gigalens.jax.analysis.undersampling.diagnose_undersampling`; nothing here is hand-rolled.

Appends to `undersampling_report.txt` (does not overwrite `undersampling_check.py`'s section,
which keeps the ss_ref=32 attempt and its failure to certify in the record).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
        python claude/DESI257/analysis/undersampling_recheck.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import diagnose  # noqa: E402
from gigalens.jax.analysis import diagnose_undersampling  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r3_unmasked"
OUT_DIR = Path(__file__).resolve().parent
REPORT_FILE = OUT_DIR / "undersampling_report.txt"
LADDER = (1, 2, 4, 8)
REF_SUPERSAMPLE = 64


def spread_hmc_draws(samples: np.ndarray, n: int) -> np.ndarray:
    """Identical to undersampling_check.py's version."""
    n_chains, n_steps, _ = samples.shape
    chains = np.linspace(0, n_chains - 1, n, dtype=int)
    steps = np.linspace(0, n_steps - 1, n, dtype=int)
    return np.stack([samples[c, s] for c, s in zip(chains, steps)], axis=0)


def main():
    rm = diagnose.rebuild_model(RUN)

    map_ck = diagnose.load_checkpoint(RUN, "map")
    z_map = np.asarray(map_ck["best_z"]).reshape(-1)
    hmc_ck = diagnose.load_checkpoint(RUN, "hmc")
    hmc_samples = hmc_ck["samples"]
    z_hmc_median, _ = diagnose.point_estimate(RUN, hmc_samples=hmc_samples)
    z_draws = spread_hmc_draws(hmc_samples, 7)
    z_batch = np.vstack([z_map[None, :], z_hmc_median[None, :], z_draws])
    labels = ["MAP", "HMC-median"] + [f"draw{i}" for i in range(7)]
    print(f"z batch: {z_batch.shape} ({', '.join(labels)}) -- same batch as undersampling_check.py")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit()
    emit("=" * 78)
    emit(f"DESI257 {RUN.name} -- undersampling RECHECK "
         f"(ladder={LADDER}, reference_supersample={REF_SUPERSAMPLE})")
    emit("=" * 78)
    emit()
    emit("Follow-up to the ss_ref=32 run above: does raising the reference to ss_ref=64 "
         "self-certify, and does extending the ladder to include ss=8 reach tolerance? "
         "Same 9-sample batch (MAP, HMC-median, 7 spread HMC draws).")
    emit()

    emit("-" * 78)
    emit(f"STEP 1: batched ladder {LADDER}, ss_ref={REF_SUPERSAMPLE}")
    emit("-" * 78)
    t0 = time.perf_counter()
    reports = diagnose_undersampling(
        rm.prob, z_batch,
        supersample_ladder=LADDER,
        reference_supersample=REF_SUPERSAMPLE,
        check_reference=True,
    )
    t1 = time.perf_counter()
    wall = t1 - t0
    emit(f"[timing] batched ladder call: {wall:.1f}s")
    assert len(reports) == 1, f"expected one imaging dataset, got {len(reports)}"
    report = reports[0]
    emit()
    emit(report.summary())
    emit()

    if not report.reference_converged:
        emit(f"*** REFERENCE (ss_ref={REF_SUPERSAMPLE}) NOT CONVERGED -- the numbers below "
             "are NOT CERTIFIED. Per the task, this script does NOT escalate further to "
             "ss_ref=128 -- if 64 does not certify, the achievability of the library's "
             "certify margin (0.2/_REFERENCE_MARGIN sigma) on this system is a separate "
             "question from round 4's supersample choice. ***")
    else:
        emit(f"Reference (ss_ref={REF_SUPERSAMPLE}) self-check: CONVERGED -- the numbers "
             "below ARE certified.")
    emit(f"    reference_self_error: {report.reference_self_error}")
    emit(f"    reference_converged: {report.reference_converged}")

    emit()
    emit("Per-rung delta_chi2 (mean, spread over the 9-sample batch) and max|delta|/sigma:")
    for r in report.rungs:
        dchi = r.delta_chi2
        emit(f"  {r.label:<28} mean {np.mean(dchi):+10.2f}   "
             f"min {dchi.min():+10.2f}   max {dchi.max():+10.2f}   "
             f"max|d|/sigma {r.max_abs_delta_over_sigma:.3f}   "
             f"%pix>tol {100*r.frac_above_tolerance:.2f}%   -> "
             f"{'PASS' if r.passes else 'FAIL'}")

    emit()
    emit(f"convention_gap_max (native vs subgrid PSF convention, at the reference): "
         f"{report.convention_gap_max}")
    emit(f"recommended_supersample: {report.recommended_supersample}")

    ss8 = next((r for r in report.rungs if r.supersample == 8), None)
    emit()
    emit("-" * 78)
    emit("ANSWER: does ss=8 reach the 0.2 sigma/px tolerance?")
    emit("-" * 78)
    if ss8 is not None:
        emit(f"ss=8: max|d|/sigma = {ss8.max_abs_delta_over_sigma:.3f}, "
             f"delta_chi2 mean {np.mean(ss8.delta_chi2):+.2f} "
             f"(spread {ss8.delta_chi2.min():+.2f}..{ss8.delta_chi2.max():+.2f}), "
             f"{100*ss8.frac_above_tolerance:.2f}% pixels above tolerance -> "
             f"{'PASSES' if ss8.passes else 'FAILS'} the 0.2 sigma/px tolerance.")
    else:
        emit("ss=8 rung not found in report.rungs -- unexpected, check ladder construction.")

    # ------------------------------------------------------------------ #
    # Ranking-trustworthiness question: does the (uncertified) ss_ref=32 #
    # ordering/magnitude hold up against the ss_ref=64 numbers?          #
    # ------------------------------------------------------------------ #
    emit()
    emit("-" * 78)
    emit("CROSS-CHECK: does the ss_ref=32 ranking (ss=1 > ss=2 > ss=4 in max|d|/sigma, by a "
         "wide margin at each step) reproduce at ss_ref=64?")
    emit("-" * 78)
    by_ss = {r.supersample: r for r in report.rungs if r.supersample is not None}
    for ss in (1, 2, 4, 8):
        if ss in by_ss:
            emit(f"  ss={ss}: max|d|/sigma = {by_ss[ss].max_abs_delta_over_sigma:.3f}")
    emit(f"reference_self_error at ss_ref={REF_SUPERSAMPLE} (subgrid): "
         f"{report.reference_self_error.get('subgrid', float('nan')):.4f} sigma -- this is "
         "the noise floor these rung numbers sit above.")

    (OUT_DIR).mkdir(exist_ok=True)
    with REPORT_FILE.open("a") as f:
        f.write("\n".join(lines) + "\n")
    emit()
    emit(f"Appended to {REPORT_FILE}")


if __name__ == "__main__":
    main()
