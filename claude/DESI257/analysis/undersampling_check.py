"""Is DESI257's `supersample=2` (config.py:369) leaving a quadrature error that matters at
the precision the headline number is quoted to (theta_E = 1.991 [1.989, 1.994]", a +/-0.002"
interval)?

Same instrument as DESI234 (HST WFC3/IR F140W, 0.065"/px), same pipeline, comparably cuspy
lens light -- and DESI234 measured `supersample=2` to FAIL at 3.13 sigma max|delta|/sigma,
delta_chi2 = -340 over 6400 px (`claude/DESI234/analysis/undersampling_certified.txt`,
round 2 -> round 3 moved DESI234 to `supersample=4`). This script asks the same question of
DESI257's `r3_unmasked` run, entirely through the library
(`gigalens.jax.analysis.undersampling`), nothing hand-rolled:

  - `diagnose.rebuild_model` reconstructs the exact `ProbModel` r3_unmasked was fitted with
    (its own configured supersample=2, its own mask -- here `mask_circles=None`, i.e.
    unmasked, per `runs/r3_unmasked/metadata.json`).
  - `diagnose_undersampling` renders a ladder of uniform supersamples (1, 2, 4) against a
    `reference_supersample=32` reference matched to each rung's PSF convention, self-certifies
    that reference by re-rendering it at half its supersample (16), and reports the per-pixel
    quadrature error in noise units plus its likelihood-level damage (`delta_chi2`) through the
    likelihood path.
  - DESI234's own ss_ref=16 reference did NOT self-certify (round 3's `undersampling_certify.py`
    had to go to ss_ref=32); we start at ss_ref=32 directly here rather than repeat that
    discovery, and report the self-check's PASS/FAIL plainly either way -- an uncertified
    reference means the numbers below are not certified, and that has to be said, not buried.
  - The parameter batch is the MAP point, the run's own HMC-median point estimate (the exact
    point `residuals.png` was rendered at -- so its delta/sigma map can be compared directly
    against that figure's observed residual), and 7 more HMC draws spread across chains and
    steps: quadrature error is parameter-dependent, so a single point cannot show the spread
    (module docstring).
  - The decisive test, same as DESI234's script: does the ss=2 rung's delta/sigma map at the
    HMC-median sample reproduce the observed residual -- same location, same sign pattern --
    in a 10x10 px box centred on the deflector core (`cfg.centroid_pix`, NOT DESI234's
    hardcoded box)? Quantified as the Pearson r between the two maps over that box, and a
    same-box peak-amplitude comparison.

COST CONTROL (CPU-only, login node, no SLURM/GPU): the batched ladder call is timed and run
FIRST. The per-sample `reference_self_error` isolation (one call per sample, as
`undersampling_certify.py` does on DESI234) is only run if the batched self-check FAILS to
certify AND the batched timing suggests it is affordable; otherwise this script stops after
the batched result and reports the timing.

Outputs (this directory):
  - `undersampling_report.txt`  -- report.summary() plus per-rung numbers, the self-check
    outcome, and the observed-vs-predicted core-box comparison.
  - `undersampling_ss2.png`     -- report.plot() for the ss=2 rung, HMC-median sample.
  - `undersampling_ss2_delta_over_sigma.npy` -- the full (bs, H, W) ss=2 delta/sigma stack.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
        python claude/DESI257/analysis/undersampling_check.py
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
BOX_HALF = 5  # 10x10 px box
LADDER = (1, 2, 4)
REF_SUPERSAMPLE = 32
COST_BUDGET_S = 30 * 60  # per-sample follow-up only attempted if this looks affordable


def spread_hmc_draws(samples: np.ndarray, n: int) -> np.ndarray:
    """``n`` unconstrained-z draws spread across chains and steps of an
    ``(n_chains, n_steps, n_params)`` HMC array -- not ``n`` draws from one chain or one
    step, which would under-sample exactly the spread this diagnostic is meant to catch."""
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
    hmc_samples = hmc_ck["samples"]  # (n_chains, n_steps, n_params)
    z_hmc_median, _ = diagnose.point_estimate(RUN, hmc_samples=hmc_samples)
    z_draws = spread_hmc_draws(hmc_samples, 7)

    # Batch order: [MAP, HMC median, 7 spread draws] -- 9 samples total. Index 1 is the
    # HMC-median point residuals.png was rendered at, so it is the one to correlate against
    # that figure's observed residual.
    z_batch = np.vstack([z_map[None, :], z_hmc_median[None, :], z_draws])
    HMC_MEDIAN_IDX = 1
    print(f"z batch: {z_batch.shape} (MAP, HMC-median, 7 spread HMC draws)")

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"DESI257 {RUN.name} -- undersampling diagnostic (supersample=2 configured)")
    emit("=" * 78)
    emit()
    emit(f"ladder={LADDER}, reference_supersample={REF_SUPERSAMPLE}, "
         f"num_pix={cfg.num_pix}, centroid_pix={cfg.centroid_pix}")
    emit()

    emit("-" * 78)
    emit("STEP 1: batched ladder, 9-sample batch (MAP, HMC-median, 7 spread draws)")
    emit("-" * 78)
    t0 = time.perf_counter()
    reports = diagnose_undersampling(
        rm.prob, z_batch,
        supersample_ladder=LADDER,
        reference_supersample=REF_SUPERSAMPLE,
        check_reference=True,
    )
    t1 = time.perf_counter()
    batched_wall = t1 - t0
    emit(f"[timing] batched ladder call: {batched_wall:.1f}s")
    assert len(reports) == 1, f"expected one imaging dataset, got {len(reports)}"
    report = reports[0]
    emit()
    emit(report.summary())
    emit()

    if not report.reference_converged:
        emit(f"*** REFERENCE (ss_ref={REF_SUPERSAMPLE}) NOT CONVERGED -- the numbers below "
             "are NOT CERTIFIED. ***")
    else:
        emit(f"Reference (ss_ref={REF_SUPERSAMPLE}) self-check: CONVERGED -- the numbers "
             "below ARE certified.")
    emit(f"    reference_self_error: {report.reference_self_error}")
    emit(f"    reference_converged: {report.reference_converged}")

    emit()
    emit("Per-rung delta_chi2 (chi2(rung) - chi2(matched reference)), mean and spread "
         "over the 9-sample batch, plus max|delta|/sigma:")
    for r in report.rungs:
        dchi = r.delta_chi2
        emit(f"  {r.label:<28} mean {np.mean(dchi):+10.2f}   "
             f"min {dchi.min():+10.2f}   max {dchi.max():+10.2f}   "
             f"max|d|/sigma {r.max_abs_delta_over_sigma:.3f}   "
             f"-> {'PASS' if r.passes else 'FAIL'}")

    emit()
    emit(f"convention_gap_max (native vs subgrid PSF convention, at the reference): "
         f"{report.convention_gap_max}")
    emit(f"recommended_supersample: {report.recommended_supersample}")

    # --- ss=2 rung: plot + save the delta/sigma stack ------------------------------------
    ss2_idx = next(i for i, r in enumerate(report.rungs) if r.supersample == 2)
    ss2 = report.rungs[ss2_idx]
    np.save(OUT_DIR / "undersampling_ss2_delta_over_sigma.npy", ss2.delta_over_sigma)
    emit()
    emit(f"ss=2 rung: convention={ss2.convention}, saved delta_over_sigma "
         f"{ss2.delta_over_sigma.shape} -> undersampling_ss2_delta_over_sigma.npy")

    fig, axes = report.plot(rung=ss2_idx, sample=HMC_MEDIAN_IDX)
    fig.savefig(OUT_DIR / "undersampling_ss2.png", dpi=150)
    emit(f"saved plot (ss=2, HMC-median sample) -> undersampling_ss2.png")

    # --- The decisive test: does ss=2's predicted quadrature error match the OBSERVED ----
    # --- residual at the deflector core, in both location and sign pattern? -------------
    image = np.asarray(rm.image)
    err_map = np.asarray(rm.prob.error_map)
    model_img = diagnose.simulate_from_unconstrained(rm, z_hmc_median)
    obs_norm_res = (image - model_img) / err_map  # same construction as residuals.png

    cy, cx = cfg.centroid_pix
    cy, cx = int(round(cy)), int(round(cx))
    r0, r1 = cy - BOX_HALF, cy + BOX_HALF
    c0, c1 = cx - BOX_HALF, cx + BOX_HALF
    box = (slice(r0, r1), slice(c0, c1))

    obs_box = obs_norm_res[box]
    pred_box = ss2.delta_over_sigma[HMC_MEDIAN_IDX][box]

    obs_flat = obs_box.ravel()
    pred_flat = pred_box.ravel()
    pearson_r = float(np.corrcoef(obs_flat, pred_flat)[0, 1])

    emit()
    emit("-" * 78)
    emit(f"DECISIVE TEST: deflector core box, {2 * BOX_HALF}x{2 * BOX_HALF} px centred at "
         f"pixel ({cy}, {cx}) = cfg.centroid_pix, HMC-median sample")
    emit("-" * 78)
    emit(f"  observed normalised-residual peak |value| in box : {np.abs(obs_box).max():.3f} sigma")
    emit(f"  ss=2 predicted delta/sigma peak |value| in box   : {np.abs(pred_box).max():.3f} sigma")
    emit(f"  Pearson r (observed vs ss=2-predicted, pixel-by-pixel over the box): "
         f"{pearson_r:+.4f}")
    emit()
    if pearson_r > 0.5:
        verdict = ("STRONG support: the ss=2 quadrature-error map reproduces the observed "
                   "residual's location and sign pattern.")
    elif pearson_r > 0.2:
        verdict = ("WEAK/PARTIAL support: some positive correlation, but not a clean "
                   "reproduction -- other contributions are likely also present.")
    else:
        verdict = ("HYPOTHESIS FAILS: the ss=2 quadrature-error map does NOT reproduce the "
                   "observed core residual; whatever residual exists there is not (mainly) "
                   "a supersampling artifact.")
    emit(f"  verdict: {verdict}")

    # ------------------------------------------------------------------ #
    # COST CONTROL: per-sample reference_self_error isolation is only    #
    # run if the batched self-check FAILED and the timing suggests it    #
    # is affordable (batched call took well under the ~30 min budget).   #
    # ------------------------------------------------------------------ #
    emit()
    emit("-" * 78)
    emit("COST CONTROL")
    emit("-" * 78)
    emit(f"Batched ladder wall-clock: {batched_wall:.1f}s (budget for a per-sample "
         f"follow-up: {COST_BUDGET_S}s).")
    if report.reference_converged:
        emit("Reference self-certified on the batched call -- no per-sample follow-up "
             "needed; the batched numbers above are the final, certified result.")
    else:
        # Rough cost model: certify.py's DESI234 experience was ~9x the batched call for
        # 9 single-sample calls (one full ladder+reference build each). Guard on that.
        projected = batched_wall * len(z_batch)
        emit(f"Reference did NOT self-certify. Projected cost of 9 single-sample follow-up "
             f"calls (same shape as batched, done once per sample): ~{projected:.1f}s.")
        if projected <= COST_BUDGET_S:
            emit("Projected cost is within budget -- running per-sample "
                 "reference_self_error isolation.")
            per_sample_self_err = []
            t0 = time.perf_counter()
            labels = ["MAP", "HMC-median"] + [f"draw{i}" for i in range(7)]
            for i, lab in enumerate(labels):
                z_one = z_batch[i:i + 1]
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
            emit(f"[timing] per-sample follow-up ({len(labels)} calls): {t1 - t0:.1f}s")
        else:
            emit("Projected cost EXCEEDS budget -- STOPPING here rather than grinding on "
                 "CPU. The reference is not certified; this script reports that plainly "
                 "and does not attempt the per-sample isolation. Getting a GPU allocation "
                 "to redo this at higher reference_supersample, or to run the per-sample "
                 "isolation, is a decision for the user.")

    (OUT_DIR / "undersampling_report.txt").write_text("\n".join(lines) + "\n")
    emit()
    emit(f"Wrote {OUT_DIR / 'undersampling_report.txt'}")


if __name__ == "__main__":
    main()
