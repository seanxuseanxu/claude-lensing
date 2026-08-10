"""Is the column-striped checkerboard in `runs/r2_s2l2/residuals.png` a PSF error, or
gigalens' `supersample>=2` subgrid-kernel quadrature error?

`r2_s2l2`'s normalised-residual panel shows a sign-alternating, pixel-scale checkerboard
right at the deflector core, reaching +/-5 sigma (see README.md's round-2 status note and
`RESULTS_v2.md`). The competing explanations are (a) a genuinely wrong PSF model and
(b) `gigalens.jax.analysis.undersampling`'s documented finding that the `subgrid_kernel`
convention (used whenever `supersample>=2`, which DESI234 runs) converges *slowly* and is
worst exactly where the model is cuspy -- and DESI234's lens-light bulge is n_sersic=5.27 at
R_sersic=0.43", about as cuspy as this pipeline models.

Method, entirely through the library (`gigalens.jax.analysis.undersampling`), nothing
hand-rolled:

  - `diagnose.rebuild_model` reconstructs the exact `ProbModel` r2_s2l2 was fitted with
    (forward mode, its own mask, its own configured supersample=2).
  - `diagnose_undersampling` renders a ladder of uniform supersamples (1, 2, 4, 8) against a
    supersample=16 reference matched to the SAME PSF convention as each rung, self-certifies
    that reference by re-rendering it at 8, and reports the per-pixel quadrature error in
    noise units plus its likelihood-level damage (`delta_chi2`) through the likelihood path.
  - The parameter batch is the MAP point, the run's own HMC-median point estimate (the exact
    point `residuals.png` was rendered at -- so its delta/sigma map can be compared directly
    against that figure's observed residual), and 7 more HMC draws spread across chains and
    steps: quadrature error is parameter-dependent, so a single point cannot show the spread
    (module docstring).
  - The decisive test: does the ss=2 rung's delta/sigma map at the HMC-median sample
    reproduce the observed checkerboard -- same location, same pixel-scale sign alternation,
    comparable amplitude -- in a 10x10 px box centred on the deflector core
    (`cfg.centroid_pix`)? Quantified as the Pearson r between the two maps over that box, and
    a same-box peak-amplitude comparison. This is reported plainly, including if it fails.

Outputs (this directory):
  - `undersampling_report.txt`  -- the report's own `.summary()` plus per-rung numbers and
    the observed-vs-predicted checkerboard comparison.
  - `undersampling_ss2.png`     -- `report.plot()` for the ss=2 rung.
  - `undersampling_ss2_delta_over_sigma.npy` -- the full (bs, H, W) ss=2 delta/sigma stack.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline \
        python claude/DESI234/analysis/undersampling_check.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import diagnose  # noqa: E402
from gigalens.jax.analysis import diagnose_undersampling  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r2_s2l2"
OUT_DIR = Path(__file__).resolve().parent
BOX_HALF = 5  # 10x10 px box


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

    reports = diagnose_undersampling(
        rm.prob, z_batch,
        supersample_ladder=(1, 2, 4, 8),
        reference_supersample=16,
        check_reference=True,
        # tolerance left at the library default (0.2 sigma/pixel, < 0.04 chi2 bias/pixel).
    )
    assert len(reports) == 1, f"expected one imaging dataset, got {len(reports)}"
    report = reports[0]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit(f"DESI234 {RUN.name} -- undersampling diagnostic (supersample=2 configured)")
    emit("=" * 78)
    emit()
    emit(report.summary())
    emit()

    if not report.reference_converged:
        emit("*** REFERENCE NOT CONVERGED -- the numbers below are NOT certified. ***")
        emit(f"    reference_self_error: {report.reference_self_error}")

    emit()
    emit("Per-rung delta_chi2 (chi2(rung) - chi2(matched reference)), mean and spread "
         "over the 9-sample batch:")
    for r in report.rungs:
        dchi = r.delta_chi2
        emit(f"  {r.label:<28} mean {np.mean(dchi):+10.2f}   "
             f"min {dchi.min():+10.2f}   max {dchi.max():+10.2f}   "
             f"max|d|/sigma {r.max_abs_delta_over_sigma:.3f}")

    emit()
    emit(f"reference_self_error (per PSF convention, sigma, self-check ss=8 vs ss=16): "
         f"{report.reference_self_error}")
    emit(f"reference_converged: {report.reference_converged}")
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
    # --- checkerboard at the deflector core, in both location and sign pattern? ----------
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
                   "checkerboard's location and sign pattern.")
    elif pearson_r > 0.2:
        verdict = ("WEAK/PARTIAL support: some positive correlation, but not a clean "
                   "reproduction -- other contributions (e.g. real PSF mismatch) are likely "
                   "also present.")
    else:
        verdict = ("HYPOTHESIS FAILS: the ss=2 quadrature-error map does NOT reproduce the "
                   "observed checkerboard; the residual is likely a genuine PSF/model error, "
                   "not a supersampling artifact.")
    emit(f"  verdict: {verdict}")

    (OUT_DIR / "undersampling_report.txt").write_text("\n".join(lines) + "\n")
    emit()
    emit(f"Wrote {OUT_DIR / 'undersampling_report.txt'}")


if __name__ == "__main__":
    main()
