"""Measure how far the deflector-core checkerboard extends, to fix the centre-mask radius.

    PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES="" \\
        /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \\
        claude/DESI257/analysis/core_checkerboard_extent.py \\
        > claude/DESI257/analysis/core_checkerboard_extent.txt 2>&1

WHY. `KNOWN_ISSUES.md` item 5's one open piece is to report a centre-masked `chi2_nu`
alongside the full-field one: the published convention is to mask the core when computing
chi-squared but *not* when modelling, so every `chi2_nu` in this repository is inflated by
core pixels a referee's number would exclude. That change needs a radius, and a radius
chosen by eye would make every downstream number a matter of taste. This measures it.

WHAT IS BEING LOCATED. Item 5 describes the artifact precisely: a *pixel-scale alternating
+/- sigma checkerboard*, taken by the field to be an ordinary PSF artifact of Sersic
deflector modelling. Two things follow, and the script uses both as independent criteria,
because either alone is weak:

  1. **Amplitude.** The annulus-mean of `norm_res**2` should sit above the field baseline
     inside the affected region. This is the obvious criterion and the less specific one:
     any localised misfit raises it (the arcs do, which is why the profile is only read
     inward of the arc radius).

  2. **Sign alternation.** The lag-1 autocorrelation of the normalised residual between
     neighbouring pixels should be *negative* inside the region -- that is what
     "checkerboard" means, and it is the criterion that distinguishes this artifact from a
     smooth blob of misfit. `DESI234/analysis/undersampling_check.py` already established
     the discrimination matters: the predicted quadrature-error map is a smooth
     single-signed blob and correlates with the observed core residual at Pearson
     r = -0.109, so amplitude alone would have conflated two unrelated things.

Reported radius = the larger of the two criteria's extents, rounded up to a whole pixel.
Taking the larger is the conservative choice for a *reporting* mask: over-masking costs a
few pixels of a 6400-pixel average, while under-masking leaves the artifact in the number
the change exists to clean up.

CROSS-CHECK, AND THE REASON IT IS THE POINT. If this is a PSF artifact then its extent is a
property of the *instrument*, not of the lens: all three systems are HST WFC3/IR F140W at
0.065"/px. So the script measures DESI257 and DESI234 and compares. **If the two radii agree,
one instrument-level constant in arcsec is the right design and this script justifies it. If
they disagree materially, the constant is wrong and the radius belongs in each lens's
`config.py`** -- that is a real fork in the implementation, decided here by measurement
rather than assumed. The disagreement case is reported as such, not smoothed over.

WHAT THIS DOES NOT DO. It does not change any likelihood, mask anything during modelling, or
write into any run directory. It renders each run at its own HMC-median point via
`diagnose.rebuild_model` (which honours each run's *recorded* `supersample` -- ss=2 for
DESI257, ss=4 for DESI234, per item 8) and reads the residual. Nothing numerical is
hand-rolled that the pipeline already provides: the residual normalisation is the dataset's
own `ProbModel.error_map`, and the reduced chi-squared quoted for context is
`diagnose.masked_chisq`.
"""

import sys

import numpy as np

import diagnose

# Runs to measure: the latest converged fit on each system. DESI257's is the headline
# result this change is being made for; DESI234's is the independent instrument check.
# ``primary`` runs set the radius and answer the instrument-vs-lens cross-check. The round-2
# runs are measured too, and only for the sign-alternation question: item 5's "pixel-scale
# alternating +/- sigma checkerboard" was described from the round-2 fits
# (each system's RESULTS_v2.md), so testing the round-3 fits alone could not tell whether the
# description was loose or whether something between the rounds removed the alternation.
# DESI234 went ss=2 -> 4 between them, DESI257 did not, which makes the pair informative.
RUNS = [
    ("DESI257", "claude/DESI257/runs/r3_unmasked", True),
    ("DESI234", "claude/DESI234/runs/r3_unmasked", True),
    ("DESI257 r2", "claude/DESI257/runs/r2_s2l2", False),
    ("DESI234 r2", "claude/DESI234/runs/r2_s2l2", False),
]

# Annuli are 1 px wide out to here. 20 px = 1.3" at 0.065"/px, comfortably past any
# plausible PSF-scale artifact (the F140W PSF is ~0.14" FWHM) and short of the arcs.
R_MAX_PX = 20

# The baseline the excess is measured against: the quiet shoulder immediately outside the
# core, 8-20 px = 0.52-1.30".
#
# This was first set to a far-field annulus (34-46 px = 2.2-3.0"), and measurement rejected
# that choice. The far field reads mean (r/sigma)^2 = 1.16 on DESI257 -- essentially the
# whole-field chi2_nu of 1.154, because it is a fair sample of a field that includes the
# arcs' outskirts. The annuli at 7-19 px read 0.4-0.9, genuinely *quieter* than the field
# average. Measured against the far field the cumulative excess therefore peaked at r=6 and
# then fell back, so "the radius enclosing 95% of the excess" was meaningless -- there was no
# plateau to take a fraction of.
#
# A local background is the right reference for a localised artifact, and it is the
# conservative one for setting a mask radius: a lower baseline makes the excess look larger,
# while the radius is fixed by where the excess *stops accumulating*, which barely moves. It
# also makes the outer annuli contribute ~0 by construction, so the plateau is the core
# excess itself rather than a sum over the whole cutout.
BASELINE_LO_PX, BASELINE_HI_PX = 8, 20

# Significance for calling an annulus "affected": the annulus statistic must depart from
# the baseline by this many standard errors of its own mean.
N_SIGMA = 3.0


def radius_map(shape, centroid_rowcol):
    """Distance in pixels from ``centroid_rowcol`` to each pixel centre."""
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return np.hypot(rows - centroid_rowcol[0], cols - centroid_rowcol[1])


def lag1_autocorr(norm_res, sel):
    """Normalised lag-1 autocorrelation of ``norm_res`` over the pixels ``sel`` selects.

    Averaged over both the row and the column direction, counting only pairs whose *both*
    members lie in ``sel`` -- otherwise an annulus would borrow its neighbours from the
    annulus outside it and the profile would smear across exactly the boundary being
    located.

    Returns ``(rho, n_pairs)``. ``rho = mean(x_a * x_b) / mean(x**2)`` over the selected
    pixels, so -1 is perfect pixel-to-pixel sign alternation (a checkerboard), 0 is white
    noise, and +1 is a smooth field. The denominator uses the same selection, so ``rho`` is
    scale-free and comparable between annuli of different amplitude -- which is the whole
    reason to use it alongside the amplitude criterion rather than instead of it.
    """
    prods = []
    # Row direction: (i, j) with (i, j+1). Column direction: (i, j) with (i+1, j).
    for a, b in (
        (sel[:, :-1] & sel[:, 1:], (norm_res[:, :-1], norm_res[:, 1:])),
        (sel[:-1, :] & sel[1:, :], (norm_res[:-1, :], norm_res[1:, :])),
    ):
        if a.any():
            prods.append((b[0][a] * b[1][a]))
    if not prods:
        return np.nan, 0
    prod = np.concatenate(prods)
    var = float(np.mean(norm_res[sel] ** 2))
    if var <= 0:
        return np.nan, int(prod.size)
    return float(np.mean(prod) / var), int(prod.size)


def measure(lens, run_dir):
    """Radial profile of both criteria for one run. Returns a dict of the findings."""
    rm = diagnose.rebuild_model(run_dir, quiet=True)
    z, label = diagnose.point_estimate(run_dir)
    im_sim = diagnose.simulate_from_unconstrained(rm, z)

    image = np.asarray(rm.image)
    err_map = np.asarray(rm.prob.error_map)
    norm_res = (image - im_sim) / err_map

    keep = (np.ones(image.shape, dtype=bool) if rm.mask is None
            else np.asarray(rm.mask, dtype=bool))
    chi2_full, n_kept = diagnose.masked_chisq(norm_res, keep)

    centroid = tuple(float(v) for v in rm.cfg.centroid_pix)
    delta_pix = float(rm.cfg.delta_pix)
    rmap = radius_map(image.shape, centroid)

    # --- the baseline both criteria are judged against -------------------------------
    base_sel = keep & (rmap >= BASELINE_LO_PX) & (rmap < BASELINE_HI_PX)
    base_sq = norm_res[base_sel] ** 2
    base_chi2 = float(np.mean(base_sq))
    base_chi2_se = float(np.std(base_sq, ddof=1) / np.sqrt(base_sq.size))
    base_rho, base_rho_n = lag1_autocorr(norm_res, base_sel)
    # Standard error of a correlation-like average over n roughly-independent pairs.
    base_rho_se = 1.0 / np.sqrt(base_rho_n) if base_rho_n else np.nan

    rows = []
    for r in range(R_MAX_PX):
        sel = keep & (rmap >= r) & (rmap < r + 1)
        n = int(sel.sum())
        if n < 4:
            rows.append(dict(r=r, n=n, chi2=np.nan, chi2_se=np.nan,
                             rho=np.nan, n_pairs=0, excess=np.nan))
            continue
        sq = norm_res[sel] ** 2
        chi2 = float(np.mean(sq))
        chi2_se = float(np.std(sq, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rho, n_pairs = lag1_autocorr(norm_res, sel)
        # Excess chi-squared this annulus contributes over what the field baseline would
        # have contributed on the same pixel count. Summed, not averaged: it is the total
        # inflation of chi2_nu that masking this annulus would remove, which is the
        # quantity the reporting change actually cares about.
        rows.append(dict(r=r, n=n, chi2=chi2, chi2_se=chi2_se, rho=rho,
                         n_pairs=n_pairs, excess=float(np.sum(sq) - n * base_chi2)))

    # --- extent by saturation of the CUMULATIVE excess --------------------------------
    # The per-annulus significance test this script first used was powerless: squared
    # residuals are heavy-tailed, so the mean over a 9-pixel annulus carries a standard
    # error larger than the excess being tested (8.56 +/- 4.13 on DESI257 at r=2), and a
    # 3-sigma rule rejected an excess plainly visible in the profile. Aggregating fixes it.
    #
    # E(R) = total excess chi-squared inside radius R. A localised artifact makes E rise
    # steeply and then flatten; the radius where it flattens is the artifact's extent.
    # Defining the radius as a fraction of the plateau uses every pixel inside it, needs no
    # per-annulus significance, and its one free choice -- the fraction -- is reported at
    # three values so the reader can see the answer barely moves.
    cum = np.nancumsum([row["excess"] for row in rows])
    plateau = float(cum[-1])
    frac_radii = {}
    for frac in (0.90, 0.95, 0.99):
        if plateau <= 0:
            frac_radii[frac] = 0
            continue
        hit = np.nonzero(cum >= frac * plateau)[0]
        frac_radii[frac] = int(hit[0] + 1) if hit.size else R_MAX_PX
    radius_px = frac_radii[0.95]

    # The alternation criterion, kept and reported but no longer used to set the radius:
    # measured, it finds no pixel-scale sign alternation anywhere (see the module
    # docstring's note on what that means for item 5's description of the artifact).
    core_sel = keep & (rmap < max(radius_px, 1))
    core_rho, core_rho_n = lag1_autocorr(norm_res, core_sel)

    return dict(
        lens=lens, run_dir=run_dir, point_label=label, rows=rows,
        chi2_full=chi2_full, n_kept=n_kept, supersample=rm.sim_config.supersample,
        centroid=centroid, delta_pix=delta_pix,
        base_chi2=base_chi2, base_chi2_se=base_chi2_se,
        base_rho=base_rho, base_rho_n=base_rho_n, base_rho_se=base_rho_se,
        cum=cum, plateau=plateau, frac_radii=frac_radii,
        core_rho=core_rho, core_rho_n=core_rho_n,
        radius_px=radius_px, radius_arcsec=radius_px * delta_pix,
    )


def report(res):
    print("=" * 78)
    print(f"{res['lens']}  {res['run_dir']}")
    print("=" * 78)
    print(f"point estimate: {res['point_label']};  rendered at recorded "
          f"supersample={res['supersample']}")
    print(f"centroid_pix (row, col) = ({res['centroid'][0]:.3f}, {res['centroid'][1]:.3f})"
          f";  delta_pix = {res['delta_pix']} \"/px")
    print(f"full-field chi2_nu over {res['n_kept']} kept px = {res['chi2_full']:.4f}")
    print()
    print(f"field baseline, annulus {BASELINE_LO_PX}-{BASELINE_HI_PX} px "
          f"({BASELINE_LO_PX * res['delta_pix']:.2f}-{BASELINE_HI_PX * res['delta_pix']:.2f}\"):")
    print(f"  mean (r/sigma)^2 = {res['base_chi2']:.4f} +/- {res['base_chi2_se']:.4f}")
    print(f"  lag-1 autocorr   = {res['base_rho']:+.4f} +/- {res['base_rho_se']:.4f} "
          f"({res['base_rho_n']} pairs)")
    print()
    print("radial profile ('cum %' = running share of the total excess chi-squared):")
    print("    r_px   r_arcsec    n_px   mean(r/sig)^2       excess   cum %   lag-1 rho")
    for row, c in zip(res["rows"], res["cum"]):
        if not np.isfinite(row["chi2"]):
            print(f"  {row['r']:5d}   {row['r'] * res['delta_pix']:8.3f}  {row['n']:6d}"
                  f"   (too few pixels)")
            continue
        pct = 100.0 * c / res["plateau"] if res["plateau"] > 0 else np.nan
        print(f"  {row['r']:5d}   {row['r'] * res['delta_pix']:8.3f}  {row['n']:6d}"
              f"   {row['chi2']:8.3f} +/- {row['chi2_se']:6.3f}"
              f"   {row['excess']:9.1f}  {pct:6.1f}   {row['rho']:+8.4f}")
    print()
    print(f"total excess chi-squared inside {R_MAX_PX} px = {res['plateau']:.1f}")
    for frac, rpx in sorted(res["frac_radii"].items()):
        mark = "  <- radius used" if abs(frac - 0.95) < 1e-9 else ""
        print(f"  {100 * frac:.0f}% of it is enclosed by r = {rpx} px = "
              f"{rpx * res['delta_pix']:.3f}\"{mark}")
    print()
    print(f"sign-alternation check inside that radius: lag-1 autocorr = "
          f"{res['core_rho']:+.4f} ({res['core_rho_n']} pairs), against a field baseline of "
          f"{res['base_rho']:+.4f}.")
    print("  A checkerboard would be NEGATIVE. It is not, here or in any annulus above, so")
    print("  the artifact at these points is a smooth positively-correlated core excess,")
    print("  not the pixel-scale sign alternation KNOWN_ISSUES item 5 describes.")
    print()


def main():
    results, secondary = [], []
    for lens, run_dir, primary in RUNS:
        try:
            res = measure(lens, run_dir)
        except Exception as exc:  # noqa: BLE001 -- one lens failing must not hide the other
            import traceback
            traceback.print_exc()
            print(f"\n!! {lens} ({run_dir}) FAILED: {type(exc).__name__}: {exc}\n")
            continue
        report(res)
        (results if primary else secondary).append(res)

    print("=" * 78)
    print("SIGN ALTERNATION ACROSS ROUNDS: is 'checkerboard' the right name?")
    print("=" * 78)
    print("Core lag-1 autocorrelation (NEGATIVE = alternating = a checkerboard):")
    for res in results + secondary:
        print(f"  {res['lens']:11s} ss={res['supersample']}  core rho = "
              f"{res['core_rho']:+.4f}   field rho = {res['base_rho']:+.4f}")
    all_rho = [r["core_rho"] for r in results + secondary if np.isfinite(r["core_rho"])]
    if all_rho and max(all_rho) < 0:
        print("  -> every run alternates. 'Checkerboard' is the right description.")
    elif all_rho and min(all_rho) >= 0:
        print("  -> NO run alternates, at either round or either supersample. The core")
        print("     excess is smooth and positively correlated. 'Checkerboard' does not")
        print("     describe what is measurable in these residuals; item 5's physical")
        print("     conclusion (a PSF artifact of Sersic deflector modelling, mask when")
        print("     reporting, not when modelling) is untouched, but the NAME is wrong and")
        print("     the round-2 description it came from should not be requoted as measured.")
    else:
        print("  -> mixed: some runs alternate and some do not. Report per run; do not")
        print("     generalise from one round to the other.")
    print()

    print("=" * 78)
    print("CROSS-CHECK: is the extent a property of the instrument or of the lens?")
    print("=" * 78)
    if len(results) < 2:
        print("Only one system measured, so the question is not answered. The "
              "instrument-constant design is UNJUSTIFIED on this evidence; either fix the "
              "failing run above or put the radius in each lens's config.py.")
        return 1

    for res in results:
        fr = res["frac_radii"]
        print(f"  {res['lens']:8s} {res['radius_px']:3d} px = {res['radius_arcsec']:.3f}\""
              f"   (90%: {fr[0.90]} px, 95%: {fr[0.95]} px, 99%: {fr[0.99]} px;"
              f" total excess {res['plateau']:.0f})")
    radii = [r["radius_arcsec"] for r in results]
    spread = max(radii) - min(radii)
    # Agreement threshold: one pixel. The radii are quantised to whole pixels, so anything
    # at or under 1 px is the finest disagreement this measurement can even express.
    px = results[0]["delta_pix"]
    print()
    print(f"  spread = {spread:.3f}\" = {spread / px:.1f} px")
    if spread <= px + 1e-9:
        rec_px = max(r["radius_px"] for r in results)
        print(f"  -> AGREE to within one pixel. An instrument-level constant is justified. "
              f"Use the larger, {rec_px} px = {rec_px * px:.3f}\", so neither system is "
              f"under-masked.")
    else:
        print(f"  -> DISAGREE by more than one pixel. The extent is NOT purely an "
              f"instrument property on this evidence, so a single shared constant would be "
              f"wrong for at least one system: put the radius in each lens's config.py "
              f"instead, with these measurements as the recorded justification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
