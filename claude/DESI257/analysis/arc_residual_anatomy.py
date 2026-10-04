"""Are DESI257's paired +/-4 sigma residual lobes displaced/mis-shaped model arcs, or
genuine source-plane structure?

`residuals.png` (`runs/r3_unmasked`, chi2_nu = 1.15 over all 6400 px) shows adjacent
opposite-sign pairs -- blue immediately next to red -- at three places: the E arc, the N
blob, and the SW object (the counter-image round 3 settled). A displaced or mis-shaped
model arc produces exactly this pattern with no source-plane structure required: shift a
bright ridge by a fraction of a pixel and the residual is negative on the side the ridge
moved away from and positive on the side it moved toward, everywhere along the ridge. That
is a different physical statement from unmodelled source clumps, with different
consequences for `theta_E`, and it must be settled before any future round adds source
freedom (e.g. a pixelated source) that would absorb a displacement without ever diagnosing
it. (The pixel-scale checkerboard at the deflector core is a separate, already-explained
issue -- `KNOWN_ISSUES.md` item 5, an ordinary Sersic-deflector PSF artifact, measured not
to be a supersampling artifact, Pearson r = +0.107 -- and is excluded from every region
below.)

This is a direct port of `claude/DESI234/analysis/arc_residual_anatomy.py`; read that
file's docstring for the full derivation. Summary: a rigid shift of the model by
``(s_row, s_col)`` pixels changes the model image, to first order, by

    M(row - s_row, col - s_col) ~= M(row, col) - s_row dM/drow - s_col dM/dcol

so the residual such a shift would produce is ``-s_row dM/drow - s_col dM/dcol``. Dividing
by the error map turns both the observed normalised residual ``r = (data-model)/err`` and
this prediction into the same sigma units:

    r ~= a_row G_row + a_col G_col,   G = (dM/d*)/err,   a = -s

Fitting ``r`` against ``(G_row, G_col)`` by OLS over a region's pixels gives ``(a_row,
a_col)`` directly in pixels; ``(s_row, s_col) = -(a_row, a_col)``. The fraction of the
region's local chi-squared this fit removes is the verdict: large means displacement,
small means the residual is uncorrelated with the model's own shape (substructure). A
second, independent tangential/radial regression about the deflector centre is also run,
since a locally-radial push is what a theta_E offset looks like, distinct from a global
translation.

What is system-specific here, changed against DESI234's script: the run
(`runs/r3_unmasked`), and THREE region selections instead of one, because DESI257's dipole
appears at three physically distinct places and the task is to check whether they are the
same displacement or independent effects.

Region geometry -- READ THE WARNING IN README.md's "Lens configuration" TABLE BEFORE
TRUSTING THESE NUMBERS: the PAs below are ARRAY-SPACE angles, measured from the +row axis
toward +col in the crop, NOT sky PA (config.py's CLUMPS comment and README.md both flag
this as "the single most misreadable field in config.py"). Converting them through the WCS
the standard way would put the markers on empty sky.

  (a) E arc:    r in [1.76, 1.92]", PA in [40, 110] deg -- README.md's own measured box for
                the secure, continuous E arc ridge, held literally (same convention as
                DESI234's script holding its S-arc box literally).
  (b) N blob:   r in [1.40, 1.70]", PA in [320, 360] u [0, 20] deg (wraps 0) -- centred on
                the measured clump at r=1.55", PA=350 (config.py CLUMPS), widened to a
                compact box that does not reach the E arc (PA >= 40) or cross deep into the
                documented 15-35 deg flux gap between the N blob and the E arc.
  (c) SW object: r in [2.40, 2.76]", PA in [195, 240] deg -- centred on the measured object
                at r=2.58", PA=218 (config.py comment / README.md), which config.py's
                FOREGROUND_CENTER derivation independently pins to r=2.603", PA=215.2 deg
                from the flux centroid; the box is widened to bracket both readings.

Every region is additionally cut to pixels above SB_SIGMA=3 in the model (so pixels the
model predicts as empty sky, with ~zero gradient, do not dilute the regression) and inside
the run's own likelihood mask (`rm.mask`, which is None/all-True for `r3_unmasked` --
nothing is masked in round 3). Pixel counts and peak model significance for each region are
printed and written to the report so the selections can be checked against "does this
actually contain the bright feature" before trusting anything downstream.

Everything physical -- the model render, the error map, the mask, the point estimate --
comes from `diagnose.rebuild_model`/`diagnose.point_estimate`/`diagnose.masked_chisq`,
which now read each run's recorded supersample (`ss=2` for all of DESI257's runs) from
`metadata.json`. Nothing here reimplements a likelihood, a bijector or the model render.

    CUDA_VISIBLE_DEVICES="" python claude/DESI257/analysis/arc_residual_anatomy.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import diagnose  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r3_unmasked"
OUT_DIR = Path(__file__).resolve().parent

SB_SIGMA = 3.0

# name, r_min, r_max (arcsec), pa_min, pa_max (deg, array-space; pa_min > pa_max means the
# box wraps through 360/0).
REGIONS = [
    ("E arc", 1.76, 1.92, 40.0, 110.0),
    ("N blob", 1.40, 1.70, 320.0, 20.0),
    ("SW object", 2.40, 2.76, 195.0, 240.0),
]


def region_selection(model_img, err_map, mask, center, delta_pix, r_min, r_max, pa_min, pa_max):
    """Boolean array: pixels in the given annulus/PA box, unmasked, above SB_SIGMA.

    PA convention matches `clumps.clump_positions`: PA = atan2(col - col0, row - row0),
    i.e. measured from the +row axis toward +col (array-space, NOT sky PA).
    """
    ny, nx = model_img.shape
    rows, cols = np.indices((ny, nx))
    row0, col0 = center
    dr, dc = rows - row0, cols - col0
    r_arcsec = np.hypot(dr, dc) * delta_pix
    pa_deg = np.degrees(np.arctan2(dc, dr)) % 360.0

    in_annulus = (r_arcsec >= r_min) & (r_arcsec <= r_max)
    if pa_min <= pa_max:
        in_pa = (pa_deg >= pa_min) & (pa_deg <= pa_max)
    else:
        in_pa = (pa_deg >= pa_min) | (pa_deg <= pa_max)
    bright = model_img > SB_SIGMA * err_map
    return in_annulus & in_pa & bright & np.asarray(mask, dtype=bool), pa_deg


def fit_shift(norm_res, g_a, g_b, sel):
    """OLS fit of ``norm_res[sel] ~ coef_a * g_a[sel] + coef_b * g_b[sel]``.

    Returns ``(coef_a, coef_b, predicted_full, chi2_before, chi2_after, n)``.
    """
    A = np.stack([g_a[sel], g_b[sel]], axis=-1)
    b = norm_res[sel]
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    coef_a, coef_b = coeffs
    predicted_full = coef_a * g_a + coef_b * g_b
    chi2_before, n = diagnose.masked_chisq(norm_res, sel)
    leftover_full = norm_res - predicted_full
    chi2_after, _ = diagnose.masked_chisq(leftover_full, sel)
    return coef_a, coef_b, predicted_full, chi2_before, chi2_after, n


def analyse_region(name, r_min, r_max, pa_min, pa_max, model_img, err_map, mask, norm_res,
                    center, delta_pix):
    sel, pa_deg = region_selection(model_img, err_map, mask, center, delta_pix,
                                    r_min, r_max, pa_min, pa_max)
    n_sel = int(sel.sum())
    if n_sel < 10:
        raise RuntimeError(
            f"[{name}] only {n_sel} pixels selected -- annulus/PA/SB cuts are too strict "
            f"to fit anything. Check center={center}, delta_pix={delta_pix}."
        )
    peak_sigma = float((model_img[sel] / err_map[sel]).max())

    grad_row, grad_col = np.gradient(model_img)
    g_row, g_col = grad_row / err_map, grad_col / err_map

    a_row, a_col, pred_rc, chi2_before, chi2_after_rc, n = fit_shift(norm_res, g_row, g_col, sel)
    s_row, s_col = -a_row, -a_col
    shift_pix = float(np.hypot(s_row, s_col))
    shift_arcsec = shift_pix * delta_pix
    frac_explained_rc = 1.0 - chi2_after_rc / chi2_before

    pa_rad = np.radians(pa_deg)
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
    g_rad = g_row * cos_pa + g_col * sin_pa
    g_tan = -g_row * sin_pa + g_col * cos_pa

    a_tan, a_rad, pred_tr, chi2_before_tr, chi2_after_tr, _ = fit_shift(norm_res, g_tan, g_rad, sel)
    s_tan, s_rad = -a_tan, -a_rad
    frac_explained_tr = 1.0 - chi2_after_tr / chi2_before_tr

    assert abs(chi2_before - chi2_before_tr) < 1e-8, "chi2_before mismatch between fits"

    best_frac = max(frac_explained_rc, frac_explained_tr)
    if best_frac > 0.5:
        verdict = "DISPLACEMENT"
    elif best_frac < 0.2:
        verdict = "SUBSTRUCTURE"
    else:
        verdict = "BOTH"

    return dict(
        name=name, r_min=r_min, r_max=r_max, pa_min=pa_min, pa_max=pa_max,
        sel=sel, n=n, peak_sigma=peak_sigma,
        a_row=a_row, a_col=a_col, s_row=s_row, s_col=s_col,
        shift_pix=shift_pix, shift_arcsec=shift_arcsec,
        chi2_before=chi2_before, chi2_after_rc=chi2_after_rc, frac_explained_rc=frac_explained_rc,
        a_tan=a_tan, a_rad=a_rad, s_tan=s_tan, s_rad=s_rad,
        chi2_after_tr=chi2_after_tr, frac_explained_tr=frac_explained_tr,
        best_frac=best_frac, verdict=verdict, pred_rc=pred_rc,
    )


def main():
    rm = diagnose.rebuild_model(RUN)
    cfg = rm.cfg
    delta_pix = cfg.delta_pix
    center = cfg.centroid_pix  # (row0, col0), pixels, within the crop

    image = np.asarray(rm.image)
    err_map = np.asarray(rm.prob.error_map)
    mask = (np.ones(image.shape, dtype=bool) if rm.mask is None
            else np.asarray(rm.mask, dtype=bool))

    z_unc, label = diagnose.point_estimate(RUN)
    print(f"Point estimate: {label}")
    model_img = diagnose.simulate_from_unconstrained(rm, z_unc)
    norm_res = (image - model_img) / err_map

    results = []
    for (name, r_min, r_max, pa_min, pa_max) in REGIONS:
        res = analyse_region(name, r_min, r_max, pa_min, pa_max, model_img, err_map, mask,
                              norm_res, center, delta_pix)
        results.append(res)

    lines = []
    lines.append(f"Run: {RUN}")
    lines.append(f"Point estimate: {label}")
    lines.append(f"Deflector centre (row, col): {center}, delta_pix={delta_pix}\"/px")
    lines.append("")

    for res in results:
        lines.append(f"=== Region: {res['name']} ===")
        lines.append(f"  selection: r in [{res['r_min']}, {res['r_max']}]\", "
                      f"PA in [{res['pa_min']}, {res['pa_max']}] deg (array-space), "
                      f"model > {SB_SIGMA} sigma, mask-kept")
        lines.append(f"  {res['n']} pixels selected, peak model significance "
                      f"{res['peak_sigma']:.1f} sigma")
        lines.append(f"  local chi2_nu before any correction: {res['chi2_before']:.4f} "
                      f"(over {res['n']} px)")
        lines.append("")
        lines.append("  --- Global (row, col) translation fit ---")
        lines.append(f"    fit coefficients: a_row={res['a_row']:+.4f}, "
                      f"a_col={res['a_col']:+.4f} (sigma per px-shift)")
        lines.append(f"    implied shift: s_row={res['s_row']:+.4f} px, "
                      f"s_col={res['s_col']:+.4f} px")
        lines.append(f"    |shift| = {res['shift_pix']:.4f} px = {res['shift_arcsec']:.4f} arcsec")
        lines.append(f"    local chi2_nu after removing this shift: {res['chi2_after_rc']:.4f}")
        lines.append(f"    fraction of local chi2 explained: {res['frac_explained_rc']:.4f}")
        lines.append("")
        lines.append("  --- Tangential/radial fit about the deflector centre ---")
        lines.append(f"    fit coefficients: a_tan={res['a_tan']:+.4f}, "
                      f"a_rad={res['a_rad']:+.4f} (sigma per px-shift)")
        lines.append(f"    implied shift: s_tan={res['s_tan']:+.4f} px, "
                      f"s_rad={res['s_rad']:+.4f} px")
        lines.append(f"    |shift| = {np.hypot(res['s_tan'], res['s_rad']):.4f} px "
                      f"= {np.hypot(res['s_tan'], res['s_rad']) * delta_pix:.4f} arcsec")
        lines.append(f"    local chi2_nu after removing this component: "
                      f"{res['chi2_after_tr']:.4f}")
        lines.append(f"    fraction of local chi2 explained: {res['frac_explained_tr']:.4f}")
        lines.append("")
        lines.append(f"  VERDICT ({res['name']}): {res['verdict']} -- best-fit shift model "
                      f"removes {res['best_frac']:.0%} of this region's local chi-squared.")
        lines.append("")

    lines.append("=== Cross-region comparison (row, col) shifts, pixels ===")
    lines.append(f"{'region':<12}{'s_row':>10}{'s_col':>10}{'|s| (px)':>12}{'|s| (arcsec)':>14}")
    for res in results:
        lines.append(f"{res['name']:<12}{res['s_row']:>+10.4f}{res['s_col']:>+10.4f}"
                      f"{res['shift_pix']:>12.4f}{res['shift_arcsec']:>14.4f}")
    lines.append("")

    # Consistency check: pairwise cosine similarity of the (row, col) shift vectors and the
    # spread in magnitude, to say whether the three regions point the same way.
    vecs = np.array([[r["s_row"], r["s_col"]] for r in results])
    norms = np.linalg.norm(vecs, axis=1)
    lines.append("Pairwise cosine similarity of (row, col) shift vectors (1 = same "
                  "direction, -1 = opposite, ~0 = unrelated):")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            cos = float(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j] + 1e-30))
            lines.append(f"  {results[i]['name']} vs {results[j]['name']}: {cos:+.3f}")
    lines.append("")

    any_displacement = any(r["verdict"] == "DISPLACEMENT" for r in results)
    all_substructure = all(r["verdict"] == "SUBSTRUCTURE" for r in results)
    if all_substructure:
        cross_verdict = ("All three regions read SUBSTRUCTURE independently -- none of the "
                          "fitted shifts explains much of its region's chi-squared, so there "
                          "is nothing consistent to compare directionally; a single rigid "
                          "displacement is not indicated anywhere.")
    else:
        cos_vals = [float(np.dot(vecs[i], vecs[j]) / (norms[i] * norms[j] + 1e-30))
                    for i in range(len(results)) for j in range(i + 1, len(results))]
        consistent = all(c > 0.5 for c in cos_vals)
        if consistent and any_displacement:
            cross_verdict = ("The fitted shifts are CONSISTENT in direction across regions "
                              "(all pairwise cosine similarities > 0.5) and at least one "
                              "region reads DISPLACEMENT -- this pattern is consistent with a "
                              "global cause (astrometric or PSF-centring error) rather than "
                              "independent per-region effects.")
        else:
            cross_verdict = ("The fitted shifts DISAGREE in direction and/or magnitude across "
                              "regions -- a single rigid displacement cannot explain all three, "
                              "so per-region verdicts (above) are the more honest reading than "
                              "any single global-shift story.")
    lines.append(f"CROSS-REGION VERDICT: {cross_verdict}")

    report = "\n".join(lines)
    print()
    print(report)
    (OUT_DIR / "arc_residual_anatomy.txt").write_text(report + "\n")

    # --- Figure: one row per region, [data, model, residual, shift-model, leftover] -------
    fig, axes = plt.subplots(len(results), 5, figsize=(22, 4.6 * len(results)))
    if len(results) == 1:
        axes = axes[None, :]
    for row_i, res in enumerate(results):
        sel = res["sel"]
        rows_sel, cols_sel = np.where(sel)
        pad = 4
        r0 = max(rows_sel.min() - pad, 0)
        r1 = min(rows_sel.max() + pad + 1, image.shape[0])
        c0 = max(cols_sel.min() - pad, 0)
        c1 = min(cols_sel.max() + pad + 1, image.shape[1])
        box = (slice(r0, r1), slice(c0, c1))

        pred_rc = res["pred_rc"]
        leftover = norm_res - pred_rc

        disp = {
            "Data": (image[box], "viridis", None),
            "Model": (model_img[box], "viridis", None),
            "Normalised residual r": (norm_res[box], "RdBu_r", 5.0),
            f"Shift model\n(|s|={res['shift_pix']:.2f}px, "
            f"{res['frac_explained_rc']:.0%})": (pred_rc[box], "RdBu_r", 5.0),
            "Leftover": (leftover[box], "RdBu_r", 5.0),
        }
        for ax, (title, (arr, cmap, vlim)) in zip(axes[row_i], disp.items()):
            if vlim is None:
                im = ax.imshow(arr, origin="lower", cmap=cmap)
            else:
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim)
            ax.contour(sel[box].astype(float), levels=[0.5], colors="k", linewidths=0.8)
            ax.set_title(f"{res['name']}: {title}" if title == "Data" else title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "DESI257 r3_unmasked: per-region residual anatomy -- " +
        ", ".join(f"{r['name']} {r['verdict']} ({r['best_frac']:.0%})" for r in results),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = OUT_DIR / "arc_residual_anatomy.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {fig_path}")
    print(f"Report: {OUT_DIR / 'arc_residual_anatomy.txt'}")


if __name__ == "__main__":
    main()
