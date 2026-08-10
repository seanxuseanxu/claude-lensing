"""Is the S-arc's paired +/-4-5 sigma residual lobes a displaced arc, or source substructure?

`RESULTS_v2.md` section 5(b) reads the discrete +/-4 sigma knots along the arc in
`runs/r2_s2l2/residuals.png` as star-forming clumps the two smooth Sersic source components
cannot reproduce. But adjacent opposite-sign pairs (blue immediately next to red) are also
exactly what a DISPLACED or MIS-SHAPED model arc produces: shift a bright ridge by a fraction
of a pixel and the residual is negative on the side the ridge moved away from and positive on
the side it moved toward, everywhere along the ridge -- no source-plane structure required.
Deciding which this is matters before round 4 adds source freedom (a pixelated source), which
would absorb a displacement without ever diagnosing it.

The test. A rigid shift of the model by (s_row, s_col) pixels changes the model image, to
first order, by

    M(row - s_row, col - s_col) ~= M(row, col) - s_row dM/drow - s_col dM/dcol

so the residual such a shift would produce is

    residual = data - M ~= -s_row dM/drow - s_col dM/dcol

Dividing by the error map turns both sides into the same sigma units the normalised residual
r = (data - model)/err is already in:

    r ~= -s_row (dM/drow)/err - s_col (dM/dcol)/err = a_row G_row + a_col G_col

with G_row = (dM/drow)/err, G_col = (dM/dcol)/err, and a_row = -s_row, a_col = -s_col. Fitting
r against G_row and G_col by ordinary least squares (both sides already in sigma units, so
unweighted) over the arc pixels gives (a_row, a_col) directly in pixels, and the implied shift
is (s_row, s_col) = (-a_row, -a_col). This is the whole point of dividing the gradient by the
error map before fitting: without it, a_row/a_col would carry units of (flux per pixel-shift)
and would not be a shift in pixels at all.

If the fit accounts for most of the arc's local chi-squared, the residual is what a
displacement looks like and should not be read as clumps. If it accounts for little, the
residual pattern is uncorrelated with the model's own shape and genuine substructure is the
more honest reading.

Gradient: `np.gradient(model_img)`, i.e. centred finite differences on the interior of the
pixel grid (one-sided at the edges, not relevant here since the arc sits well inside the
80x80 crop). The model image is a Sersic profile lensed and PSF-convolved -- a smooth
function on a 0.065"/pix grid against an arc that is 2-3 px wide -- so a plain finite
difference on the rendered array is an accurate cheap estimate of the local gradient. Getting
the *analytic* derivative would mean re-differentiating the Sersic + EPL ray-tracing by hand,
which is exactly the "unvalidated hand-rolled numerics" CLAUDE.md warns against; the rendered
array is gigalens' own output, so differencing it stays inside validated code.

Tangential/radial decomposition. array-space PA (see below) at each arc pixel gives a radial
unit vector e_r = (cos(PA), sin(PA)) and tangential unit vector e_t = (-sin(PA), cos(PA)) in
the (row, col) basis (e_t = d/dPA of e_r, the direction of increasing PA). Projecting
(G_row, G_col) onto these per pixel and re-fitting r against (G_tan, G_rad) is a second,
independent regression -- NOT a change of basis of the row/col fit, because e_r and e_t rotate
with PA along the arc while the row/col basis does not. A global (row, col) shift is what a
source-position offset looks like; a locally-radial push is what a theta_E (or overall mass
normalisation) offset looks like, since it changes the critical curve's radius rather than its
centre. Reporting both is cheap (same design-matrix machinery, a different pair of columns)
and the task explicitly asks for it.

Arc pixel selection. `config.py`'s `CLUMPS` dict and its surrounding comment (config.py
lines 129-146) measure the S-arc as a continuous ridge from array-space PA 125 deg to PA
235 deg -- **measured from the +row axis toward +col in the crop, NOT sky PA** -- staying
within radius 1.66-1.82" the whole way. That documented ridge, not the three individual
clump radii, is the annulus/PA box used below, intersected with the run's own likelihood mask
(`rm.mask`) and a surface-brightness cut, model_img > 3*err_map, so pixels where the model
predicts no significant flux (and hence ~zero gradient) do not dilute the regression with
noise. `claude/pipeline/clumps.py` documents and uses the identical (r, PA) -> (row, col)
conversion (row = row0 + r_pix*cos(PA), col = col0 + r_pix*sin(PA)); this script reimplements
only the annulus/PA *selection* mask, not that conversion's role as a coordinate transform,
since clumps.py itself has no "which pixels are in this range" helper to call into.

Everything physical here -- the model render, the error map, the mask, the point estimate --
comes from `diagnose.rebuild_model`/`diagnose.point_estimate`/`diagnose.masked_chisq`, the
same functions `residual_figure` and `sw_region_chisq.py` use. Nothing here reimplements a
likelihood or a bijector.

    CUDA_VISIBLE_DEVICES=2 python claude/DESI234/analysis/arc_residual_anatomy.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import diagnose  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r2_s2l2"
OUT_DIR = Path(__file__).resolve().parent

# The documented S-arc ridge (config.py lines 140-143): continuous from PA 125 to PA 235
# (array-space, +row toward +col), staying within radius 1.66-1.82" the whole way. Held
# literally here rather than re-derived, for the same reason sw_region_chisq.py holds its
# mask circle literally: this script's job is to test a hypothesis about that ridge, not to
# re-measure where it is.
ARC_R_MIN, ARC_R_MAX = 1.66, 1.82  # arcsec
ARC_PA_MIN, ARC_PA_MAX = 125.0, 235.0  # degrees, array-space

# Arc pixels must also carry model surface brightness at least this many sigma above the
# noise, so pixels the model predicts as empty sky (and whose gradient is therefore ~0, pure
# noise) do not dilute the regression.
SB_SIGMA = 3.0


def arc_selection(model_img, err_map, mask, center, delta_pix):
    """Boolean array: pixels in the documented S-arc ridge, unmasked, above SB_SIGMA.

    ``center`` is ``(row0, col0)`` in pixel coordinates. PA convention matches
    ``clumps.clump_positions``: PA = atan2(col - col0, row - row0), i.e. measured from the
    +row axis toward +col.
    """
    ny, nx = model_img.shape
    rows, cols = np.indices((ny, nx))
    row0, col0 = center
    dr, dc = rows - row0, cols - col0
    r_arcsec = np.hypot(dr, dc) * delta_pix
    pa_deg = np.degrees(np.arctan2(dc, dr)) % 360.0

    in_annulus = (r_arcsec >= ARC_R_MIN) & (r_arcsec <= ARC_R_MAX)
    in_pa = (pa_deg >= ARC_PA_MIN) & (pa_deg <= ARC_PA_MAX)
    bright = model_img > SB_SIGMA * err_map
    return in_annulus & in_pa & bright & np.asarray(mask, dtype=bool), pa_deg


def fit_shift(norm_res, g_a, g_b, sel):
    """OLS fit of ``norm_res[sel] ~ coef_a * g_a[sel] + coef_b * g_b[sel]``.

    Returns ``(coef_a, coef_b, predicted_full, chi2_before, chi2_after, n)``, where
    ``predicted_full`` is the fitted model evaluated over the WHOLE array (for plotting),
    and the two chi-squareds are ``diagnose.masked_chisq`` over ``sel`` before and after
    subtracting the fit -- i.e. mean(r**2) and mean((r - predicted)**2) over arc pixels.
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

    sel, pa_deg = arc_selection(model_img, err_map, mask, center, delta_pix)
    n_sel = int(sel.sum())
    if n_sel < 10:
        raise RuntimeError(
            f"Only {n_sel} arc pixels selected -- annulus/PA/SB cuts are too strict to "
            f"fit anything. Check center={center}, delta_pix={delta_pix}."
        )

    # --- Row/col (global translation) fit --------------------------------------------
    grad_row, grad_col = np.gradient(model_img)  # centred finite differences, edges one-sided
    g_row, g_col = grad_row / err_map, grad_col / err_map

    a_row, a_col, pred_rc, chi2_before, chi2_after_rc, n_arc = fit_shift(
        norm_res, g_row, g_col, sel
    )
    s_row, s_col = -a_row, -a_col
    shift_pix = np.hypot(s_row, s_col)
    shift_arcsec = shift_pix * delta_pix
    frac_explained_rc = 1.0 - chi2_after_rc / chi2_before

    # --- Tangential/radial fit, about the deflector centre ----------------------------
    pa_rad = np.radians(pa_deg)
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)
    # e_r = (cos PA, sin PA), e_t = (-sin PA, cos PA) in the (row, col) basis.
    g_rad = g_row * cos_pa + g_col * sin_pa
    g_tan = -g_row * sin_pa + g_col * cos_pa

    a_tan, a_rad, pred_tr, chi2_before_tr, chi2_after_tr, _ = fit_shift(
        norm_res, g_tan, g_rad, sel
    )
    s_tan, s_rad = -a_tan, -a_rad
    frac_explained_tr = 1.0 - chi2_after_tr / chi2_before_tr

    # Sign check: chi2_before from both fits must agree, since it depends only on sel and
    # norm_res, not on which basis is being fit.
    assert abs(chi2_before - chi2_before_tr) < 1e-8, "chi2_before mismatch between fits"

    # --- Report -------------------------------------------------------------------------
    lines = []
    lines.append(f"Run: {RUN}")
    lines.append(f"Point estimate: {label}")
    lines.append(f"Deflector centre (row, col): {center}, delta_pix={delta_pix}\"/px")
    lines.append("")
    lines.append(f"Arc selection: r in [{ARC_R_MIN}, {ARC_R_MAX}]\", "
                  f"PA in [{ARC_PA_MIN}, {ARC_PA_MAX}] deg (array-space), "
                  f"model > {SB_SIGMA} sigma, mask-kept")
    lines.append(f"  {n_arc} pixels selected")
    lines.append("")
    lines.append(f"Arc chi2_nu before any correction: {chi2_before:.4f} (over {n_arc} px)")
    lines.append("")
    lines.append("--- Global (row, col) translation fit ---")
    lines.append(f"  fit coefficients: a_row={a_row:+.4f}, a_col={a_col:+.4f} (sigma per px-shift)")
    lines.append(f"  implied shift: s_row={s_row:+.4f} px, s_col={s_col:+.4f} px")
    lines.append(f"  |shift| = {shift_pix:.4f} px = {shift_arcsec:.4f} arcsec")
    lines.append(f"  arc chi2_nu after removing this shift: {chi2_after_rc:.4f}")
    lines.append(f"  fraction of arc chi2 explained: {frac_explained_rc:.4f}")
    lines.append("")
    lines.append("--- Tangential/radial fit about the deflector centre ---")
    lines.append(f"  fit coefficients: a_tan={a_tan:+.4f}, a_rad={a_rad:+.4f} (sigma per px-shift)")
    lines.append(f"  implied shift: s_tan={s_tan:+.4f} px, s_rad={s_rad:+.4f} px")
    lines.append(f"  |shift| = {np.hypot(s_tan, s_rad):.4f} px "
                  f"= {np.hypot(s_tan, s_rad) * delta_pix:.4f} arcsec")
    lines.append(f"  arc chi2_nu after removing this component: {chi2_after_tr:.4f}")
    lines.append(f"  fraction of arc chi2 explained: {frac_explained_tr:.4f}")
    lines.append("")

    best_frac = max(frac_explained_rc, frac_explained_tr)
    if best_frac > 0.5:
        verdict = ("DISPLACEMENT: the gradient-projection (shift) model removes "
                   f"{best_frac:.0%} of the arc's chi-squared. The paired +/- lobes are "
                   "consistent with the model arc sitting in slightly the wrong place, "
                   "not with discrete source-plane structure.")
    elif best_frac < 0.2:
        verdict = ("SUBSTRUCTURE: the gradient-projection (shift) model removes only "
                   f"{best_frac:.0%} of the arc's chi-squared. The residual is largely "
                   "uncorrelated with the model's own local shape, consistent with genuine "
                   "source-plane structure the smooth Sersic components cannot reproduce.")
    else:
        verdict = ("BOTH: the gradient-projection (shift) model removes "
                   f"{best_frac:.0%} of the arc's chi-squared -- a real but partial "
                   "displacement, with substantial residual chi-squared left over that a "
                   "shift alone does not explain.")
    lines.append(f"VERDICT: {verdict}")

    report = "\n".join(lines)
    print()
    print(report)
    (OUT_DIR / "arc_residual_anatomy.txt").write_text(report + "\n")

    # --- Figure ---------------------------------------------------------------------
    rows_sel, cols_sel = np.where(sel)
    pad = 4
    r0, r1 = max(rows_sel.min() - pad, 0), min(rows_sel.max() + pad + 1, image.shape[0])
    c0, c1 = max(cols_sel.min() - pad, 0), min(cols_sel.max() + pad + 1, image.shape[1])
    box = (slice(r0, r1), slice(c0, c1))

    leftover = norm_res - pred_rc

    disp = {
        "Data (arc region)": (image[box], "viridis", None),
        "Model": (model_img[box], "viridis", None),
        "Normalised residual r": (norm_res[box], "RdBu_r", 5.0),
        f"Best-fit shift model\n(|s|={shift_pix:.2f}px, {frac_explained_rc:.0%} of arc chi2)":
            (pred_rc[box], "RdBu_r", 5.0),
        "Leftover after shift": (leftover[box], "RdBu_r", 5.0),
    }

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, (title, (arr, cmap, vlim)) in zip(axes, disp.items()):
        if vlim is None:
            im = ax.imshow(arr, origin="lower", cmap=cmap)
        else:
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=-vlim, vmax=vlim)
        # Outline the arc selection within the displayed box.
        ax.contour(sel[box].astype(float), levels=[0.5], colors="k", linewidths=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"DESI234 r2_s2l2: S-arc residual anatomy -- shift ({s_row:+.2f},{s_col:+.2f}) px "
        f"explains {frac_explained_rc:.0%} of arc chi2",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = OUT_DIR / "arc_residual_anatomy.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {fig_path}")
    print(f"Report: {OUT_DIR / 'arc_residual_anatomy.txt'}")


if __name__ == "__main__":
    main()
