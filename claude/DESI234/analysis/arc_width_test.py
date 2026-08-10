"""Second-order follow-up to arc_residual_anatomy.py: is the S-arc MIS-SHAPED (wrong radial
width) rather than displaced or clumpy?

arc_residual_anatomy.py found the arc's paired +/-4-5 sigma residual lobes are NOT a rigid
displacement: a first-order (dm/dt, dm/dr) shift-gradient fit explains only ~0.5-0.7% of the
arc's chi-squared. But a first-derivative basis is blind to a specific alternative: a model
arc rendered too narrow or too wide *radially* puts its residual on the model's SECOND
derivative (a wrong-width Gaussian-like ridge differenced against the data looks locally like
a rescaled second derivative, not a shift), and would pass the first-order test as cleanly as
genuine source-plane clumps do. This script adds the three second-order terms and asks whether
they buy real explanatory power over the null of "any three arbitrary smooth regressors on
these 118 correlated pixels."

Basis (same t/r frame as arc_residual_anatomy.py, same 118 arc pixels, same sigma-unit
normalisation by dividing through by err_map):

    r ~ a*dm/dt + b*dm/dr + c*d2m/dr2 + d*d2m/dt2 + e*d2m/dtdr

Derivatives. First derivatives: `np.gradient(model_img)` (centred finite differences,
identical to arc_residual_anatomy.py), giving grad_row, grad_col. The Cartesian Hessian is
built the same way one derivative up: `np.gradient` of each first-derivative array along each
axis,

    d2_rr = d(grad_row)/drow,  d2_cc = d(grad_col)/dcol
    d2_rc = 1/2 [ d(grad_row)/dcol + d(grad_col)/drow ]   (symmetrised mixed partial)

Local tangential/radial second derivatives are then the Hessian projected onto the same
per-pixel e_r = (cos PA, sin PA), e_t = (-sin PA, cos PA) basis arc_residual_anatomy.py uses:

    d2m/dr2  = e_r^T H e_r
    d2m/dt2  = e_t^T H e_t
    d2m/dtdr = e_t^T H e_r

This projects the Cartesian Hessian pointwise; it does NOT add the extra curvature term that
comes from e_r, e_t themselves rotating with PA along the arc (that correction is second order
in 1/radius_pix and the arc sits at ~28 px radius, so it is a small effect deliberately not
chased here -- the point of this test is the model's own second derivative, not a fully
covariant curvilinear Laplacian). All five basis columns are divided by err_map before fitting,
exactly as arc_residual_anatomy.py's G_row/G_col/G_tan/G_rad were, so the fit stays in sigma
units and R^2 / chi2-fraction are read directly off it.

Width linearisation (item 3). Model a locally 1-D radial profile as narrowing/widening by a
fractional scale (1+eps): m(r) -> m(r/(1+eps)) ~= m(r) - eps*r*dm/dr for eps small, so to
leading order a width change is degenerate with a *radial* first-derivative term, not the
second derivative reported here. The genuinely second-order piece of a width change is the
next term in that same Taylor expansion, m(r/(1+eps)) - m(r) - (-eps r dm/dr) ~= (eps^2/2) r^2
d2m/dr2 + eps r dm/dr *shape correction*... in practice reporting a clean "width error in
pixels" from c alone requires an assumed profile scale r; here the local arc width is ~2-3 px
(radial annulus is 0.16" = 2.5 px wide), so eps ~= sqrt(2 |c| / <(r dm/dr)^2-ish>) is not
attempted analytically -- instead this script reports the DIRECT, assumption-light estimate:
eps_c = c / <d2m/dr2 scale>, taking the coefficient c at face value against the RMS of d2m/dr2
itself over the arc pixels, eps ~ c (dimensionless in the units the regression already fit in,
since G_rr is already a d2m/dr2/err column) times the local arc half-width (~1.25 px), giving a
width error in pixels = |c| * 1.25 px * (RMS G_rr / RMS G_r-normalisation) -- reported directly
by the script from the fitted values, not hand-derived further, so the number is traceable to
the fit rather than to an extra unvalidated formula.

Null (item 4, the actual point of the test). Five free parameters fit to 118 pixels that are
themselves spatially correlated (adjacent arc pixels are not independent draws) will produce
some nonzero incremental R^2 from ANY three additional smooth regressors, clump or no clump.
So this script draws ~200 random triples of smooth basis vectors matched in spatial smoothness
to the real second-derivative fields -- concretely, random linear combinations (with
independent standard-normal coefficients) of the SAME five spatial derivative fields
(dm/dt, dm/dr, d2m/dr2, d2m/dt2, d2m/dtdr) restricted to the arc footprint, re-orthogonalised
against (dm/dt, dm/dr) by projecting them out (so the null triples play the same role as the
three second-order terms: they are added ON TOP of a fixed first-order fit) -- and reports the
resulting incremental-R^2 distribution's mean and 95th percentile. The measured second-order
increment is judged against that null, not against zero.

    CUDA_VISIBLE_DEVICES=3 python claude/DESI234/analysis/arc_width_test.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose  # noqa: E402
import arc_residual_anatomy as ara  # noqa: E402

RUN = ara.RUN
OUT_DIR = ara.OUT_DIR
N_NULL = 200
RNG_SEED = 0


def r_squared(resid_before, resid_after):
    """R^2 = 1 - SS_res/SS_tot, both demeaned, over whatever pixels resid_before covers."""
    # resid_after is (r - pred); pred already carries any constant shift the OLS fit wants,
    # so demeaning resid_after again would double count. Use SS_tot from the demeaned target
    # and SS_res from the actual fit residual (standard R^2 with an implicit design that did
    # not include an intercept column -- see main() for why that's fine here).
    ss_tot = np.sum((resid_before - resid_before.mean()) ** 2)
    ss_res = np.sum(resid_after ** 2)
    return 1.0 - ss_res / ss_tot


def ols_fit(y, X):
    """Least squares y ~ X @ coeffs. Returns (coeffs, pred, resid)."""
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    return coeffs, pred, y - pred


def main():
    rm = diagnose.rebuild_model(RUN)
    cfg = rm.cfg
    delta_pix = cfg.delta_pix
    center = cfg.centroid_pix

    image = np.asarray(rm.image)
    err_map = np.asarray(rm.prob.error_map)
    mask = (np.ones(image.shape, dtype=bool) if rm.mask is None
            else np.asarray(rm.mask, dtype=bool))

    z_unc, label = diagnose.point_estimate(RUN)
    model_img = diagnose.simulate_from_unconstrained(rm, z_unc)
    norm_res = (image - model_img) / err_map

    sel, pa_deg = ara.arc_selection(model_img, err_map, mask, center, delta_pix)
    n_arc = int(sel.sum())

    pa_rad = np.radians(pa_deg)
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)

    grad_row, grad_col = np.gradient(model_img)
    g_row, g_col = grad_row / err_map, grad_col / err_map
    g_rad = g_row * cos_pa + g_col * sin_pa
    g_tan = -g_row * sin_pa + g_col * cos_pa

    d2_rr = np.gradient(grad_row, axis=0)
    d2_cc = np.gradient(grad_col, axis=1)
    d2_rc = 0.5 * (np.gradient(grad_row, axis=1) + np.gradient(grad_col, axis=0))

    c2, s2 = cos_pa ** 2, sin_pa ** 2
    cs = cos_pa * sin_pa
    hess_rr = c2 * d2_rr + 2 * cs * d2_rc + s2 * d2_cc          # e_r^T H e_r
    hess_tt = s2 * d2_rr - 2 * cs * d2_rc + c2 * d2_cc          # e_t^T H e_t
    hess_tr = -cs * d2_rr + (c2 - s2) * d2_rc + cs * d2_cc      # e_t^T H e_r

    g_rr = hess_rr / err_map
    g_tt = hess_tt / err_map
    g_tr = hess_tr / err_map

    r = norm_res[sel]
    X1 = np.stack([g_tan[sel], g_rad[sel]], axis=-1)
    X2 = np.stack([g_tan[sel], g_rad[sel], g_rr[sel], g_tt[sel], g_tr[sel]], axis=-1)

    coeffs1, pred1, resid1 = ols_fit(r, X1)
    coeffs2, pred2, resid2 = ols_fit(r, X2)

    chi2_before = float(np.mean(r ** 2))
    chi2_after1 = float(np.mean(resid1 ** 2))
    chi2_after2 = float(np.mean(resid2 ** 2))
    frac1 = 1.0 - chi2_after1 / chi2_before
    frac2 = 1.0 - chi2_after2 / chi2_before

    R2_1 = r_squared(r, resid1)
    R2_2 = r_squared(r, resid2)
    incr_R2 = R2_2 - R2_1

    c_coef = coeffs2[2]
    half_width_px = 0.5 * (1.82 - 1.66) / delta_pix
    width_err_px = abs(c_coef) * half_width_px / max(np.sqrt(np.mean(g_rad[sel] ** 2)), 1e-30)

    # Null construction. An earlier version of this null drew random LINEAR COMBINATIONS of
    # the same 5 derivative fields (dm/dt, dm/dr, d2m/dr2, d2m/dt2, d2m/dtdr). That is wrong:
    # after projecting out (dm/dt, dm/dr), any 3 random combinations of those 5 fields
    # generically still SPAN the exact same 3-D residual subspace that (d2m/dr2, d2m/dt2,
    # d2m/dtdr) span -- so every draw reproduced the measured incremental R^2 exactly
    # (mean = 95th pct = measured value, a giveaway the null was degenerate). The fix: draw
    # null regressors that are smooth but structurally INDEPENDENT of the real derivative
    # fields -- Gaussian-smoothed white noise, with the smoothing scale matched to the real
    # second-derivative fields' own spatial correlation length so the null triples have
    # comparable "wiggliness," not comparable algebra. Correlation length is estimated per
    # field as L = sqrt(Var(field)/Var(d field/dpixel)) (exact for a Gaussian-autocorrelated
    # field); a field built by convolving white noise with a Gaussian kernel of std sigma_k
    # has autocorrelation length L = sigma_k*sqrt(2), so sigma_k = L_target/sqrt(2).
    def corr_length(field):
        gr = np.gradient(field, axis=0)
        gc = np.gradient(field, axis=1)
        var_f = np.var(field)
        var_g = 0.5 * (np.var(gr) + np.var(gc))
        return np.sqrt(var_f / var_g)

    L_target = float(np.mean([corr_length(g_rr), corr_length(g_tt), corr_length(g_tr)]))
    sigma_k = L_target / np.sqrt(2.0)

    rng = np.random.default_rng(RNG_SEED)
    null_incr = np.empty(N_NULL)
    for i in range(N_NULL):
        triple_cols = []
        for _ in range(3):
            noise = rng.standard_normal(model_img.shape)
            smooth = gaussian_filter(noise, sigma=sigma_k, mode="nearest")
            triple_cols.append(smooth[sel])
        triple = np.stack(triple_cols, axis=-1)  # (n_arc, 3)
        proj_coeffs, _, _, _ = np.linalg.lstsq(X1, triple, rcond=None)
        triple_orth = triple - X1 @ proj_coeffs
        Xn = np.concatenate([X1, triple_orth], axis=1)
        _, pred_n, resid_n = ols_fit(r, Xn)
        R2_n = r_squared(r, resid_n)
        null_incr[i] = R2_n - R2_1

    null_mean = float(np.mean(null_incr))
    null_p95 = float(np.percentile(null_incr, 95))
    passes_null = incr_R2 > null_p95

    lines = []
    lines.append(f"Run: {RUN}")
    lines.append(f"Point estimate: {label}")
    lines.append(f"Arc pixels: {n_arc} (identical selection to arc_residual_anatomy.py)")
    lines.append("")
    lines.append("--- First-order fit: r ~ a*dm/dt + b*dm/dr ---")
    lines.append(f"  coefficients: a={coeffs1[0]:+.4f}, b={coeffs1[1]:+.4f}")
    lines.append(f"  R^2 = {R2_1:.4f}")
    lines.append(f"  fraction of arc chi2 explained = {frac1:.4f}")
    lines.append("")
    lines.append("--- Full second-order fit: r ~ a*dm/dt + b*dm/dr + c*d2m/dr2 + d*d2m/dt2 + e*d2m/dtdr ---")
    lines.append(f"  coefficients: a={coeffs2[0]:+.4f}, b={coeffs2[1]:+.4f}, "
                  f"c={coeffs2[2]:+.4f}, d={coeffs2[3]:+.4f}, e={coeffs2[4]:+.4f}")
    lines.append(f"  R^2 = {R2_2:.4f}")
    lines.append(f"  fraction of arc chi2 explained = {frac2:.4f}")
    lines.append("")
    lines.append(f"Incremental R^2 from the 3 second-order terms: {incr_R2:.4f}")
    lines.append("")
    lines.append("--- Implied radial width error (item 3) ---")
    lines.append("  Linearisation: m(r/(1+eps)) ~= m(r) - eps*r*dm/dr to first order in eps "
                  "(a width rescale is degenerate with a RADIAL first derivative, not the "
                  "second). c (the d2m/dr2 coefficient) is reported directly; converting it to "
                  "a width error in pixels uses width_err_px = |c| * (half arc annulus width "
                  f"in px = {half_width_px:.3f}) / RMS(dm/dr in sigma units over arc pixels).")
    lines.append(f"  c = {c_coef:+.4f}, RMS(g_rad over arc) = "
                  f"{np.sqrt(np.mean(g_rad[sel] ** 2)):.4f}")
    lines.append(f"  width_err_px ~= {width_err_px:.4f} px")
    lines.append("")
    lines.append(f"--- Null: {N_NULL} random smooth triples (Gaussian-smoothed white noise, "
                  f"matched correlation length L={L_target:.3f} px [smoothing sigma_k="
                  f"{sigma_k:.3f} px], orthogonalised against the first-order fit) ---")
    lines.append(f"  null incremental-R^2: mean = {null_mean:.4f}, 95th pct = {null_p95:.4f}")
    lines.append(f"  measured incremental R^2 ({incr_R2:.4f}) "
                  f"{'EXCEEDS' if passes_null else 'DOES NOT exceed'} the null 95th percentile")
    lines.append("")
    if passes_null and incr_R2 > 0.05:
        verdict = ("MIS-SHAPED: the second-order terms buy incremental R^2 beyond what "
                   "arbitrary smooth regressors of matched complexity buy on these 118 "
                   "correlated pixels. Some of the residual is consistent with the arc's "
                   "radial width being wrong, not purely source-plane clumps.")
    else:
        verdict = ("NOT MIS-SHAPED (by this test): the second-order terms' incremental R^2 "
                   "does not clear the null built from equally complex arbitrary smooth "
                   "regressors. Combined with arc_residual_anatomy.py's first-order result, "
                   "neither a displacement nor a radial-width error accounts for the arc "
                   "residual pattern; genuine source-plane substructure remains the more "
                   "honest reading.")
    lines.append(f"VERDICT: {verdict}")

    report = "\n".join(lines)
    print(report)
    (OUT_DIR / "arc_width_test.txt").write_text(report + "\n")
    print(f"\nReport: {OUT_DIR / 'arc_width_test.txt'}")


if __name__ == "__main__":
    main()
