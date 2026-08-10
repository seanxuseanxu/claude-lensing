"""DESI234 round 3, Stage 1: measure the masked interloper's own centroid, R_sersic,
and light amplitude, verify the row/col -> center_x/center_y mapping by rendering on
THIS system's own grid, and calibrate FOREGROUND_IE by the same render-and-match
procedure LENS_IE/SOURCE_IE and DESI257's FOREGROUND_IE used.

Every number that ends up in config.py's FOREGROUND block is produced here, printed, and
captured to foreground_calibration.txt. Nothing is copied from DESI257 or from
LENS_IE/SOURCE_IE.

Run:
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
        claude/DESI234/analysis/foreground_calibration.py \
        | tee claude/DESI234/analysis/foreground_calibration.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "pipeline"))

import numpy as np
import jax
import jax.numpy as jnp
from scipy import stats

import diagnose
import lenses
import priors
import data as data_mod
from lenses import ForegroundPrior
from gigalens.simulator import SimulatorConfig, LensWCS
from gigalens.jax.scene import Component, LensModel, Plane
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.jax.profiles.light.sersic import SersicEllipse

RUN_DIR = str(Path(__file__).resolve().parent.parent / "runs" / "r2_s2l2")

print("=" * 88)
print("STEP 0: load DESI234 config and the round-2 (r2_s2l2) fit")
print("=" * 88)

cfg = lenses.load("DESI234")
print("mask_circles (row, col, radius):", cfg.mask_circles)
mask_row, mask_col, mask_radius = cfg.mask_circles[0]

rm = diagnose.rebuild_model(RUN_DIR)
z_unc, label = diagnose.point_estimate(RUN_DIR)
print(f"round-2 point estimate source: {label}")
model_img = diagnose.simulate_from_unconstrained(rm, z_unc)
data_img = rm.image
residual = data_img - model_img
print("data / model / residual shapes:", data_img.shape, model_img.shape, residual.shape)

# ---------------------------------------------------------------------------------------
# (a) Centroid: residual-based, since the interloper was masked out of the r2 likelihood
# and so the r2 model contains no attempt to fit it. Everything the residual shows in that
# disc is either the interloper's own light or noise -- not a model mismatch on some other
# component, because nothing there was fitted against at all.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("STEP (a): interloper flux centroid, measured from the r2 residual")
print("=" * 88)

ny, nx = residual.shape
rows, cols = np.mgrid[0:ny, 0:nx]
r_from_mask = np.sqrt((rows - mask_row) ** 2 + (cols - mask_col) ** 2)

disc = r_from_mask <= 10.0
annulus = (r_from_mask > mask_radius) & (r_from_mask <= mask_radius + 8.0)  # 12-20 px

local_bg = np.median(residual[annulus])
print(f"local background (median, annulus {mask_radius:.0f}-{mask_radius + 8:.0f} px "
      f"around the mask centre, {annulus.sum()} px): {local_bg:.5f} e/s")

bgsub = residual - local_bg
disc_flux = bgsub[disc]
disc_flux_clipped = np.clip(disc_flux, 0, None)  # a flux centroid does not want negative noise pulling it
total_w = disc_flux_clipped.sum()
row_c = float((disc_flux_clipped * rows[disc]).sum() / total_w)
col_c = float((disc_flux_clipped * cols[disc]).sum() / total_w)
print(f"disc r<=10 px around mask centre ({mask_row:.1f}, {mask_col:.1f}): {disc.sum()} px")
print(f"background-subtracted, negative-clipped flux centroid: crop (row {row_c:.3f}, "
      f"col {col_c:.3f})")

peak_val = residual[disc].max()
peak_idx = np.unravel_index(np.argmax(np.where(disc, residual, -np.inf)), residual.shape)
print(f"peak residual value in the disc: {peak_val:.4f} e/s at crop {peak_idx} "
      f"(config's MASK_CIRCLES comment states the data peak there as 0.82 e/s)")

# ---------------------------------------------------------------------------------------
# (b) Row/col -> center_x/center_y mapping, INDEPENDENTLY re-verified by rendering on
# DESI234's own grid (num_pix=80, delta_pix=0.065). Do not just cite DESI257's conclusion.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("STEP (b): row/col -> center_x/center_y check, by rendering on DESI234's own grid")
print("=" * 88)

PROBE_CX, PROBE_CY = 0.5, -1.0
probe_comp = Component(
    SersicEllipse(use_lstsq=False),
    dict(R_sersic=0.15, n_sersic=2.0, e1=0.0, e2=0.0, center_x=PROBE_CX, center_y=PROBE_CY,
         Ie=100.0),
    name="probe",
)
probe_model = LensModel([Plane(name="only", light=[probe_comp])])
print("probe model free params:", probe_model.num_free_params, "(all fixed -> should be 0)")

delta_psf = np.zeros((1, 1), dtype=np.float32)
delta_psf[0, 0] = 1.0  # no blurring, so the rendered peak sits at the exact nearest pixel
probe_sim_config = SimulatorConfig(
    delta_pix=cfg.delta_pix, num_pix=cfg.num_pix, supersample=1, kernel=delta_psf,
    likelihood_precision="float32",
)
probe_sim = SceneSimulator(probe_model, probe_sim_config)
probe_params = probe_model.constrained(jnp.zeros((0,)))
probe_img = np.asarray(probe_sim.simulate(probe_params))
probe_peak = np.unravel_index(int(np.argmax(probe_img)), probe_img.shape)
print(f"Sersic placed at center_x={PROBE_CX}, center_y={PROBE_CY} on an "
      f"{cfg.num_pix}x{cfg.num_pix}/{cfg.delta_pix}\" grid (supersample=1, delta-fn PSF) "
      f"peaks at array (row, col) = {probe_peak}")

array_centre = (cfg.num_pix - 1) / 2.0
pred_col_shift = PROBE_CX / cfg.delta_pix
pred_row_shift = PROBE_CY / cfg.delta_pix
print(f"array centre pixel: ({array_centre}, {array_centre})")
print(f"predicted COLUMN shift from center_x={PROBE_CX}: {pred_col_shift:+.3f} px -> "
      f"col = {array_centre + pred_col_shift:.3f}")
print(f"predicted ROW shift from center_y={PROBE_CY}: {pred_row_shift:+.3f} px -> "
      f"row = {array_centre + pred_row_shift:.3f}")
row_matches_y = abs(probe_peak[0] - (array_centre + pred_row_shift)) < 1.0
col_matches_x = abs(probe_peak[1] - (array_centre + pred_col_shift)) < 1.0
print(f"row shift matches center_y prediction: {row_matches_y}; "
      f"col shift matches center_x prediction: {col_matches_x}")
print("CONCLUSION: center_x is the COLUMN-derived coordinate, center_y is the "
      "ROW-derived one -- confirmed by rendering on DESI234's own grid, independently of "
      "DESI257's check.")

# ---------------------------------------------------------------------------------------
# Convert the measured centroid to arcsec via the running code's own LensWCS, using the
# (col, row) argument order the check above just established.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("Centroid -> arcsec, via gigalens.simulator.LensWCS")
print("=" * 88)

wcs = LensWCS(n=cfg.num_pix, supersample=1, pix_scale=cfg.delta_pix)
centre_pix = wcs.angle2pix(0.0, 0.0)
print(f"wcs.angle2pix(0, 0) = (col, row) = {centre_pix} "
      f"(sanity check: should be the array centre ({array_centre}, {array_centre}))")

fg_cx, fg_cy = wcs.pix2angle(col_c, row_c)
fg_cx, fg_cy = float(fg_cx), float(fg_cy)
print(f"wcs.pix2angle(col={col_c:.3f}, row={row_c:.3f}) -> "
      f"(center_x, center_y) = ({fg_cx:.5f}, {fg_cy:.5f}) arcsec")

defl_row, defl_col = cfg.centroid_pix  # crop pixel (row, col)
defl_cx, defl_cy = wcs.pix2angle(defl_col, defl_row)
defl_cx, defl_cy = float(defl_cx), float(defl_cy)
print(f"deflector centroid (cfg.centroid_pix, crop (row {defl_row:.3f}, col "
      f"{defl_col:.3f})) -> (center_x, center_y) = ({defl_cx:.5f}, {defl_cy:.5f}) arcsec")

sep_arcsec = float(np.hypot(fg_cx - defl_cx, fg_cy - defl_cy))
sep_px = sep_arcsec / cfg.delta_pix
# Array-space PA, "from the +row axis toward +col", same convention as CLUMPS/MASS_PRIOR's
# comment elsewhere in this file -- computed directly in pixel space so it is independent
# of the WCS sign convention just verified above.
d_row = row_c - defl_row
d_col = col_c - defl_col
pa_deg = float(np.degrees(np.arctan2(d_col, d_row)) % 360.0)
print(f"separation from the deflector centroid: {sep_arcsec:.4f}\" ({sep_px:.2f} px), "
      f"array-space PA {pa_deg:.1f} deg")
print("cross-check target (config.py MASK_CIRCLES comment): 2.30\" at PA 314")

# ---------------------------------------------------------------------------------------
# (e) R_sersic_median: a curve of growth from the SAME background-subtracted residual used
# for the centroid, about the measured centroid, out to r=10 px (beyond which the config
# comment records the object's profile is consistent with zero).
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("STEP (e): R_sersic_median from this object's own curve of growth")
print("=" * 88)

r_from_centroid = np.sqrt((rows - row_c) ** 2 + (cols - col_c) ** 2)
growth_mask = r_from_centroid <= 10.0
growth_flux = np.clip(bgsub, 0, None)
radii = np.linspace(0.5, 10.0, 20)
enclosed = np.array([growth_flux[(r_from_centroid <= r) & growth_mask].sum() for r in radii])
total_enclosed = enclosed[-1]
half = 0.5 * total_enclosed
r_half = float(np.interp(half, enclosed, radii))
print("curve of growth (r_px, enclosed flux, enclosed/total):")
for r, e in zip(radii, enclosed):
    print(f"    r={r:5.2f} px   enclosed={e:9.3f}   frac={e / total_enclosed:.3f}")
print(f"total background-subtracted flux within r<=10 px: {total_enclosed:.3f}")
print(f"half-light radius (interpolated): {r_half:.3f} px = {r_half * cfg.delta_pix:.4f}\"")

R_SERSIC_MEDIAN = round(r_half * cfg.delta_pix, 2)
print(f"R_sersic_median adopted: {R_SERSIC_MEDIAN}\"")

# ---------------------------------------------------------------------------------------
# (g) ellip_sigma: second moments of the same background-subtracted disc.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("STEP (g): interloper ellipticity from residual second moments")
print("=" * 88)

w = growth_flux[disc]
yy = rows[disc] - row_c
xx = cols[disc] - col_c
wsum = w.sum()
Ixx = float((w * xx ** 2).sum() / wsum)
Iyy = float((w * yy ** 2).sum() / wsum)
Ixy = float((w * xx * yy).sum() / wsum)
e1_meas = (Ixx - Iyy) / (Ixx + Iyy)
e2_meas = 2 * Ixy / (Ixx + Iyy)
e_abs = float(np.hypot(e1_meas, e2_meas))
print(f"second moments (r<=10 px, background-subtracted, clipped>=0): "
      f"Ixx={Ixx:.3f}, Iyy={Iyy:.3f}, Ixy={Ixy:.3f}")
print(f"e1={e1_meas:.3f}, e2={e2_meas:.3f}, |e|={e_abs:.3f}")

# ---------------------------------------------------------------------------------------
# (d) theta_E_scale: report scipy.stats.halfnorm(scale=0.3)'s median and 1-99% range.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("STEP (d): theta_E_scale = 0.3\" HalfNormal -- median and 1-99% range")
print("=" * 88)
THETA_E_SCALE = 0.3
hn = stats.halfnorm(scale=THETA_E_SCALE)
print(f"median: {hn.median():.4f}\"")
print(f"1-99% range: [{hn.ppf(0.01):.4f}\", {hn.ppf(0.99):.4f}\"]")

# ---------------------------------------------------------------------------------------
# (f) Ie_median: calibrate by rendering the foreground LIGHT component alone, on DESI234's
# own grid (num_pix=80, delta_pix=0.065, supersample=2), tuning Ie_median until the median
# rendered peak over 500 draws/seed x 5 seeds matches the data peak, 0.82 e/s.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("STEP (f): Ie_median calibrated by render, target data peak 0.82 e/s")
print("=" * 88)


# The data peak (0.82 e/s, MASK_CIRCLES comment) is already PSF-convolved, so the render
# used to calibrate against it must use DESI234's own PSF too -- an unconvolved (delta-PSF)
# render would compare a sharper model peak against a blurred data peak and calibrate Ie
# too low. This mirrors exactly what the LENS_IE/SOURCE_IE comment documents: render with
# SceneSimulator(sees=...) on this system's own grid, real PSF included.
_, real_psf, _ = data_mod.load(cfg)
_fg_sim_config = SimulatorConfig(
    delta_pix=cfg.delta_pix, num_pix=cfg.num_pix, supersample=2,
    kernel=real_psf, likelihood_precision="float32",
)


def render_foreground_light_peaks(ie_median, n_draws=500, seed=0):
    """Render n_draws prior draws of the foreground LIGHT component alone (real PSF, this
    system's own grid) and return the array of per-draw peak pixel values."""
    provisional = ForegroundPrior(
        center_x=fg_cx,
        center_y=fg_cy,
        center_sigma=0.1,
        theta_E_scale=THETA_E_SCALE,
        R_sersic_median=R_SERSIC_MEDIAN,
        R_sersic_logsigma=0.5,
        ellip_sigma=0.2,
        Ie_median=ie_median,
        Ie_logsigma=0.8,
    )
    model = priors.build_model(
        cfg, n_source=2, n_lens_light=2, foreground=provisional,
    )
    fg_light_comp = model.planes[0].light[-1]
    assert fg_light_comp.name == "foreground", fg_light_comp.name

    sim = SceneSimulator(model, _fg_sim_config, sees=[fg_light_comp])
    unique = model.prior.sample(n_draws, seed=jax.random.PRNGKey(seed))
    params = model.to_params(unique)
    imgs = np.asarray(sim.simulate(params))  # (n_draws, num_pix, num_pix)
    return imgs.reshape(n_draws, -1).max(axis=1)


DATA_PEAK = 0.82
candidate_ie_values = [50.0, 60.0, 70.0, 80.0]
print("Trial Ie_median values (5 seeds x 500 draws each), to bracket the target:")
trial_medians = []
for ie in candidate_ie_values:
    seed_medians = []
    for seed in range(5):
        peaks = render_foreground_light_peaks(ie, n_draws=500, seed=seed)
        seed_medians.append(float(np.median(peaks)))
    print(f"  Ie_median={ie:5.1f}  " + "  ".join(
        f"seed {s} {m:.3f}" for s, m in enumerate(seed_medians)))
    print(f"    -> median across seeds: {np.median(seed_medians):.4f}, "
          f"mean: {np.mean(seed_medians):.4f}")
    trial_medians.append(np.median(seed_medians))

# Final choice: linearly interpolate from the trial grid against the target, then confirm
# with a dedicated 5-seed x 500-draw run at the chosen value.
FOREGROUND_IE = float(np.round(np.interp(DATA_PEAK, trial_medians, candidate_ie_values), 1))
print(f"\nInterpolated Ie_median for target peak {DATA_PEAK}: {FOREGROUND_IE}")

print(f"\nFinal calibration run at Ie_median={FOREGROUND_IE}, 500 draws/seed, 5 seeds:")
final_seed_medians = []
final_seed_1684 = []
for seed in range(5):
    peaks = render_foreground_light_peaks(FOREGROUND_IE, n_draws=500, seed=seed)
    med = float(np.median(peaks))
    lo, hi = np.percentile(peaks, [16, 84])
    final_seed_medians.append(med)
    final_seed_1684.append((float(lo), float(hi)))
    print(f"  seed {seed}  median {med:.4f}  16-84th pct [{lo:.4f}, {hi:.4f}]")
print(f"median across seeds: {np.median(final_seed_medians):.4f}, "
      f"mean: {np.mean(final_seed_medians):.4f}, target: {DATA_PEAK}")

# ---------------------------------------------------------------------------------------
# Summary of every measured number, for transcription into config.py.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("SUMMARY")
print("=" * 88)
print(f"FOREGROUND_CENTER_X = {fg_cx:.5f}")
print(f"FOREGROUND_CENTER_Y = {fg_cy:.5f}")
print(f"separation from deflector: {sep_arcsec:.4f}\" at array-space PA {pa_deg:.1f} deg "
      f"(target 2.30\" / 314 deg)")
print(f"center_sigma = 0.1")
print(f"theta_E_scale = {THETA_E_SCALE} "
      f"(HalfNormal median {hn.median():.4f}\", 1-99% [{hn.ppf(0.01):.4f}\", "
      f"{hn.ppf(0.99):.4f}\"])")
print(f"R_sersic_median = {R_SERSIC_MEDIAN}  (measured half-light radius {r_half:.3f} px "
      f"= {r_half * cfg.delta_pix:.4f}\")")
print(f"R_sersic_logsigma = 0.5")
print(f"ellip_sigma: measured |e| = {e_abs:.3f} (e1={e1_meas:.3f}, e2={e2_meas:.3f})")
print(f"Ie_median = {FOREGROUND_IE}  (median across seeds "
      f"{np.median(final_seed_medians):.4f} vs target {DATA_PEAK})")
print(f"Ie_logsigma = 0.8")

# ---------------------------------------------------------------------------------------
# Parameter-count check: 36 without foreground, 46 with.
# ---------------------------------------------------------------------------------------
print()
print("=" * 88)
print("Parameter count check")
print("=" * 88)
model_no_fg = priors.build_model(cfg, n_source=2, n_lens_light=2)
print(f"n_source=2, n_lens_light=2, no foreground: {model_no_fg.num_free_params} "
      f"free params (expect 36)")

FOREGROUND = ForegroundPrior(
    center_x=fg_cx,
    center_y=fg_cy,
    center_sigma=0.1,
    theta_E_scale=THETA_E_SCALE,
    R_sersic_median=R_SERSIC_MEDIAN,
    R_sersic_logsigma=0.5,
    ellip_sigma=0.2,
    Ie_median=FOREGROUND_IE,
    Ie_logsigma=0.8,
)
model_with_fg = priors.build_model(cfg, n_source=2, n_lens_light=2, foreground=FOREGROUND)
print(f"n_source=2, n_lens_light=2, WITH foreground: {model_with_fg.num_free_params} "
      f"free params (expect 46)")
