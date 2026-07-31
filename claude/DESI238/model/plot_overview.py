"""One 2x2 overview figure for a DESI238 fit: data, model+critical curve, normalised
residual, and source plane + caustics.

This duplicates ``diagnose.residual_figure``'s data/model/normalised-residual panels
(same err_map convention, same point estimate) rather than importing them, because the
fourth panel needs the raw lens parameters that ``residual_figure`` never surfaces, and
because this file is not allowed to modify ``diagnose.py``. The model-rebuild logic
(metadata.json -> build_prior/build_phys_model) is copied verbatim from
``clumps.compare`` for the same reason: reconstructing the wrong prior variant silently
produces plausible, wrong physical values (see that module's docstring).

Critical curves and caustics are computed on a fresh, non-supersampled coordinate grid
built here -- NOT the simulator's internal grid, which is tied to a batch size and is
supersampled. The Jacobian of the lens mapping is obtained by finite differences
(``np.gradient``) rather than autodiff, which is simpler to get right by eye but has
exactly one trap: ``np.gradient``'s ``axis`` argument and "differentiate with respect to
x or y" are easy to cross up when the grid is built with ``np.meshgrid``'s default
``indexing='xy'``. That mapping is verified explicitly below (see ``_jacobian_det``)
rather than assumed, because a transposed Jacobian produces a critical curve that still
looks plausible -- it is not obviously wrong by eye, only by the theta_E cross-check.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm

import compat  # noqa: F401 -- applies JAX 0.11 shims before gigalens.jax.* imports

import jax.numpy as jnp

from gigalens.jax.model import ForwardProbModel

import data
import diagnose
import priors


def build_model(run_dir):
    """Rebuild ``(prior, phys_model, prob_model, sim_config, image)`` for ``run_dir``.

    Copied from ``clumps.compare`` rather than imported: the fallback defaults for a
    pre-revision run (no ``n_source`` key) matter, and this keeps that logic in one place
    per file rather than risking drift if it's refactored elsewhere.
    """
    run_cfg = diagnose.load_run_config(run_dir)
    n_source = int(run_cfg.get("n_source", 1))
    n_lens_light = int(run_cfg.get("n_lens_light", 1))
    lens_light_n_max = float(run_cfg.get("lens_light_n_max", 6.0))
    theta_e_center = float(run_cfg.get("theta_e_center", 1.5))
    theta_e_width = float(run_cfg.get("theta_e_width", 0.35))
    source_prior_version = int(
        run_cfg.get("source_prior_version", 2 if "n_source" in run_cfg else 1)
    )
    lens_light_prior_version = int(
        run_cfg.get("lens_light_prior_version",
                    priors.lens_light_prior_version_for(n_lens_light))
    )
    print(
        f"Model variant: n_source={n_source}, n_lens_light={n_lens_light}, "
        f"lens_light_n_max={lens_light_n_max}, source_prior_v{source_prior_version}, "
        f"lens_light_prior_v{lens_light_prior_version}"
    )

    image, psf, sim_config = data.load()
    prior = priors.build_prior(
        theta_E_center=theta_e_center,
        theta_E_width=theta_e_width,
        n_source=n_source,
        n_lens_light=n_lens_light,
        lens_light_n_max=lens_light_n_max,
        source_prior_version=source_prior_version,
        lens_light_prior_version=lens_light_prior_version,
    )
    phys_model = priors.build_phys_model(n_source=n_source, n_lens_light=n_lens_light)
    prob_model = ForwardProbModel(
        prior,
        jnp.asarray(image),
        background_rms=data.BACKGROUND_RMS,
        exp_time=data.EXP_TIME,
    )
    return prior, phys_model, prob_model, sim_config, image


def _deflection(phys_model, lens_params, Xg, Yg):
    """Sum ``lens.deriv`` over ``phys_model.lenses`` on an arbitrary (Xg, Yg) grid.

    Mirrors ``LensSimulator._beta``'s loop exactly, but takes plain numpy grids instead
    of the simulator's fixed, supersampled ``img_X``/``img_Y`` -- this needs its own
    grid, sized for the critical curve rather than the 80x80 image.
    """
    ax = np.zeros_like(Xg)
    ay = np.zeros_like(Xg)
    for lens, p in zip(phys_model.lenses, lens_params):
        fx, fy = lens.deriv(jnp.asarray(Xg), jnp.asarray(Yg), **p)
        ax = ax + np.asarray(fx)
        ay = ay + np.asarray(fy)
    return ax, ay


def _jacobian_det(Xg, Yg, ax, ay, dx):
    """det(A), A = I - d(alpha)/d(theta), by finite differences.

    TRAP: ``Xg, Yg = np.meshgrid(x, y)`` (default ``indexing='xy'``) makes array axis 0
    range over y and axis 1 range over x -- i.e. ``Xg[i, j] = x[j]``, ``Yg[i, j] = y[i]``.
    ``np.gradient(f, dx)`` with no ``axis`` argument differentiates axis 0 first, axis 1
    second, so ``np.gradient(ax, dx)`` returns ``(d(ax)/dy, d(ax)/dx)`` in that order --
    the OPPOSITE of what the variable names suggest at a glance. Get this backwards and
    the Jacobian is transposed: the resulting curve still looks like a plausible critical
    curve (same rough size and shape for a mildly elliptical lens) but its radius drifts
    from theta_E, which is exactly the cross-check this file insists on below.

    Verified concretely for this file's grid convention (checked, not assumed): with
    ``indexing='xy'``, ``d/dy`` is the axis-0 derivative and ``d/dx`` is the axis-1
    derivative, matching ``np.gradient``'s default order one-to-one.
    """
    day_dy, day_dx = np.gradient(ay, dx, dx)
    dax_dy, dax_dx = np.gradient(ax, dx, dx)
    a11 = 1 - dax_dx
    a22 = 1 - day_dy
    a12 = -dax_dy
    a21 = -day_dx
    return a11 * a22 - a12 * a21


def critical_curves_and_caustics(phys_model, lens_params, half_extent=3.0, n=600):
    """Return ``(Xg, Yg, detA, curves, caustics)``.

    ``curves`` and ``caustics`` are lists of ``(N, 2)`` arrays, one per disjoint contour
    segment of ``detA = 0``, in (x, y) arcsec. Caustics are each curve vertex mapped
    through ``beta = theta - alpha(theta)``, with ``alpha`` recomputed directly at the
    vertices (cheap, and exact -- no interpolation error from the coarser detA grid).
    """
    axis = np.linspace(-half_extent, half_extent, n, dtype=np.float64)
    dx = axis[1] - axis[0]
    Xg, Yg = np.meshgrid(axis, axis)  # indexing='xy': Xg varies along axis 1, Yg along axis 0
    ax, ay = _deflection(phys_model, lens_params, Xg, Yg)
    detA = _jacobian_det(Xg, Yg, ax, ay, dx)

    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(Xg, Yg, detA, levels=[0.0])
    curves = [seg for segs in cs.allsegs for seg in segs if len(seg) > 2]
    plt.close(fig_tmp)

    caustics = []
    for curve in curves:
        cx, cy = curve[:, 0], curve[:, 1]
        vax, vay = _deflection(phys_model, lens_params, cx, cy)
        caustics.append(np.stack([cx - vax, cy - vay], axis=-1))

    return Xg, Yg, detA, curves, caustics


def curve_radius(curve, center):
    """Mean distance of a curve's vertices from ``center`` (arcsec)."""
    r = np.hypot(curve[:, 0] - center[0], curve[:, 1] - center[1])
    return float(np.mean(r))


def check_theta_E(curves, center, theta_E):
    """Print each critical curve's mean radius against theta_E; identify the tangential
    one (the larger of the (usually) two curves) and flag loudly if it disagrees by more
    than 10%.

    An EPL+shear lens generically produces two critical curves, tangential (outer,
    larger) and radial (inner, smaller); only the tangential one is expected to track
    theta_E closely.
    """
    if not curves:
        print("VALIDATION FAILED: no critical curve found at all.")
        return
    radii = sorted((curve_radius(c, center) for c in curves), reverse=True)
    tangential = radii[0]
    print(f"Critical curve radii (arcsec, sorted): {['%.4f' % r for r in radii]}")
    print(f"Tangential (largest) radius = {tangential:.4f}, theta_E = {theta_E:.4f}")
    frac = abs(tangential - theta_E) / theta_E
    if frac > 0.10:
        print(
            f"VALIDATION FAILED LOUDLY: tangential critical-curve radius disagrees with "
            f"theta_E by {frac * 100:.1f}% (> 10%). The Jacobian orientation is probably "
            f"wrong -- see _jacobian_det's docstring. DO NOT trust this figure's critical "
            f"curve / caustics."
        )
    else:
        print(f"Validation OK: tangential radius agrees with theta_E to {frac * 100:.2f}%.")


def render_source_plane(phys_model, source_params, extent, n=400):
    """Render summed source-light surface brightness on a fresh (x, y) grid in the source
    plane -- i.e. calling ``light()`` directly on plain coordinates, not through the
    lens. ``extent = (xmin, xmax, ymin, ymax)`` in arcsec.
    """
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    Xg, Yg = np.meshgrid(xs, ys)  # indexing='xy', consistent with imshow(extent=...) below
    img = np.zeros_like(Xg)
    for lightModel, p in zip(phys_model.source_light, source_params):
        img = img + np.asarray(lightModel.light(jnp.asarray(Xg), jnp.asarray(Yg), **p))
    return Xg, Yg, img


def make_figure(run_dir, out_dir):
    prior, phys_model, prob_model, sim_config, image = build_model(run_dir)

    z_unc, label = diagnose.point_estimate(run_dir)
    im_sim = diagnose.simulate_from_unconstrained(prob_model, phys_model, sim_config, z_unc)

    x = prob_model.bij.forward(list(jnp.asarray(np.asarray(z_unc).reshape(1, -1)).T))
    lens_params, lens_light_params, source_params = x[0], x[1], x[2]
    epl_params = {k: float(np.asarray(v).squeeze()) for k, v in lens_params[0].items()}
    theta_E = epl_params["theta_E"]
    lens_center = (epl_params["center_x"], epl_params["center_y"])

    # Scalar-ised params for the finite-difference deflection grid (batch size 1 -> plain
    # floats), and for source rendering (kept as length-1 arrays; light() broadcasts).
    lens_params_scalar = [
        {k: float(np.asarray(v).squeeze()) for k, v in p.items()} for p in lens_params
    ]

    npix, delta = data.NUM_PIX, data.DELTA_PIX
    half_extent_image = npix / 2 * delta

    Xg, Yg, detA, curves, caustics = critical_curves_and_caustics(
        phys_model, lens_params_scalar, half_extent=half_extent_image, n=600
    )
    check_theta_E(curves, lens_center, theta_E)

    # --- Panel 3 data: normalised residual, same convention as diagnose.residual_figure ---
    err_map = np.sqrt(data.BACKGROUND_RMS ** 2 + np.clip(im_sim, 0, np.inf) / data.EXP_TIME)
    norm_res = (image - im_sim) / err_map
    chi2 = float(np.mean(norm_res ** 2))
    print(f"Reduced chi-squared (mean of normalised-residual^2): {chi2:.4f}")

    # --- Panel 4 data: source plane, sized from the fitted source centres/radii and the
    # caustic extent, deliberately NOT sharing axes with the image-plane panels (arcsec
    # scale is similar but the physically relevant window is much smaller). ---
    src_centers = []
    src_radii = []
    for p in source_params:
        src_centers.append((float(np.asarray(p["center_x"]).squeeze()),
                             float(np.asarray(p["center_y"]).squeeze())))
        src_radii.append(float(np.asarray(p["R_sersic"]).squeeze()))
    cxs = [c[0] for c in src_centers]
    cys = [c[1] for c in src_centers]
    pad = max(src_radii) * 4 if src_radii else 0.3
    xmin, xmax = min(cxs) - pad, max(cxs) + pad
    ymin, ymax = min(cys) - pad, max(cys) + pad
    if caustics:
        all_caustic = np.concatenate(caustics, axis=0)
        xmin = min(xmin, np.percentile(all_caustic[:, 0], 1))
        xmax = max(xmax, np.percentile(all_caustic[:, 0], 99))
        ymin = min(ymin, np.percentile(all_caustic[:, 1], 1))
        ymax = max(ymax, np.percentile(all_caustic[:, 1], 99))

    Xs, Ys, src_img = render_source_plane(
        phys_model, source_params, (xmin, xmax, ymin, ymax), n=400
    )

    _draw(
        out_dir, image, im_sim, norm_res, chi2, label,
        Xg, Yg, curves, caustics, lens_center, theta_E,
        Xs, Ys, src_img, src_centers, (xmin, xmax, ymin, ymax),
        npix, delta,
    )
    return chi2


def _draw(out_dir, image, im_sim, norm_res, chi2, label,
          Xg, Yg, curves, caustics, lens_center, theta_E,
          Xs, Ys, src_img, src_centers, src_extent, npix, delta):
    """Build and save the 2x2 figure. Image-plane axes are in arcsec, centred on the
    simulator's coordinate convention (mean coordinate 0,0), matching the grid used for
    the critical curve so the overlay lines up pixel-for-pixel with no separate
    registration step.
    """
    half = npix / 2 * delta
    extent_img = [-half, half, -half, half]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # asinh stretch for the two image-plane brightness panels.
    #
    # The deflector core reaches 7.2 e/s while the arcs sit near 0.1 and the sky noise is
    # 0.0075. On a linear scale the core saturates and the arcs -- the only part of the
    # image that constrains the lens -- are nearly invisible. asinh is linear below
    # `linear_width` and logarithmic above, so the background keeps its familiar linear
    # appearance while two decades of arc and core structure become visible at once.
    #
    # linear_width is tied to the noise, not to a percentile: the point of the linear
    # region is to cover "pixels consistent with background", which is what
    # BACKGROUND_RMS measures. A percentile would drift with how much bright galaxy
    # happens to be in the crop.
    #
    # vmin is slightly negative rather than 0 because the drizzled science image has
    # genuinely negative sky pixels (min -0.0034 e/s here); clipping at 0 would render
    # 0.1% of the frame as uniform floor and hide the noise symmetry that makes an
    # over-subtracted background obvious.
    vmax = float(np.percentile(image, 99.9))
    vmin = -5.0 * data.BACKGROUND_RMS
    linear_width = 5.0 * data.BACKGROUND_RMS
    bright_norm = AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax)

    # Panel 1: observed
    ax = axes[0, 0]
    im0 = ax.imshow(image, origin="lower", extent=extent_img, norm=bright_norm)
    ax.set_title("Observed (asinh)")
    ax.plot(*lens_center, "+", color="white", ms=12, mew=1.5)
    plt.colorbar(im0, ax=ax, fraction=0.046, label="e/s")

    # Panel 2: model + critical curve. Shares `bright_norm` with panel 1 on purpose --
    # a different stretch between data and model would make them incomparable by eye,
    # which is the whole point of putting them side by side.
    ax = axes[0, 1]
    im1 = ax.imshow(im_sim, origin="lower", extent=extent_img, norm=bright_norm)
    for curve in curves:
        ax.plot(curve[:, 0], curve[:, 1], color="lime", lw=1.2)
    ax.plot(*lens_center, "+", color="white", ms=12, mew=1.5)
    ax.set_title(f"Model ({label}, asinh) + critical curve")
    plt.colorbar(im1, ax=ax, fraction=0.046, label="e/s")

    # Panel 3: normalised residual
    ax = axes[1, 0]
    im2 = ax.imshow(norm_res, origin="lower", extent=extent_img, cmap="RdBu_r",
                     vmin=-5, vmax=5)
    ax.set_title(rf"Normalised residual ((obs$-$model)/$\sigma$), $\chi^2_\nu$={chi2:.3f}")
    plt.colorbar(im2, ax=ax, fraction=0.046, label=r"$\sigma$")

    # Panel 4: source plane + caustics
    ax = axes[1, 1]
    # Also AsinhNorm rather than the hand-rolled arcsinh this previously used. Same
    # stretch, but the colourbar now reads in physical surface brightness instead of
    # "asinh(SB / 99.5th percentile)", which no one can convert back in their head.
    #
    # The source plane needs its own linear_width: it is intrinsic surface brightness
    # with no noise in it, so BACKGROUND_RMS is meaningless here. A small fraction of the
    # peak is the natural choice.
    src_vmax = float(np.percentile(src_img, 99.9))
    im3 = ax.imshow(src_img, origin="lower",
                     extent=[src_extent[0], src_extent[1], src_extent[2], src_extent[3]],
                     cmap="viridis", aspect="equal",
                     norm=AsinhNorm(linear_width=max(src_vmax * 1e-3, 1e-8),
                                    vmin=0.0, vmax=src_vmax))
    for cau in caustics:
        ax.plot(cau[:, 0], cau[:, 1], color="orange", lw=1.2)
    for i, (cx, cy) in enumerate(src_centers):
        ax.plot(cx, cy, "x", color="red", ms=10, mew=2)
        ax.annotate(f"src{i}", (cx, cy), color="red", fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Source plane (asinh) + caustics")
    plt.colorbar(im3, ax=ax, fraction=0.046, label="surface brightness (e/s/pix)")

    for ax in axes.flat:
        ax.set_xlabel('arcsec')
        ax.set_ylabel('arcsec')

    fig.tight_layout()
    path = Path(out_dir) / "lensing_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="2x2 overview figure: data/model+critcurve/residual/source+caustics"
    )
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    ap.add_argument("--out", default=None, help="Output directory (default: run_dir)")
    args = ap.parse_args()

    out_dir = args.out or args.run_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    make_figure(args.run_dir, out_dir)


if __name__ == "__main__":
    main()
