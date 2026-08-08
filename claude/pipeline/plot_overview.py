"""One 2x2 overview figure for a lens fit: data, model+critical curve, normalised
residual, and source plane + caustics.

This duplicates ``diagnose.residual_figure``'s data/model/normalised-residual panels
(same err_map convention, same point estimate) rather than importing them, because the
fourth panel needs the raw lens parameters that ``residual_figure`` never surfaces, and
because this file is not allowed to modify ``diagnose.py``. The model reconstruction is
now done by calling ``diagnose.rebuild_model``.

The normalised residual divides by ``rm.prob.error_map``, the dataset's own per-pixel
sigma, which ``ImageData`` derives **once from the observed image**. It does not depend on
the model, so it is not recomputed from ``im_sim`` here. Deriving it from the model image
instead -- the pre-port convention -- would still produce a plausible-looking residual
panel with a slightly different chi-squared, which is exactly the kind of wrong that never
raises.

Critical curves, caustics and the ray-traced cutout border all come from
``gigalens.jax.analysis``: ``LensSolver.critical_and_caustics`` and
``image_border_in_source_plane``. This file used to compute det(A) itself, on its own
grid, with a finite-difference Jacobian. That version got the tangential radius right to
2e-4 arcsec but also reported a second, non-existent critical curve near the origin (see
``check_theta_E``), and it carried a trap worth recording now that it is gone:
``np.gradient`` differentiates axis 0 first, so under
``np.meshgrid``'s default ``indexing='xy'`` it returns ``(d/dy, d/dx)``, the opposite of
what the variable names suggest. Getting it backwards transposes the Jacobian and yields a
critical curve that still looks entirely plausible -- wrong only by the theta_E
cross-check, which is why that check is kept below even though the curve now comes from
the library. ``LensSolver`` uses forward-mode AD on ``LensModel.total_deflection`` and is
pinned bit-identical to the simulator's own trace by gigalens' test suite, so it is both
exact and incapable of drifting from the model that was fitted.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm

import jax.numpy as jnp

from gigalens.jax.analysis import LensSolver, image_border_in_source_plane

import diagnose
import lenses


def curve_radius(curve, center):
    """Mean distance of a curve's vertices from ``center`` (arcsec).

    ``curve`` is a ``(xs, ys)`` pair, the form ``LensSolver.critical_and_caustics``
    returns -- not the ``(N, 2)`` array the hand-rolled contour code used to produce.
    """
    xs, ys = curve
    return float(np.mean(np.hypot(xs - center[0], ys - center[1])))


def check_theta_E(curves, center, theta_E, gamma=None):
    """Print each critical curve's mean radius against theta_E; identify the tangential
    one (the largest) and flag loudly if it disagrees by more than 10%.

    HOW MANY CURVES TO EXPECT, which is a check in itself. The radial eigenvalue of a
    circular EPL is ``1 - (2-gamma)(theta/theta_E)^(1-gamma)``, which vanishes only where
    ``(theta/theta_E)^(1-gamma) = 1/(2-gamma)``. For **gamma >= 2 the right-hand side is
    non-positive, so there is no radial critical curve at all** -- a gamma > 2 fit must
    produce exactly ONE curve, the tangential one at theta_E. Two curves are expected
    only for gamma < 2. An extra inner curve at a radius of order the grid spacing is
    the signature of a contour finder tripping over the ``alpha ~ theta^(2-gamma)``
    divergence at the origin, not of real lens structure; this file's earlier
    finite-difference implementation did exactly that, which is part of why the curves
    now come from LensSolver.

    Only the tangential curve is expected to track theta_E closely.
    """
    if not curves:
        print("VALIDATION FAILED: no critical curve found at all.")
        return
    radii = sorted((curve_radius(c, center) for c in curves), reverse=True)
    tangential = radii[0]
    print(f"Critical curve radii (arcsec, sorted): {['%.4f' % r for r in radii]}")
    if gamma is not None and gamma >= 2.0 and len(radii) > 1:
        print(
            f"VALIDATION FAILED: gamma = {gamma:.3f} >= 2 admits no radial critical "
            f"curve, so exactly one curve is expected, but {len(radii)} were found. The "
            f"extra one(s) are a numerical artifact of the deflection's divergence at the "
            f"origin -- see check_theta_E's docstring. Distrust the caustics."
        )
    print(f"Tangential (largest) radius = {tangential:.4f}, theta_E = {theta_E:.4f}")
    frac = abs(tangential - theta_E) / theta_E
    if frac > 0.10:
        print(
            f"VALIDATION FAILED LOUDLY: tangential critical-curve radius disagrees with "
            f"theta_E by {frac * 100:.1f}% (> 10%). DO NOT trust this figure's critical "
            f"curve / caustics. With the curves coming from LensSolver this is more "
            f"likely to mean the fit itself is bad (an unconverged run, or a lens "
            f"centroid far from the light) than a geometry bug."
        )
    else:
        print(f"Validation OK: tangential radius agrees with theta_E to {frac * 100:.2f}%.")


def source_component_params(params, plane):
    """Light-component parameter dicts for ``plane``, in component-index order.

    ``model.constrained`` returns ``planes``, ``light`` and ``mass`` as **dicts keyed by
    integer index**, not lists. Iterating one yields the keys (ints), and ``params["planes"]``
    does not even iterate in index order -- so a ``for p in ...["light"]`` loop that looks
    exactly like a list iteration silently hands you integers. Index by ``range`` instead.
    """
    light = params["planes"][plane]["light"]
    return [light[j] for j in range(len(light))]


def render_source_plane(model, params, extent, n=400):
    """Render summed source-light surface brightness on a fresh (x, y) grid in the source
    plane -- i.e. calling ``light()`` directly on plain coordinates, not through the
    lens. ``extent = (xmin, xmax, ymin, ymax)`` in arcsec.

    The source plane is the last plane of the scene, and its light components are
    ``model.planes[-1].light`` with parameters at the matching path in the structured
    ``params`` dict. Indexing the plane list from the end rather than hardcoding ``1``
    keeps this correct if a deflector plane is ever inserted.
    """
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    Xg, Yg = np.meshgrid(xs, ys)  # indexing='xy', consistent with imshow(extent=...) below
    img = np.zeros_like(Xg)
    src = len(model.planes) - 1
    for j, comp in enumerate(model.planes[src].light):
        p = params["planes"][src]["light"][j]
        img = img + np.asarray(comp.profile.light(jnp.asarray(Xg), jnp.asarray(Yg), **p))
    return Xg, Yg, img


def make_figure(run_dir, out_dir, lens=None):
    rm = diagnose.rebuild_model(run_dir, lens=lens)
    model, cfg, image = rm.model, rm.cfg, rm.image

    z_unc, label = diagnose.point_estimate(run_dir)
    im_sim = diagnose.simulate_from_unconstrained(rm, z_unc)

    # Unbatched z -> a structured params dict of plain scalars, which is what both the
    # finite-difference deflection grid and the source render below want. Passing a
    # batched (1, n) vector instead would give every parameter a length-1 leading axis and
    # broadcast the critical-curve grid into a spurious extra dimension.
    params = model.constrained(jnp.asarray(np.asarray(z_unc).reshape(-1)))
    epl_params = {k: float(np.asarray(v)) for k, v in params["planes"][0]["mass"][0].items()}
    theta_E = epl_params["theta_E"]
    lens_center = (epl_params["center_x"], epl_params["center_y"])

    npix, delta = cfg.num_pix, cfg.delta_pix
    half_extent_image = npix / 2 * delta

    # LensSolver defaults its tracing window to the cutout's own angular extent and its
    # scale to delta_pix/supersample, so it needs neither here. It reads deflection from
    # rm.prob's model -- the same object the fit used.
    solver = LensSolver(rm.prob)
    curves, caustics = solver.critical_and_caustics(params)
    check_theta_E(curves, lens_center, theta_E, gamma=epl_params.get("gamma"))

    # --- Panel 3 data: normalised residual, same convention as diagnose.residual_figure.
    # error_map is the dataset's, fixed from the observed image -- not recomputed from
    # im_sim. See the module docstring. ---
    err_map = np.asarray(rm.prob.error_map)
    norm_res = (image - im_sim) / err_map
    chi2 = float(np.mean(norm_res ** 2))
    print(f"Reduced chi-squared (mean of normalised-residual^2): {chi2:.4f}")

    # --- Panel 4 data: source plane, sized from the fitted source centres/radii and the
    # caustic extent, deliberately NOT sharing axes with the image-plane panels (arcsec
    # scale is similar but the physically relevant window is much smaller). ---
    src_plane = len(model.planes) - 1
    src_centers = []
    src_radii = []
    # TRAP: params["planes"][i]["light"] is a dict keyed by component index, not a list --
    # and params["planes"] itself does not even iterate in index order. Iterating it
    # directly yields the integer keys, so index explicitly.
    for p in source_component_params(params, src_plane):
        src_centers.append((float(np.asarray(p["center_x"])),
                             float(np.asarray(p["center_y"]))))
        src_radii.append(float(np.asarray(p["R_sersic"])))
    cxs = [c[0] for c in src_centers]
    cys = [c[1] for c in src_centers]
    pad = max(src_radii) * 4 if src_radii else 0.3
    xmin, xmax = min(cxs) - pad, max(cxs) + pad
    ymin, ymax = min(cys) - pad, max(cys) + pad
    # The window is widened to the ray-traced image of the cutout border, which answers
    # the question this panel is actually asking: a source is observable only if one of
    # its images lands in the cutout, so beta(dR) is the edge of the observable region.
    # The caustic is a property of the mass model, the border a property of the
    # observation -- two different questions, and the border is the one that bounds where
    # a source could be.
    #
    # It is also robust in a way a caustic-derived window is not. The border is traced at
    # the cutout boundary, where the deflection is small and smooth; a caustic is the
    # image of a critical curve, so any spurious critical curve near the origin (where
    # alpha ~ theta^(2-gamma) diverges for gamma > 2) maps to an enormous caustic and
    # would blow the window out. See check_theta_E for why gamma > 2 admits exactly one
    # real critical curve.
    border_x, border_y = image_border_in_source_plane(solver, params)
    xmin = min(xmin, float(border_x.min()))
    xmax = max(xmax, float(border_x.max()))
    ymin = min(ymin, float(border_y.min()))
    ymax = max(ymax, float(border_y.max()))

    Xs, Ys, src_img = render_source_plane(
        model, params, (xmin, xmax, ymin, ymax), n=400
    )

    _draw(
        out_dir, image, im_sim, norm_res, chi2, label, cfg,
        curves, caustics, (border_x, border_y), lens_center, theta_E,
        Xs, Ys, src_img, src_centers, (xmin, xmax, ymin, ymax),
        npix, delta,
    )
    return chi2


def _draw(out_dir, image, im_sim, norm_res, chi2, label, cfg,
          curves, caustics, border, lens_center, theta_E,
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
    vmin = -5.0 * cfg.background_rms
    linear_width = 5.0 * cfg.background_rms
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
    for cxs, cys in curves:
        ax.plot(cxs, cys, color="lime", lw=1.2)
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
    for cxs, cys in caustics:
        ax.plot(cxs, cys, color="orange", lw=1.2)
    ax.plot(border[0], border[1], color="white", lw=1.2, ls="--")
    for i, (cx, cy) in enumerate(src_centers):
        ax.plot(cx, cy, "x", color="red", ms=10, mew=2)
        ax.annotate(f"src{i}", (cx, cy), color="red", fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Source plane (asinh) + caustics, dashed = cutout border")
    # Pin the axes to the computed window: `ax.plot` of a caustic otherwise autoscales the
    # panel out to its most distant vertex, so choosing a window would have no effect at
    # all. Any off-window caustic segment runs off the edge, which is the intent.
    ax.set_xlim(src_extent[0], src_extent[1])
    ax.set_ylim(src_extent[2], src_extent[3])
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
    lenses.add_argument(ap, required=False)
    args = ap.parse_args()

    out_dir = args.out or args.run_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    make_figure(args.run_dir, out_dir, lens=args.lens)


if __name__ == "__main__":
    main()
