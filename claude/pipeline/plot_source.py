"""Unlensed vs. lensed source components for a lens fit.

Produces one figure, ``source_components.png``, for a two-source run: each source-plane
Sersic component next to its image-plane (lensed) contribution, plus a total row. This
answers a question the existing diagnostics don't: ``residuals.png`` shows the *sum* of
everything, and ``clumps.py`` shows deflector-subtracted flux at fixed clump apertures --
neither shows what a single source component actually looks like before or after lensing,
which matters for judging whether two components produce genuinely separate images (the
"detached clumps" claim in RESULTS_v3.md) rather than one smeared blob.

Model reconstruction is done by calling ``diagnose.rebuild_model``, which handles all
fallback defaults for runs that predate ``metadata.json`` keys.

Two traps specific to *this* figure that don't show up in clumps.py or diagnose.py:

- ``SceneSimulator.simulate`` returns a PSF-convolved, average-pooled, conversion-factor-
  scaled image -- but building a single component's contribution means calling
  ``component.profile.light(...)`` directly, which skips all three of those steps. The
  pooling and conversion factor are reproduced by hand below so the per-component lensed
  panels are on the same flux scale as the real model image; PSF convolution is *not*
  reproduced (see the docstring on ``render_lensed_component``) and the figure says so.
- The unlensed source-plane view needs its own coordinate grid -- ``sim.img_X/img_Y`` are
  image-plane coordinates. Evaluating ``light()`` on those would silently return a
  correctly-shaped but physically meaningless array (the profile evaluated at the wrong
  place), not an error.

The deflected positions come from ``sim.trace(params)``, which delegates to
``LensModel.trace_planes`` -- the one ray trace the simulator and the lens-equation solver
also use. It returns a ``{plane_index: (x, y)}`` dict, so the source plane's coordinates
are looked up by that plane's index rather than unpacked positionally.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

import diagnose
import lenses


def build_model(run_dir, lens=None):
    """Reconstruct the model using diagnose.rebuild_model."""
    rm = diagnose.rebuild_model(run_dir, lens=lens, quiet=True)
    print(
        f"Model variant: n_source={rm.n_source}, n_lens_light={rm.n_lens_light}, "
        f"source_prior_v{rm.source_prior_version}, lens_light_prior_v{rm.lens_light_prior_version}"
    )
    return rm


def render_lensed_component(sim, beta_x, beta_y, light_profile, params):
    """Return one source component's image-plane contribution, on the *pipeline* flux scale.

    ``light_profile.light(beta_x, beta_y, **params)`` alone is at supersampled resolution,
    not PSF-convolved, and not multiplied by ``conversion_factor`` -- so it is not
    comparable to the real model image or to ``diagnose.simulate_from_unconstrained``'s
    output on its own. This reproduces the two steps of ``SceneSimulator.simulate`` that
    matter for a *per-component* comparison (average-pool by ``sim.supersample``, then
    multiply by ``sim.conversion_factor``) but deliberately skips PSF convolution: the
    kernel mixes flux across components once applied, so a convolved single-component
    panel would not sum cleanly to the convolved total. This is noted on the figure.
    """
    from objax.constants import ConvPadding
    from objax.functional import average_pool_2d

    img = light_profile.light(beta_x, beta_y, **params)  # (H, W, 1), H=W=supersample*num_pix
    img = jnp.transpose(img, (2, 0, 1))[:, jnp.newaxis, ...]  # (1, 1, H, W) for pooling
    if sim.supersample != 1:
        img = average_pool_2d(img, size=sim.supersample, padding=ConvPadding.SAME)
    img = jnp.squeeze(img) * sim.conversion_factor
    return np.asarray(img)


def source_plane_grid(num_pix, half_extent):
    """Return (X, Y) coordinate arrays in arcsec, shape (num_pix, num_pix, 1).

    Mean-centred like the simulator's own ``wcs.pixel_grid()`` builds the
    image-plane grid, so that "arcsec from the origin" means the same thing in both
    planes. Y increases with row index here, matching that same convention (row 0 is one
    extreme of the axis, not flipped for image display -- imshow's ``origin="lower"``
    below keeps the two consistent).
    """
    lo = np.linspace(-half_extent, half_extent, num_pix, dtype=np.float32)
    X, Y = np.meshgrid(lo, lo)
    return X[..., np.newaxis], Y[..., np.newaxis]


def make_figure(run_dir, out_dir, lens=None):
    """Build and save ``source_components.png``; return the two validation numbers.

    Layout is 3 rows (component 0, component 1, total) x 2 cols (unlensed source-plane,
    lensed image-plane). Row 2 ("total") is the sum of the two component rows, not a
    third independent render -- this is what the sum-check below verifies is legitimate.
    """
    run_dir = Path(run_dir)
    rm = build_model(run_dir, lens=lens)
    model, cfg, sim_config, n_source = rm.model, rm.cfg, rm.sim_config, rm.n_source
    if n_source != 2:
        raise ValueError(
            f"plot_source.py assumes a 2-row/2-col two-source layout; run has n_source={n_source}"
        )

    z_unc, label = diagnose.point_estimate(run_dir)
    print(f"Point estimate: {label}")

    # Unbatched z -> scalar params. The simulator's grid carries a trailing singleton
    # batch axis that the params broadcast onto, so one instance renders one sample; there
    # is no bs= argument any more.
    params = model.constrained(jnp.asarray(np.asarray(z_unc).reshape(-1)))
    sim = rm.simulator

    src_plane = len(model.planes) - 1
    src_components = model.planes[src_plane].light
    # TRAP: params["planes"][i]["light"] is a dict keyed by component index, not a list, so
    # `zip(src_components, that_dict)` would pair each component with an integer key rather
    # than its parameters -- and fail somewhere further down, as a TypeError about ints.
    light_params = params["planes"][src_plane]["light"]
    source_params = [light_params[j] for j in range(len(light_params))]

    # trace() returns {plane_index: (x, y)}; take the source plane's entry by key rather
    # than by position, since the dict spans every plane in the scene.
    beta_x, beta_y = sim.trace(params)[src_plane]

    # Source-plane window: cover both fitted centres plus several R_sersic, so a compact
    # component near the edge of a smaller window isn't clipped.
    centers = []
    radii = []
    for p in source_params:
        cx, cy = float(np.asarray(p["center_x"])), float(np.asarray(p["center_y"]))
        r = float(np.asarray(p["R_sersic"]))
        centers.append((cx, cy))
        radii.append(r)
    half_extent = max(abs(c[0]) + 6 * r for c, r in zip(centers, radii))
    half_extent = max(half_extent, max(abs(c[1]) + 6 * r for c, r in zip(centers, radii)))
    half_extent = max(half_extent, 0.3)  # floor, in case both components are tiny/central
    print(f"Source-plane window: +/-{half_extent:.3f} arcsec")

    src_X, src_Y = source_plane_grid(sim_config.num_pix, half_extent)

    image_extent = sim_config.num_pix * cfg.delta_pix / 2

    # Unlensed source-plane light for each component, evaluated on the source-plane grid.
    unlensed = [
        np.asarray(comp.profile.light(src_X, src_Y, **p)).squeeze()
        for comp, p in zip(src_components, source_params)
    ]
    # Lensed image-plane contribution for each component, on the pipeline flux scale
    # (pooled + conversion-factor-scaled, not PSF-convolved -- see render_lensed_component).
    lensed = [
        render_lensed_component(sim, beta_x, beta_y, comp.profile, p)
        for comp, p in zip(src_components, source_params)
    ]

    # Validation 1: per-component lensed contributions (each pooled+scaled independently)
    # must sum to the SAME total computed by summing the raw source light first, then
    # pooling+scaling once. Pooling is linear, so these two orders must agree to float
    # precision; a mismatch would mean render_lensed_component's per-component pipeline
    # is wrong, not just imprecise.
    lensed_sum = lensed[0] + lensed[1]
    raw_total = jnp.zeros_like(sim.img_X)
    for comp, p in zip(src_components, source_params):
        raw_total = raw_total + comp.profile.light(beta_x, beta_y, **p)
    lensed_total_direct = render_lensed_component(
        sim, beta_x, beta_y,
        _IdentityLight(raw_total), {}
    )
    sum_check_max_diff = float(np.max(np.abs(lensed_sum - lensed_total_direct)))
    print(f"Validation (component sum vs. joint total, source light only): "
          f"max abs diff = {sum_check_max_diff:.3e}")

    # Validation 2: full reconstruction against diagnose.simulate_from_unconstrained, the
    # call every other diagnostic in this repo trusts as ground truth.
    #
    # Be honest about what this is worth now: pre-port it compared an independently
    # constructed LensSimulator against diagnose's, and could catch a real disagreement.
    # Under the scene API both sides go through the same rm.simulator and the same
    # model.constrained, so it is very nearly tautological and should be expected to come
    # out at exactly 0.0. It is kept only as a cheap guard that the params dict built here
    # is the one simulate() accepts -- a shape or plane-indexing mistake above would
    # surface here rather than as a subtly wrong figure. It is NOT evidence that the
    # per-component rendering is right; validation 1 is what tests that.
    im_reference = diagnose.simulate_from_unconstrained(rm, z_unc)
    im_manual = np.asarray(sim.simulate(params)).reshape(sim_config.num_pix, sim_config.num_pix)
    full_max_diff = float(np.max(np.abs(im_manual - im_reference)))
    print(f"Validation (manual SceneSimulator.simulate vs. diagnose.simulate_from_unconstrained): "
          f"max abs diff = {full_max_diff:.3e}")

    fig_path = _draw_figure(
        out_dir, run_dir, cfg, unlensed, lensed, centers, half_extent, image_extent,
        sum_check_max_diff, full_max_diff,
    )
    return fig_path, sum_check_max_diff, full_max_diff


class _IdentityLight:
    """Wrap a precomputed array so ``render_lensed_component`` can pool+scale it.

    ``render_lensed_component`` expects a profile object with a ``.light(x, y, **p)``
    method; this lets it pool and flux-scale an already-summed raw light array (used only
    for the sum-check, where the "total" needs the identical pooling/scaling code path as
    each individual component, not a hand-rolled duplicate that could drift out of sync).
    """

    def __init__(self, array):
        self._array = array

    def light(self, x, y, **kwargs):
        return self._array


def _asinh_norm(arr, linthresh_frac=1e-3):
    """asinh-scale an array for display: linear near zero, log in the wings.

    A compact Sersic spans several decades of surface brightness within a handful of
    pixels, so a linear colour scale shows only the peak and a log scale can't represent
    the (irrelevant but present) near-zero floor. ``linthresh`` is set relative to the
    array's own max rather than a fixed number, since the two components differ in peak
    brightness by roughly a factor of 5 (Ie 18.4 vs. 86.8 at the point estimate).
    """
    vmax = float(np.max(arr))
    linthresh = max(vmax * linthresh_frac, 1e-12)
    return matplotlib.colors.SymLogNorm(linthresh=linthresh, vmin=0, vmax=vmax, base=10)


def _draw_figure(out_dir, run_dir, cfg, unlensed, lensed, centers, half_extent, image_extent,
                  sum_check_max_diff, full_max_diff):
    """3x2 figure: rows = [source 0, source 1, total], cols = [source-plane, image-plane].

    Source-plane panels use asinh/log colour scaling (labelled in each title) because of
    the dynamic range noted in ``_asinh_norm``; image-plane panels use a plain linear
    scale since after lensing the arcs are much less sharply peaked. Both column types
    get their own colourbar per row so the total row (which has higher summed flux than
    either component row) doesn't wash out the component rows.
    """
    fig, axes = plt.subplots(3, 2, figsize=(9, 12.5))
    ext_src = [-half_extent, half_extent, -half_extent, half_extent]
    ext_img = [-image_extent, image_extent, -image_extent, image_extent]

    unlensed_total = unlensed[0] + unlensed[1]
    lensed_total = lensed[0] + lensed[1]
    rows = [
        ("Source component 0", unlensed[0], lensed[0]),
        ("Source component 1", unlensed[1], lensed[1]),
        ("Total (0 + 1)", unlensed_total, lensed_total),
    ]

    for i, (title, src_img, img_img) in enumerate(rows):
        ax_src, ax_img = axes[i, 0], axes[i, 1]

        norm = _asinh_norm(src_img)
        im0 = ax_src.imshow(src_img, origin="lower", extent=ext_src, cmap="magma", norm=norm)
        ax_src.set_title(f"{title}: unlensed (source plane, asinh/log scale)", fontsize=9)
        ax_src.set_xlabel("arcsec")
        ax_src.set_ylabel("arcsec")
        cb0 = fig.colorbar(im0, ax=ax_src, fraction=0.046, pad=0.04)
        cb0.set_label("surface brightness (electrons/s/pix, intrinsic)")
        if i < 2:
            cx, cy = centers[i]
            ax_src.plot(cx, cy, "+", color="cyan", ms=12, mew=2)

        im1 = ax_img.imshow(img_img, origin="lower", extent=ext_img, cmap="viridis")
        ax_img.set_title(f"{title}: lensed (image plane, no PSF, linear scale)", fontsize=9)
        ax_img.set_xlabel("arcsec")
        ax_img.set_ylabel("arcsec")
        cb1 = fig.colorbar(im1, ax=ax_img, fraction=0.046, pad=0.04)
        cb1.set_label("surface brightness (electrons/s/pix)")

    fig.suptitle(
        f"{cfg.name} {Path(run_dir).name}: source components, unlensed vs. lensed\n"
        f"component-sum check {sum_check_max_diff:.1e}  |  "
        f"full-reconstruction check {full_max_diff:.1e}  |  "
        f"image-plane panels skip PSF convolution",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = Path(out_dir) / "source_components.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory, e.g. runs/bulgedisk_long")
    parser.add_argument("--out", default=None, help="Output directory (default: run_dir)")
    lenses.add_argument(parser, required=False)
    args = parser.parse_args()

    out_dir = args.out if args.out is not None else args.run_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig_path, sum_check_max_diff, full_max_diff = make_figure(args.run_dir, out_dir, lens=args.lens)
    print(f"\nFigure: {fig_path}")
    print(f"Component-sum max abs diff: {sum_check_max_diff:.6e}")
    print(f"Full-reconstruction max abs diff: {full_max_diff:.6e}")


if __name__ == "__main__":
    main()
