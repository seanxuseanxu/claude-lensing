"""Azimuthal-median clump photometry for a lens fit.

This is what tells you whether a new source model actually fixed an underfit arc, as
opposed to improving a global chi-squared by redistributing light somewhere else. It
previously existed only as throwaway ``python -c`` commands; this file makes it
reproducible.

Two things that silently go wrong if you are not careful:

- **A config's clump PA values are array-space angles**, measured from the +row axis
  toward +col in the crop -- NOT sky position angles. Converting them through the image's
  WCS (the "obvious" thing to do with a PA) places every marker on empty sky. The
  conversion used below is exactly:

      row = row0 + (r / delta_pix) * cos(radians(PA))
      col = col0 + (r / delta_pix) * sin(radians(PA))

- **Reading a run's checkpoints through the wrong prior silently produces plausible,
  wrong numbers.** A run's metadata.json records which prior variant it was fitted
  against, and runs predating a given key used the values in force at the time -- not
  today's defaults. Getting a fallback wrong doesn't raise; it runs the stored
  unconstrained samples through a bijector with a different support and reports a
  different, wrong constrained value. That reconstruction is centralised in
  ``diagnose.rebuild_model``, which this file calls rather than reimplementing. See
  diagnose.py's module docstring and priors.py's SOURCE_PRIOR_VERSION docs.

The photometry below is a *flux* comparison and never divides by the noise, so nothing
here depends on the noise convention. If that changes, take the error map from
``rm.prob.error_map`` -- fixed from the observed image -- rather than recomputing one
from the model image.

Why azimuthal-median subtraction at all: the deflector's own smooth light dominates the
image by roughly an order of magnitude over the arc/clump flux, so raw pixel differences at
clump locations are swamped by the residual of the *deflector* light, not the clumps.
(A mask does not help here and is not the point: ``ImageData`` does take a ``mask=``, which
``data.build_mask`` supplies from ``cfg.mask_circles``, but masking excludes interlopers
from the likelihood -- it does nothing about the deflector's light *under* the arc, which
is what this subtraction is for.) Azimuthal
median subtraction removes the axisymmetric part of the deflector without touching
non-axisymmetric structure (arcs, clumps) -- as long as it's applied identically to data
and model, which is what makes the resulting ratio meaningful.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import diagnose
import lenses


def azimuthal_median_subtract(img, center):
    """Return ``img`` minus its azimuthal median profile about ``center``.

    ``center`` is ``(row, col)`` in pixel coordinates (not necessarily integer).
    Pixels are binned by *integer* radius (``round(distance)``) from the centre, the
    median is taken in each annulus, and the resulting 1-D profile is linearly
    interpolated back to each pixel's (real-valued) radius before subtracting. Binning by
    integer radius rather than a finer grid keeps every annulus populated with enough
    pixels near the centre (where the deflector light -- and the need for a robust
    median -- matters most) that the median isn't computed on a handful of points.
    """
    img = np.asarray(img)
    ny, nx = img.shape
    row0, col0 = center

    rows, cols = np.indices((ny, nx))
    r = np.sqrt((rows - row0) ** 2 + (cols - col0) ** 2)
    r_int = np.round(r).astype(int)

    r_max = r_int.max()
    profile = np.full(r_max + 1, np.nan)
    for k in range(r_max + 1):
        sel = r_int == k
        if np.any(sel):
            profile[k] = np.median(img[sel])
    # Fill any empty bins (possible for large k, where an integer-radius annulus at the
    # corners of a square image can be sparse) by interpolating over valid ones so the
    # subtraction below never subtracts a NaN.
    valid = ~np.isnan(profile)
    profile[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), profile[valid])

    bin_edges = np.arange(r_max + 1)
    model = np.interp(r.ravel(), bin_edges, profile).reshape(ny, nx)
    return img - model


def clump_positions(cfg, center):
    """Return ``{name: (row, col)}`` for the clumps, in array-space pixel coords.

    Uses the array-space PA convention documented in the module docstring and in
    RESULTS.md -- NOT a WCS/sky conversion. Clump radii and position angles come from
    ``cfg.clumps``, which stores them as (r_arcsec, pa_deg) pairs. The PA convention is
    array-space: measured from the +row axis toward +col within the crop. Converting these
    through the image WCS the standard way (North through East) places the markers on
    empty sky; that is a known trap documented in clumps.py's module docstring and in
    RESULTS.md.
    """
    row0, col0 = center
    positions = {}
    for name, (r_arcsec, pa_deg) in cfg.clumps.items():
        r_pix = r_arcsec / cfg.delta_pix
        pa = np.radians(pa_deg)
        row = row0 + r_pix * np.cos(pa)
        col = col0 + r_pix * np.sin(pa)
        positions[name] = (row, col)
    return positions


def aperture_flux(img, row, col, radius=3.0):
    """Sum of pixels whose centre lies within ``radius`` px of ``(row, col)``.

    Pixel centres are at integer (row, col) coordinates -- consistent with
    ``np.indices`` used in ``azimuthal_median_subtract`` -- so this is a plain circular
    aperture, no sub-pixel weighting.
    """
    img = np.asarray(img)
    ny, nx = img.shape
    rows, cols = np.indices((ny, nx))
    sel = (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2
    return float(np.sum(img[sel]))


def compare(run_dir, out_dir, radius=3.0, lens=None):
    """Az-median-subtract data and the model's point estimate; print and plot the table.

    Reconstructs the model variant from ``run_dir``'s ``metadata.json`` using
    diagnose.rebuild_model, which handles all fallback defaults for runs that predate
    ``metadata.json`` keys. Reading a run through the wrong prior silently reports wrong
    physical values -- see the module docstring.
    """
    rm = diagnose.rebuild_model(run_dir, lens=lens)
    cfg = rm.cfg

    z_unc, label = diagnose.point_estimate(run_dir)
    print(f"Point estimate: {label}")
    im_sim = diagnose.simulate_from_unconstrained(rm, z_unc)

    center = cfg.centroid_pix
    data_sub = azimuthal_median_subtract(rm.image, center)
    model_sub = azimuthal_median_subtract(im_sim, center)

    positions = clump_positions(cfg, center)

    print(f"\n=== Azimuthal-median clump photometry (r <= {radius} px) ===")
    print(f"{'clump':<10}{'data':>10}{'model':>10}{'ratio':>10}")
    rows_out = []
    for name, (row, col) in positions.items():
        d = aperture_flux(data_sub, row, col, radius=radius)
        m = aperture_flux(model_sub, row, col, radius=radius)
        ratio = m / d if d != 0 else float("nan")
        rows_out.append((name, d, m, ratio))
        print(f"{name:<10}{d:>10.2f}{m:>10.2f}{ratio:>10.2f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = _az_median_figure(data_sub, model_sub, positions, out_dir)
    print(f"\nFigure: {fig_path}")

    return rows_out


def _az_median_figure(data_sub, model_sub, positions, out_dir):
    """Two-panel figure: az-median-subtracted data (left) vs. model (right).

    Shared symmetric colour scale so the two panels are directly comparable by eye, not
    just by the printed table. Clump positions are marked with green circles at the
    ``aperture_flux`` radius so the figure visually documents what the table measures.
    """
    vlim = float(np.percentile(np.abs(data_sub), 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, img, title in zip(axes, (data_sub, model_sub), ("Data", "Model")):
        ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(f"{title} (azimuthal-median subtracted)")
        for name, (row, col) in positions.items():
            circ = plt.Circle((col, row), 3.0, edgecolor="green", facecolor="none", lw=1.5)
            ax.add_patch(circ)
            ax.annotate(name, (col, row), color="green", fontsize=8,
                        xytext=(3, 3), textcoords="offset points")

    fig.tight_layout()
    path = Path(out_dir) / "az_median_compare.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Azimuthal-median clump photometry for a lens fit"
    )
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    ap.add_argument("--out", default=None, help="Output directory (default: run_dir)")
    ap.add_argument("--radius", type=float, default=3.0, help="Aperture radius, px")
    lenses.add_argument(ap, required=False)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir

    compare(run_dir, out_dir, radius=args.radius, lens=args.lens)


if __name__ == "__main__":
    main()
