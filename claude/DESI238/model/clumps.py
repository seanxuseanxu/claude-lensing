"""Azimuthal-median clump photometry for a DESI238 fit.

This is the analysis behind the table in RESULTS.md ("Residual quality: correcting an
early misread") and the thing that will actually tell us whether a new source model fixed
the N arc underfit. It previously existed only as throwaway ``python -c`` commands; this
file makes it reproducible.

Two things that silently go wrong if you are not careful, both inherited from
diagnose.py and repeated here on purpose rather than imported blind:

- **The clump PA values in MODELING_PLAN.md are array-space angles**, measured from the
  +row axis toward +col in the 80x80 crop -- NOT sky position angles. Converting them
  through the image's WCS (the "obvious" thing to do with a PA) places every marker on
  empty sky. See RESULTS.md, "Residual quality: correcting an early misread". The
  conversion used below is exactly:

      row = row0 + (r / delta_pix) * cos(radians(PA))
      col = col0 + (r / delta_pix) * sin(radians(PA))

- **Reading a run's checkpoints through the wrong prior silently produces plausible,
  wrong numbers.** A run's metadata.json records which prior variant it was fitted
  against (n_source, lens_light_n_max, source_prior_version); a run with no "n_source"
  key at all (seed0, seed1, inner3) predates that key and used source_prior_version=1,
  the narrow original source prior -- NOT the current default. Getting this fallback
  wrong doesn't raise an error; it just runs the stored unconstrained samples through a
  bijector with a different support and reports a different (wrong) constrained value.
  See diagnose.py's module docstring and priors.py's SOURCE_PRIOR_VERSION docs for the
  concrete example (source.n_sersic: 0.5729 true vs. 0.6146 misread).

Why azimuthal-median subtraction at all: the deflector's own smooth light dominates the
image by roughly an order of magnitude over the arc/clump flux, and gigalens fits no mask
(``grep -rn "mask" gigalens/src`` returns nothing), so raw pixel differences at clump
locations are swamped by the residual of the *deflector* light, not the clumps. Azimuthal
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

import compat  # noqa: F401  -- applies JAX 0.11 shims on import

import jax.numpy as jnp

from gigalens.jax.model import ForwardProbModel

import data
import diagnose
import priors


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


def clump_positions(center):
    """Return ``{name: (row, col)}`` for the four clumps, in array-space pixel coords.

    Uses the array-space PA convention documented in the module docstring and in
    RESULTS.md -- NOT a WCS/sky conversion. ``priors.CLUMP_RADII`` gives radii in
    arcsec but not PA, so the PA values here are copied from MODELING_PLAN.md's table
    directly (they are not otherwise recorded in the codebase).
    """
    row0, col0 = center
    # name -> (r arcsec, PA degrees, array-space convention)
    clumps = {
        "SW arc": (1.06, 224.0),
        "E blob": (2.11, 88.0),
        "W clump": (1.64, 278.0),
        "N arc": (1.25, 339.0),
    }
    positions = {}
    for name, (r_arcsec, pa_deg) in clumps.items():
        r_pix = r_arcsec / data.DELTA_PIX
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


def compare(run_dir, out_dir, radius=3.0):
    """Az-median-subtract data and the model's point estimate; print and plot the table.

    Reconstructs the model variant from ``run_dir``'s ``metadata.json`` using exactly
    diagnose.main's fallback logic (``load_run_config`` + the ``n_source``/
    ``source_prior_version`` defaults), because reading a run through the wrong prior
    silently reports wrong physical values -- see the module docstring.
    """
    run_cfg = diagnose.load_run_config(run_dir)
    n_source = int(run_cfg.get("n_source", 1))
    lens_light_n_max = float(run_cfg.get("lens_light_n_max", 6.0))
    theta_e_center = float(run_cfg.get("theta_e_center", 1.5))
    theta_e_width = float(run_cfg.get("theta_e_width", 0.35))
    # Absence of "n_source" identifies a pre-revision run (seed0/seed1/inner3), fitted
    # against the narrow v1 source prior -- see diagnose.main's identical fallback.
    source_prior_version = int(
        run_cfg.get("source_prior_version", 2 if "n_source" in run_cfg else 1)
    )
    print(
        f"Model variant: n_source={n_source}, lens_light_n_max={lens_light_n_max}, "
        f"source_prior_v{source_prior_version}"
    )

    image, psf, sim_config = data.load()
    prior = priors.build_prior(
        theta_E_center=theta_e_center,
        theta_E_width=theta_e_width,
        n_source=n_source,
        lens_light_n_max=lens_light_n_max,
        source_prior_version=source_prior_version,
    )
    phys_model = priors.build_phys_model(n_source=n_source)
    prob_model = ForwardProbModel(
        prior,
        jnp.asarray(image),
        background_rms=data.BACKGROUND_RMS,
        exp_time=data.EXP_TIME,
    )

    z_unc, label = diagnose.point_estimate(run_dir)
    print(f"Point estimate: {label}")
    im_sim = diagnose.simulate_from_unconstrained(prob_model, phys_model, sim_config, z_unc)

    center = data.CENTROID_PIX
    data_sub = azimuthal_median_subtract(image, center)
    model_sub = azimuthal_median_subtract(im_sim, center)

    positions = clump_positions(center)

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
        description="Azimuthal-median clump photometry for a DESI238 fit"
    )
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    ap.add_argument("--out", default=None, help="Output directory (default: run_dir)")
    ap.add_argument("--radius", type=float, default=3.0, help="Aperture radius, px")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir

    compare(run_dir, out_dir, radius=args.radius)


if __name__ == "__main__":
    main()
