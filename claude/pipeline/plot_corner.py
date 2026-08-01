"""Full and mass-only corner plots for a completed lens fit.

``diagnose.py`` already draws one corner plot, but it hardcodes a 6-parameter mass subset
(``theta_E, gamma, e1, e2, gamma1, gamma2`` -- no centroid) and has no path to plotting
every parameter. This script produces both:

- ``corner_full.png`` -- all parameters (36 for a ``bulgedisk`` run: 2 lens-mass
  components x N + 2 lens-light Sersic components + 2 source Sersic components).
- ``corner_mass.png`` -- the 8 lens-mass parameters, including the centroid that
  ``diagnose.corner_plot``'s default subset omits.

Model reconstruction is done by calling ``diagnose.rebuild_model``, which handles all
fallback defaults for runs that predate ``metadata.json`` keys.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

import compat  # noqa: F401  -- applies JAX 0.11 shims on import



import diagnose
import lenses

MASS_PARAMS = [
    "lens.theta_E",
    "lens.gamma",
    "lens.e1",
    "lens.e2",
    "lens.center_x",
    "lens.center_y",
    "lens.gamma1",
    "lens.gamma2",
]

# Cap on total (draws x chains) samples fed to `corner`. `corner` bins every pair of
# parameters into a 2-D histogram/KDE, so its cost scales with sample count but not
# steeply -- the cap here is about keeping the full 36x36 figure's plotting time and
# rendered point count sane, not about a hard memory limit.
MAX_SAMPLES_FULL = 200_000


def build_prob_model(run_dir, lens=None):
    """Reconstruct prior/prob_model using diagnose.rebuild_model."""
    rm = diagnose.rebuild_model(run_dir, lens=lens)
    return rm.prior, rm.prob_model, rm.pnames


def thin_to_cap(samples, max_total):
    """Thin an (n_draws, n_chains, n_params) array along the draw axis to `max_total`.

    Thinning the draw axis rather than dropping whole chains preserves the mix of chains
    (relevant for eyeballing multimodality across starts) while cutting the dominant
    axis -- draws are typically far more numerous than chains and highly autocorrelated,
    so thinning them loses little independent information relative to dropping chains.
    """
    n_draws, n_chains, n_params = samples.shape
    total = n_draws * n_chains
    if total <= max_total:
        return samples, 1
    factor = int(np.ceil(total / max_total))
    return samples[::factor], factor


def make_corner(con, labels, truths, out_path, *, show_titles, dpi, figsize=None):
    import corner

    n = len(labels)
    if figsize is None:
        # ~0.9 in/param keeps a 36-param figure large enough to inspect on screen while
        # staying well short of matplotlib's pixel-size ceiling at reasonable dpi.
        side = max(6.0, 0.9 * n)
        figsize = (side, side)

    label_kwargs = {"fontsize": 6 if n > 12 else 10}
    tick_kwargs = {"labelsize": 5 if n > 12 else 9}

    fig = corner.corner(
        con,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=show_titles,
        title_fmt=".3f" if show_titles else None,
        label_kwargs=label_kwargs,
        title_kwargs={"fontsize": 6} if show_titles and n > 12 else {},
        fig=None,
    )
    fig.set_size_inches(*figsize)
    for ax in fig.get_axes():
        ax.tick_params(axis="both", which="major", **tick_kwargs)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Full and mass-only corner plots for a lens HMC run"
    )
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    ap.add_argument("--out", default=None, help="Output directory (default: run_dir)")
    lenses.add_argument(ap, required=False)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hmc = diagnose.load_checkpoint(run_dir, "hmc")
    if hmc is None:
        raise FileNotFoundError(f"No hmc.npz in {run_dir}; corner plots need HMC samples.")

    prior, prob_model, pnames = build_prob_model(run_dir, lens=args.lens)

    samples = hmc["samples"]  # (n_draws, n_chains, n_params)
    n_draws, n_chains, n_params = samples.shape
    print(f"HMC samples: {n_draws} draws x {n_chains} chains x {n_params} params")

    thinned, factor = thin_to_cap(samples, MAX_SAMPLES_FULL)
    n_used = thinned.shape[0] * thinned.shape[1]
    print(f"Thinning factor for corner_full: {factor} "
          f"({n_draws * n_chains} -> {n_used} samples)")

    con_full = diagnose.to_constrained(prob_model, thinned.reshape(-1, n_params))

    z_point, label = diagnose.point_estimate(run_dir)
    print(f"Point estimate for truth lines: {label}")
    con_point = diagnose.to_constrained(prob_model, z_point.reshape(1, -1))[0]

    # --- Full corner: all parameters ---
    full_path = out_dir / "corner_full.png"
    make_corner(
        con_full,
        labels=pnames,
        truths=con_point,
        out_path=full_path,
        show_titles=False,
        dpi=110,
    )
    print(f"Full corner ({n_params} params): {full_path} "
          f"({full_path.stat().st_size / 1e6:.2f} MB)")

    # --- Mass corner: 8 lens-mass parameters, unthinned (small enough as-is) ---
    idx = [pnames.index(n) for n in MASS_PARAMS]
    missing = [n for n in MASS_PARAMS if n not in pnames]
    if missing:
        raise KeyError(f"Expected mass parameters missing from pnames: {missing}")

    con_mass_full = diagnose.to_constrained(prob_model, samples.reshape(-1, n_params))
    con_mass = con_mass_full[:, idx]
    truths_mass = con_point[idx]

    mass_path = out_dir / "corner_mass.png"
    make_corner(
        con_mass,
        labels=[pnames[i] for i in idx],
        truths=truths_mass,
        out_path=mass_path,
        show_titles=True,
        dpi=150,
        figsize=(12, 12),
    )
    print(f"Mass corner (8 params): {mass_path} "
          f"({mass_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
