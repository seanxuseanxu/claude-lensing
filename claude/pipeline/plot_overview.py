"""CLI wrapper around ``diagnose.residual_figure`` that writes ``lensing_overview.png``.

This used to be a second, hand-duplicated implementation of the four-panel figure --
``diagnose.residual_figure`` didn't surface the raw lens parameters (critical curves,
caustics, the source render) that this file's fourth panel needed. Now that
``residual_figure`` itself produces observed+critical-curve, model+critical-curve,
unlensed-source+caustics and normalised-residual panels (2026-08-08, at the request that
``diagnose.py``'s automatic ``residuals.png`` needed no separate regeneration to be the
final-result figure), there is exactly one figure implementation, in ``diagnose.py``. This
file is kept only so the documented ``plot_overview.py <run_dir>`` invocation in
``README.md`` keeps working and keeps writing the filename existing result docs
(``claude/DESI238/RESULTS_v5.md`` etc.) already link to.
"""

import argparse
from pathlib import Path

import diagnose
import lenses


def make_figure(run_dir, out_dir, lens=None):
    rm = diagnose.rebuild_model(run_dir, lens=lens)
    z_unc, label = diagnose.point_estimate(run_dir)
    im_sim = diagnose.simulate_from_unconstrained(rm, z_unc)
    chi2, path = diagnose.residual_figure(
        out_dir, rm.cfg, rm.model, rm.prob, rm.image, rm.prob.error_map, im_sim, label,
        z_unc, mask=rm.mask, filename="lensing_overview.png",
    )
    n_kept = rm.image.size if rm.mask is None else int(rm.mask.sum())
    print(f"Reduced chi-squared at point estimate: {chi2:.4f} "
          f"(over {n_kept}/{rm.image.size} unmasked pixels)")
    print(f"Figure: {path}")
    return chi2


def main():
    ap = argparse.ArgumentParser(
        description="2x2 overview figure: data+critcurve/model+critcurve/source+caustics/residual"
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
