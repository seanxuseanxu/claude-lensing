"""Reduced chi-squared over the 403 pixels rounds 1-2 masked, for a run that saw them.

This is the number `RESULTS_v3.md` section 5 staked round 3's prediction on: if the SW object
is the fourth image, the *existing* source components reproduce it through the same mass
model, and these pixels are ordinary. If it is a foreground galaxy, no lensed configuration
can put light there and they carry a large coherent positive residual.

Method. The residual is built exactly the way `diagnose.residual_figure` builds it -- the
model rendered by `SceneSimulator` at the run's own HMC-median parameters, differenced
against the data and divided by the same `err_map` the likelihood uses -- and reduced by
`diagnose.masked_chisq`, the pipeline's single chi-squared formula. Nothing here rolls its
own normalisation (CLAUDE.md). The comparison region is `data.build_mask`'s own disc, so the
"previously masked" set is the identical 403 pixels rounds 1-2 dropped, not a re-derived one.

The control matters as much as the number: chi2_nu inside the disc is only meaningful against
chi2_nu outside it, on the same image, at the same parameters. A model that fits the whole
field badly would also fit the disc badly, and that would say nothing about the SW object.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline \
        python claude/DESI257/analysis/sw_region_chisq.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import data as data_mod  # noqa: E402
import diagnose  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r3_unmasked"

# The disc rounds 1 and 2 masked, as it stood in config.py before round 3 removed it. Held
# literally rather than read from the config, because the config no longer has it -- and
# reading it from r2_s2l2's metadata.json would tie this script to that run's file layout.
ROUND2_MASK_CIRCLES = ((8.0, 15.0, 12.0),)


def main():
    rm = diagnose.rebuild_model(RUN)
    cfg = rm.cfg

    image = np.asarray(rm.image)
    # The likelihood's own error map, reached the same way diagnose.main does
    # (diagnose.py:1228) -- not rebuilt from the data, which would be a second, unvalidated
    # noise model sitting alongside the one the fit actually used.
    err_map = np.asarray(rm.prob.error_map)

    z_unc, _ = diagnose.point_estimate(RUN)
    model_img = np.asarray(diagnose.simulate_from_unconstrained(rm, z_unc))

    norm_res = (image - model_img) / err_map

    # build_mask returns True for KEPT pixels, so the round-2 mask's False set is exactly
    # the 403 pixels that were dropped -- the region of interest here.
    keep_r2 = data_mod.build_mask(
        type(cfg)(**{**cfg.__dict__, "mask_circles": ROUND2_MASK_CIRCLES}), image.shape
    )
    keep_r2 = np.asarray(keep_r2, dtype=bool)
    inside = ~keep_r2

    chi2_in, n_in = diagnose.masked_chisq(norm_res, inside)
    chi2_out, n_out = diagnose.masked_chisq(norm_res, keep_r2)
    chi2_all, n_all = diagnose.masked_chisq(norm_res, np.ones_like(inside))

    res_in = norm_res[inside]
    print(f"run: {RUN}")
    print(f"previously-masked disc : chi2_nu {chi2_in:8.4f}  over {n_in} px")
    print(f"everywhere else        : chi2_nu {chi2_out:8.4f}  over {n_out} px")
    print(f"whole image            : chi2_nu {chi2_all:8.4f}  over {n_all} px")
    print()
    # A foreground galaxy the model cannot produce shows up as a MEAN residual well above
    # zero, not merely as a large chi-squared: chi-squared is blind to sign, and the whole
    # discriminating content of this test is whether unmodelled flux is left over.
    print(f"mean normalised residual inside disc : {res_in.mean():+.4f} sigma")
    print(f"mean normalised residual outside     : {norm_res[keep_r2].mean():+.4f} sigma")
    print(f"max  normalised residual inside disc : {res_in.max():+.4f} sigma")
    print(f"min  normalised residual inside disc : {res_in.min():+.4f} sigma")
    print(f"fraction of disc pixels beyond +3 sigma: {(res_in > 3).mean():.4f}")
    print(f"fraction of disc pixels beyond -3 sigma: {(res_in < -3).mean():.4f}")
    print()
    # Peak data flux in the disc, against the model's own peak there: the SW object reaches
    # 0.957 e/s in the data (README.md). If the model reproduces it, its peak matches.
    print(f"data  peak inside disc: {image[inside].max():.4f} e/s")
    print(f"model peak inside disc: {model_img[inside].max():.4f} e/s")

    # --- Controls: the same statistic on the OTHER lensed images -------------------------
    #
    # This is the part that makes the number above mean anything. chi2_nu = 2.05 inside the
    # disc reads as a misfit against the field average of 1.15 -- and that is the wrong
    # comparison, because the field average is dominated by blank sky. The right comparison
    # is the same statistic on the other bright compact structure in the same image, at the
    # same parameters and the same error map. Reporting the disc without these controls is
    # how a perfectly ordinary chi-squared gets read as evidence.
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    def disc(cy, cx, r=12.0):
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    def data_peak(r0, r1, c0, c1):
        """Centre a control disc on the DATA's own local maximum, so the comparison regions
        are the images themselves rather than three hand-placed guesses."""
        sub = image[r0:r1, c0:c1]
        i = np.unravel_index(np.argmax(sub), sub.shape)
        return (r0 + i[0], c0 + i[1])

    controls = [
        ("SW object (prev. masked)", (8.0, 15.0)),
        ("N blob", data_peak(50, 72, 28, 52)),
        ("E arc", data_peak(28, 52, 58, 78)),
        ("deflector core", (cfg.centroid_pix[0], cfg.centroid_pix[1])),
    ]
    print()
    print(f"{'region':<26} {'chi2_nu':>8} {'n px':>5} {'mean res':>10} {'peak data':>10}")
    for name, (cy, cx) in controls:
        m = disc(cy, cx)
        c2, n = diagnose.masked_chisq(norm_res, m)
        print(f"{name:<26} {c2:8.3f} {n:5d} {norm_res[m].mean():+10.4f} "
              f"{image[m].max():10.3f}")
    print(f"{'whole image':<26} {chi2_all:8.3f} {n_all:5d}")


if __name__ == "__main__":
    main()
