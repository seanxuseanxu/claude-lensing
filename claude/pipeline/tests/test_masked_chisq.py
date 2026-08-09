"""Regression test: ``diagnose.masked_chisq`` / ``diagnose.residual_figure`` respect the mask.

Run directly (pytest is not installed in this env; needs a real lens config and gigalens,
so unlike this directory's other tests it is not CPU-only):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline \\
        python claude/pipeline/tests/test_masked_chisq.py --lens DESI238

Moved out of ``verify.py`` (was check 7) on 2026-08-09: this asserts a fact about pipeline
code -- that two call sites agree with ``ProbModel``'s own reduced chi-squared -- true
independent of which lens is loaded, and it cost 10s of every preflight run for a claim
that does not change per lens.

``diagnose.residual_figure`` and ``plot_overview.py`` used to average squared normalised
residuals over EVERY pixel, ignoring the fit's boolean mask. Since masked pixels are
typically the field's worst residuals (that is why they were masked), this inflated the
reported reduced chi-squared badly: DESI234 read 4.225 for a true 1.560, DESI257 read 7.112
for a true 1.533 (2026-08-08). Both call sites are now fixed to sum ``(r/sigma)^2`` over
unmasked pixels only, divided by the unmasked count -- the same definition
``ProbModel.log_prob``'s second return value uses.

This test exercises the actual fixed function, ``diagnose.residual_figure``, rather than
reimplementing the formula: it builds a synthetic observation that is the model's own
render plus catastrophic outliers at a few pixels, masks exactly those pixels out, and
asserts ``residual_figure``'s returned chi2 agrees with ``ProbModel``'s own ``red_chi2``
(the ground truth, per CLAUDE.md's gigalens-is-truth rule) to tight tolerance. It also
computes the old, unmasked-mean value directly here -- not through the fixed code -- purely
to confirm the synthetic case actually discriminates: if that value did not disagree with
the masked one well outside tolerance, this test would still pass under the reverted bug,
and would be worthless.

``residual_figure`` and ``plot_overview.py``'s panel now share one formula,
``diagnose.masked_chisq``, rather than each hand-writing ``mean(norm_res[keep]**2)``. This
test calls that helper directly too (on the same synthetic residual array), so both call
sites are covered by construction: nothing here would pass if either had its own,
possibly-diverged copy of the formula.
"""

import argparse
import sys
import tempfile

import numpy as np

import jax.random

from gigalens.jax.scene_prob_model import ImageData, ProbModel

import diagnose
import lenses
import verify

RESULTS = []


def check(name):
    def wrap(fn):
        def run(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True, detail or ""))
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                import traceback
                traceback.print_exc()

        return run

    return wrap


@check("Residual chi-squared respects the mask")
def test_masked_chisq(cfg):
    image, sim_config, model, prob = verify.build(cfg, n_source=1, n_lens_light=1)
    z = model.bijector.inverse(model.prior.sample(1, seed=jax.random.PRNGKey(2)))

    sim = prob.simulators[0]
    # A batch of 1 squeezes to (H, W) rather than (1, H, W) -- the batch axis rides on the
    # params and broadcasts onto a singleton grid axis.
    im_sim = np.asarray(sim.simulate(model.constrained(z)))
    if im_sim.ndim == 3:
        im_sim = im_sim[0]

    # Synthetic "observed" image: the model's own render plus ordinary per-pixel noise,
    # then a few catastrophic outliers -- the DESI234/257 situation, where an unmodelled
    # interloper under mask_circles carries the field's largest residuals.
    rng = np.random.default_rng(0)
    sigma = max(0.02 * float(np.abs(im_sim).max()), 1e-6)
    err_map = np.full(image.shape, sigma, dtype=np.float32)
    obs = (im_sim + rng.normal(scale=sigma, size=im_sim.shape)).astype(np.float32)

    mask = np.ones(image.shape, dtype=bool)
    for r, c in ((3, 3), (3, 4), (4, 3)):
        obs[r, c] += 500.0 * sigma  # a 500-sigma outlier: masked, not fit
        mask[r, c] = False
    obs = obs.astype(np.float32)

    ds = ImageData(obs, sim_config, error_map=err_map, mask=mask, sees="all")
    prob2 = ProbModel(model, ds, mode="forward")
    _, red_chi2_true = prob2.log_prob(z)
    red_chi2_true = float(np.asarray(red_chi2_true)[0])

    with tempfile.TemporaryDirectory() as tmp:
        chi2_masked, _ = diagnose.residual_figure(
            tmp, cfg, model, prob2, obs, err_map, im_sim, "verify-synthetic", z,
            mask=mask,
        )

    # The shared helper itself, called directly on the same normalised residual -- exactly
    # what plot_overview.py's panel now does instead of its own copy of the formula. This
    # is the assertion that makes both call sites covered by construction.
    norm_res = (obs - im_sim) / err_map
    chi2_helper, n_kept_helper = diagnose.masked_chisq(norm_res, mask)
    assert n_kept_helper == int(mask.sum()), (
        f"masked_chisq counted {n_kept_helper} kept pixels, expected {int(mask.sum())}"
    )
    assert abs(chi2_helper - red_chi2_true) < 1e-4, (
        f"diagnose.masked_chisq gave {chi2_helper:.6f}, ProbModel's red_chi2 gave "
        f"{red_chi2_true:.6f}"
    )

    # NOT the fixed code path -- computed here only to show the test would have caught the
    # bug, by confirming the two definitions genuinely disagree on this synthetic case.
    chi2_all_pixels = float(np.mean(((obs - im_sim) / err_map) ** 2))

    assert abs(chi2_masked - red_chi2_true) < 1e-4, (
        f"diagnose.residual_figure gave {chi2_masked:.6f}, ProbModel's red_chi2 gave "
        f"{red_chi2_true:.6f} -- the masked chi-squared definitions disagree"
    )
    discrepancy = abs(chi2_all_pixels - red_chi2_true)
    assert discrepancy > 0.5, (
        f"all-pixel chi2 ({chi2_all_pixels:.3f}) is too close to the masked value "
        f"({red_chi2_true:.3f}, discrepancy {discrepancy:.3f}) for this synthetic case "
        f"to discriminate the masking bug"
    )

    return (
        f"masked chi2 {chi2_masked:.4f} == ProbModel red_chi2 {red_chi2_true:.4f}; "
        f"unmasked-mean (pre-fix) value {chi2_all_pixels:.4f} differs by "
        f"{discrepancy:.2f} -- confirms the test discriminates"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", default="DESI238", help="Lens config to build a scene from")
    args = ap.parse_args()
    cfg = lenses.load(args.lens)

    test_masked_chisq(cfg)

    print()
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
