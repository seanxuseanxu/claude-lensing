"""Pre-flight checks for one lens's fit, before spending a GPU-hour on it.

Every check here corresponds to a way the fit can fail *silently* -- producing numbers
that look plausible and are wrong. They are cheap.

Run from the repo root:

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=claude/pipeline \\
        python claude/pipeline/verify.py --lens DESI238 --n-source 2 --n-lens-light 2

Checks 2-4 are parametrised over the model variant (``--n-source``/``--n-lens-light``,
default 1/1, mirroring ``run_fit.py``'s flags). By default they run once, against the
variant about to be fitted -- pass ``--all-variants`` to sweep the four variants
``run_fit.py`` supports instead, which is worth doing after a change to this pipeline
itself rather than to one lens's config (measured at 210s for the sweep vs. ~75s for one
variant on DESI238, A100).

Checks 1 and 5 lean on facts about the specific system -- whether a companion galaxy is in
frame, and which stored runs are known to have diverged. Those come from the lens config
(``companion_max``, ``svi_guard_reference``) and report themselves skipped when a new lens
has not yet supplied them. A skipped check is honest; a check silently weakened to pass is
not.

``claude/pipeline/tests/test_masked_chisq.py`` used to be check 7 here. It asserts a fact
about pipeline code (that ``diagnose.masked_chisq``/``diagnose.residual_figure`` agree with
``ProbModel``'s own reduced chi-squared), true independent of which lens is loaded, so it
moved to the regression-test suite rather than costing 10s of every preflight run.
"""

import argparse
import sys
import traceback

import numpy as np

import jax
import jax.numpy as jnp
import optax

from gigalens.jax.inference import HMC, MAP, SVI
from gigalens.jax.scene_prob_model import ImageData, ProbModel

import data
import lenses
import priors
import run_fit
import svi_health

RESULTS = []


def check(name):
    """Decorator: run a check, record pass/fail, never abort the suite on one failure.

    ``name`` may contain ``{n_source}`` / ``{n_lens_light}`` for checks parametrised over
    the model variant -- the label actually recorded is formatted against whatever keyword
    arguments the caller passes to the wrapped function (e.g. ``check_params(n_source=2)``).
    """

    def wrap(fn):
        def run(*args, **kwargs):
            label = name.format(**kwargs) if kwargs else name
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((label, True, detail or ""))
            except Exception as exc:  # noqa: BLE001 -- a failed check must not stop the rest
                RESULTS.append((label, False, f"{type(exc).__name__}: {exc}"))
                traceback.print_exc()

        run.__name__ = fn.__name__
        return run

    return wrap


def build(cfg, n_source, n_lens_light):
    """The same construction ``run_fit.py`` performs, in one place.

    Returns ``(image, sim_config, model, prob)``. Written once here so a check cannot
    verify a model that differs from the one the fit will actually use.
    """
    image, psf, sim_config = data.load(cfg)
    model = priors.build_model(
        cfg, n_source=n_source, n_lens_light=n_lens_light
    )
    mask = data.build_mask(cfg, image.shape)
    ds = ImageData(
        image,
        sim_config,
        background_rms=cfg.background_rms,
        exp_time=cfg.exp_time,
        mask=mask,
        sees="all",
    )
    return image, sim_config, model, ProbModel(model, ds, mode="forward")


def corner_report(image, num_pix, companion_max, mask):
    """Check all four corners of ``image`` for a companion galaxy's flux.

    gigalens has no pixel masking by default, so a companion galaxy inside the frame is
    fitted as though it were part of the lens system unless excluded via ``mask_circles``
    (``ImageData``'s ``mask=``). Whether one is in frame is a fact about this field, so the
    threshold comes from the config; ``companion_max=None`` means it hasn't been supplied,
    and the check reports itself skipped rather than passing a check it never ran.

    All four corners are checked, not just lower-left: one system's interloper sits in the
    upper-left, where a single-corner check is blind. Fails on the worst.

    Returns ``(worst_name, corner_max, note)``; raises ``AssertionError`` if
    ``companion_max`` is set and exceeded.
    """
    q = num_pix // 4
    corners = {
        "lower-left": image[:q, :q],
        "lower-right": image[:q, -q:],
        "upper-left": image[-q:, :q],
        "upper-right": image[-q:, -q:],
    }
    raw_maxima = {name: float(patch.max()) for name, patch in corners.items()}

    if mask is not None:
        mask_corners = {
            "lower-left": mask[:q, :q],
            "lower-right": mask[:q, -q:],
            "upper-left": mask[-q:, :q],
            "upper-right": mask[-q:, -q:],
        }
        # A corner with no unmasked pixels at all has nothing to check; treat it as
        # contributing nothing to the likelihood there.
        maxima = {
            name: float(patch[mask_corners[name]].max()) if mask_corners[name].any() else 0.0
            for name, patch in corners.items()
        }
    else:
        maxima = raw_maxima

    worst_name = max(maxima, key=maxima.get)
    corner_max = maxima[worst_name]
    raw_str = ", ".join(f"{n}={v:.3f}" for n, v in raw_maxima.items())
    checked_str = ", ".join(f"{n}={v:.3f}" for n, v in maxima.items())

    if companion_max is None:
        note = (
            f"corner maxima (unmasked): {checked_str} (raw: {raw_str}); "
            f"no companion_max set -- not checked"
        )
    else:
        assert corner_max < companion_max, (
            f"{worst_name} corner max {corner_max:.3f} exceeds companion_max "
            f"{companion_max} -- a companion may be in frame there, and (unless masked) "
            f"it would be fitted as part of the system. All corners (unmasked): "
            f"{checked_str}; raw: {raw_str}"
        )
        note = (
            f"worst corner {worst_name} max {corner_max:.3f} < {companion_max} "
            f"(unmasked: {checked_str}; raw: {raw_str})"
        )

    return worst_name, corner_max, note


@check("1. Data conditioning")
def check_data(cfg):
    image = data.load_image(cfg)
    psf = data.load_psf(cfg)

    want = (cfg.num_pix, cfg.num_pix)
    assert image.shape == want, f"crop shape {image.shape}, expected {want}"
    assert image.dtype == np.float32, f"dtype {image.dtype}, expected float32"
    # JAX rejects big-endian arrays outright. byteorder '=' or '|' means native.
    assert image.dtype.byteorder in "=|", f"non-native byteorder {image.dtype.byteorder!r}"
    assert np.isfinite(image).all(), "non-finite pixels in the crop"

    assert abs(psf.sum() - 1.0) < 1e-6, f"psf sums to {psf.sum()}, expected 1"
    assert psf.shape[0] % 2 == 1, "psf must be odd-sized for a centred convolution"

    mask = data.build_mask(cfg, image.shape)
    _, _, corner_note = corner_report(image, cfg.num_pix, cfg.companion_max, mask)

    return (f"crop {image.shape} max {image.max():.2f}, psf {psf.shape} sum 1.0, "
            f"{corner_note}")


@check("2. Parameter count and ordering [{n_source}src {n_lens_light}ll]")
def check_params(cfg, n_source=1, n_lens_light=1):
    """``model.z_param_names`` is authoritative; this cross-checks it against the model.

    Under the old API the flat order was reconstructed by hand and the probe below was the
    only thing standing between a plausible plot and a mislabelled one. The scene API
    derives the names from the same tree-flatten that orders the z columns, so the probe is
    now a *cross-check* of the library rather than of this pipeline. It is kept because it
    is the only test that would catch a column/name skew, and that skew mislabels every
    downstream figure without raising anywhere.

    Also asserts the round trip ``unconstrained(constrained(z)) == z``. A bijector that
    does not invert would corrupt every stored checkpoint's interpretation.
    """
    image, sim_config, model, prob = build(cfg, n_source, n_lens_light)
    n = model.num_free_params
    names = list(model.z_param_names)
    assert len(names) == n, f"{len(names)} names for {n} columns"
    assert len(model.z_labels) == n, "z_labels is not in lockstep with z_param_names"

    sample = model.prior.sample(4, seed=jax.random.PRNGKey(0))
    z = model.bijector.inverse(sample)
    assert z.shape == (4, n), f"got {z.shape}"

    # Round trip through the structured constrained layout.
    z_round = model.unconstrained(model.constrained(z))
    assert np.allclose(np.asarray(z), np.asarray(z_round), rtol=1e-4, atol=1e-4), (
        f"bijector round trip is not the identity: max |dz| = "
        f"{float(jnp.max(jnp.abs(z - z_round))):.3e}"
    )

    # Probe the ordering: perturb each unconstrained coordinate and see which constrained
    # parameter moves. Exactly one must, and it must be the one named. Batched as a single
    # `forward` call on `eye(n)` rather than n separate calls -- same n perturbations, one
    # dispatch, since check 2 was the second-most expensive part of the per-variant loop.
    base = model.bijector.forward(jnp.zeros((n,)))
    moved_batch = model.bijector.forward(jnp.eye(n))
    for i, expected in enumerate(names):
        found = [
            k for k in moved_batch
            if not np.allclose(np.asarray(moved_batch[k][i]), np.asarray(base[k]))
        ]
        # If two coordinates moved, the last one silently overwriting `found` would let
        # this check pass on a genuinely broken ordering -- so assert exactly one moved.
        assert len(found) == 1, (
            f"index {i}: expected exactly one moved param, got {found} "
            f"(z_param_names says {expected})"
        )
        assert found[0] == expected, (
            f"index {i}: z_param_names says {expected}, probe says {found[0]}"
        )

    return f"{n} parameters, ordering and bijector round trip verified"


@check("3. Simulator flux scale [{n_source}src {n_lens_light}ll]")
def check_simulator(cfg, n_source=1, n_lens_light=1):
    image, sim_config, model, prob = build(cfg, n_source, n_lens_light)

    bs = 64
    sample = model.prior.sample(bs, seed=jax.random.PRNGKey(1))
    z = model.bijector.inverse(sample)
    # One simulator, any batch size: the batch axis rides on the params and broadcasts
    # onto a singleton grid axis, so there is no per-stage `bs=` instance any more.
    sim = prob.simulators[0]
    out = sim.simulate(model.constrained(z))

    want = (bs, cfg.num_pix, cfg.num_pix)
    assert out.shape == want, f"got {out.shape}, expected {want}"
    assert bool(jnp.isfinite(out).all()), "non-finite pixels in simulated images"

    # This is the check that catches a flux-scale error before it wastes a fit. Ie must be
    # of order 100, not order 1, because the simulator multiplies by
    # conversion_factor = delta_pix^2 = 0.004225.
    med_peak = float(jnp.median(out.max(axis=(1, 2))))
    data_peak = float(image.max())
    ratio = med_peak / data_peak
    assert 1 / 3 < ratio < 3, (
        f"[{n_source}src {n_lens_light}ll] median prior peak {med_peak:.2f} vs data peak "
        f"{data_peak:.2f} (ratio {ratio:.3f}) -- Ie priors are mis-scaled"
    )

    lp, chisq = prob.log_prob(z)
    assert bool(jnp.isfinite(lp).all()), "non-finite log_prob on prior draws"
    assert lp.shape == (bs,) and chisq.shape == (bs,)

    return (f"median prior peak {med_peak:.2f} vs data {data_peak:.2f} "
            f"(ratio {ratio:.2f}), log_prob finite")


@check("4. Short end-to-end MAP -> SVI -> HMC [{n_source}src {n_lens_light}ll]")
def check_end_to_end(cfg, n_source=1, n_lens_light=1):
    """MAP -> SVI -> HMC plumbing, plus the MAP-descends assertion this check absorbed.

    A former, separate check (numbered 4 pre-2026-08) ran its own 20-step, 64-particle MAP
    solely to assert chi-squared descended, then this check ran a second, independent
    10-step MAP right after it. Since this check already runs a MAP as its first stage,
    the descent assertion below is checked on *that* MAP instead of a second one -- same
    claim, once, on a MAP that was going to run anyway.
    """
    image, sim_config, model, prob = build(cfg, n_source, n_lens_light)
    n = model.num_free_params

    n_map = 10
    opt = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            optax.polynomial_schedule(-1e-2, -1e-2 / 3, power=0.5, transition_steps=n_map)
        ),
    )
    best_z, _, chisq_hist = MAP(prob, opt, n_samples=32, num_steps=n_map, output_type="best")

    lp0, chi0 = prob.log_prob(
        model.bijector.inverse(model.prior.sample(32, seed=jax.random.PRNGKey(0)))
    )
    best_before = float(jnp.min(chi0))
    best_after = float(np.nanmin(np.asarray(chisq_hist)))
    assert best_after < best_before, (
        f"chi-squared did not improve: {best_before:.1f} -> {best_after:.1f}"
    )

    # SVI(prob_model, start, optimizer, ...) -- start SECOND. The schedule mirrors
    # run_fit.py's warmup-then-decay shape rather than the old ramp-up.
    n_svi = 10
    warmup, peak = 1, -2e-3
    svi_opt = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            optax.join_schedules(
                [
                    optax.linear_schedule(peak / 200, peak, transition_steps=warmup),
                    optax.polynomial_schedule(
                        init_value=peak, end_value=peak / 10, power=0.5,
                        transition_steps=n_svi - warmup,
                    ),
                ],
                boundaries=[warmup],
            )
        ),
    )
    qz, loss_hist = SVI(prob, best_z, svi_opt, n_vi=32, num_steps=n_svi)
    assert qz.mean().shape == (n,), f"qz mean {qz.mean().shape}"
    assert np.asarray(loss_hist).reshape(-1).size == n_svi

    # The SVI guard is NOT applied here, and only here. Ten SVI steps cannot converge, so
    # this covariance is ill-conditioned by construction (~1e26-1e33 at these sizes) and
    # the production guard rejects it correctly. What this check is for is the plumbing --
    # shapes, dtypes, the axis normalisation -- not the quality of a 10-step fit. Check 5
    # is what verifies the guard itself.
    num_results, n_hmc = 10, 4
    raw = HMC(
        prob, qz, n_hmc=n_hmc, num_burnin_steps=5, num_results=num_results, seed=0,
    )
    n_chains = (n_hmc // jax.device_count()) * jax.device_count()
    samples = run_fit.normalize_hmc_samples(raw, n_chains, num_results, n)
    assert samples.shape == (n_chains, num_results, n), f"hmc {samples.shape}"
    assert np.isfinite(samples).all(), "non-finite HMC samples"

    return (f"MAP ({n},) -> SVI {tuple(qz.mean().shape)} -> "
            f"HMC {tuple(samples.shape)} (chains first)")


@check("5. SVI covariance guard fires on the known-diverged run")
def check_svi_guard(cfg):
    """The guard must be observed to trigger, not merely to exist.

    ``cfg.svi_guard_reference`` names stored runs whose SVI conditioning is known to be
    healthy or diverged; ``svi_health.check_svi_covariance`` must classify each correctly.
    For DESI238 the diverged one is ``twosource``, whose covariance came out at 8.8e+11,
    whose float32 inverse was not a valid precision matrix, and whose 29 parameters
    therefore all failed Rhat at once.

    Those runs predate the scene-API port, and this check still reads them -- deliberately.
    A covariance's conditioning does not depend on the z-column order, so the calibration
    survives a change of parameterisation even though the runs' *physical* values do not.
    This is the one place a pre-port checkpoint is legitimately read.

    This check exists because the first version of the guard tested positive definiteness
    and silently *passed* that run -- its covariance is in fact PD. Without an assertion
    pinned to a run known to be bad, that would have shipped. A lens with no reference runs
    yet cannot make this assertion, and says so rather than passing vacuously.

    The margin is also asserted. The guard is worthless if the threshold happens to sit a
    few percent above the worst healthy run, so this requires an order of magnitude of
    clearance on both sides.
    """
    expected = dict(cfg.svi_guard_reference)
    if not expected:
        return "SKIPPED: no svi_guard_reference runs in the lens config"
    if not any(not ok for ok in expected.values()):
        raise AssertionError(
            "svi_guard_reference names only healthy runs, so the guard would never be "
            "observed to fire. Include at least one run known to have diverged."
        )
    runs = cfg.runs_dir
    conds, checked = {}, []
    for name, should_pass in expected.items():
        path = runs / name / "svi.npz"
        if not path.exists():
            continue
        L = np.load(path)["scale_tril"]
        try:
            _, cond = svi_health.check_svi_covariance(scale_tril=L)
            passed = True
        except RuntimeError:
            s = np.linalg.svd(np.asarray(L, dtype=np.float64), compute_uv=False)
            cond = float(s.max() ** 2 / s.min() ** 2)
            passed = False
        assert passed == should_pass, (
            f"{name}: guard {'passed' if passed else 'raised'} at condition {cond:.2e}, "
            f"expected it to {'pass' if should_pass else 'raise'}"
        )
        conds[name] = cond
        checked.append(name)

    diverged = [n for n in checked if not expected[n]]
    if not diverged:
        raise AssertionError(
            f"none of the diverged reference runs "
            f"({[n for n, ok in expected.items() if not ok]}) has an svi.npz under "
            f"{runs}, so the guard was never observed to fire. This check is only "
            f"meaningful against a run known to be bad."
        )
    healthy = [n for n in checked if expected[n]]
    if not healthy:
        raise AssertionError(
            f"no healthy reference run found under {runs}, so the threshold's margin "
            f"above normal conditioning cannot be checked."
        )

    worst_ok = max(conds[n] for n in healthy)
    assert worst_ok < svi_health.COND_MAX / 10, (
        f"only {svi_health.COND_MAX / worst_ok:.1f}x margin between the worst healthy run "
        f"({worst_ok:.2e}) and the threshold ({svi_health.COND_MAX:.0e}) -- too tight to "
        f"separate the populations reliably"
    )
    worst_bad = min(conds[n] for n in diverged)
    assert worst_bad > svi_health.COND_MAX * 10, (
        f"the mildest diverged run is at {worst_bad:.2e}, only "
        f"{worst_bad / svi_health.COND_MAX:.1f}x over the threshold"
    )
    return (f"correct on {len(checked)} stored runs; worst healthy {worst_ok:.1e}, "
            f"limit {svi_health.COND_MAX:.0e}, mildest diverged {worst_bad:.1e}")


def main():
    ap = argparse.ArgumentParser(description="Pre-flight checks for one lens's fit")
    lenses.add_argument(ap, required=True)
    ap.add_argument(
        "--n-source", type=int, default=1,
        help="Source-plane light components for the checked variant (default: 1)",
    )
    ap.add_argument(
        "--n-lens-light", type=int, default=1,
        help="Lens-plane light components for the checked variant (default: 1)",
    )
    ap.add_argument(
        "--all-variants", action="store_true",
        help="Sweep all four variants run_fit.py supports ((1,1),(2,1),(1,2),(2,2)) "
             "instead of just --n-source/--n-lens-light. Slower (~210s vs ~75s on "
             "DESI238); use after a change to the pipeline itself, not to one lens.",
    )
    args = ap.parse_args()
    cfg = lenses.load(args.lens)

    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Lens: {cfg.name}  ({cfg.notes})\n" if cfg.notes else f"Lens: {cfg.name}\n")
    # Check 1 (data conditioning) doesn't depend on the model variant; the rest are
    # parametrised over it so a bug specific to a multi-component block can't hide behind a
    # single-component run. The (2, 2) case is the 36-parameter bulge+disk model.
    check_data(cfg)
    variants = (
        ((1, 1), (2, 1), (1, 2), (2, 2)) if args.all_variants
        else ((args.n_source, args.n_lens_light),)
    )
    for n_source, n_lens_light in variants:
        check_params(cfg, n_source=n_source, n_lens_light=n_lens_light)
        check_simulator(cfg, n_source=n_source, n_lens_light=n_lens_light)
        check_end_to_end(cfg, n_source=n_source, n_lens_light=n_lens_light)
    check_svi_guard(cfg)

    print("\n" + "=" * 72)
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print("=" * 72)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
