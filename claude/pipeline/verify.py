"""Pre-flight checks for one lens's fit, before spending a GPU-hour on it.

Every check here corresponds to a way the fit can fail *silently* -- producing numbers
that look plausible and are wrong. They are cheap.

Run from the repo root:

    NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=gigalens/src:claude/pipeline \\
        python claude/pipeline/verify.py --lens DESI238

Checks 1 and 6 lean on facts about the specific system -- whether a companion galaxy is in
frame, and which stored runs are known to have diverged. Those come from the lens config
(``companion_max``, ``svi_guard_reference``) and report themselves skipped when a new lens
has not yet supplied them. A skipped check is honest; a check silently weakened to pass is
not.
"""

import sys
import traceback
from pathlib import Path

import numpy as np

import compat

import jax
import jax.numpy as jnp
import optax

from gigalens.jax.model import ForwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.jax.inference import ModellingSequence

import argparse

import data
import lenses
import priors

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

    # gigalens has no pixel masking, so a companion galaxy inside the frame is fitted as
    # though it were part of the lens system. Whether one is in frame is a fact about this
    # field, so the threshold comes from the config; a lens that has not supplied one says
    # so rather than passing a check it never ran.
    corner_max = float(image[: cfg.num_pix // 4, : cfg.num_pix // 4].max())
    if cfg.companion_max is None:
        corner_note = f"corner max {corner_max:.3f} (no companion_max set -- not checked)"
    else:
        assert corner_max < cfg.companion_max, (
            f"corner max {corner_max:.3f} exceeds companion_max {cfg.companion_max} -- "
            f"a companion may be in frame, and it would be fitted as part of the system"
        )
        corner_note = f"corner max {corner_max:.3f} < {cfg.companion_max}"

    return (f"crop {image.shape} max {image.max():.2f}, psf {psf.shape} sum 1.0, "
            f"{corner_note}")


@check("2. Parameter count and ordering [{n_source}src {n_lens_light}ll]")
def check_params(cfg, n_source=1, n_lens_light=1):
    prior = priors.build_prior(cfg, n_source=n_source, n_lens_light=n_lens_light)
    n = priors.n_params(prior)
    image = data.load_image(cfg)
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=cfg.background_rms, exp_time=cfg.exp_time,
    )
    sample = prior.sample(4, seed=jax.random.PRNGKey(0))
    z = jnp.stack(prob.bij.inverse(sample)).T
    assert z.shape == (4, n), f"got {z.shape}"

    names = priors.param_names(prior)
    assert len(names) == n

    # The declared ordering must match what bij.forward actually produces. Probe it: set
    # one unconstrained coordinate at a time and see which constrained key moves. A
    # mismatch here mislabels every diagnostic plot without raising an error anywhere.
    #
    # The probe's label must be built by the SAME rule as priors.param_names -- a
    # sub-block index is appended only when two sub-blocks in that group share a key
    # (see param_names' docstring) -- otherwise, with n_source=2, this check (which exists
    # specifically to catch mislabelling) would itself mislabel the two source components.
    structure = priors._structure(prior)
    groups = ["lens", "lens_light", "source"]
    collides = [
        len([k for keys in block for k in keys])
        != len(set(k for keys in block for k in keys))
        for block in structure
    ]

    base = prob.bij.forward(list(jnp.zeros((1, n)).T))
    for i, expected in enumerate(names):
        z_probe = jnp.zeros((1, n)).at[0, i].set(1.0)
        moved = prob.bij.forward(list(z_probe.T))
        found = []
        for gi, block in enumerate(moved):
            for si, sub in enumerate(block):
                for key, val in sub.items():
                    if not np.allclose(np.asarray(val), np.asarray(base[gi][si][key])):
                        prefix = f"{groups[gi]}{si}" if collides[gi] else groups[gi]
                        found.append(f"{prefix}.{key}")
        # If two coordinates moved, the last one silently overwriting `found` would let
        # this check pass on a genuinely broken ordering -- so assert exactly one moved.
        assert len(found) == 1, (
            f"index {i}: expected exactly one moved param, got {found} "
            f"(param_names says {expected})"
        )
        assert found[0] == expected, (
            f"index {i}: param_names says {expected}, probe says {found[0]}"
        )

    return f"{n} parameters, ordering verified against a perturbation probe"


@check("3. Simulator flux scale [{n_source}src {n_lens_light}ll]")
def check_simulator(cfg, n_source=1, n_lens_light=1):
    image, psf, sim_config = data.load(cfg)
    prior = priors.build_prior(cfg, n_source=n_source, n_lens_light=n_lens_light)
    phys_model = priors.build_phys_model(n_source=n_source, n_lens_light=n_lens_light)
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=cfg.background_rms, exp_time=cfg.exp_time,
    )

    bs = 64
    sample = prior.sample(bs, seed=jax.random.PRNGKey(1))
    z = jnp.stack(prob.bij.inverse(sample)).T
    sim = LensSimulator(phys_model, sim_config, bs=bs)
    out = sim.simulate(prob.bij.forward(list(z.T)))

    want = (bs, cfg.num_pix, cfg.num_pix)
    assert out.shape == want, f"got {out.shape}, expected {want}"
    assert bool(jnp.isfinite(out).all()), "non-finite pixels in simulated images"

    # This is the check that catches a flux-scale error before it wastes a fit. Ie must be
    # of order 100, not order 1, because LensSimulator multiplies by
    # conversion_factor = delta_pix^2 = 0.004225.
    med_peak = float(jnp.median(out.max(axis=(1, 2))))
    data_peak = float(image.max())
    ratio = med_peak / data_peak
    assert 1 / 3 < ratio < 3, (
        f"[{n_source}src {n_lens_light}ll] median prior peak {med_peak:.2f} vs data peak "
        f"{data_peak:.2f} (ratio {ratio:.3f}) -- Ie priors are mis-scaled"
    )

    lp, chisq = prob.log_prob(sim, z)
    assert bool(jnp.isfinite(lp).all()), "non-finite log_prob on prior draws"
    assert lp.shape == (bs,) and chisq.shape == (bs,)

    return (f"median prior peak {med_peak:.2f} vs data {data_peak:.2f} "
            f"(ratio {ratio:.2f}), log_prob finite")


@check("4. Shim: MAP runs and descends [{n_source}src {n_lens_light}ll]")
def check_map(cfg, n_source=1, n_lens_light=1):
    compat.require_single_device()
    image, psf, sim_config = data.load(cfg)
    prior = priors.build_prior(cfg, n_source=n_source, n_lens_light=n_lens_light)
    n = priors.n_params(prior)
    phys_model = priors.build_phys_model(n_source=n_source, n_lens_light=n_lens_light)
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=cfg.background_rms, exp_time=cfg.exp_time,
    )
    seq = ModellingSequence(phys_model, prob, sim_config)

    n_steps = 20
    opt = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            optax.polynomial_schedule(
                init_value=-1e-2, end_value=-1e-2 / 3, power=0.5,
                transition_steps=n_steps,
            )
        ),
    )
    # Without the compat shims this raises AttributeError (jax.tree_map) or TypeError
    # (nanmin keepdims) before completing a single step.
    est = seq.MAP(opt, n_samples=64, num_steps=n_steps)
    assert est.shape == (64, n), f"got {est.shape}"

    sim = LensSimulator(phys_model, sim_config, bs=64)
    lp0, chi0 = prob.log_prob(sim, jnp.stack(prob.bij.inverse(
        prior.sample(64, seed=jax.random.PRNGKey(0)))).T)
    lp1, chi1 = prob.log_prob(sim, est)
    best_before, best_after = float(jnp.min(chi0)), float(jnp.min(chi1))
    assert best_after < best_before, (
        f"chi-squared did not improve: {best_before:.1f} -> {best_after:.1f}"
    )

    return f"MAP {est.shape}, best chi2 {best_before:.1f} -> {best_after:.1f}"


@check("5. Short end-to-end MAP -> SVI -> HMC [{n_source}src {n_lens_light}ll]")
def check_end_to_end(cfg, n_source=1, n_lens_light=1):
    compat.require_single_device()
    image, psf, sim_config = data.load(cfg)
    prior = priors.build_prior(cfg, n_source=n_source, n_lens_light=n_lens_light)
    n = priors.n_params(prior)
    phys_model = priors.build_phys_model(n_source=n_source, n_lens_light=n_lens_light)
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=cfg.background_rms, exp_time=cfg.exp_time,
    )
    seq = ModellingSequence(phys_model, prob, sim_config)

    opt = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            optax.polynomial_schedule(-1e-2, -1e-2 / 3, power=0.5, transition_steps=10)
        ),
    )
    est = seq.MAP(opt, n_samples=32, num_steps=10)
    sim = LensSimulator(phys_model, sim_config, bs=32)
    best = est[jnp.argmax(prob.log_prob(sim, est)[0])][None, :]

    # SVI exercises the jax.tree_map path, which MAP does not reach. Argument order is
    # SVI(start, optimizer, ...) in JAX -- the reverse of the TF backend. The schedule
    # mirrors run_fit.py's warmup-then-decay shape rather than the old ramp-up.
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
    qz, loss_hist = seq.SVI(best, svi_opt, n_vi=32, num_steps=n_svi)
    assert qz.mean().shape == (n,), f"qz mean {qz.mean().shape}"
    assert len(loss_hist) == n_svi

    # Upstream HMC is unusable (pmap context-mesh error, even at one device); this is the
    # pmap-free replacement.
    #
    # max_cond is disabled here, and only here. Ten SVI steps cannot converge, so this
    # covariance is ill-conditioned by construction (~1e26-1e33 at these sizes) and the
    # production guard rejects it correctly. What this check is for is the plumbing --
    # shapes, dtypes, the pmap-free kernel stack -- not the quality of a 10-step fit.
    # Check 6 is what verifies the guard itself.
    samples = compat.hmc_single_device(
        prob, phys_model, sim_config, qz,
        n_hmc=4, num_burnin_steps=5, num_results=10, seed=0,
        max_cond=np.inf,
    )
    assert samples.shape == (10, 4, n), f"hmc {samples.shape}"
    assert bool(jnp.isfinite(samples).all()), "non-finite HMC samples"

    return f"MAP {tuple(est.shape)} -> SVI {tuple(qz.mean().shape)} -> HMC {tuple(samples.shape)}"


@check("6. SVI covariance guard fires on the known-diverged run")
def check_svi_guard(cfg):
    """The guard must be observed to trigger, not merely to exist.

    ``cfg.svi_guard_reference`` names stored runs whose SVI conditioning is known to be
    healthy or diverged; ``compat.check_svi_covariance`` must classify each correctly. For
    DESI238 the diverged one is ``twosource``, whose covariance came out at 8.8e+11, whose
    float32 inverse was not a valid precision matrix, and whose 29 parameters therefore all
    failed Rhat at once.

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
            _, cond = compat.check_svi_covariance(scale_tril=L)
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
    assert worst_ok < compat.COND_MAX / 10, (
        f"only {compat.COND_MAX / worst_ok:.1f}x margin between the worst healthy run "
        f"({worst_ok:.2e}) and the threshold ({compat.COND_MAX:.0e}) -- too tight to "
        f"separate the populations reliably"
    )
    worst_bad = min(conds[n] for n in diverged)
    assert worst_bad > compat.COND_MAX * 10, (
        f"the mildest diverged run is at {worst_bad:.2e}, only "
        f"{worst_bad / compat.COND_MAX:.1f}x over the threshold"
    )
    return (f"correct on {len(checked)} stored runs; worst healthy {worst_ok:.1e}, "
            f"limit {compat.COND_MAX:.0e}, mildest diverged {worst_bad:.1e}")


def main():
    ap = argparse.ArgumentParser(description="Pre-flight checks for one lens's fit")
    lenses.add_argument(ap, required=True)
    args = ap.parse_args()
    cfg = lenses.load(args.lens)

    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Lens: {cfg.name}  ({cfg.notes})\n" if cfg.notes else f"Lens: {cfg.name}\n")
    # Check 1 (data conditioning) doesn't depend on the model variant; the rest are
    # parametrised over it so a bug specific to a multi-component block -- the
    # source0/source1 or lens_light0/lens_light1 naming split, say -- can't hide behind a
    # single-component run. The (2, 2) case is the 36-parameter bulge+disk model.
    check_data(cfg)
    for n_source, n_lens_light in ((1, 1), (2, 1), (1, 2), (2, 2)):
        check_params(cfg, n_source=n_source, n_lens_light=n_lens_light)
        check_simulator(cfg, n_source=n_source, n_lens_light=n_lens_light)
        check_map(cfg, n_source=n_source, n_lens_light=n_lens_light)
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
