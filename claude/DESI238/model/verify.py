"""Pre-flight checks for the DESI238 pipeline.

Every check here corresponds to a way the fit can fail *silently* -- producing numbers
that look plausible and are wrong. They are cheap; run them before spending a GPU-hour.

Run from the repo root:

    NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=gigalens/src:claude/DESI238/model \\
        python claude/DESI238/model/verify.py
"""

import sys
import traceback

import numpy as np

import compat

import jax
import jax.numpy as jnp
import optax

from gigalens.jax.model import ForwardProbModel
from gigalens.jax.simulator import LensSimulator
from gigalens.jax.inference import ModellingSequence

import data
import priors

RESULTS = []


def check(name):
    """Decorator: run a check, record pass/fail, never abort the suite on one failure.

    ``name`` may contain ``{n_source}`` for checks parametrised over the model variant --
    the label actually recorded is formatted against whatever keyword arguments the caller
    passes to the wrapped function (e.g. ``check_params(n_source=2)``).
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
def check_data():
    image = data.load_image()
    psf = data.load_psf()

    assert image.shape == (80, 80), f"crop shape {image.shape}, expected (80, 80)"
    assert image.dtype == np.float32, f"dtype {image.dtype}, expected float32"
    # JAX rejects big-endian arrays outright. byteorder '=' or '|' means native.
    assert image.dtype.byteorder in "=|", f"non-native byteorder {image.dtype.byteorder!r}"
    assert np.isfinite(image).all(), "non-finite pixels in the crop"

    assert psf.shape == (27, 27), f"psf shape {psf.shape}"
    assert abs(psf.sum() - 1.0) < 1e-6, f"psf sums to {psf.sum()}, expected 1"
    assert psf.shape[0] % 2 == 1, "psf must be odd-sized for a centred convolution"

    # The companion galaxy sits at cutout ~(11, 11), outside this crop. gigalens has no
    # pixel masking, so a companion inside the frame would be fit as part of the system.
    corner_max = float(image[:20, :20].max())
    assert corner_max < 0.15, f"corner max {corner_max:.3f} e/s -- companion may be in frame"

    return (f"crop {image.shape} max {image.max():.2f} e/s, psf sum 1.0, "
            f"corner max {corner_max:.3f}")


@check("2. Parameter count and ordering [{n_source} source]")
def check_params(n_source=1):
    prior = priors.build_prior(n_source=n_source)
    n = priors.n_params(prior)
    image = data.load_image()
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=data.BACKGROUND_RMS, exp_time=data.EXP_TIME,
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


@check("3. Simulator flux scale [{n_source} source]")
def check_simulator(n_source=1):
    image, psf, sim_config = data.load()
    prior = priors.build_prior(n_source=n_source)
    phys_model = priors.build_phys_model(n_source=n_source)
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=data.BACKGROUND_RMS, exp_time=data.EXP_TIME,
    )

    bs = 64
    sample = prior.sample(bs, seed=jax.random.PRNGKey(1))
    z = jnp.stack(prob.bij.inverse(sample)).T
    sim = LensSimulator(phys_model, sim_config, bs=bs)
    out = sim.simulate(prob.bij.forward(list(z.T)))

    assert out.shape == (bs, 80, 80), f"got {out.shape}"
    assert bool(jnp.isfinite(out).all()), "non-finite pixels in simulated images"

    # This is the check that catches a flux-scale error before it wastes a fit. Ie must be
    # of order 100, not order 1, because LensSimulator multiplies by
    # conversion_factor = delta_pix^2 = 0.004225.
    med_peak = float(jnp.median(out.max(axis=(1, 2))))
    data_peak = float(image.max())
    ratio = med_peak / data_peak
    assert 1 / 3 < ratio < 3, (
        f"[{n_source} source] median prior peak {med_peak:.2f} vs data peak "
        f"{data_peak:.2f} (ratio {ratio:.3f}) -- Ie priors are mis-scaled"
    )

    lp, chisq = prob.log_prob(sim, z)
    assert bool(jnp.isfinite(lp).all()), "non-finite log_prob on prior draws"
    assert lp.shape == (bs,) and chisq.shape == (bs,)

    return (f"median prior peak {med_peak:.2f} vs data {data_peak:.2f} "
            f"(ratio {ratio:.2f}), log_prob finite")


@check("4. Shim: MAP runs and descends [{n_source} source]")
def check_map(n_source=1):
    compat.require_single_device()
    image, psf, sim_config = data.load()
    prior = priors.build_prior(n_source=n_source)
    n = priors.n_params(prior)
    phys_model = priors.build_phys_model(n_source=n_source)
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=data.BACKGROUND_RMS, exp_time=data.EXP_TIME,
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


@check("5. Short end-to-end MAP -> SVI -> HMC [{n_source} source]")
def check_end_to_end(n_source=1):
    compat.require_single_device()
    image, psf, sim_config = data.load()
    prior = priors.build_prior(n_source=n_source)
    n = priors.n_params(prior)
    phys_model = priors.build_phys_model(n_source=n_source)
    prob = ForwardProbModel(
        prior, jnp.asarray(image),
        background_rms=data.BACKGROUND_RMS, exp_time=data.EXP_TIME,
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
    # SVI(start, optimizer, ...) in JAX -- the reverse of the TF backend.
    svi_opt = optax.chain(
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            optax.polynomial_schedule(-1e-6, -3e-3, power=2, transition_steps=10)
        ),
    )
    qz, loss_hist = seq.SVI(best, svi_opt, n_vi=32, num_steps=10)
    assert qz.mean().shape == (n,), f"qz mean {qz.mean().shape}"
    assert len(loss_hist) == 10

    # Upstream HMC is unusable (pmap context-mesh error, even at one device); this is the
    # pmap-free replacement.
    samples = compat.hmc_single_device(
        prob, phys_model, sim_config, qz,
        n_hmc=4, num_burnin_steps=5, num_results=10, seed=0,
    )
    assert samples.shape == (10, 4, n), f"hmc {samples.shape}"
    assert bool(jnp.isfinite(samples).all()), "non-finite HMC samples"

    return f"MAP {tuple(est.shape)} -> SVI {tuple(qz.mean().shape)} -> HMC {tuple(samples.shape)}"


def main():
    print(f"JAX {jax.__version__}, devices: {jax.devices()}\n")
    # Check 1 (data conditioning) doesn't depend on n_source; the rest are parametrised
    # over both model variants so a bug specific to two source components (e.g. the
    # source0/source1 naming split) can't hide behind a single-source-only run.
    check_data()
    for n_source in (1, 2):
        check_params(n_source=n_source)
        check_simulator(n_source=n_source)
        check_map(n_source=n_source)
        check_end_to_end(n_source=n_source)

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
