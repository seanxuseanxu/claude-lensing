"""JAX 0.11 compatibility shims for the vendored GIGA-Lens JAX backend.

``gigalens.jax.inference`` was written against JAX 0.2.24 and does not run under the
0.11.0 installed in ``gigajax6.0``. Rather than patch ``gigalens/src`` (which stays clean
against upstream), apply the fixes here and import this module *before*
``gigalens.jax.inference``.

Three breakages, all reproduced by traceback:

1. ``inference.py:98`` -- ``jax.tree_map`` was removed in JAX 0.11. It lives at
   ``jax.tree_util.tree_map``. Hit by ``SVI``, not by ``MAP``.

2. ``inference.py:68`` -- ``float(jnp.nanmin(loss, keepdims=True))`` raises
   ``TypeError: Only scalar arrays can be converted to Python scalars; got arr.ndim=1``.
   ``keepdims=True`` leaves a length-1 axis that ``float()`` used to accept and no longer
   does. Only affects the progress-bar string.

3. ``inference.py:55`` -- ``jnp.array(jnp.split(params, dev_cnt, axis=0))`` under ``pmap``
   raises ``ValueError: Sharding passed to jit does not match the sharding on the
   respective arg``: jit wants ``P(axis)``, the argument arrives as ``P(None, axis)``.

4. ``inference.py:HMC`` -- a *separate* pmap failure, and one that bites even with a
   single device, so ``require_single_device()`` does not save you from it. ``run_chain``
   is wrapped in ``@pmap``, but ``q_z`` and ``momentum_distribution`` are built *outside*
   it. Their arrays carry ``Auto`` sharding while the pmap body runs under a ``Manual``
   mesh, and TFP's ``scale_tril`` matvec trips on the mismatch:
   ``ValueError: Context mesh AbstractMesh(..., axis_types=(Manual,)) should match the
   mesh of sharding AbstractMesh(..., axis_types=(Auto,)) passed to broadcast_in_dim``.

Breakages 1 and 2 are patched below. Breakage 3 is *not* -- the multi-device path is
unused (see ``require_single_device()``). Timings show the model is too small to saturate
one H200 (bs=1024 at 80x80, supersample=2 costs 0.064 s/step), so splitting one batch
across GPUs buys nothing; run independent single-GPU jobs instead. Breakage 4 is worked
around by ``hmc_single_device()``, which reimplements ``ModellingSequence.HMC`` without
the ``pmap`` wrapper. Verified: the identical kernel stack samples cleanly that way.
"""

import os

# Must precede any JAX/CUDA initialisation: the node emits NVLS / Fabric Manager errors
# otherwise. They are harmless here but flood stderr.
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

# Narrow a multi-GPU allocation down to one device, before JAX initialises.
#
# The intended way to use several GPUs here is several independent processes, one per GPU
# -- the model is too small to benefit from splitting a batch, and the pmap path is broken
# anyway (breakages 3 and 4 below). But a shared node often hands you
# CUDA_VISIBLE_DEVICES="1,2", and JAX would then see two devices and trip
# require_single_device(). Setting GIGALENS_DEVICE=<n> picks the n-th *visible* GPU, so
#
#   GIGALENS_DEVICE=0 python run_fit.py --seed 0 --label seed0 &
#   GIGALENS_DEVICE=1 python run_fit.py --seed 1 --label seed1 &
#
# gives two single-device jobs without the caller having to work out absolute GPU indices.
# JAX reads CUDA_VISIBLE_DEVICES at first use, so this must happen before `import jax`.
_slot = os.environ.get("GIGALENS_DEVICE")
if _slot is not None:
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if _visible:
        _devices = [d for d in _visible.split(",") if d.strip()]
        _i = int(_slot)
        if not 0 <= _i < len(_devices):
            raise RuntimeError(
                f"GIGALENS_DEVICE={_i} out of range: CUDA_VISIBLE_DEVICES exposes "
                f"{len(_devices)} device(s) ({_visible})."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = _devices[_i]

import jax
import numpy as np
from jax import numpy as jnp


def _apply_shims():
    # (1) jax.tree_map removed in 0.11.
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree_util.tree_map

    # (2) float(jnp.nanmin(x, keepdims=True)) no longer works. Patch the name as seen
    # from inside gigalens.jax.inference only -- the global jnp.nanmin is left alone.
    import gigalens.jax.inference as _inference

    if getattr(_inference.jnp.nanmin, "_gigalens_shimmed", False):
        return

    _real_nanmin = jnp.nanmin

    def _nanmin(a, *args, **kwargs):
        kwargs.pop("keepdims", None)
        return _real_nanmin(a, *args, **kwargs)

    _nanmin._gigalens_shimmed = True
    _inference.jnp.nanmin = _nanmin


_apply_shims()


def require_single_device():
    """Fail loudly if more than one JAX device is visible.

    ``MAP``, ``SVI`` and ``HMC`` all ``pmap`` across ``jax.device_count()``. With more
    than one device that path hits breakage 3 above and dies mid-run, after the JIT
    compile has already been paid for. Better to refuse at startup.

    Set ``CUDA_VISIBLE_DEVICES`` to a single GPU. To use several GPUs, launch several
    independent processes -- one per GPU -- rather than one process over many devices.
    """
    n = jax.device_count()
    if n != 1:
        raise RuntimeError(
            f"{n} JAX devices visible, but the pmap path in gigalens.jax.inference is "
            f"broken under JAX {jax.__version__} (sharding mismatch at inference.py:55). "
            f"Set CUDA_VISIBLE_DEVICES to one GPU; run multiple GPUs as separate "
            f"processes."
        )
    return jax.devices()[0]


#: Condition number above which the SVI covariance is rejected outright.
#:
#: The threshold is set by measurement, not by taste. Across the five runs to date the two
#: populations are separated by three orders of magnitude:
#:
#:     nsersic10  9.4e+05   inner3  1.1e+07   seed1  4.8e+07   seed0  5.9e+08   (22 par, ok)
#:     twosource  8.8e+11                                             (29 par, diverged)
#:
#: 1e10 sits in the gap with ~17x margin below and ~90x above.
COND_MAX = 1e10


def check_svi_covariance(cov=None, scale_tril=None, max_cond=COND_MAX):
    """Raise unless the SVI covariance is usable as an HMC preconditioner.

    Pass ``scale_tril`` when you have it -- see the note on precision at the end. Returns
    ``(min_eig, cond)``.

    ``hmc_single_device`` builds the momentum distribution from
    ``inv(q_z.covariance())``, in float32, with JAX x64 off. What kills a run is the
    *conditioning* of that inverse, not the sign of the covariance's eigenvalues:

        run         cov condition   max|inv(C) @ C - I|
        nsersic10      9.4e+05            0.07
        seed0          5.9e+08            0.98
        twosource      8.8e+11           55.3

    At 8.8e+11 the float32 inverse is not an inverse at all. Its raw spectrum carries five
    negative eigenvalues reaching -3.2e+07, so HMC sampled against an indefinite mass
    matrix, and all 29 of that run's parameters failed Rhat. That was read at the time as
    slow mixing; it was not.

    A positive-definiteness test does NOT catch this, which is worth stating plainly
    because the obvious version of this guard was written first and silently passed the one
    run it existed for. ``twosource``'s covariance is positive definite: min eigenvalue
    +5.7e-13 as TFP computes it, +1.5e-13 in float64. The -1.8e-12 originally attributed to
    it came from forming ``L @ L.T`` in float32 during the audit -- rounding noise in a
    matrix whose smallest eigenvalues sit twelve orders below its largest, not a property
    of the fit. Conditioning is the real discriminant, so conditioning is the test.

    Both failures are hard errors. A repaired or floored covariance would yield an HMC run
    that looks fine downstream and is not, which costs more than a failed stage.

    Eigenvalues come from an SVD of the Cholesky factor when ``scale_tril`` is given
    (squaring the singular values), because forming ``L @ L.T`` loses half the available
    precision exactly where this matrix has none to spare. Falling back to ``cov`` is
    supported but strictly worse conditioned; it is what produced the spurious negative
    eigenvalue described above.
    """
    if (cov is None) == (scale_tril is None):
        raise TypeError("pass exactly one of cov= or scale_tril=")

    if scale_tril is not None:
        L = np.asarray(scale_tril, dtype=np.float64)
        if not np.isfinite(L).all():
            raise RuntimeError(
                "SVI scale_tril contains non-finite entries; the SVI stage diverged."
            )
        s = np.linalg.svd(L, compute_uv=False)
        # A Cholesky factor's singular values are the square roots of the covariance's
        # eigenvalues, and are non-negative by construction -- so a zero here is a
        # rank-deficient (singular) covariance, reported below as a non-PD failure.
        min_eig, max_eig = float(s.min() ** 2), float(s.max() ** 2)
    else:
        c = np.asarray(cov, dtype=np.float64)
        c = 0.5 * (c + c.T)
        if not np.isfinite(c).all():
            raise RuntimeError(
                "SVI covariance contains non-finite entries; the SVI stage diverged."
            )
        w = np.linalg.eigvalsh(c)
        min_eig, max_eig = float(w.min()), float(w.max())

    cond = abs(max_eig / min_eig) if min_eig > 0 else np.inf
    if min_eig <= 0:
        raise RuntimeError(
            f"SVI covariance is not positive definite (min eigenvalue {min_eig:.3e}, "
            f"condition {cond:.3e}). The SVI stage diverged; HMC preconditioned on this "
            f"would produce meaningless Rhat. Re-run SVI, lowering --svi-peak-lr."
        )
    if cond > max_cond:
        raise RuntimeError(
            f"SVI covariance condition number {cond:.3e} exceeds the limit {max_cond:.0e} "
            f"(min eigenvalue {min_eig:.3e}). Its float32 inverse will not be a valid "
            f"precision matrix, so HMC would sample against an effectively indefinite "
            f"mass matrix and every Rhat would be meaningless -- this is what happened to "
            f"the twosource run. Healthy runs here sit at 1e6-6e8. Re-run SVI, lowering "
            f"--svi-peak-lr."
        )
    return min_eig, cond


def hmc_single_device(
    prob_model,
    phys_model,
    sim_config,
    q_z,
    init_eps=0.3,
    init_l=3,
    n_hmc=50,
    num_burnin_steps=250,
    num_results=750,
    max_leapfrog_steps=30,
    seed=0,
    max_cond=COND_MAX,
):
    """``ModellingSequence.HMC`` with the ``pmap`` wrapper removed.

    Kernel stack, adaptation schedule and defaults are copied verbatim from
    ``gigalens/src/gigalens/jax/inference.py``; the only change is that ``sample_chain``
    is called directly instead of inside ``@pmap``. That removes breakage 4 above.

    Returns an array of shape ``(num_results, n_hmc, n_params)`` -- note this differs
    from the upstream ``(dev_cnt, n_hmc // dev_cnt, num_results, n_params)``. Values are
    in **unconstrained** space; apply ``prob_model.bij.forward`` before interpreting them.

    The chains are the ``n_hmc`` axis: they share one seed but start from independent
    draws of ``q_z``, which is what makes R-hat across that axis meaningful.

    ``max_cond`` raises the SVI-covariance conditioning limit. Leave it at the default for
    any real fit -- it exists so a deliberately-truncated smoke test, whose SVI has had no
    chance to converge, can still exercise the sampler.
    """
    import tensorflow_probability.substrates.jax as tfp
    from tensorflow_probability.substrates.jax import distributions as tfd
    from tensorflow_probability.substrates.jax import experimental as tfe

    import gigalens.jax.simulator as sim

    lens_sim = sim.LensSimulator(phys_model, sim_config, bs=n_hmc)

    @jax.jit
    def log_prob(z):
        return prob_model.log_prob(lens_sim, z)[0]

    # Hard-fail before building the preconditioner rather than after sampling on it.
    # Prefer the Cholesky factor: squaring its singular values keeps the precision that
    # forming L @ L.T throws away, and these matrices have none to spare.
    L = getattr(q_z, "scale_tril", None)
    if L is not None:
        min_eig, cond = check_svi_covariance(scale_tril=L, max_cond=max_cond)
    else:
        min_eig, cond = check_svi_covariance(cov=q_z.covariance(), max_cond=max_cond)
    print(f"SVI covariance: min eigenvalue {min_eig:.3e}, condition {cond:.3e}")

    momentum_distribution = tfd.MultivariateNormalFullCovariance(
        loc=jnp.zeros_like(q_z.mean()),
        covariance_matrix=jnp.linalg.inv(q_z.covariance()),
    )

    key = jax.random.PRNGKey(seed)
    start = q_z.sample(n_hmc, seed=key)
    num_adaptation_steps = int(num_burnin_steps * 0.8)

    mc_kernel = tfe.mcmc.PreconditionedHamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        momentum_distribution=momentum_distribution,
        step_size=init_eps,
        num_leapfrog_steps=init_l,
    )
    mc_kernel = tfe.mcmc.GradientBasedTrajectoryLengthAdaptation(
        mc_kernel,
        num_adaptation_steps=num_adaptation_steps,
        max_leapfrog_steps=max_leapfrog_steps,
    )
    mc_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=mc_kernel, num_adaptation_steps=num_adaptation_steps
    )

    out = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=start,
        trace_fn=lambda _, pkr: None,
        seed=key,
        kernel=mc_kernel,
    )
    # sample_chain returns a StatesAndTrace namedtuple when trace_fn is not None-typed.
    states = out.all_states if hasattr(out, "all_states") else out
    return jnp.asarray(states)
