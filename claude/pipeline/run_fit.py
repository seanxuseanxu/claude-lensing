"""Fit one lens system through the MAP / SVI / (HMC | MAMS | MCLMC) pipeline.

Which system is set by ``--lens``, naming a directory under ``claude/`` that defines a
``config.py``. Results land in ``claude/<LENS>/runs/<label>/``, and the lens name is
recorded in ``metadata.json`` so every downstream tool can rebuild the same model from the
run directory alone.

The sampling stage is still called "hmc" for --stages / checkpoint-naming purposes;
--sampler picks which algorithm runs it (default "hmc", so existing invocations are
unaffected). MAMS and MCLMC are the library's microcanonical samplers -- see
``gigalens.jax.inference.mams``/``.mclmc`` and ``nanolens/docs/how-to/run-mams.md`` /
``run-mclmc.md``.

The three inference stages are checkpointed independently, so a crash in the sampling
stage does not lose the MAP and SVI results. Stages can be resumed: if --stages omits "map"
but svi is requested, MAP's checkpoint is loaded automatically. All saved parameters are in
UNCONSTRAINED space (the flat ``z`` the samplers move in). Callers apply
``model.bijector.forward`` -- or ``model.constrained`` for the structured params dict --
to map back to physical space; diagnose.py does this.

``metadata.json`` records ``"api": "scene"``. Runs written before the scene-API port carry
no such key and are NOT readable by this code: the port changed both the z-column order
and the noise convention, so the same stored numbers mean something else. Read those at
the pre-port commit, or re-fit.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gigalens.jax.inference import HMC, MAMS, MAP, MCLMC, SVI
from gigalens.jax.scene_prob_model import ImageData, ProbModel

import data
import lenses
import priors
import svi_health


def get_git_commit():
    """Return the current git commit hash, or None if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def save_map_checkpoint(path, best_z, best_lp, chisq_hist, param_names, config):
    """Save MAP results to npz with metadata.

    ``MAP(..., output_type="best")`` returns the single best particle over all starts and
    all steps, chosen with ``jnp.nanargmax`` internally, so there is no batch of starts and
    no ``best_idx`` to store any more -- the old checkpoint's ``map_estimate`` /
    ``best_idx`` pair is gone. ``chisq_hist`` is the per-step reduced chi-squared of the
    best-scoring particle, kept because it is what shows whether MAP was still descending
    when it stopped.
    """
    np.savez(
        path,
        best_z=best_z,
        best_lp=best_lp,
        chisq_hist=chisq_hist,
        param_names=param_names,
        config=json.dumps(config),
    )


def save_svi_checkpoint(path, qz, loss_hist):
    """Save SVI results to npz. Store mean, scale_tril, and loss history."""
    np.savez(
        path,
        mean=np.asarray(qz.mean()),
        scale_tril=np.asarray(qz.scale_tril),
        loss_hist=np.asarray(loss_hist).reshape(-1),
    )


def normalize_hmc_samples(samples, n_chains, num_results, n_params):
    """Return HMC output as ``(n_chains, n_steps, n_params)`` -- chains FIRST.

    This is the one axis convention that has to be got right by construction rather than
    by eye. ``diagnose_convergence`` (and ArviZ under it) want chains first; TFP puts
    steps first; and a silent swap yields a plausible-looking but wrong split-R-hat, which
    is the failure mode that convinces you a converged run has not converged.

    ``gigalens.jax.inference.HMC`` returns a 4-D array shaped
    ``(n_steps, n_devices, n_chains_per_device, n_params)``. Its own tail-end reshape is
    written for a ``process_allgather`` that merges the process axis; single-process (the
    normal case here) it does not, so the merge folds the length-1 process axis into the
    step axis and the device/chain axes survive separately. Rather than reason about that
    at the call site, both layouts are accepted and each is checked against the shape it
    must have -- an array that matches neither raises rather than being reshaped hopefully.
    """
    a = np.asarray(samples)
    if a.ndim == 4:
        # (n_steps, n_devices, n_chains_per_device, n_params)
        want = (num_results, n_chains, n_params)
        got = (a.shape[0], a.shape[1] * a.shape[2], a.shape[3])
        if got != want:
            raise RuntimeError(
                f"HMC returned {a.shape}, which does not read as (n_steps, n_devices, "
                f"n_chains_per_device, n_params) = {want} once devices and per-device "
                f"chains are merged (got {got})."
            )
        # chain index = device * chains_per_device + c, then steps and chains swapped.
        return a.reshape(num_results, n_chains, n_params).transpose(1, 0, 2)
    if a.ndim == 3:
        # Already chains-first (what a fixed upstream would return).
        if a.shape != (n_chains, num_results, n_params):
            raise RuntimeError(
                f"HMC returned {a.shape}, expected (n_chains, n_steps, n_params) = "
                f"{(n_chains, num_results, n_params)}."
            )
        return a
    raise RuntimeError(
        f"HMC returned a {a.ndim}-D array {a.shape}; expected 3-D (chains, steps, params) "
        f"or 4-D (steps, devices, chains-per-device, params)."
    )


def normalize_mams_mclmc_samples(sampler_name, samples, n_chains, num_results, n_params):
    """Return MAMS/MCLMC output as ``(n_chains, n_steps, n_params)`` -- chains FIRST.

    Unlike ``HMC``, ``MAMS_JIT``/``MCLMC_JIT`` do their own device/chain merge inside
    ``run_sharded`` (the chain axis is what ``shard_map`` partitions on, and the per-step
    history is transposed to chains-first before the device axis is gathered back to
    replicated) and their own burn-in trim (``all_samples[:, -num_results:, :]``), so the
    public return is already ``(n_chains, num_results, num_free_params)`` -- both
    ``run-mams.md`` and ``run-mclmc.md`` state this explicitly. That was confirmed against
    this pipeline empirically with a tiny smoke run rather than taken on the docs' word
    alone. No reshape is needed, but the shape is checked anyway: a mismatch here means an
    assumption about chain count or the device floor is wrong, not that a transpose fixes it.
    """
    a = np.asarray(samples)
    want = (n_chains, num_results, n_params)
    if a.shape != want:
        raise RuntimeError(
            f"{sampler_name} returned {a.shape}; expected {want} = (n_chains, num_results, "
            f"n_params), chains first. This sampler does not need the device/chain merge "
            f"HMC needs, so a mismatch here is not a known reshape -- it means the chain "
            f"count or device floor assumption is wrong."
        )
    return a


def save_hmc_checkpoint(path, samples):
    """Save HMC samples to npz, shape (n_chains, n_steps, n_params), unconstrained."""
    np.savez(path, samples=np.asarray(samples))


def load_map_checkpoint(path):
    """Load MAP checkpoint. Returns (best_z, param_names)."""
    stored = np.load(path, allow_pickle=True)
    if "best_z" not in stored:
        raise ValueError(
            f"{path} predates the scene-API port (no 'best_z'); its samples are in the "
            f"old z-column order and cannot be resumed here. Re-run the MAP stage."
        )
    return stored["best_z"], stored["param_names"]


def load_svi_checkpoint(path):
    """Load SVI checkpoint. Returns (mean, scale_tril, loss_hist)."""
    stored = np.load(path, allow_pickle=True)
    return stored["mean"], stored["scale_tril"], stored["loss_hist"]


def load_hmc_checkpoint(path):
    """Load HMC checkpoint. Returns raw samples array."""
    return np.load(path, allow_pickle=True)["samples"]


def main():
    parser = argparse.ArgumentParser(
        description="Fit one lens through the MAP / SVI / HMC pipeline"
    )
    lenses.add_argument(parser, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=1024, help="MAP batch size")
    parser.add_argument("--map-steps", type=int, default=350, help="MAP optimization steps")
    parser.add_argument("--n-vi", type=int, default=1000, help="SVI batch size")
    parser.add_argument("--svi-steps", type=int, default=1500, help="SVI optimization steps")
    parser.add_argument("--n-hmc", type=int, default=50, help="Sampling stage chains")
    parser.add_argument("--burnin", type=int, default=250, help="Sampling stage burnin steps")
    parser.add_argument(
        "--num-results", type=int, default=750, help="Sampling stage results to keep"
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="hmc",
        choices=["hmc", "mams", "mclmc"],
        help="Sampling-stage algorithm. MAMS and MCLMC share HMC's SVI-seeded start and "
             "--n-hmc/--burnin/--num-results, but tune their own step size and trajectory "
             "length internally rather than taking --max-leapfrog.",
    )
    parser.add_argument(
        "--target-acceptance",
        type=float,
        default=0.9,
        help="MAMS only: acceptance rate the dual-averaging step-size adaptation targets.",
    )
    parser.add_argument(
        "--desired-energy-variance",
        type=float,
        default=5e-4,
        help="MCLMC only: energy-variance setpoint for its step-size controller.",
    )
    parser.add_argument(
        "--theta-e-center",
        type=float,
        default=None,
        help="Einstein radius prior center (arcsec). Default: the lens config's value.",
    )
    parser.add_argument(
        "--theta-e-width",
        type=float,
        default=None,
        help="Einstein radius prior width (log space). Default: the lens config's value.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="map,svi,hmc",
        help="Comma-separated list of stages to run (map, svi, hmc)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: claude/<LENS>/runs/<label>)",
    )
    parser.add_argument("--label", type=str, default="baseline", help="Run label")
    parser.add_argument(
        "--n-source",
        type=int,
        default=1,
        help="Number of Sersic source components (1 or 2)",
    )
    parser.add_argument(
        "--n-lens-light",
        type=int,
        default=1,
        help="Number of Sersic deflector-light components (1, or 2 for bulge+disk)",
    )
    parser.add_argument(
        "--lens-light-n-max",
        type=float,
        default=10.0,
        help="Upper bound of the lens-light n_sersic Uniform prior (n_lens_light=1 only)",
    )
    parser.add_argument(
        "--svi-peak-lr",
        type=float,
        default=2e-3,
        help="Peak SVI learning rate, reached at the end of warmup. Lower this if the "
             "ELBO destabilises at high parameter counts.",
    )
    parser.add_argument(
        "--svi-warmup-frac",
        type=float,
        default=0.1,
        help="Fraction of SVI steps spent ramping up to --svi-peak-lr",
    )
    parser.add_argument(
        "--max-leapfrog",
        type=int,
        default=30,
        help="Ceiling on HMC leapfrog steps per trajectory. 30 was tuned at 22 "
             "parameters; use 60 at 29 or more.",
    )
    parser.add_argument(
        "--svi-max-cond",
        type=float,
        default=svi_health.COND_MAX,
        help="SVI covariance conditioning limit. Raise it only for a deliberately "
             "truncated smoke test, whose SVI has had no chance to converge.",
    )

    args = parser.parse_args()

    # Parse stages
    stages_str = args.stages.lower()
    stages = [s.strip() for s in stages_str.split(",")]
    valid_stages = {"map", "svi", "hmc"}
    for s in stages:
        if s not in valid_stages:
            raise ValueError(f"Invalid stage: {s}. Must be one of {valid_stages}")

    cfg = lenses.load(args.lens)

    # Set output directory. Derived from the lens config, not from this file's location --
    # the pipeline lives outside any one lens's directory.
    if args.out is None:
        out_dir = cfg.runs_dir / args.label
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the theta_E prior now rather than passing None through. metadata.json must
    # record the value actually fitted: diagnose.py reads these back to rebuild the prior,
    # and a null there would silently substitute today's default for whatever this run used.
    theta_e_center = (
        cfg.mass_prior.theta_E_median if args.theta_e_center is None
        else args.theta_e_center
    )
    theta_e_width = (
        cfg.mass_prior.theta_E_logsigma if args.theta_e_width is None
        else args.theta_e_width
    )

    # Load data and build the model. The model owns its prior, its bijector and its
    # z-column order, so there is nothing to keep in step by hand.
    image, psf, sim_config = data.load(cfg)
    model = priors.build_model(
        cfg,
        theta_E_center=theta_e_center,
        theta_E_width=theta_e_width,
        n_source=args.n_source,
        n_lens_light=args.n_lens_light,
        lens_light_n_max=args.lens_light_n_max,
    )
    param_names = list(model.z_param_names)

    # ImageData derives sigma = sqrt(background_rms^2 + clip(observed, 0, inf) / exp_time)
    # ONCE, from the observed image. The pre-port ForwardProbModel recomputed it from the
    # *simulated* image every step, making the variance parameter-dependent. This is the
    # old BackwardProbModel convention and the only one the scene API offers; it is a real
    # change to the likelihood, not a refactor.
    ds = ImageData(
        image,
        sim_config,
        background_rms=cfg.background_rms,
        exp_time=cfg.exp_time,
        sees="all",
    )
    prob = ProbModel(model, ds, mode="forward")

    config = {
        # The port marker. diagnose.rebuild_model refuses a run without it rather than
        # decoding pre-port samples through the new bijector.
        "api": "scene",
        "lens": cfg.name,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "map_steps": args.map_steps,
        "n_vi": args.n_vi,
        "svi_steps": args.svi_steps,
        "sampler": args.sampler,
        "n_hmc": args.n_hmc,
        "burnin": args.burnin,
        "num_results": args.num_results,
        "target_acceptance": args.target_acceptance,
        "desired_energy_variance": args.desired_energy_variance,
        "theta_e_center": theta_e_center,
        "theta_e_width": theta_e_width,
        "stages": stages,
        "label": args.label,
        "git_commit": get_git_commit(),
        "n_source": args.n_source,
        "n_lens_light": args.n_lens_light,
        "lens_light_n_max": args.lens_light_n_max,
        "n_free_params": int(model.num_free_params),
        "svi_peak_lr": args.svi_peak_lr,
        "svi_warmup_frac": args.svi_warmup_frac,
        "max_leapfrog": args.max_leapfrog,
        "n_devices": int(jax.device_count()),
        # Pin the prior versions explicitly. Checkpoints store unconstrained samples, so
        # reading them back through a different prior support silently yields different
        # physical values -- diagnose.py needs to know which one produced this run.
        "source_prior_version": priors.SOURCE_PRIOR_VERSION,
        "lens_light_prior_version": priors.lens_light_prior_version_for(
            args.n_lens_light
        ),
    }
    print(f"Lens: {cfg.name}  ->  {out_dir}")
    print(f"JAX {jax.__version__}, devices: {jax.devices()}")
    print(f"Model: {len(param_names)} parameters "
          f"(n_source={args.n_source}, n_lens_light={args.n_lens_light})")

    timings = {}

    # MAP stage
    if "map" in stages:
        print("=" * 60)
        print("Starting MAP stage")
        print("=" * 60)
        t0 = time.time()

        # Build optimizer: optax negative learning rate composes with scale_by_adam.
        optimizer = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(
                optax.polynomial_schedule(
                    init_value=-1e-2,
                    end_value=-1e-2 / 3,
                    power=0.5,
                    transition_steps=args.map_steps,
                )
            ),
        )

        # output_type="best" returns the single best particle over all starts and all
        # steps, selected with jnp.nanargmax INSIDE MAP, and prints the count of
        # non-finite chains. That retires the hand-rolled np.nanargmax block this script
        # used to carry (jnp.argmax propagates NaN, so one diverged start out of 1024
        # silently selected garbage and every subsequent ELBO came out NaN). The finiteness
        # assertion below stays: it is the one thing MAP's own reporting does not enforce.
        best_z, best_lp, chisq_hist = MAP(
            prob,
            optimizer,
            n_samples=args.n_samples,
            num_steps=args.map_steps,
            seed=args.seed,
            output_type="best",
        )
        best_z = np.asarray(jax.device_get(best_z))
        best_lp = float(jax.device_get(best_lp))
        chisq_hist = np.asarray(jax.device_get(chisq_hist))

        if not np.isfinite(best_z).all() or not np.isfinite(best_lp):
            raise RuntimeError(
                "MAP returned a non-finite best particle; every start diverged. The "
                "priors or the data conditioning are wrong -- SVI cannot start from this."
            )

        t1 = time.time()
        timings["map"] = t1 - t0

        print(f"MAP completed in {timings['map']:.1f}s")
        print(f"Best log-posterior: {best_lp:.4g}")

        # Save checkpoint
        map_npz = out_dir / "map.npz"
        save_map_checkpoint(
            map_npz,
            best_z,
            np.asarray(best_lp),
            chisq_hist,
            np.asarray(param_names, dtype=object),
            config,
        )
        print(f"MAP checkpoint saved to {map_npz}")

    else:
        # Load MAP checkpoint
        map_npz = out_dir / "map.npz"
        if not map_npz.exists():
            raise FileNotFoundError(f"MAP checkpoint not found: {map_npz}")
        best_z, param_names_loaded = load_map_checkpoint(map_npz)
        best_z = np.asarray(best_z)
        if len(param_names_loaded) != len(param_names):
            raise ValueError(
                f"{map_npz} holds {len(param_names_loaded)} parameters but the model built "
                f"here has {len(param_names)}; the checkpoint is from a different model."
            )
        if not np.isfinite(best_z).all():
            raise RuntimeError(f"MAP start in {map_npz} has non-finite parameters.")
        print(f"Loaded MAP checkpoint from {map_npz}")

    # SVI stage
    if "svi" in stages:
        print("=" * 60)
        print("Starting SVI stage")
        print("=" * 60)
        t0 = time.time()

        # Warmup, then decay -- the same shape MAP uses.
        #
        # The previous schedule ramped *upward* (init -1e-6, end -3e-3, power 2), so the
        # largest steps landed at the very end of the run, where the ELBO should be
        # settling. At 22 parameters it survived that; at 29 it did not. The twosource run
        # sat at ELBO -17,850 for most of the run, swung to +10,550, and ended at -16,659
        # with sd = 7,384 over the final 300 steps. The resulting covariance had a negative
        # eigenvalue, HMC preconditioned on its inverse, and all 29 parameters failed Rhat.
        # See svi_health.check_svi_covariance, which now refuses that outright.
        #
        # Peak LR here is below the 3e-3 the old schedule reached, and it arrives early --
        # where the ELBO is far from the optimum and a large step is safe -- rather than at
        # the end.
        #
        # Learning rates are NEGATIVE throughout: optax composes them with scale_by_adam
        # as a scaling, so descent needs the sign.
        warmup = max(1, int(args.svi_steps * args.svi_warmup_frac))
        peak = -abs(args.svi_peak_lr)
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(peak / 200.0, peak, transition_steps=warmup),
                optax.polynomial_schedule(
                    init_value=peak,
                    end_value=peak / 10.0,
                    power=0.5,
                    transition_steps=args.svi_steps - warmup,
                ),
            ],
            boundaries=[warmup],
        )
        print(f"SVI schedule: warmup {warmup} steps to lr {abs(peak):.1e}, "
              f"then decay to {abs(peak) / 10:.1e} over {args.svi_steps - warmup} steps")
        optimizer = optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(schedule)
        )

        # SVI(prob_model, start, optimizer, ...) -- start SECOND, optimizer THIRD.
        qz, loss_hist = SVI(
            prob,
            jnp.asarray(best_z),
            optimizer,
            n_vi=args.n_vi,
            num_steps=args.svi_steps,
            seed=args.seed,
        )

        t1 = time.time()
        timings["svi"] = t1 - t0

        print(f"SVI completed in {timings['svi']:.1f}s")

        # Save the checkpoint BEFORE the health check, so a diverged run is still
        # inspectable after the raise rather than being thrown away.
        svi_npz = out_dir / "svi.npz"
        save_svi_checkpoint(svi_npz, qz, loss_hist)
        print(f"SVI checkpoint saved to {svi_npz}")

        # Report the two numbers that distinguish a converged SVI from a diverged one, and
        # fail here rather than letting HMC run on an indefinite mass matrix. The library
        # has no such guard of its own -- HMC inverts this covariance unconditionally.
        tail = np.asarray(loss_hist).reshape(-1)[-300:]
        print(f"ELBO loss: final {float(tail[-1]):.1f}, "
              f"sd over last {len(tail)} steps {float(np.std(tail)):.1f} "
              f"(healthy 22-param runs sit near 10)")
        min_eig, cond = svi_health.check_svi_covariance(
            scale_tril=qz.scale_tril, max_cond=args.svi_max_cond
        )
        print(f"SVI covariance: min eigenvalue {min_eig:.3e}, condition {cond:.3e} "
              f"(healthy runs 1e6-6e8, limit {args.svi_max_cond:.0e})")

    else:
        # Load SVI checkpoint
        svi_npz = out_dir / "svi.npz"
        if not svi_npz.exists():
            raise FileNotFoundError(f"SVI checkpoint not found: {svi_npz}")
        mean, scale_tril, loss_hist = load_svi_checkpoint(svi_npz)
        # Reconstruct qz distribution
        from tensorflow_probability.substrates.jax import distributions as tfd

        qz = tfd.MultivariateNormalTriL(loc=mean, scale_tril=scale_tril)
        print(f"Loaded SVI checkpoint from {svi_npz}")
        min_eig, cond = svi_health.check_svi_covariance(
            scale_tril=scale_tril, max_cond=args.svi_max_cond
        )
        print(f"SVI covariance: min eigenvalue {min_eig:.3e}, condition {cond:.3e}")

    # Sampling stage. All three samplers are seeded from the same SVI surrogate qz --
    # HMC via inv(qz.covariance()) as its momentum precision, MAMS/MCLMC via qz as both
    # the chains' starting distribution and the initial inverse mass matrix.
    #
    # Saved shape is (n_chains, n_steps, n_params) -- chains FIRST, which is what
    # gigalens.jax.analysis.diagnose_convergence (and ArviZ) want. HMC returns the
    # opposite (see normalize_hmc_samples); MAMS/MCLMC already return chains-first (see
    # normalize_mams_mclmc_samples) because their device/chain merge and burn-in trim
    # happen inside the library call, not here.
    if "hmc" in stages:
        print("=" * 60)
        print(f"Starting sampling stage ({args.sampler})")
        print("=" * 60)
        t0 = time.time()

        n_chains = (args.n_hmc // jax.device_count()) * jax.device_count()

        if args.sampler == "hmc":
            samples = HMC(
                prob,
                qz,
                init_eps=0.3,
                init_l=3,
                n_hmc=args.n_hmc,
                num_burnin_steps=args.burnin,
                num_results=args.num_results,
                max_leapfrog_steps=args.max_leapfrog,
                seed=args.seed,
            )
            samples = normalize_hmc_samples(
                jax.device_get(samples), n_chains, args.num_results, model.num_free_params
            )
        elif args.sampler == "mams":
            samples = MAMS(
                prob,
                qz,
                n_hmc=args.n_hmc,
                num_burnin_steps=args.burnin,
                num_results=args.num_results,
                target_acceptance=args.target_acceptance,
                seed=args.seed,
            )
            samples = normalize_mams_mclmc_samples(
                "MAMS", jax.device_get(samples), n_chains, args.num_results,
                model.num_free_params,
            )
        elif args.sampler == "mclmc":
            samples = MCLMC(
                prob,
                qz,
                n_hmc=args.n_hmc,
                num_burnin_steps=args.burnin,
                num_results=args.num_results,
                desired_energy_variance=args.desired_energy_variance,
                seed=args.seed,
            )
            samples = normalize_mams_mclmc_samples(
                "MCLMC", jax.device_get(samples), n_chains, args.num_results,
                model.num_free_params,
            )
        else:
            raise ValueError(f"Unknown sampler: {args.sampler}")

        t1 = time.time()
        timings["hmc"] = t1 - t0

        print(f"{args.sampler.upper()} completed in {timings['hmc']:.1f}s")

        # Save checkpoint. Filename stays "hmc.npz" regardless of --sampler --
        # diagnose.py and the plot_*.py tools look for that name, not for one keyed on
        # the algorithm, and metadata.json's "sampler" field is what tells you which
        # algorithm actually produced it.
        hmc_npz = out_dir / "hmc.npz"
        save_hmc_checkpoint(hmc_npz, samples)
        print(f"{args.sampler.upper()} checkpoint saved to {hmc_npz}  {samples.shape}")

    # Write metadata
    config["timings"] = timings
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    print("=" * 60)
    print("All requested stages completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
