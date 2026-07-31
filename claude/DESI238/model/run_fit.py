"""Fit the DESI238 lens system through the MAP / SVI / HMC pipeline.

The three inference stages are checkpointed independently, so a crash in HMC does not
lose the MAP and SVI results. Stages can be resumed: if --stages omits "map" but svi is
requested, MAP's checkpoint is loaded automatically. All returned parameters are in
UNCONSTRAINED space (the space that prior.sample produces). Callers apply
prob_model.bij.forward to map back to physical space; diagnose.py does this.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

# CRITICAL: compat must be imported before any gigalens.jax module.
import compat

compat.require_single_device()

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import ForwardProbModel

import data
import priors


def get_git_commit():
    """Return the current git commit hash, or None if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def save_map_checkpoint(path, map_estimate, lp_values, chisq_values, best_idx, param_names, config):
    """Save MAP results to npz with metadata."""
    np.savez(
        path,
        map_estimate=map_estimate,
        lp_values=lp_values,
        chisq_values=chisq_values,
        best_idx=best_idx,
        param_names=param_names,
        config=json.dumps(config),
    )


def save_svi_checkpoint(path, qz, loss_hist):
    """Save SVI results to npz. Store mean, scale_tril, and loss history."""
    np.savez(
        path,
        mean=np.asarray(qz.mean()),
        scale_tril=np.asarray(qz.scale_tril),
        loss_hist=np.asarray(loss_hist),
    )


def save_hmc_checkpoint(path, samples):
    """Save HMC samples to npz, shape (num_results, n_hmc, n_params), unconstrained."""
    np.savez(path, samples=np.asarray(samples))


def load_map_checkpoint(path):
    """Load MAP checkpoint. Returns (map_estimate, best_idx, param_names)."""
    data = np.load(path, allow_pickle=True)
    return data["map_estimate"], data["best_idx"], data["param_names"]


def load_svi_checkpoint(path):
    """Load SVI checkpoint. Returns (mean, scale_tril, loss_hist)."""
    data = np.load(path, allow_pickle=True)
    return data["mean"], data["scale_tril"], data["loss_hist"]


def load_hmc_checkpoint(path):
    """Load HMC checkpoint. Returns raw samples array."""
    data = np.load(path, allow_pickle=True)
    return data["samples"]


def main():
    parser = argparse.ArgumentParser(
        description="Fit DESI238 through MAP / SVI / HMC pipeline"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=1024, help="MAP batch size")
    parser.add_argument("--map-steps", type=int, default=350, help="MAP optimization steps")
    parser.add_argument("--n-vi", type=int, default=1000, help="SVI batch size")
    parser.add_argument("--svi-steps", type=int, default=1500, help="SVI optimization steps")
    parser.add_argument("--n-hmc", type=int, default=50, help="HMC samples per device")
    parser.add_argument("--burnin", type=int, default=250, help="HMC burnin steps")
    parser.add_argument("--num-results", type=int, default=750, help="HMC results to keep")
    parser.add_argument(
        "--theta-e-center",
        type=float,
        default=1.5,
        help="Einstein radius prior center (arcsec)",
    )
    parser.add_argument(
        "--theta-e-width",
        type=float,
        default=0.35,
        help="Einstein radius prior width (log space)",
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
        help="Output directory (default: claude/DESI238/model/runs/<label>)",
    )
    parser.add_argument("--label", type=str, default="baseline", help="Run label")
    parser.add_argument(
        "--n-source",
        type=int,
        default=1,
        help="Number of Sersic source components (1 or 2)",
    )
    parser.add_argument(
        "--lens-light-n-max",
        type=float,
        default=10.0,
        help="Upper bound of the lens-light n_sersic Uniform prior",
    )

    args = parser.parse_args()

    # Parse stages
    stages_str = args.stages.lower()
    stages = [s.strip() for s in stages_str.split(",")]
    valid_stages = {"map", "svi", "hmc"}
    for s in stages:
        if s not in valid_stages:
            raise ValueError(f"Invalid stage: {s}. Must be one of {valid_stages}")

    # Set output directory
    if args.out is None:
        out_dir = Path(__file__).parent / "runs" / args.label
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data and build model
    image, psf, sim_config = data.load()
    # n_source must match between build_prior and build_phys_model: LensSimulator zips
    # profiles against parameters positionally, so a mismatch silently drops the extra
    # source component instead of raising.
    prior = priors.build_prior(
        theta_E_center=args.theta_e_center,
        theta_E_width=args.theta_e_width,
        n_source=args.n_source,
        lens_light_n_max=args.lens_light_n_max,
    )
    phys_model = priors.build_phys_model(n_source=args.n_source)
    param_names = priors.param_names(prior)

    prob_model = ForwardProbModel(
        prior,
        observed_image=image,
        background_rms=data.BACKGROUND_RMS,
        exp_time=data.EXP_TIME,
    )

    sequence = ModellingSequence(phys_model, prob_model, sim_config)

    config = {
        "seed": args.seed,
        "n_samples": args.n_samples,
        "map_steps": args.map_steps,
        "n_vi": args.n_vi,
        "svi_steps": args.svi_steps,
        "n_hmc": args.n_hmc,
        "burnin": args.burnin,
        "num_results": args.num_results,
        "theta_e_center": args.theta_e_center,
        "theta_e_width": args.theta_e_width,
        "stages": stages,
        "label": args.label,
        "git_commit": get_git_commit(),
        "n_source": args.n_source,
        "lens_light_n_max": args.lens_light_n_max,
        # Pin the prior version explicitly. Checkpoints store unconstrained samples, so
        # reading them back through a different prior support silently yields different
        # physical values -- diagnose.py needs to know which one produced this run.
        "source_prior_version": priors.SOURCE_PRIOR_VERSION,
    }

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

        map_estimate = sequence.MAP(
            optimizer, start=None, n_samples=args.n_samples, num_steps=args.map_steps, seed=args.seed
        )

        # Evaluate MAP on the full batch to get log_prob for each sample
        from gigalens.jax.simulator import LensSimulator

        lens_sim = LensSimulator(phys_model, sim_config, bs=args.n_samples)
        lp, chisq = prob_model.log_prob(lens_sim, map_estimate)

        # Find best by log_prob, ignoring diverged starts.
        #
        # jnp.argmax propagates NaN: a single start that blew up returns ITS index, not
        # the best one. That is not hypothetical -- the first nsersic10 run had exactly
        # one NaN among 1024 starts, argmax returned it, and SVI began from a garbage
        # point and produced an ELBO of NaN for 1500 steps. The MAP progress bar read
        # chi2 = 2.003 the whole time, so the failure was invisible until SVI died.
        # nanargmax skips them.
        lp_np = np.asarray(lp)
        n_bad = int((~np.isfinite(lp_np)).sum())
        if n_bad == len(lp_np):
            raise RuntimeError(
                f"All {n_bad} MAP starts produced non-finite log_prob. The priors or the "
                f"data conditioning are wrong; SVI cannot be started from this."
            )
        if n_bad:
            print(f"WARNING: {n_bad}/{len(lp_np)} MAP starts diverged to non-finite "
                  f"log_prob and were excluded from the best-sample selection.")
        best_idx = int(np.nanargmax(np.where(np.isfinite(lp_np), lp_np, np.nan)))
        best_start = map_estimate[best_idx][None, :]  # Add leading axis for SVI

        if not bool(jnp.isfinite(best_start).all()):
            raise RuntimeError(
                f"MAP start {best_idx} has non-finite parameters despite a finite "
                f"log_prob. Refusing to hand this to SVI."
            )

        t1 = time.time()
        timings["map"] = t1 - t0

        print(f"MAP completed in {timings['map']:.1f}s")
        print(f"Best reduced chi-squared: {float(chisq[best_idx]):.3f}")

        # Save checkpoint
        map_npz = out_dir / "map.npz"
        save_map_checkpoint(
            map_npz,
            np.asarray(map_estimate),
            np.asarray(lp),
            np.asarray(chisq),
            np.asarray(best_idx),
            np.asarray(param_names, dtype=object),
            config,
        )
        print(f"MAP checkpoint saved to {map_npz}")

    else:
        # Load MAP checkpoint
        map_npz = out_dir / "map.npz"
        if not map_npz.exists():
            raise FileNotFoundError(f"MAP checkpoint not found: {map_npz}")
        map_estimate, best_idx, param_names_loaded = load_map_checkpoint(map_npz)

        # Recompute the best index rather than trusting the stored one. Checkpoints
        # written before the nanargmax fix above hold a best_idx chosen by jnp.argmax,
        # which returns the index of a NaN when any start diverged -- resuming from such
        # a file feeds SVI a garbage start and every ELBO step comes out NaN. Recomputing
        # is cheap and makes old checkpoints safe to resume.
        stored = np.load(map_npz, allow_pickle=True)
        lp_np = np.asarray(stored["lp_values"])
        n_bad = int((~np.isfinite(lp_np)).sum())
        if n_bad == len(lp_np):
            raise RuntimeError(
                f"All {n_bad} MAP starts in {map_npz} have non-finite log_prob; "
                f"re-run the MAP stage."
            )
        recomputed = int(np.nanargmax(np.where(np.isfinite(lp_np), lp_np, np.nan)))
        if recomputed != int(best_idx):
            print(f"NOTE: stored best_idx {int(best_idx)} has non-finite or sub-optimal "
                  f"log_prob ({n_bad}/{len(lp_np)} starts diverged); using {recomputed} "
                  f"instead.")
        best_idx = recomputed
        best_start = map_estimate[best_idx][None, :]
        if not np.isfinite(best_start).all():
            raise RuntimeError(f"MAP start {best_idx} has non-finite parameters.")
        print(f"Loaded MAP checkpoint from {map_npz} (best_idx {best_idx}, "
              f"chi2 {float(np.asarray(stored['chisq_values'])[best_idx]):.3f})")

    # SVI stage
    if "svi" in stages:
        print("=" * 60)
        print("Starting SVI stage")
        print("=" * 60)
        t0 = time.time()

        # JAX SVI signature: SVI(start, optimizer, n_vi=..., num_steps=...)
        # NOTE: start FIRST, optimizer SECOND. This is opposite to TensorFlow backend.
        optimizer = optax.chain(
            optax.scale_by_adam(),
            optax.scale_by_schedule(
                optax.polynomial_schedule(
                    init_value=-1e-6,
                    end_value=-3e-3,
                    power=2,
                    transition_steps=args.svi_steps,
                )
            ),
        )

        qz, loss_hist = sequence.SVI(
            best_start, optimizer, n_vi=args.n_vi, num_steps=args.svi_steps, seed=args.seed
        )

        t1 = time.time()
        timings["svi"] = t1 - t0

        print(f"SVI completed in {timings['svi']:.1f}s")

        # Save checkpoint
        svi_npz = out_dir / "svi.npz"
        save_svi_checkpoint(svi_npz, qz, loss_hist)
        print(f"SVI checkpoint saved to {svi_npz}")

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

    # HMC stage.
    #
    # ModellingSequence.HMC is NOT used. Its `run_chain` is wrapped in @pmap while `q_z`
    # and the momentum distribution are built outside, so their Auto-sharded arrays meet a
    # Manual-sharded pmap mesh and TFP's scale_tril matvec raises a context-mesh
    # ValueError. That happens even with a single device, so require_single_device() is no
    # defence. compat.hmc_single_device reproduces the identical kernel stack and
    # adaptation schedule without the pmap wrapper; it is verified to sample cleanly.
    #
    # Shape differs from upstream: (num_results, n_hmc, n_params), chains on axis 1.
    if "hmc" in stages:
        print("=" * 60)
        print("Starting HMC stage")
        print("=" * 60)
        t0 = time.time()

        samples = compat.hmc_single_device(
            prob_model,
            phys_model,
            sim_config,
            qz,
            init_eps=0.3,
            init_l=3,
            n_hmc=args.n_hmc,
            num_burnin_steps=args.burnin,
            num_results=args.num_results,
            max_leapfrog_steps=30,
            seed=args.seed,
        )

        t1 = time.time()
        timings["hmc"] = t1 - t0

        print(f"HMC completed in {timings['hmc']:.1f}s")

        # Save checkpoint
        hmc_npz = out_dir / "hmc.npz"
        save_hmc_checkpoint(hmc_npz, samples)
        print(f"HMC checkpoint saved to {hmc_npz}")

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
