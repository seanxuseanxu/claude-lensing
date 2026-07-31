"""Print a posterior summary table (median, mean, sd, 16/84%, Rhat, ESS) for one run.

Reuses diagnose.py's loading/transform machinery (load_checkpoint, load_run_config,
to_constrained, priors.param_names, and the tfp Rhat/ESS calls it uses in
convergence_table) rather than reimplementing any of it. diagnose.py is safe to import:
its argument parsing and side effects are gated under `if __name__ == "__main__":`, so
`import diagnose` only defines functions/constants.

Column order matches priors.param_names(prior): blocks in order lens, shear, lens_light,
source, alphabetical by key within each block (e.g. "lens.theta_E", "shear.gamma1",
"lens_light0.R_sersic", "source0.n_sersic"). This is the same convention diagnose.py's
convergence_table prints under.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow_probability.substrates.jax as tfp

import diagnose
import data
import priors
from gigalens.jax.model import ForwardProbModel
import jax.numpy as jnp


def main():
    ap = argparse.ArgumentParser(description="Posterior summary table for one run")
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    run_cfg = diagnose.load_run_config(run_dir)
    n_source = int(run_cfg.get("n_source", 1))
    n_lens_light = int(run_cfg.get("n_lens_light", 1))
    lens_light_n_max = float(run_cfg.get("lens_light_n_max", 6.0))
    theta_e_center = float(run_cfg.get("theta_e_center", 1.5))
    theta_e_width = float(run_cfg.get("theta_e_width", 0.35))
    source_prior_version = int(run_cfg.get("source_prior_version",
                                           2 if "n_source" in run_cfg else 1))
    lens_light_prior_version = int(
        run_cfg.get("lens_light_prior_version",
                    priors.lens_light_prior_version_for(n_lens_light))
    )

    image, psf, sim_config = data.load()
    prior = priors.build_prior(
        theta_E_center=theta_e_center,
        theta_E_width=theta_e_width,
        n_source=n_source,
        n_lens_light=n_lens_light,
        lens_light_n_max=lens_light_n_max,
        source_prior_version=source_prior_version,
        lens_light_prior_version=lens_light_prior_version,
    )
    prob_model = ForwardProbModel(
        prior,
        jnp.asarray(image),
        background_rms=data.BACKGROUND_RMS,
        exp_time=data.EXP_TIME,
    )
    pnames = priors.param_names(prior)

    hmc = diagnose.load_checkpoint(run_dir, "hmc")
    if hmc is None:
        raise FileNotFoundError(f"No hmc.npz in {run_dir}")
    samples = hmc["samples"]  # (num_results, n_hmc, n_params)
    n_draws, n_chains, n_params = samples.shape
    print(f"Loaded HMC samples with shape {samples.shape} "
          f"(num_results={n_draws}, n_hmc={n_chains}, n_params={n_params})")

    con = diagnose.to_constrained(prob_model, samples.reshape(-1, n_params))
    con = con.reshape(n_draws, n_chains, n_params)

    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(con, independent_chain_ndims=1))
    ess = np.asarray(tfp.mcmc.effective_sample_size(con, cross_chain_dims=1))

    flat = con.reshape(-1, n_params)
    lo, med, hi = np.percentile(flat, [16, 50, 84], axis=0)
    mean = np.mean(flat, axis=0)
    sd = np.std(flat, axis=0)

    header = ["parameter", "median", "mean", "sd", "p16", "p84", "rhat", "ess"]
    rows = []
    for i, name in enumerate(pnames):
        rows.append([
            name,
            float(med[i]),
            float(mean[i]),
            float(sd[i]),
            float(lo[i]),
            float(hi[i]),
            float(rhat[i]),
            float(ess[i]),
        ])

    print("\nParameter naming convention: priors.param_names(prior) order -- blocks "
          "[lens, shear, lens_light, source], alphabetical by key within each block "
          "(e.g. 'lens.theta_E', 'lens_light0.R_sersic', 'source0.n_sersic').")

    print("\t".join(header))
    for r in rows:
        print(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}"
              f"\t{r[6]:.6f}\t{r[7]:.1f}")

    out_path = run_dir / "params.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
