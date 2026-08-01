"""Print a posterior summary table (median, mean, sd, 16/84%, Rhat, ESS) for one run.

Reuses diagnose.py's loading/transform machinery (load_checkpoint, rebuild_model,
to_constrained, and the tfp Rhat/ESS calls it uses in convergence_table) rather than
reimplementing any of it. diagnose.py is safe to import: its argument parsing and side
effects are gated under `if __name__ == "__main__":`, so `import diagnose` only defines
functions/constants.

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
import lenses


def main():
    ap = argparse.ArgumentParser(description="Posterior summary table for one run")
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    lenses.add_argument(ap, required=False)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    rm = diagnose.rebuild_model(run_dir, lens=args.lens, quiet=True)
    prob_model = rm.prob_model
    pnames = rm.pnames

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
