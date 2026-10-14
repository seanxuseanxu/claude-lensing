"""Print a posterior summary table (median, mean, sd, 16/84%, Rhat, ESS) for one run.

Reuses diagnose.py's loading/transform machinery (load_checkpoint, rebuild_model,
to_constrained) rather than reimplementing any of it. diagnose.py is safe to import: its
argument parsing and side effects are gated under `if __name__ == "__main__":`, so
`import diagnose` only defines functions/constants.

R-hat and ESS come from ``gigalens.jax.analysis.diagnose_convergence`` -- the same call
``diagnose.convergence_table`` makes, so the two tools cannot disagree about whether a run
converged. They are computed in **unconstrained** space (where the sampler actually moved);
the quantiles below are in constrained space, which is the space the numbers get published
in.

Rows are in ``model.z_param_names`` order: path-keyed and sorted, so within the lens plane
``light`` precedes ``mass``. Each row carries both names -- the display label
(``lens/mass/host/theta_E``) and the functional path key (``planes/0/mass/0/theta_E``),
which is the stable identifier to join a CSV on, since it does not move when a component
is renamed.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from gigalens.jax.analysis import diagnose_convergence

import diagnose
import lenses


def main():
    ap = argparse.ArgumentParser(description="Posterior summary table for one run")
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    lenses.add_argument(ap, required=False)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    rm = diagnose.rebuild_model(run_dir, lens=args.lens, quiet=True)

    hmc = diagnose.load_checkpoint(run_dir, "hmc", rm)
    if hmc is None:
        raise FileNotFoundError(f"No hmc.npz in {run_dir}")
    samples = np.asarray(hmc["samples"])  # (n_chains, n_steps, n_params)
    n_chains, n_draws, n_params = samples.shape
    print(f"Loaded HMC samples with shape {samples.shape} "
          f"(n_chains={n_chains}, n_steps={n_draws}, n_params={n_params})")

    con = diagnose.to_constrained(rm.model, samples.reshape(-1, n_params))

    report = diagnose_convergence(samples, model=rm.model)
    rhat, ess_bulk, ess_tail = report.rhat, report.ess_bulk, report.ess_tail

    lo, med, hi = np.percentile(con, [16, 50, 84], axis=0)
    mean = np.mean(con, axis=0)
    sd = np.std(con, axis=0)

    header = ["parameter", "path", "median", "mean", "sd", "p16", "p84",
              "rhat", "ess_bulk", "ess_tail"]
    rows = []
    for i, name in enumerate(rm.labels):
        rows.append([
            name,
            rm.pnames[i],
            float(med[i]),
            float(mean[i]),
            float(sd[i]),
            float(lo[i]),
            float(hi[i]),
            float(rhat[i]),
            float(ess_bulk[i]),
            float(ess_tail[i]),
        ])

    print("\nParameter naming: the 'parameter' column is model.z_labels "
          "(<plane>/<kind>/<component>/<param>, e.g. 'lens/mass/host/theta_E'), which "
          "tracks the display names given in priors.build_model. The 'path' column is "
          "model.z_param_names (e.g. 'planes/0/mass/0/theta_E') -- the functional key, "
          "which is what to join on, since renaming a component moves the label but not "
          "the path. Rows are in z-column order: sorted by path, so lens *light* precedes "
          "lens *mass* within the lens plane.")

    print("\t".join(header))
    for r in rows:
        print(f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}"
              f"\t{r[6]:.6f}\t{r[7]:.6f}\t{r[8]:.1f}\t{r[9]:.1f}")

    out_path = run_dir / "params.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
