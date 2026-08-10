"""Is round 3's R-hat failure FROZEN chains or MULTIPLE MODES?

The two look identical in a summary R-hat table and have opposite fixes:

  frozen      -- a chain barely moves. Within-chain sd is ~0 (orders of magnitude below
                 the between-chain spread), and bulk ESS collapses toward n_chains
                 because each chain contributes ~one effective draw. The fix is
                 sampler/step-size/burn-in, not the model.
  multimodal  -- a chain explores normally about a shifted mean. Within-chain sd is
                 comparable to a single mode's width; the chains fall into a small number
                 of clusters, and R-hat computed WITHIN a cluster comes back near 1. The
                 fix is a model change (or an admission of degeneracy), not more burn-in.

The r3_unmasked numbers already hint at frozen (bulk ESS 50.7 on 50 chains, i.e. ~1.01
effective draws per chain) while r3_fg does not (bulk ESS 71-380 on 50 chains). This
script measures it rather than inferring it from ESS.

Three parts, in order of how decisive they are:

  A. per-chain theta_E median and within-chain sd, plus the frozen/multimodal ratio
     R = between-chain sd of the per-chain medians / median within-chain sd.
  B. the foreground component across chains (r3_fg only): does the SIS theta_E / Sersic Ie
     pair split into a "present" and an "absent" cluster? That is the bimodality
     hypothesis for the interloper stated in RESULTS_v3.md section 4b.
  C. k-means clustering of chains on their standardised per-chain medians, followed by
     R-hat recomputed WITHIN each cluster. If within-cluster R-hat is near 1, the chains
     are sampling several well-mixed modes; if it stays large, they are frozen.

Everything numerical goes through diagnose.rebuild_model / to_constrained and gigalens'
own diagnose_convergence -- no hand-rolled bijector or R-hat here (CLAUDE.md).

Run from the repo root:

    PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES=0 \
      python claude/DESI234/analysis/mode_structure.py \
      > claude/DESI234/analysis/mode_structure.txt 2> claude/DESI234/analysis/mode_structure.err.txt
"""

from pathlib import Path

import numpy as np

import diagnose
from gigalens.jax.analysis import diagnose_convergence

RUNS_DIR = Path("/global/u1/s/seanjx/claude-lensing/claude/DESI234/runs")
RUNS = ["r3_unmasked", "r3_fg"]

# A chain whose within-chain sd is this many times below the between-chain spread of the
# per-chain medians is called frozen. The threshold is a reporting convention, not a test:
# the per-chain table is printed in full so the reader can see the separation directly.
FROZEN_RATIO = 10.0


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load(label):
    run_dir = RUNS_DIR / label
    rm = diagnose.rebuild_model(run_dir)
    z = np.asarray(diagnose.load_checkpoint(run_dir, "hmc")["samples"])
    n_chains, n_steps, n_params = z.shape
    con = np.asarray(diagnose.to_constrained(rm.model, z.reshape(-1, n_params)))
    con = con.reshape(n_chains, n_steps, n_params)
    return rm, z, con


# ---------------------------------------------------------------------------- A --------

def per_chain_stats(con, labels, idx, name, label):
    """Per-chain median and within-chain sd for one constrained parameter."""
    x = con[:, :, idx]                       # (n_chains, n_steps)
    med = np.median(x, axis=1)
    sd = np.std(x, axis=1)
    between = float(np.std(med))
    within = float(np.median(sd))
    ratio = between / within if within > 0 else np.inf

    print(f"\n[{label}] {name}  (column {idx})")
    print(f"{'chain':>6}{'median':>14}{'within-sd':>14}{'sd/within_med':>16}")
    order = np.argsort(med)
    for c in order:
        print(f"{c:>6}{med[c]:>14.5f}{sd[c]:>14.3e}{(sd[c]/within if within>0 else np.nan):>16.3f}")
    print(f"\n  between-chain sd of the per-chain medians : {between:.5e}")
    print(f"  median within-chain sd                    : {within:.5e}")
    print(f"  ratio (between / within)                  : {ratio:.2f}")
    verdict = ("FROZEN-like: within-chain motion is negligible against the between-chain spread"
               if ratio > FROZEN_RATIO else
               "MODE-like: chains move by an amount comparable to their separation")
    print(f"  verdict at ratio > {FROZEN_RATIO:g}: {verdict}")
    print(f"  full range of per-chain medians: {med.min():.5f} .. {med.max():.5f}")
    return dict(median=med, within_sd=sd, between=between, within=within, ratio=ratio)


# ---------------------------------------------------------------------------- C --------

def kmeans(X, k, seed=0, n_init=10, n_iter=200):
    """Plain Lloyd k-means on standardised rows. Small enough (50 x n_params) that a
    dependency is not worth it; n_init restarts guard the local minimum."""
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = X[rng.choice(len(X), size=k, replace=False)].copy()
        assign = np.zeros(len(X), dtype=int)
        for _ in range(n_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
            new = d.argmin(1)
            if np.array_equal(new, assign):
                break
            assign = new
            for j in range(k):
                if np.any(assign == j):
                    centers[j] = X[assign == j].mean(0)
        inertia = float(((X - centers[assign]) ** 2).sum())
        if inertia < best_inertia:
            best, best_inertia = (assign.copy(), centers.copy()), inertia
    return best[0], best[1], best_inertia


def cluster_chains(z, con, model, labels, label, k_max=5):
    """Cluster chains on their standardised per-chain medians (constrained space), then
    recompute R-hat WITHIN each cluster on the UNCONSTRAINED samples -- diagnose_convergence
    takes unconstrained z, exactly as the full-run check does."""
    section(f"[{label}] C. chain clustering and within-cluster R-hat")
    n_chains, n_steps, n_params = con.shape
    M = np.median(con, axis=1)                        # (n_chains, n_params)
    s = M.std(0)
    s[s == 0] = 1.0
    X = (M - M.mean(0)) / s

    print(f"{'k':>3}{'inertia':>14}{'cluster sizes':>40}")
    fits = {}
    for k in range(1, k_max + 1):
        assign, centers, inertia = kmeans(X, k, seed=0)
        sizes = [int(np.sum(assign == j)) for j in range(k)]
        fits[k] = (assign, inertia, sizes)
        print(f"{k:>3}{inertia:>14.2f}{str(sizes):>40}")

    for k in range(2, k_max + 1):
        assign, inertia, sizes = fits[k]
        print(f"\n--- k = {k}: within-cluster R-hat (clusters with >= 4 chains only) ---")
        print(f"cluster assignment by chain: {assign.tolist()}")
        for j in range(k):
            members = np.flatnonzero(assign == j)
            if len(members) < 4:
                print(f"  cluster {j}: {len(members)} chains -- too few for R-hat, skipped")
                continue
            rep = diagnose_convergence(z[members], model=model)
            rh = np.asarray(rep.rhat)
            eb = np.asarray(rep.ess_bulk)
            worst = int(np.argmax(rh))
            print(f"  cluster {j}: {len(members):>2} chains  max R-hat {rh.max():>8.4f} "
                  f"(worst: {labels[worst]})  #>1.01 {int((rh > diagnose.RHAT_THRESHOLD).sum())}/{n_params}  "
                  f"min bulk ESS {eb.min():>7.1f}")
    return fits


def main():
    results = {}
    for label in RUNS:
        rm, z, con = load(label)
        labels = rm.labels
        n_chains, n_steps, n_params = con.shape
        section(f"RUN {label}: {n_chains} chains x {n_steps} steps x {n_params} params")

        # --- A: theta_E, the number the round is about -------------------------------
        section(f"[{label}] A. per-chain theta_E: frozen or multimodal?")
        idx = diagnose.column(rm.model, "mass/0/theta_E", kind="mass")
        a = per_chain_stats(con, labels, idx, "lens/mass/host/theta_E", label)

        # A companion parameter, for the record: the same statistic on the single worst
        # R-hat parameter of this run, in case theta_E is atypical of the failure.
        rep = diagnose_convergence(z, model=rm.model)
        worst = int(np.argmax(np.asarray(rep.rhat)))
        if worst != idx:
            section(f"[{label}] A'. same statistic on the worst-R-hat parameter")
            per_chain_stats(con, labels, worst, labels[worst], label)

        # --- B: the foreground component across chains (r3_fg only) -------------------
        if label == "r3_fg":
            section(f"[{label}] B. the interloper across chains (bimodality hypothesis)")
            names = list(rm.model.z_param_names)
            fg_theta = [n for n in names
                        if n.endswith("/theta_E") and "/mass/" in n and n != "planes/0/mass/0/theta_E"]
            assert len(fg_theta) == 1, fg_theta
            i_ft = names.index(fg_theta[0])
            plane = fg_theta[0].split("/")[1]
            ie = sorted([n for n in names if n.endswith("/Ie") and f"/{plane}/" in n and "/light/" in n],
                        key=lambda n: int(n.split("/")[3]))[-1]
            i_ie = names.index(ie)
            print(f"foreground SIS theta_E column: {fg_theta[0]}")
            print(f"foreground Sersic Ie column  : {ie}")
            per_chain_stats(con, labels, i_ft, fg_theta[0], label)
            per_chain_stats(con, labels, i_ie, ie, label)

            # Does any chain sit near the nested null (theta_E -> 0, Ie -> 0)?
            t = con[:, :, i_ft]
            f = con[:, :, i_ie]
            print(f"\nper-chain fraction of draws with SIS theta_E < 0.05 arcsec:")
            frac = (t < 0.05).mean(1)
            print("  " + ", ".join(f"{c}:{frac[c]:.3f}" for c in range(len(frac)) if frac[c] > 0)
                  or "  none -- no chain visits the null in theta_E")
            print(f"  max over chains: {frac.max():.4f}")
            print(f"per-chain median (SIS theta_E, Sersic Ie) correlation across chains: "
                  f"{np.corrcoef(np.median(t,1), np.median(f,1))[0,1]:+.4f}")

        # --- C: clustering ------------------------------------------------------------
        fits = cluster_chains(z, con, rm.model, labels, label)
        results[label] = dict(theta_E=a, fits=fits)

    section("SUMMARY")
    for label, r in results.items():
        a = r["theta_E"]
        print(f"{label}: theta_E between-chain sd {a['between']:.4e}, median within-chain sd "
              f"{a['within']:.4e}, ratio {a['ratio']:.2f} -> "
              f"{'FROZEN-like' if a['ratio'] > FROZEN_RATIO else 'MODE-like'}")


if __name__ == "__main__":
    main()
