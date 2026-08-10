"""How far short DESI234's round-2 MAP stopped, and what the SVI stage looked like.

`r2_s2l2/map.npz` records `best_lp = 17376.09` with a `chisq_hist` tail that still looks
like it is descending. DESI257 hit the same symptom (`claude/pipeline/KNOWN_ISSUES.md` item
6): its MAP finished 681.8 nats below the best HMC draw of the chain it seeded, and that
deficit substantially decided which mode 48 of 50 chains started in. This script runs the
same check here, through `diagnose.rebuild_model` and `diagnose.map_convergence_check` --
nothing here reimplements a bijector or a likelihood (`CLAUDE.md`).

It also resolves the KNOWN_ISSUES item-1 metadata trap: `r2_s2l2/metadata.json` claims
`map_steps=350, n_samples=1024` (argparse defaults, written by a later `--stages hmc`
overwrite), but `RESULTS_v2.md` SS2.1 says 1500 steps at n_vi batch 4096, and `map.npz`'s own
`chisq_hist` has length 1500 -- confirming the README, not the metadata.

Second part: `svi_health`/`diagnose.svi_report` on both `r1_s1l1` and `r2_s2l2`, to settle
whether `r1_s1l1`'s SVI covariance was genuinely diverged (which would justify populating
`verify.py`'s `svi_guard_reference` for DESI234) or whether it was merely a bad fit that
failed on R-hat for an unrelated reason. `svi_health.py` itself exports only
`check_svi_covariance` (a guard, no CLI); `diagnose.svi_report` is the runnable wrapper
around it and is used here directly.
"""
import sys
sys.path.insert(0, "/global/u1/s/seanjx/claude-lensing/claude/pipeline")
import io
import contextlib

import numpy as np
import jax.numpy as jnp

import diagnose
import svi_health

LENS_DIR = "/global/u1/s/seanjx/claude-lensing/claude/DESI234"
RUNS = {
    "r1_s1l1": f"{LENS_DIR}/runs/r1_s1l1",
    "r2_s2l2": f"{LENS_DIR}/runs/r2_s2l2",
}


def section(title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def map_deficit_report(run_dir, label):
    """Run diagnose.map_convergence_check, plus the explicit last-500-of-1500 tail slope
    the round-3 plan quoted (-4.4e-5/step), so that figure is confirmed or corrected."""
    rm = diagnose.rebuild_model(run_dir, quiet=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diagnose.map_convergence_check(run_dir, rm)
    captured = buf.getvalue()
    print(captured, end="")

    mp = diagnose.load_checkpoint(run_dir, "map")
    hmc = diagnose.load_checkpoint(run_dir, "hmc")
    chisq_hist = np.asarray(mp["chisq_hist"], dtype=np.float64).reshape(-1)
    n_steps = len(chisq_hist)

    meta = diagnose.load_run_config(run_dir)
    meta_map_steps = meta.get("map_steps")
    print(f"\nchisq_hist length: {n_steps} steps "
          f"(metadata.json claims map_steps={meta_map_steps}: "
          f"{'CONSISTENT' if n_steps == meta_map_steps else 'CONTRADICTS metadata.json'})")
    if label == "r2_s2l2":
        print("  r2_s2l2's metadata.json is known-corrupted (KNOWN_ISSUES.md item 1: a "
              "later --stages hmc run overwrote it with argparse defaults). RESULTS_v2.md "
              "SS2.1 states the true staging was map_steps=1500 at n_vi batch 4096; "
              f"chisq_hist's actual length of {n_steps} "
              f"{'CONFIRMS' if n_steps == 1500 else 'CONTRADICTS'} that, independent of "
              "the corrupted metadata field.")

    def tail_slope(window):
        y = chisq_hist[-window:]
        x = np.arange(len(y), dtype=np.float64)
        slope, _ = np.polyfit(x, y, 1)
        return slope, len(y)

    last500_slope, n500 = tail_slope(500)
    print(f"chisq_hist tail slope, explicit last {n500} of {n_steps} steps: "
          f"{last500_slope:.3e}/step "
          f"(plan quoted -4.4e-5/step)")

    # diagnose.py's own convergence-check window (see map_convergence_check source: it does
    # not itself print a slope over a fixed window distinct from what's above, so also
    # report the last 150 steps -- the window KNOWN_ISSUES item 6 quotes for DESI257 -- as a
    # second, independently-chosen check.
    last150_slope, n150 = tail_slope(150)
    print(f"chisq_hist tail slope, last {n150} steps (DESI257 KNOWN_ISSUES item-6 window): "
          f"{last150_slope:.3e}/step")

    stored_lp = float(np.asarray(mp["best_lp"]))
    best_z = np.asarray(mp["best_z"]).reshape(-1)
    recomputed_lp, _ = rm.prob.log_prob(jnp.asarray(best_z))
    recomputed_lp = float(np.asarray(recomputed_lp))
    print(f"map.npz best_lp (stored):     {stored_lp:.3f}")
    print(f"map.npz best_lp (recomputed): {recomputed_lp:.3f}")

    if hmc is not None:
        samples = np.asarray(hmc["samples"])
        flat = samples.reshape(-1, samples.shape[-1])
        sub = flat[:: max(1, len(flat) // 3000)]
        lp, _ = rm.prob.log_prob(jnp.asarray(sub))
        lp = np.asarray(lp)
        max_lp = float(lp.max())
        median_lp = float(np.median(lp))
        deficit = max_lp - recomputed_lp
        print(f"sampler log posterior over {len(sub)} thinned draws: "
              f"max {max_lp:.3f}, median {median_lp:.3f}")
        print(f"deficit (max sampler lp - MAP best_lp): {deficit:.1f} nats "
              f"({'MAP stopped short' if deficit > 0 else 'MAP cleared the sampler'})")
    else:
        print("no hmc.npz -- cannot compute deficit against sampler draws")

    return rm


def svi_report(run_dir, label):
    svi = diagnose.load_checkpoint(run_dir, "svi")
    if svi is None:
        print(f"{label}: no svi.npz")
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        diagnose.svi_report(run_dir)
    print(buf.getvalue(), end="")

    # Explicit verdict on the diverged-vs-bad-fit question, independent of svi_report's own
    # wording, since that is the specific fact verify.py's guard needs.
    try:
        min_eig, cond = svi_health.check_svi_covariance(scale_tril=svi["scale_tril"])
        print(f"{label}: SVI covariance PASSES check_svi_covariance "
              f"(min eig {min_eig:.3e}, cond {cond:.3e}, limit {svi_health.COND_MAX:.0e}) "
              f"-- NOT diverged by this guard's definition.")
    except RuntimeError as exc:
        print(f"{label}: SVI covariance FAILS check_svi_covariance -- {exc}")


def main():
    for label in ("r1_s1l1", "r2_s2l2"):
        section(f"MAP deficit: {label}")
        map_deficit_report(RUNS[label], label)

    for label in ("r1_s1l1", "r2_s2l2"):
        section(f"SVI health: {label}")
        svi_report(RUNS[label], label)


if __name__ == "__main__":
    main()
