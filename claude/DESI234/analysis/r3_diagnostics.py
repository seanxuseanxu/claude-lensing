"""DESI234 round 3 diagnostics: r3_unmasked (36 params) vs r3_fg (46 params).

Both runs share a likelihood (unmasked, supersample=4), so their log posteriors are
directly comparable with each other -- but not with rounds 1-2, which were masked.

Uses diagnose.py's own functions throughout (rebuild_model, to_constrained,
map_convergence_check, masked_chisq, boundary_pinning_check, diagnose_convergence) rather
than reimplementing any of the bijector, likelihood, chi-squared or R-hat/ESS machinery.
Chi-squared goes through diagnose.masked_chisq only (KNOWN_ISSUES.md item 3).

Run with PYTHONPATH=claude/pipeline. Writes its full report to
claude/DESI234/analysis/r3_diagnostics.txt via stdout redirection.
"""

import io
import contextlib
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import diagnose
from gigalens.jax.analysis import diagnose_convergence

RUNS_DIR = Path("/global/u1/s/seanjx/claude-lensing/claude/DESI234/runs")
RUNS = ["r3_unmasked", "r3_fg"]


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def report_run(label):
    run_dir = RUNS_DIR / label
    section(f"RUN: {label}  ({run_dir})")

    rm = diagnose.rebuild_model(run_dir)

    hmc = diagnose.load_checkpoint(run_dir, "hmc")
    hmc_samples = np.asarray(hmc["samples"])  # (n_chains, n_steps, n_params)
    n_chains, n_steps, n_params = hmc_samples.shape
    print(f"hmc.npz shape: {hmc_samples.shape}")
    print(f"n_free_params (metadata): {rm.run_cfg.get('n_free_params')}, "
          f"model.num_free_params: {rm.model.num_free_params}")

    # --- 1. map_convergence_check -------------------------------------------------
    section(f"[{label}] 1. MAP convergence check (full output)")
    diagnose.map_convergence_check(run_dir, rm, hmc_samples=hmc_samples)

    # Also report tail slope on last 500 (in addition to the last-1000-equivalent
    # map_convergence_check already prints as "last N/5000").
    mp = diagnose.load_checkpoint(run_dir, "map")
    chisq_hist = np.asarray(mp["chisq_hist"], dtype=np.float64).reshape(-1)
    for tail_len in (500, 1000):
        tail = chisq_hist[-tail_len:]
        slope = float(tail[-1] - tail[0]) / max(1, tail_len - 1)
        print(f"MAP chisq_hist tail slope (last {tail_len}/{len(chisq_hist)}): "
              f"{slope:+.6f}/step ({'still descending' if slope < 0 else 'flat or rising'})")

    # --- 2. R-hat / ESS, all 50 chains, all params ----------------------------------
    section(f"[{label}] 2. R-hat / bulk+tail ESS, all {n_chains} chains, all {n_params} params")
    con = diagnose.to_constrained(rm.model, hmc_samples.reshape(-1, n_params))
    con3d = np.asarray(con).reshape(n_chains, n_steps, n_params)

    report = diagnose_convergence(hmc_samples, model=rm.model)
    rhat, ess_bulk, ess_tail = np.asarray(report.rhat), np.asarray(report.ess_bulk), np.asarray(report.ess_tail)
    labels = rm.labels
    width = max(len(l) for l in labels)
    print(f"{'parameter':<{width}}{'Rhat':>9}{'ESSbulk':>10}{'ESStail':>10}")
    for i, name in enumerate(labels):
        flag = "  <-- Rhat>1.01" if rhat[i] > diagnose.RHAT_THRESHOLD else ""
        print(f"{name:<{width}}{rhat[i]:>9.4f}{ess_bulk[i]:>10.1f}{ess_tail[i]:>10.1f}{flag}")

    max_rhat = float(np.max(rhat))
    n_bad = int(np.sum(rhat > diagnose.RHAT_THRESHOLD))
    min_bulk_ess = float(np.min(ess_bulk))
    min_tail_ess = float(np.min(ess_tail))
    worst_idx = np.argsort(rhat)[::-1][:5]
    med_all = np.median(con3d.reshape(-1, n_params), axis=0)
    print(f"\nmax R-hat: {max_rhat:.4f}   #params with Rhat>1.01: {n_bad}/{n_params}")
    print(f"min bulk ESS: {min_bulk_ess:.1f}   min tail ESS: {min_tail_ess:.1f}")
    print("5 worst-Rhat parameters:")
    for i in worst_idx:
        print(f"  {labels[i]:<{width}} Rhat={rhat[i]:.4f}  median={med_all[i]:.5f}  "
              f"ESSbulk={ess_bulk[i]:.1f}  ESStail={ess_tail[i]:.1f}")

    # --- 3. boundary pinning ---------------------------------------------------------
    section(f"[{label}] 3. Boundary pinning check (full output)")
    flat_con = con3d.reshape(-1, n_params)
    diagnose.boundary_pinning_check(rm.model, flat_con, rm.pnames)

    # --- 4. masked_chisq at posterior median -----------------------------------------
    section(f"[{label}] 4. masked_chisq at posterior median")
    med_z = np.median(hmc_samples.reshape(-1, n_params), axis=0)
    im_sim = diagnose.simulate_from_unconstrained(rm, med_z)
    image = np.asarray(rm.image)
    err_map = np.asarray(rm.prob.error_map)
    norm_res = (image - im_sim) / err_map
    keep = (np.ones(image.shape, dtype=bool) if rm.mask is None
            else np.asarray(rm.mask, dtype=bool))
    chi2_nu, n_kept = diagnose.masked_chisq(norm_res, keep)
    n_kept_recorded = rm.run_cfg.get("n_masked_pixels")
    print(f"chi2_nu at posterior median: {chi2_nu:.4f}")
    print(f"n_kept pixels (rebuilt mask): {n_kept}, recorded n_masked_pixels: {n_kept_recorded}")
    assert n_kept_recorded is None or int(n_kept_recorded) == n_kept, "mismatch -- rebuild_model should have raised already"

    # --- 5. theta_E posterior ---------------------------------------------------------
    section(f"[{label}] 5. theta_E (host EPL) posterior")
    idx_theta_E = diagnose.column(rm.model, "mass/0/theta_E", kind="mass")
    lo, med, hi = np.percentile(flat_con[:, idx_theta_E], [16, 50, 84])
    print(f"theta_E: median {med:.4f}, 16pct {lo:.4f}, 84pct {hi:.4f}  "
          f"({med:.3f} +{hi-med:.3f}/-{med-lo:.3f})")

    # --- best_lp / max / median sampler log posterior for model comparison ----------
    map_lp, _ = rm.prob.log_prob(jnp.asarray(np.asarray(mp["best_z"]).reshape(-1)))
    map_lp = float(np.asarray(map_lp))
    from plot_corner import thin_to_cap
    thinned, factor = thin_to_cap(hmc_samples, 3000)
    flatz = thinned.reshape(-1, thinned.shape[-1])
    hmc_lp, _ = rm.prob.log_prob(jnp.asarray(flatz))
    hmc_lp = np.asarray(hmc_lp)

    result = dict(
        label=label, rm=rm, n_params=n_params, max_rhat=max_rhat, n_bad=n_bad,
        min_bulk_ess=min_bulk_ess, min_tail_ess=min_tail_ess, chi2_nu=chi2_nu,
        theta_E=(med, lo, hi), map_lp=map_lp, hmc_lp_max=float(np.max(hmc_lp)),
        hmc_lp_med=float(np.median(hmc_lp)), con3d=con3d, flat_con=flat_con,
        hmc_samples=hmc_samples,
    )
    return result


def foreground_report(res):
    """Item 6: foreground SIS theta_E and Sersic Ie for r3_fg, nested-model check."""
    rm = res["rm"]
    flat_con = res["flat_con"]
    section("6. r3_fg foreground component posterior (nested-model check)")

    names = list(rm.model.z_param_names)
    print("Full z_param_names list (for the record, exact names used below):")
    for n in names:
        print(f"  {n}")

    # Structural lookup: any mass-site theta_E that is NOT the host's mass/0/theta_E.
    theta_E_mass_names = [n for n in names if n.endswith("/theta_E") and "/mass/" in n]
    fg_theta_E_name = [n for n in theta_E_mass_names if n != "planes/0/mass/0/theta_E"]
    if len(fg_theta_E_name) != 1:
        # fall back: use diagnose.column with kind="mass" is ambiguous by design here,
        # so search directly instead.
        print(f"NOTE: expected exactly one non-host mass theta_E, found {fg_theta_E_name}; "
              f"all mass theta_E columns: {theta_E_mass_names}")
    fg_theta_E_name = fg_theta_E_name[0] if len(fg_theta_E_name) == 1 else theta_E_mass_names[-1]
    idx_fg_theta_E = names.index(fg_theta_E_name)

    # Foreground light Sersic Ie: light-site Ie that is not part of the lens-light or
    # source planes -- i.e. under the same plane/component the SIS theta_E is under, or
    # identifiable as the sole "extra" Ie beyond n_lens_light + n_source component counts.
    # Structural approach: find all light-site Ie columns and print them, then pick the
    # one belonging to the foreground component's plane/index (same plane index as the
    # SIS mass component, since priors.py places the foreground SIS+Sersic pair together).
    fg_plane = fg_theta_E_name.split("/")[1]
    ie_names = [n for n in names if n.endswith("/Ie") and f"/{fg_plane}/" in n and "/light/" in n]
    print(f"\nForeground SIS theta_E column: {fg_theta_E_name}")
    print(f"Candidate foreground light Ie columns (same plane {fg_plane}): {ie_names}")

    # Disambiguate from lens light: the host lens light is n_lens_light=2 components at
    # low index; foreground is appended after, per priors.build_model's component order
    # (see priors.py). Take the highest-indexed light Ie in that plane as the foreground's,
    # and print all candidates so this choice is auditable rather than a guess.
    if len(ie_names) == 0:
        raise KeyError("No candidate foreground Ie column found under the SIS's plane.")
    fg_ie_name = sorted(ie_names, key=lambda n: int(n.split("/")[3]))[-1]
    idx_fg_ie = names.index(fg_ie_name)
    print(f"Foreground Sersic Ie column selected: {fg_ie_name}")

    theta_E_samples = flat_con[:, idx_fg_theta_E]
    ie_samples = flat_con[:, idx_fg_ie]

    lo_t, med_t, hi_t = np.percentile(theta_E_samples, [16, 50, 84])
    lo_i, med_i, hi_i = np.percentile(ie_samples, [16, 50, 84])
    sd_t, sd_i = float(np.std(theta_E_samples)), float(np.std(ie_samples))
    dist_t_sd = med_t / sd_t if sd_t > 0 else np.inf
    dist_i_sd = med_i / sd_i if sd_i > 0 else np.inf
    frac_near_zero = float(np.mean(theta_E_samples < 0.05))

    print(f"\n{fg_theta_E_name}: median {med_t:.5f}, 16pct {lo_t:.5f}, 84pct {hi_t:.5f}, "
          f"sd {sd_t:.5f}, median/sd {dist_t_sd:.3f}")
    print(f"fraction of posterior mass with theta_E < 0.05 arcsec: {frac_near_zero:.4f}")
    print(f"{fg_ie_name}: median {med_i:.6e}, 16pct {lo_i:.6e}, 84pct {hi_i:.6e}, "
          f"sd {sd_i:.6e}, median/sd {dist_i_sd:.3f}")

    return dict(theta_E_name=fg_theta_E_name, theta_E=(med_t, lo_t, hi_t), theta_E_sd=sd_t,
                theta_E_dist_sd=dist_t_sd, frac_near_zero=frac_near_zero,
                ie_name=fg_ie_name, ie=(med_i, lo_i, hi_i), ie_sd=sd_i, ie_dist_sd=dist_i_sd)


def svi_nan_report():
    section("8. SVI NaN in r3_fg -- step index, count, pre-NaN convergence, surrogate finiteness")
    svi_fg = np.load(RUNS_DIR / "r3_fg" / "svi.npz", allow_pickle=True)
    loss = np.asarray(svi_fg["loss_hist"], dtype=np.float64).reshape(-1)
    finite = np.isfinite(loss)
    n_total = len(loss)
    n_nonfinite = int(np.sum(~finite))
    first_bad = int(np.argmax(~finite)) if n_nonfinite > 0 else None
    print(f"r3_fg svi.npz loss_hist: {n_total} steps, {n_nonfinite} non-finite")
    if first_bad is not None:
        print(f"first non-finite step index: {first_bad}")
        pre = loss[:first_bad]
        if len(pre) >= 300:
            tail = pre[-300:]
            print(f"loss over 300 steps before the first non-finite value: "
                  f"final {tail[-1]:.2f}, sd {np.std(tail):.2f}")
            print(f"first 10 pre-NaN losses: {pre[:10]}")
            print(f"last 10 pre-NaN losses: {tail[-10:]}")
        else:
            print(f"only {len(pre)} finite steps before the first non-finite value: {pre}")
    else:
        print("no non-finite values found in loss_hist")

    mean = np.asarray(svi_fg["mean"], dtype=np.float64)
    scale_tril = np.asarray(svi_fg["scale_tril"], dtype=np.float64)
    mean_finite = bool(np.all(np.isfinite(mean)))
    tril_finite = bool(np.all(np.isfinite(scale_tril)))
    print(f"\nsaved surrogate mean finite: {mean_finite} (shape {mean.shape})")
    print(f"saved surrogate scale_tril finite: {tril_finite} (shape {scale_tril.shape})")
    if not mean_finite:
        print(f"  non-finite entries in mean: {np.sum(~np.isfinite(mean))}")
    if not tril_finite:
        print(f"  non-finite entries in scale_tril: {np.sum(~np.isfinite(scale_tril))}")

    return dict(n_total=n_total, n_nonfinite=n_nonfinite, first_bad=first_bad,
                mean_finite=mean_finite, tril_finite=tril_finite)


def main():
    results = {}
    for label in RUNS:
        results[label] = report_run(label)

    fg_extra = foreground_report(results["r3_fg"])

    section("7. Model comparison: r3_unmasked (36 params) vs r3_fg (46 params)")
    ru, rf = results["r3_unmasked"], results["r3_fg"]
    print(f"r3_unmasked: n_params={ru['n_params']}  median HMC log-post={ru['hmc_lp_med']:.3f}  "
          f"max HMC log-post={ru['hmc_lp_max']:.3f}  MAP recomputed lp={ru['map_lp']:.3f}")
    print(f"r3_fg      : n_params={rf['n_params']}  median HMC log-post={rf['hmc_lp_med']:.3f}  "
          f"max HMC log-post={rf['hmc_lp_max']:.3f}  MAP recomputed lp={rf['map_lp']:.3f}")
    print(f"\nDifference (r3_fg - r3_unmasked): median {rf['hmc_lp_med']-ru['hmc_lp_med']:+.3f} nats, "
          f"max {rf['hmc_lp_max']-ru['hmc_lp_max']:+.3f} nats, "
          f"MAP {rf['map_lp']-ru['map_lp']:+.3f} nats")
    print(f"r3_fg has {rf['n_params']-ru['n_params']} more parameters than r3_unmasked.")
    print("diagnose.py does not implement an information criterion (AIC/BIC/WAIC); none "
          "is computed here per instructions.")

    svi_res = svi_nan_report()

    section("SUMMARY (for reply table construction)")
    for label in RUNS:
        r = results[label]
        print(f"{label}: max_rhat={r['max_rhat']:.4f} n_bad={r['n_bad']} "
              f"min_bulk_ess={r['min_bulk_ess']:.1f} chi2_nu={r['chi2_nu']:.4f} "
              f"theta_E={r['theta_E']} "
              f"map_deficit_nats={r['hmc_lp_max']-r['map_lp']:.3f}")
    print(f"\nfg foreground: {fg_extra}")
    print(f"\nSVI NaN: {svi_res}")


if __name__ == "__main__":
    main()
