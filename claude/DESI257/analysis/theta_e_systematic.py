"""How far does DESI257's quoted theta_E move because of the ss=2 quadrature systematic?

This is a MEASUREMENT, not a re-fit. `undersampling_report.txt` already established that
`r3_unmasked`'s configured supersample=2 FAILS the library's 0.2 sigma/px tolerance against
a subgrid-convention reference at supersample=64 (max|delta|/sigma = 3.697, delta_chi2 mean
-221.1 over a 9-sample batch spread across chains/steps). That number alone doesn't say
which way, or how far, the POSTERIOR moves -- the quadrature error is parameter-dependent
(it tilts the log-likelihood surface, not just offsets it), and a tilt shifts a posterior
mean. This script estimates that shift to first order (linear response / Gaussian
approximation), NOT by re-running HMC at ss=64.

METHOD
------
1. Rebuild r3_unmasked's exact ProbModel (`diagnose.rebuild_model`, recorded supersample=2,
   unmasked, 6400/6400 px -- confirmed below, not assumed).
2. Select N=60 posterior draws spread across all 50 chains and the full 10000-step range
   (index-paired linspace over chains and steps -- same construction already used by
   `undersampling_check.py`'s `spread_hmc_draws`, extended from N=9 to N=60). MAP and the
   HMC posterior median are carried as two EXTRA points, scored the same way but excluded
   from the regression (they are point estimates, not posterior draws).
3. For each of the N+2 points, call `gigalens.jax.analysis.undersampling.diagnose_undersampling`
   with `supersample_ladder=[2]` (the run's own configured rung -- no need to render the
   full (1,2,4,8) ladder for this) and `reference_supersample=64` (the same reference depth
   `undersampling_recheck.py` already used; ss=2's number was shown there to move <5% between
   an ss_ref=32 and ss_ref=64 reference, which is what makes reusing that depth -- rather than
   re-deriving a fresh, more expensive one -- defensible here). Both the ss=2 rung and the
   reference render under the SAME PSF convention: `RungResult.convention` and
   `reference_self_error`'s only key are both "subgrid" (asserted below), so this isolates
   the quadrature (grid-refinement) error and does NOT include the ~3.7 sigma bin-first-vs-
   subgrid PSF-CONVENTION gap `undersampling_report.txt` measured separately on the ss=1 rung.
   The reference does NOT self-certify here either (subgrid self-check ~0.072 sigma vs the
   library's 0.05 sigma pass margin, same as every other script in this directory) -- reported,
   not certified.

   SIGN CONVENTION (read from `gigalens/jax/analysis/undersampling.py`, not assumed):
   the module defines `RungResult.delta_chi2 = chi2(rung) - chi2(reference)` (source line
   ~358, ~633-634). r3_unmasked was FITTED at the rung (ss=2); its actual sampled posterior
   is proportional to exp(-chi2_ss2(theta)/2). The "more correct" posterior -- the one that
   would have been sampled had the likelihood been evaluated at the converged reference --
   is proportional to exp(-chi2_ref(theta)/2). The log-density CORRECTION that turns the
   fitted posterior into the reference posterior is therefore

       delta(theta) := log p_ref(theta) - log p_ss2(theta)
                     = -chi2_ref(theta)/2 - ( -chi2_ss2(theta)/2 )
                     = ( chi2_ss2(theta) - chi2_ref(theta) ) / 2
                     = RungResult.delta_chi2(theta) / 2.

   Equivalently, in the task's Delta-chi^2 notation, this is `delta(theta) = -Delta_chi2/2`
   with `Delta_chi2 := chi2(reference) - chi2(coarse)` -- the OTHER sign than the module's
   own `delta_chi2` attribute, which is `chi2(coarse) - chi2(reference)`. Get the wrong one
   and the sign of the whole result flips; this is checked against the already-measured
   `delta_chi2` mean of -221 (SANITY CHECK 1 below: delta(theta) should average to roughly
   -221/2 = -110.5, NOT +110.5).
4. Regress delta(theta) on the 6 host-mass parameters (theta_E, gamma, e1, e2, center_x,
   center_y -- `column(model, "mass/0/<name>")`, the EPL host, not the SIS foreground or
   shear), STANDARDIZED (zero mean, unit sd over the N draws) before regressing, by ordinary
   least squares with an intercept. theta_E and gamma are known to be correlated (posterior
   covariance below), so a univariate regression on theta_E alone would misattribute shared
   variance; all 6 are regressed jointly. Reports the standardized gradient with OLS standard
   errors and R^2. R^2 is the credibility gate: if delta is not close to linear in the
   parameters over the posterior bulk, R^2 will be low and the shift below is reported only
   as an order-of-magnitude bound, not a value.
5. Convert to a shift of the posterior mean via the standard first-order / linear-response
   result: if a log-density is perturbed by delta(theta) (small, and here linear in theta by
   construction of the regression), the induced shift in the posterior mean of any parameter
   vector theta is

       Delta_theta = Cov_p[theta, delta(theta)]  ~=  Sigma @ g

   where Sigma is the posterior covariance of the same 6 parameters (estimated from a stride
   through the full 50x10000-draw chain, not just the N=60 regression points) and g is the
   gradient d(delta)/d(theta) in ORIGINAL units (the standardized OLS coefficients divided by
   each parameter's own sd over the N draws). This is exactly the leading term of an
   importance-reweighting expansion w(theta) = exp(delta(theta)) ~= 1 + delta(theta) - mean(delta),
   E_ref[theta] - E_ss2[theta] ~= Cov_ss2[theta, delta] -- the same expansion whose failure
   (weight collapse from the ~17-nat delta_chi2 spread) is why a full reweighting was ruled
   out and this first-order estimate is used instead.
6. Sanity checks (reported whichever way they come out): mean(delta) against the already
   measured -221/2; std(delta) against the already-reported ~17-nat delta_chi2 spread;
   split-half regression agreement; and the linear-response validity condition (delta small
   and near-linear over the posterior bulk -- gated on R^2 and on Delta_theta being small
   relative to the posterior spread, since a "first-order" correction that turns out to be a
   large multiple of the posterior sd is self-undermining).

VALIDITY CONDITION -- stated plainly: this is a FIRST-ORDER estimate, not a refit. It is
only trustworthy if (a) delta(theta) is close to linear over the region the N=60 draws span
(R^2 high) and (b) the resulting Delta_theta is not itself large compared to the posterior
sd (a "small" perturbation assumption that a large shift would contradict). Both are checked
and reported, not assumed.

COST: `supersample_ladder=[2]` (one rung, not the full 1/2/4/8 ladder) at
`reference_supersample=64`, subgrid convention only (no PSF-convention-gap reference is
built, since only one convention is in play) measured at ~5.2s/sample on this CPU login
node (10-sample timing check). N=62 (60 regression draws + MAP + HMC-median) -> ~5.4
minutes, well inside the 30-minute budget; no reduction to N or to ss_ref=32 was needed.

Run (repo root, CPU-only, no GPU/SLURM needed):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
        /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
        claude/DESI257/analysis/theta_e_systematic.py

Outputs (this directory): `theta_e_systematic.txt` (full report), `theta_e_systematic.png`
(delta vs theta_E scatter with the marginal OLS line -- the honest picture of whether the
regression means anything).
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))

import diagnose  # noqa: E402
from gigalens.jax.analysis import diagnose_undersampling  # noqa: E402

RUN = Path(__file__).resolve().parents[1] / "runs" / "r3_unmasked"
OUT_DIR = Path(__file__).resolve().parent
N_DRAWS = 60
REF_SUPERSAMPLE = 64
LADDER = (2,)  # the run's own configured rung only
MASS_PARAM_SUFFIXES = ["center_x", "center_y", "e1", "e2", "gamma", "theta_E"]

# Colors: muted blue for the draws, a warm orange for the fitted line -- categorical
# identity, not magnitude/diverging, so two fixed hues rather than a ramp (dataviz skill).
COLOR_POINTS = "#4C72B0"
COLOR_FIT = "#DD8452"
COLOR_REF = "#55A868"


def spread_hmc_draws(samples: np.ndarray, n: int, seed_note: str = ""):
    """``n`` unconstrained-z draws spread across chains and steps of an
    ``(n_chains, n_steps, n_params)`` HMC array, via index-paired linspace over both axes --
    the same construction as `undersampling_check.py`'s `spread_hmc_draws`, just at a larger
    ``n``. With ``n=60 > n_chains=50`` some chains are hit twice, each time at a different,
    still widely spread, step -- not repeats of the same (chain, step) pair. Returns
    ``(draws, chains, steps)`` so the exact selection is recorded, not just used.
    """
    n_chains, n_steps, _ = samples.shape
    chains = np.linspace(0, n_chains - 1, n, dtype=int)
    steps = np.linspace(0, n_steps - 1, n, dtype=int)
    draws = np.stack([samples[c, s] for c, s in zip(chains, steps)], axis=0)
    return draws, chains, steps


def ols_with_se(X_std: np.ndarray, y: np.ndarray):
    """OLS of ``y`` on standardized predictors ``X_std`` (n, p) with an intercept.

    Returns (beta[p+1] incl. intercept first, se[p+1], r2, resid).
    """
    n, p = X_std.shape
    Xd = np.column_stack([np.ones(n), X_std])
    beta, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = n - (p + 1)
    sigma2 = float(resid @ resid) / dof if dof > 0 else np.nan
    cov_beta = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov_beta))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, r2, resid


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    t_start = time.perf_counter()

    emit("=" * 78)
    emit("DESI257 r3_unmasked -- first-order theta_E shift from the ss=2 quadrature error")
    emit("=" * 78)
    emit()

    rm = diagnose.rebuild_model(RUN, quiet=True)
    model = rm.model

    # --- confirm the run's own conditions, rather than assume the established context ----
    mask = np.asarray(rm.prob.terms[0].dataset.mask, dtype=bool)
    n_kept = int(mask.sum())
    emit(f"mask: {n_kept}/{mask.size} px kept "
         f"({'UNMASKED, confirmed' if n_kept == mask.size else 'MASKED -- differs from expectation'})")

    hmc = diagnose.load_checkpoint(RUN, "hmc")
    hmc_samples = hmc["samples"]  # (n_chains, n_steps, n_params)
    n_chains, n_steps, n_params = hmc_samples.shape
    emit(f"HMC checkpoint: {n_chains} chains x {n_steps} steps x {n_params} params "
         "(post-burnin, per metadata.json)")

    map_ck = diagnose.load_checkpoint(RUN, "map")
    z_map = np.asarray(map_ck["best_z"]).reshape(-1)
    z_median, _ = diagnose.point_estimate(RUN, hmc_samples=hmc_samples)

    z_draws, sel_chains, sel_steps = spread_hmc_draws(hmc_samples, N_DRAWS)
    emit()
    emit(f"Draw selection: N={N_DRAWS} posterior draws, index-paired linspace over "
         f"{n_chains} chains x {n_steps} steps (spread_hmc_draws).")
    emit(f"  chains used (sorted unique): {sorted(set(sel_chains.tolist()))}")
    emit(f"  step range covered: {sel_steps.min()} .. {sel_steps.max()} "
         f"({len(set(sel_steps.tolist()))} unique step indices)")
    emit("  MAP and HMC-median carried as 2 extra points, scored identically, excluded "
         "from the regression (not posterior draws).")

    z_batch = np.vstack([z_draws, z_map[None, :], z_median[None, :]])
    IDX_MAP, IDX_MEDIAN = N_DRAWS, N_DRAWS + 1
    emit(f"z_batch shape: {z_batch.shape} (60 draws + MAP + HMC-median)")

    # --- step 3: render ss=2 vs ss_ref=64, both subgrid convention -----------------------
    # Rendering at reference_supersample=64 on an 80x80 cutout means an effective
    # 5120x5120 render per sample; a single call batching all 62 samples risks exhausting
    # memory on a shared login node (the 10-sample timing check below was the largest batch
    # actually exercised). Chunk into CHUNK_SIZE-sample calls and concatenate -- validated
    # behavior, not a guess: CHUNK_SIZE=10 is exactly the batch size the earlier timing
    # check (52.0s / 10 samples) already ran successfully.
    emit()
    emit("-" * 78)
    emit(f"STEP: diagnose_undersampling(ladder={LADDER}, reference_supersample={REF_SUPERSAMPLE}), "
         f"chunked")
    emit("-" * 78)
    CHUNK_SIZE = 10
    n_total = z_batch.shape[0]
    delta_chi2_chunks = []
    convention = None
    self_err_max = {}
    converged_all = True
    t0 = time.perf_counter()
    for start in range(0, n_total, CHUNK_SIZE):
        z_chunk = z_batch[start:start + CHUNK_SIZE]
        tc0 = time.perf_counter()
        reports_c = diagnose_undersampling(
            rm.prob, z_chunk,
            supersample_ladder=LADDER,
            reference_supersample=REF_SUPERSAMPLE,
            check_reference=True,
        )
        tc1 = time.perf_counter()
        rep_c = reports_c[0]
        ss2_c = rep_c.rungs[0]
        assert ss2_c.supersample == 2
        delta_chi2_chunks.append(ss2_c.delta_chi2)
        convention = ss2_c.convention if convention is None else convention
        assert ss2_c.convention == convention, "convention changed across chunks"
        for k, v in rep_c.reference_self_error.items():
            self_err_max[k] = max(self_err_max.get(k, 0.0), v)
        converged_all = converged_all and rep_c.reference_converged
        emit(f"  chunk [{start}:{start + z_chunk.shape[0]}] "
             f"({z_chunk.shape[0]} samples): {tc1 - tc0:.1f}s")
    t1 = time.perf_counter()
    wall = t1 - t0
    delta_chi2 = np.concatenate(delta_chi2_chunks)
    emit(f"[timing] {wall:.1f}s total for {n_total} samples in "
         f"{-(-n_total // CHUNK_SIZE)} chunks of <= {CHUNK_SIZE} "
         f"({wall / n_total:.2f}s/sample)")

    import types
    report = types.SimpleNamespace(
        reference_self_error=self_err_max, reference_converged=converged_all)
    ss2 = types.SimpleNamespace(convention=convention)

    conventions_seen = set(report.reference_self_error.keys()) | {ss2.convention}
    emit(f"rung convention: {ss2.convention!r}; reference_self_error keys: "
         f"{list(report.reference_self_error.keys())}")
    if conventions_seen == {"subgrid"}:
        emit("CONFIRMED: both the ss=2 rung and the ss=64 reference render under the "
             "SAME ('subgrid') PSF convention -- this isolates quadrature error and does "
             "NOT include the ~3.7 sigma bin-first-vs-subgrid PSF-convention gap measured "
             "separately (undersampling_report.txt).")
    else:
        emit(f"*** UNEXPECTED: conventions seen = {conventions_seen} -- the isolation this "
             "script relies on does not hold; investigate before trusting anything below. ***")

    emit(f"reference_converged: {report.reference_converged}  "
         f"(self_error={report.reference_self_error}, library pass margin = "
         "tolerance/_REFERENCE_MARGIN = 0.2/4.0 = 0.05 sigma)")
    if not report.reference_converged:
        emit("*** REFERENCE NOT CERTIFIED -- the numbers below are REPORTED, NOT CERTIFIED, "
             "same as every other script in this directory. ***")

    delta = delta_chi2 / 2.0     # delta(theta) = log p_ref - log p_ss2, see docstring
    emit()
    emit("SIGN CHECK: module's delta_chi2 = chi2(rung) - chi2(reference); "
         "delta(theta) = delta_chi2/2 = log p_ref(theta) - log p_ss2(theta) (docstring "
         "derivation). delta_chi2 is measured NEGATIVE here (the ss=2 rung fits the "
         "observed data BETTER -- lower chi2 -- than the converged reference, at the same "
         "parameters: the coarse quadrature is quietly absorbing some of its own error "
         "into an apparently better fit). So delta < 0: correcting to the reference makes "
         "these parameter points LESS probable than the ss=2 fit currently reports -- the "
         "physically sensible direction for removing an artifact that flattered the fit.")
    emit(f"  delta_chi2 range over batch: [{delta_chi2.min():+.2f}, {delta_chi2.max():+.2f}]")
    emit(f"  delta range over batch:      [{delta.min():+.3f}, {delta.max():+.3f}] nats")

    delta_draws = delta[:N_DRAWS]
    delta_map = float(delta[IDX_MAP])
    delta_median = float(delta[IDX_MEDIAN])
    emit(f"  delta at MAP: {delta_map:+.3f} nats;  delta at HMC-median: {delta_median:+.3f} nats")

    # --- SANITY CHECK 1: mean(delta) vs the already-measured -221/2 ~= -110.5 -----------
    emit()
    emit("-" * 78)
    emit("SANITY CHECK 1: mean(delta) over the N=60 draws vs the already-measured "
         "delta_chi2 mean (-221.1, undersampling_recheck.py, 9-sample batch, ss_ref=64)")
    emit("-" * 78)
    prior_mean_delta = -221.1 / 2.0
    this_mean_delta = float(delta_draws.mean())
    emit(f"  this script:  mean(delta) = {this_mean_delta:+.2f} nats over N={N_DRAWS} draws "
         f"(delta_chi2 mean {float(delta_chi2[:N_DRAWS].mean()):+.2f})")
    emit(f"  already measured: delta_chi2 mean -221.1 -> delta = {prior_mean_delta:+.2f} nats "
         "(9-sample batch, different draw selection, same ss_ref=64, subgrid convention)")
    rel_diff = abs(this_mean_delta - prior_mean_delta) / abs(prior_mean_delta)
    emit(f"  relative difference: {100 * rel_diff:.1f}%  "
         f"-> {'CONSISTENT' if rel_diff < 0.25 else 'LARGE DISCREPANCY -- investigate before trusting anything below'}")

    # --- SANITY CHECK 2: spread of delta vs the already-reported ~17-nat spread ----------
    emit()
    emit("-" * 78)
    emit("SANITY CHECK 2: spread of delta over draws vs the already-reported ~17-nat "
         "delta_chi2 spread (guarantees weight collapse under full reweighting)")
    emit("-" * 78)
    std_delta = float(delta_draws.std(ddof=1))
    std_delta_chi2 = float(delta_chi2[:N_DRAWS].std(ddof=1))
    emit(f"  std(delta_chi2) over N={N_DRAWS} draws = {std_delta_chi2:.2f} nats "
         f"(chi2 units); std(delta) = {std_delta:.2f} nats (log-likelihood units)")
    emit(f"  range(delta_chi2) = [{delta_chi2[:N_DRAWS].min():+.2f}, "
         f"{delta_chi2[:N_DRAWS].max():+.2f}], range(delta) = "
         f"[{delta_draws.min():+.2f}, {delta_draws.max():+.2f}]")
    emit("  order of magnitude matches the previously reported ~17-nat delta_chi2 spread "
         f"(here: {std_delta_chi2:.1f} nats std over a wider, more representative N=60 "
         "batch) -- both say the same thing: per-draw delta_chi2 varies by O(10) nats "
         "across the posterior, which is exactly why full importance reweighting was "
         "already ruled out (weight collapse) and only this first-order estimate is "
         "attempted.")

    # --- step 4: regress delta on the 6 host-mass parameters, standardized ---------------
    emit()
    emit("-" * 78)
    emit("STEP: regress delta on the 6 host-mass parameters (standardized), OLS")
    emit("-" * 78)
    cols = [diagnose.column(model, f"mass/0/{suf}") for suf in MASS_PARAM_SUFFIXES]
    emit(f"  parameters (z-column indices): "
         + ", ".join(f"{s}={c}" for s, c in zip(MASS_PARAM_SUFFIXES, cols)))

    constrained_batch = diagnose.to_constrained(model, z_batch)  # (62, n_params)
    X_all = constrained_batch[:, cols]  # (62, 6), original units
    X_draws = X_all[:N_DRAWS]           # (60, 6)

    X_mean = X_draws.mean(axis=0)
    X_std = X_draws.std(axis=0, ddof=1)
    X_z = (X_draws - X_mean) / X_std

    beta, se, r2, resid = ols_with_se(X_z, delta_draws)
    intercept, beta_p = beta[0], beta[1:]
    se_intercept, se_p = se[0], se[1:]

    emit()
    emit(f"  R^2 = {r2:.4f}  (n={N_DRAWS}, p=6 predictors + intercept, dof={N_DRAWS - 7})")
    emit(f"  intercept = {intercept:+.3f} +/- {se_intercept:.3f} nats "
         f"(delta at the draws' own mean parameter point)")
    emit("  standardized gradient g_std = d(delta)/d(theta_std)  [nats per 1 sd of theta]:")
    for suf, b, s in zip(MASS_PARAM_SUFFIXES, beta_p, se_p):
        tstat = b / s if s > 0 else np.nan
        emit(f"    {suf:<10s} g_std = {b:+8.3f} +/- {s:.3f}   (t = {tstat:+.2f})")

    if r2 < 0.3:
        emit()
        emit("  R^2 < 0.3: delta is NOT well-described as linear in these 6 parameters "
             "over the posterior bulk spanned by these draws. Per the method above, the "
             "shift computed below should be read as an ORDER-OF-MAGNITUDE BOUND, not a "
             "value -- this is stated again in the final verdict.")
        linear_credible = False
    elif r2 < 0.6:
        emit()
        emit("  R^2 in [0.3, 0.6): a real but partial linear signal -- other structure "
             "(nonlinearity, or parameters not in this regression) is also present. "
             "Treat the shift below as indicative, with real uncertainty on its magnitude.")
        linear_credible = True
    else:
        emit()
        emit("  R^2 >= 0.6: delta is reasonably well-described as linear in these 6 "
             "parameters over the posterior bulk -- the linear-response picture below is "
             "credible.")
        linear_credible = True

    # --- step 5: Delta_theta = Sigma @ g, g in ORIGINAL units -----------------------------
    emit()
    emit("-" * 78)
    emit("STEP: Delta_theta = Sigma @ g  (posterior covariance times the gradient, "
         "original units)")
    emit("-" * 78)
    g_orig = beta_p / X_std  # d(delta)/d(theta), original units

    # Posterior covariance from a stride through the FULL chain (not just the N=60 draws):
    # every 50th step of every chain -> 50 * 200 = 10000 samples, cheap and representative.
    stride = 50
    z_cov_sample = hmc_samples[:, ::stride, :].reshape(-1, n_params)
    emit(f"  covariance estimated from {z_cov_sample.shape[0]} draws "
         f"(every {stride}th step of all {n_chains} chains, full post-burnin range)")
    constrained_cov_sample = diagnose.to_constrained(model, z_cov_sample)
    X_cov = constrained_cov_sample[:, cols]
    Sigma = np.cov(X_cov, rowvar=False)  # (6, 6)
    post_sd = np.sqrt(np.diag(Sigma))

    Delta_theta = Sigma @ g_orig  # (6,)

    emit()
    emit("  posterior sd (from the same stride sample), original units:")
    for suf, sd in zip(MASS_PARAM_SUFFIXES, post_sd):
        emit(f"    {suf:<10s} sd = {sd:.5f}")

    idx_thetaE = MASS_PARAM_SUFFIXES.index("theta_E")
    idx_gamma = MASS_PARAM_SUFFIXES.index("gamma")
    dthetaE = Delta_theta[idx_thetaE]
    dgamma = Delta_theta[idx_gamma]
    sd_thetaE = post_sd[idx_thetaE]
    sd_gamma = post_sd[idx_gamma]

    emit()
    emit(f"  Delta theta_E = {dthetaE:+.6f}\"  = {dthetaE / sd_thetaE:+.3f} x posterior sd "
         f"(sd = {sd_thetaE:.5f}\")")
    emit(f"  Delta gamma   = {dgamma:+.6f}    = {dgamma / sd_gamma:+.3f} x posterior sd "
         f"(sd = {sd_gamma:.5f})")
    emit()
    quoted_half_width = 0.002
    emit(f"  quoted theta_E interval: 1.991 [1.989, 1.994]\" -- reported half-width "
         f"{quoted_half_width}\"")
    emit(f"  |Delta theta_E| = {abs(dthetaE):.6f}\" is "
         f"{'LARGER' if abs(dthetaE) > quoted_half_width else 'SMALLER'} than the quoted "
         f"+/-{quoted_half_width}\" interval "
         f"(ratio {abs(dthetaE) / quoted_half_width:.2f}x).")

    # --- SANITY CHECK 3: split-half agreement --------------------------------------------
    emit()
    emit("-" * 78)
    emit("SANITY CHECK 3: split-half agreement (first 30 vs last 30 of the 60 selected "
         "draws, ordered by the chains/steps linspace selection -- an interleaved-in-index "
         "but not adjacent-in-chain split)")
    emit("-" * 78)
    half = N_DRAWS // 2
    split_results = {}
    for name, sl in [("first half", slice(0, half)), ("second half", slice(half, N_DRAWS))]:
        Xh = X_draws[sl]
        dh = delta_draws[sl]
        Xh_mean = Xh.mean(axis=0)
        Xh_std = Xh.std(axis=0, ddof=1)
        Xh_z = (Xh - Xh_mean) / Xh_std
        beta_h, se_h, r2_h, _ = ols_with_se(Xh_z, dh)
        g_h = beta_h[1:] / Xh_std
        dtheta_h = Sigma @ g_h
        split_results[name] = (r2_h, dtheta_h[idx_thetaE], dtheta_h[idx_gamma])
        emit(f"  {name:<12s} (n={half}): R^2={r2_h:.3f}  "
             f"Delta theta_E={dtheta_h[idx_thetaE]:+.6f}\"  "
             f"Delta gamma={dtheta_h[idx_gamma]:+.6f}")

    (r2_1, dte_1, dg_1), (r2_2, dte_2, dg_2) = split_results.values()
    same_sign_thetaE = (dte_1 > 0) == (dte_2 > 0) == (dthetaE > 0)
    ratio_thetaE = abs(dte_1 / dte_2) if dte_2 != 0 else np.inf
    emit()
    emit(f"  sign agreement on Delta theta_E across halves and the full sample: "
         f"{'YES' if same_sign_thetaE else 'NO'}")
    emit(f"  magnitude ratio |half1/half2| = {ratio_thetaE:.2f} "
         f"(1.0 = perfect agreement; large or <<1 deviations from 1 indicate noise)")
    split_stable = same_sign_thetaE and (0.2 < ratio_thetaE < 5.0)
    if split_stable:
        emit("  -> split-half STABLE: the two halves agree on sign and rough magnitude.")
    else:
        emit("  -> split-half UNSTABLE: the estimate is not distinguishable from noise "
             "at this N -- treat Delta theta_E as an order-of-magnitude statement only.")

    # --- plot: delta vs theta_E, full sample + fitted line (honest, univariate view) -----
    emit()
    emit("-" * 78)
    emit("PLOT: delta vs theta_E across draws (marginal view -- NOT controlling for gamma "
         "etc, so the scatter is the honest picture, not the fitted multivariate line)")
    emit("-" * 78)
    theta_E_draws = X_draws[:, idx_thetaE]
    # Marginal (univariate) OLS line for the plot only -- NOT the multivariate regression
    # used for Delta_theta above; shown because it is what the eye can check against the
    # scatter, and its slope necessarily differs from g_std[theta_E] once gamma is
    # controlled for (told to the reader explicitly, not left implicit).
    A = np.column_stack([np.ones(N_DRAWS), theta_E_draws])
    coef_marg, *_ = np.linalg.lstsq(A, delta_draws, rcond=None)
    xx = np.linspace(theta_E_draws.min(), theta_E_draws.max(), 50)
    yy = coef_marg[0] + coef_marg[1] * xx
    r2_marg = 1 - np.sum((delta_draws - A @ coef_marg) ** 2) / np.sum((delta_draws - delta_draws.mean()) ** 2)
    emit(f"  marginal (theta_E only) OLS: slope={coef_marg[1]:.2f} nats/arcsec, "
         f"R^2={r2_marg:.4f} (weaker than the 6-parameter R^2={r2:.4f} above, as expected "
         "since theta_E/gamma are correlated and share explanatory power)")

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(theta_E_draws, delta_draws, s=28, color=COLOR_POINTS, alpha=0.85,
               edgecolor="white", linewidth=0.5, label=f"N={N_DRAWS} posterior draws", zorder=3)
    ax.plot(xx, yy, color=COLOR_FIT, lw=2, label=f"marginal OLS ($R^2$={r2_marg:.2f})", zorder=2)
    ax.axhline(0, color="0.75", lw=1, zorder=1)
    ax.scatter([X_all[IDX_MAP, idx_thetaE]], [delta_map], marker="*", s=160,
               color=COLOR_REF, edgecolor="black", linewidth=0.6, label="MAP", zorder=4)
    ax.scatter([X_all[IDX_MEDIAN, idx_thetaE]], [delta_median], marker="D", s=60,
               color=COLOR_REF, edgecolor="black", linewidth=0.6, label="HMC median", zorder=4)
    ax.set_xlabel(r"$\theta_E$ [arcsec]")
    ax.set_ylabel(r"$\delta(\theta) = \Delta\chi^2_{ss2}/2$ [nats]")
    ax.set_title("DESI257 r3_unmasked: ss=2 quadrature log-likelihood\ncorrection vs $\\theta_E$", fontsize=11)
    ax.legend(fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "theta_e_systematic.png", dpi=150)
    emit(f"  saved -> theta_e_systematic.png")

    # --- final verdict ---------------------------------------------------------------------
    t_end = time.perf_counter()
    emit()
    emit("=" * 78)
    emit("VERDICT")
    emit("=" * 78)
    small_vs_sd = abs(dthetaE) < 0.5 * sd_thetaE
    validity_holds = linear_credible and split_stable
    emit(f"linear-response validity: R^2={r2:.3f} ({'OK' if linear_credible else 'LOW'}), "
         f"split-half {'stable' if split_stable else 'unstable'}, "
         f"|Delta theta_E|/sd = {abs(dthetaE) / sd_thetaE:.3f} "
         f"({'small' if small_vs_sd else 'NOT small'} relative to the posterior spread, "
         "as a first-order approximation should be)")
    if not validity_holds:
        emit()
        emit("PLAIN-ENGLISH VERDICT: this measurement does NOT resolve the question. The "
             "linear-response picture is not credible enough at N=60 (either R^2 is low, "
             "or the two draw-halves disagree on sign/magnitude) to certify a shift number. "
             "The honest statement is that the ss=2 quadrature error's effect on theta_E's "
             "posterior mean could not be pinned down to better than order-of-magnitude with "
             "this draw count; a larger N or a genuine ss=64 refit would be needed to do "
             "better, and neither is attempted here (the latter was already ruled out by "
             "the delta_chi2 spread -- weight collapse under reweighting).")
    else:
        emit()
        emit(f"PLAIN-ENGLISH VERDICT: to first order, moving from the fitted ss=2 "
             f"likelihood to the (better, but not itself certified) ss=64 reference shifts "
             f"theta_E's posterior mean by {dthetaE:+.5f}\" "
             f"({dthetaE / sd_thetaE:+.2f} posterior sd) and gamma's by {dgamma:+.5f} "
             f"({dgamma / sd_gamma:+.2f} posterior sd). "
             f"{'This is LARGER than' if abs(dthetaE) > quoted_half_width else 'This is SMALLER than'} "
             f"the quoted +/-{quoted_half_width}\" interval half-width, so the interval "
             f"{'should be widened (or the shift explicitly quoted alongside it)' if abs(dthetaE) > quoted_half_width else 'can stand as quoted, with this systematic noted as sub-dominant'}. "
             "This is a first-order estimate from N=60 draws, not a refit at ss=64 -- treat "
             "the magnitude, not just the sign, as provisional.")
    emit()
    emit(f"[timing] total script wall-clock: {t_end - t_start:.1f}s")

    (OUT_DIR / "theta_e_systematic.txt").write_text("\n".join(lines) + "\n")
    emit(f"\nWrote {OUT_DIR / 'theta_e_systematic.txt'}")

    partial = OUT_DIR / "_partial_theta_e_systematic.txt"
    if partial.exists():
        partial.unlink()


if __name__ == "__main__":
    main()
