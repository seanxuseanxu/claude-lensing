# DESI238, round 5: the first fit under the GIGA-Lens scene API

**Date:** 2026-08-08. **Supersedes** `RESULTS_v4.md` for every number quoted here.

This is the first DESI238 result produced by the ported pipeline. The port changed two
things that make earlier checkpoints unreadable rather than merely stale — the flat-`z`
column order and the noise convention — so nothing here is a re-analysis of round 4's
samples. It is an independent re-fit, and that makes the agreement with round 4 evidence
rather than bookkeeping.

**Headline:** `theta_E = 1.478 +/- 0.003"`, with **all 36 parameters at R-hat <= 1.003**.

---

## 1. What was fitted

| | |
|---|---|
| Data | HST WFC3/IR F140W, 80x80 crop at 0.065"/pix, EXPTIME 1197.699 s |
| Model | EPL + external shear; 2 Sersic source components; bulge+disk deflector light |
| Free parameters | 36 |
| Pipeline | `run_fit.py --lens DESI238 --n-source 2 --n-lens-light 2` |
| Code | `claude/pipeline/` at commit `e6b7f57` plus the uncommitted `--sampler` change |
| Preflight | `verify.py --lens DESI238` — 18/18 checks pass |

The model and the conditioning are unchanged from round 4. The config was not retuned to
get this result; only the inference budget and the sampler changed.

---

## 2. The result

Headline run: `runs/mclmc_long` — MCLMC, 64 chains x 12,000 draws, 4,000 burn-in, one A100,
**467 s**.

Reduced chi-squared at the point estimate **0.968**.

### 2.1 Lens mass

| parameter | median | -1sig | +1sig | R-hat | bulk-ESS |
|---|---|---|---|---|---|
| `theta_E` | **1.4784** | 0.0025 | 0.0025 | 1.002 | 79,827 |
| `gamma` | 1.7996 | 0.0204 | 0.0206 | 1.002 | 79,343 |
| `e1` | 0.0220 | 0.0060 | 0.0059 | 1.002 | 78,728 |
| `e2` | 0.0208 | 0.0052 | 0.0052 | 1.002 | 78,732 |
| `center_x` | 0.0118 | 0.0029 | 0.0029 | 1.002 | 78,653 |
| `center_y` | -0.0031 | 0.0030 | 0.0030 | 1.002 | 80,011 |
| `shear/gamma1` | 0.1428 | 0.0043 | 0.0043 | 1.002 | 79,871 |
| `shear/gamma2` | -0.0133 | 0.0036 | 0.0036 | 1.002 | 78,606 |

### 2.2 Deflector light (bulge + disk)

| parameter | median | -1sig | +1sig | R-hat | bulk-ESS |
|---|---|---|---|---|---|
| `bulge/Ie` | 281.66 | 22.59 | 23.46 | 1.001 | 79,179 |
| `bulge/R_sersic` | 0.3021 | 0.0175 | 0.0191 | 1.001 | 78,375 |
| `bulge/n_sersic` | 3.0522 | 0.1013 | 0.1050 | 1.001 | 79,249 |
| `bulge/e1` | -0.0052 | 0.0035 | 0.0037 | 1.001 | 76,138 |
| `bulge/e2` | -0.0572 | 0.0064 | 0.0060 | 1.001 | 71,833 |
| `bulge/center_x` | 0.0109 | 0.0007 | 0.0007 | 1.002 | 79,122 |
| `bulge/center_y` | 0.0083 | 0.0006 | 0.0006 | 1.002 | 77,698 |
| `disk/Ie` | 9.9151 | 0.9108 | 0.9458 | 1.001 | 76,343 |
| `disk/R_sersic` | 2.3923 | 0.0951 | 0.1066 | 1.001 | 75,927 |
| `disk/n_sersic` | 2.0678 | 0.1324 | 0.1431 | 1.002 | 60,812 |
| `disk/e1` | -0.0822 | 0.0053 | 0.0049 | 1.001 | 79,890 |
| `disk/e2` | 0.1382 | 0.0074 | 0.0082 | 1.001 | 79,633 |
| `disk/center_x` | 0.0169 | 0.0041 | 0.0042 | 1.003 | 78,434 |
| `disk/center_y` | 0.0025 | 0.0038 | 0.0038 | 1.002 | 77,960 |

`disk/n_sersic` sits at 78.4% of its Uniform(0.5, 2.5) support — not pinned by the 2%
criterion, but close enough to the upper bound that the bound should be revisited if the
disk shape is ever the quantity of interest. It is not, here.

### 2.3 Source

| parameter | median | -1sig | +1sig | R-hat | bulk-ESS |
|---|---|---|---|---|---|
| `src0/Ie` | 18.058 | 0.617 | 0.612 | 1.002 | 78,082 |
| `src0/R_sersic` | 0.2139 | 0.0065 | 0.0067 | 1.002 | 78,780 |
| `src0/n_sersic` | 1.1013 | 0.0360 | 0.0376 | 1.002 | 78,026 |
| `src0/e1` | -0.0381 | 0.0084 | 0.0084 | 1.002 | 79,158 |
| `src0/e2` | 0.1821 | 0.0079 | 0.0079 | 1.002 | 77,765 |
| `src0/center_x` | 0.1460 | 0.0033 | 0.0033 | 1.002 | 79,376 |
| `src0/center_y` | 0.0081 | 0.0027 | 0.0028 | 1.002 | 79,986 |
| `src1/Ie` | 90.244 | 5.004 | 5.237 | 1.001 | 75,684 |
| `src1/R_sersic` | 0.0687 | 0.0027 | 0.0027 | 1.002 | 77,901 |
| `src1/n_sersic` | **>= 0.37 (bound)** | 0.0485 | 0.0699 | 1.003 | 18,692 |
| `src1/e1` | 0.3130 | 0.0222 | 0.0224 | 1.002 | 78,130 |
| `src1/e2` | -0.0834 | 0.0262 | 0.0253 | 1.002 | 79,165 |
| `src1/center_x` | 0.4137 | 0.0100 | 0.0100 | 1.002 | 78,469 |
| `src1/center_y` | 0.1489 | 0.0055 | 0.0056 | 1.001 | 79,002 |

**All 36 parameters R-hat <= 1.003; min bulk-ESS 18,692, min tail-ESS 14,586.**

No label switching: the two source components separate at 20.2 sigma in `R_sersic`, with
0 crossings in 0/64 chains. R-hat is therefore reported on the raw chains.

---

## 3. Agreement with round 4

| | round 4 (pre-port) | round 5 (scene API) |
|---|---|---|
| `theta_E` | 1.480 +/- 0.003" | 1.478 +/- 0.003" |
| worst R-hat over 36 params | ~1.1 (`< 1.1` claimed) | **1.003** |
| chi-squared_v | 0.989 | 0.968 |

The two theta_E values agree to 0.002", well inside either error bar, having been produced
by different likelihood normalisation, a different parameter ordering, a different sampler
and different code. That is the strongest available evidence that the port did not change
the physics. It is worth saying plainly that this was the outcome at risk: the noise
convention moved from a parameter-dependent variance to one derived once from the observed
image, and that is a real change to the likelihood, not a refactor.

Round 4's R-hat and this one are **not** numerically comparable — convergence now comes
from ArviZ rank-normalized split R-hat via `diagnose_convergence`, where the pre-port
pipeline used a classic PSRF that over-reports on these marginals. The comparison in the
table is of pass/fail against a stated threshold, not of the statistic.

---

## 4. What actually produced the convergence

Round 4 reached worst R-hat 1.338 at 36 parameters and stalled there. The fix was **not**
more sampling. It was remedy 2 from the guide's section 4.3 — a better variational
surrogate.

`runs/prod_svi`, all four GPUs, MAP 1,500 steps at 4,096 particles then SVI 12,000 steps at
n_vi 2,048 with peak LR 1e-3 (21 min total):

| | round 4's `twosource` (diverged) | round 4 typical healthy | **round 5 `prod_svi`** |
|---|---|---|---|
| SVI covariance condition | 8.8e+11 | 1e6 – 6e8 | **2.3e+06** |
| ELBO tail sd (last 300) | 7,384 | ~10 | **0.1** |
| MAP chi-squared_v | — | — | 0.987 |

The ELBO tail sd of 0.1 at 36 parameters is two orders of magnitude better than a healthy
22-parameter run. Every sampler below was preconditioned on this one covariance, which is
why the sampler comparison is controlled: the surrogate is held fixed.

**The general lesson:** at 36 parameters the binding constraint was the preconditioner, not
the chain length. Spending 21 minutes on MAP+SVI bought a result that no amount of extra
HMC on a mediocre surrogate had reached.

---

## 5. Sampler comparison

All four runs start from the identical `prod_svi` MAP+SVI checkpoint. Trials are 32 chains
x 3,000 draws, 1,000 burn-in, one A100 each, run concurrently on the four GPUs.

| run | sampler | chains x draws | wall | worst R-hat | min bulk-ESS | target met |
|---|---|---|---|---|---|---|
| `trial_hmc` | HMC | 32 x 3,000 | 704 s | 1.003 | 10,928 | yes |
| `trial_mams` | MAMS | 32 x 3,000 | 750 s | 1.004 | 7,000 | yes |
| `trial_mams_s1` | MAMS (seed 1) | 32 x 3,000 | 712 s | 1.011 | — | no |
| `trial_mclmc` | MCLMC | 32 x 3,000 | **81 s** | 1.012 | — | no |
| `mclmc_long` | MCLMC | 64 x 12,000 | 467 s | **1.003** | **18,692** | yes |

MCLMC is roughly **9x faster per draw** than HMC here. At matched wall-clock it is not
close: `mclmc_long` delivers 8x the samples of `trial_hmc` in two thirds of the time, and
reaches the same worst R-hat with 1.7x the minimum ESS. MCLMC is the right default for this
problem.

The mass parameters agree across all four samplers to <= 0.00004" on `theta_E` and
proportionally on everything else — far inside any single run's posterior width. Independent
kernels landing on the same posterior is the check that the sampling is not the thing
setting the answer.

---

## 6. Why the trial budget is not enough — read this before quoting a short run

Five HMC runs at the trial budget (32 chains x 3,000 draws), identical in every respect except
the seed:

| seed | worst R-hat | passes |
|---|---|---|
| 0 (`trial_hmc`) | 1.003 | yes |
| 1 (`repl_seed1`) | 1.019 | no |
| 2 (`repl_seed2`) | 1.009 | yes |
| 3 (`repl_seed3`) | 1.093 | no |
| 4 (`repl_seed4`) | 1.019 | no |

**Only 2 of 5 seeds clear the threshold at that budget**, with min bulk-ESS ranging 2,735 to
10,928. That scatter is the signature of a budget too small to have converged — 3,000 draws
does not resolve these marginals, so which side of 1.01 a run lands on is partly luck.

The remedy is draws, and `mclmc_long` is the headline for that reason: 4x the draws and 2x the
chains give worst R-hat 1.003 at min ESS 18,692, a margin wide enough that seed-scale scatter
is no longer in play. Seed-to-seed variation at the short budget is useful as a *diagnostic
that the short budget is inadequate*, which is how it is used here — not as a claim needing
its own multi-seed protocol.

What is and is not measured, stated plainly: all 36 parameters converge to R-hat <= 1.003 in
the headline run, and the mass parameters agree across four independent samplers. A repeat of
the headline run under a second seed was not performed. If a lower R-hat is wanted, the move
is a longer run, not a seed survey.

Anyone re-running this should use the `mclmc_long` budget or larger, and should not treat a
single 3,000-draw run as converged.

---

## 7. `src1` is unresolved — report its index as a bound

`src1` has `R_sersic = 0.0687 +/- 0.0027"`, which is **smaller than the 0.065" pixel**. Its
`n_sersic` sits at 0.369, which is 1.2% of the way into its Uniform(0.3, 6.0) support —
pinned at the lower bound by `diagnose.py`'s automated 2% check, in every sampler
independently.

That floor of 0.3 is not arbitrary and **should not simply be widened**. It is the lowest
index at which the installed rational-(3,2) Sersic `b_n` approximation is verified
(5.5e-05 over n in [0.3, 8]; `claude/pipeline/test_sersic_bn.py` validates it against the
defining relation). Going below 0.3 means extending that test first, otherwise a fitted
index below the floor reports a profile-accuracy artefact rather than a morphology.

What `src1` is, on the evidence:

- **Unresolved.** Sub-pixel `R_sersic`, index driven to the floor — the fit is asking for a
  flatter core than a Sersic in the verified range can give. That is the signature of a
  point-like clump, not a resolved galaxy.
- **Real but minor.** It carries 22.3 e/s against `src0`'s 178.2 e/s — 11.1% of source flux,
  1.8% of all light in the scene. It sits 0.32" from `src0` in the source plane, inside
  `src0`'s own `R_sersic`: a knot riding on the host.
- **Not driving theta_E.** Pearson r(`src1/n_sersic`, `theta_E`) = **0.018**, and |r| < 0.02
  against all six mass parameters. The pinning does not propagate into the headline number.

**Therefore:** quote `src1/n_sersic` as a bound (`<= 0.37`, prior-limited), describe `src1`
as an unresolved clump, and do not quote its index as a measurement. Its position, flux and
size are measured and well converged (R-hat <= 1.002).

There is a genuine posterior correlation between `src1/center_x`, `src1/center_y` and
`gamma` (r = 0.957, 0.869) and moderately with `theta_E` (r = -0.428, -0.594). Both sides are
well converged, so this is a property of the posterior, not a sampling failure — but it
means the source-position prior is doing some work on the mass slope, and it is worth a
sensitivity test that has not been run.

---

## 8. Residuals and remaining structure

Two features survive in the residual map (`runs/trial_hmc/residuals.png`), neither newly
introduced by this round:

1. **A tight alternating-sign striped pattern at the deflector core**, ~10x10 px, peaking
   near +/-5 sigma. It is confined to the PSF core and does not extend, which reads as a
   PSF/profile mismatch at the peak rather than a missing extended component. This is *not*
   the coherent radial S-curve that motivated the bulge+disk split in round 4 — that one is
   gone. Root cause not established.
2. **The N arc is underfit by ~16%** (clump flux model/data: SW arc 1.03, E blob 0.99,
   W clump 1.02, N arc 0.84). The other three are within 3%. This shortfall was already
   present in round 4 and is unchanged. `src1`'s images land near the W clump and E blob,
   both near unity, so there is no evidence `src1` is cannibalising N-arc flux.

Both are open items. Neither is large enough to move `theta_E` at the quoted precision, but
neither has been chased to a root cause, and the N-arc deficit in particular is the kind of
coherent, localised shortfall that the guide's section 5 says should be treated as a prompt
for a missing component rather than as noise.

---

## 9. Caveats and things not verified

- **The four-image interpretation is an input, not a result.** theta_E = 1.478" is
  conditioned on all four clumps being counter-images of one source, which is the repository
  owner's decision recorded in `config.py`. The falsification run against the inner three
  clumps was not repeated this round.
- **chi-squared_v = 0.968 is not evidence the noise model is right.** The background RMS
  carries a +/-10% spread across blank patches, which is +/-20% on chi-squared. Not
  propagated into the quoted error.
- **The quoted error on theta_E is statistical only.** No model systematic is attached; the
  0.036" systematic asserted in an earlier round was retracted at commit `8b080ea` and has
  not been replaced.
- **The headline run was not repeated under a second seed** (section 6). Its margin is wide;
  if a tighter R-hat is wanted the move is a longer run.
- **Sensitivity of `gamma` to the source-position prior: not measured** (section 7).
- **A tooling bug was reported but not independently verified by me.** The diagnostics pass
  reports that `gigalens.jax.analysis.diagnose_convergence` mutates the array returned by
  `diagnose.to_constrained` in place, corrupting the `theta_E` column, and that this makes
  `dump_params.py`'s printed *mean/sd* for `mclmc_long` (1.4881 +/- 0.0100) inconsistent
  with its own percentiles (implying 0.0025). `diagnose.py`'s percentile-based table is
  unaffected and is what section 2 quotes. **Use median/percentile values, not
  `dump_params.py`'s mean/sd, for this run.** This sits in the read-only `nanolens` package
  and is an upstream report, not something to patch from here. It has not been reproduced
  by a second person and should be confirmed before being filed.

---

## 10. Pipeline change made this round

`run_fit.py` gained `--sampler {hmc,mams,mclmc}`, default `hmc`, so every existing
invocation is unaffected. `metadata.json` records `"sampler"`. The checkpoint file stays
`hmc.npz` regardless of sampler, because the analysis tools key on that name.

The one trap worth recording: **HMC and MCLMC/MAMS return different axis layouts.** HMC
returns `(n_steps, n_devices, n_chains_per_device, n_params)`; MAMS and MCLMC already return
`(n_chains, num_results, n_params)`, chains first. This was determined empirically with tiny
runs, not taken from the docs. Both normalizers *validate* the layout against the shape it
must have and raise otherwise, rather than reshaping hopefully — a silent chain/step swap
produces a plausible-looking but wrong split-R-hat, which is the failure mode that would
convince you an unconverged run had converged.

---

## 11. Reproducing this

```bash
export PYTHONPATH=claude/pipeline
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python

# Preflight. 18/18 expected.
$PY claude/pipeline/verify.py --lens DESI238

# Surrogate: all 4 GPUs, ~21 min. This stage is what buys the convergence.
$PY claude/pipeline/run_fit.py --lens DESI238 --label prod_svi \
    --n-source 2 --n-lens-light 2 --stages map,svi \
    --n-samples 4096 --map-steps 1500 --n-vi 2048 --svi-steps 12000 --svi-peak-lr 1e-3

# Headline sample: one GPU, ~8 min.
cp claude/DESI238/runs/prod_svi/{map,svi}.npz claude/DESI238/runs/mclmc_long/
CUDA_VISIBLE_DEVICES=0 $PY claude/pipeline/run_fit.py --lens DESI238 --label mclmc_long \
    --n-source 2 --n-lens-light 2 --stages hmc --sampler mclmc \
    --n-hmc 64 --burnin 4000 --num-results 12000

$PY claude/pipeline/diagnose.py claude/DESI238/runs/mclmc_long
```

Check, in this order: SVI condition number is O(1e6) and ELBO tail sd is O(1); then the
R-hat table; then the pinning report. An SVI that fails the first check makes the rest
meaningless.
