# DESI234 analysis scripts

Short scripts, each producing numbers quoted in `../RESULTS_v3.md`. They exist so those
numbers are reproducible rather than asserted. All of them go through
`claude/pipeline/diagnose.py`'s `rebuild_model` / `to_constrained` and gigalens' own
`ProbModel.log_prob` — nothing here reimplements a bijector, a likelihood or a normalisation
(`CLAUDE.md`), and nothing here writes outside the repo.

Run from the repo root, one GPU each; the five are independent and can run concurrently:

```bash
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python
CUDA_VISIBLE_DEVICES=0 $PY claude/DESI234/analysis/<script>.py
```

| script | what it measures | where the numbers land |
|---|---|---|
| `undersampling_check.py` | `diagnose_undersampling` ladder (1,2,4,8) at `ss_ref=16` on a 9-sample batch, plus **the decisive test**: Pearson r between the ss=2 predicted `delta/sigma` map and the observed residual, in a 10x10 px box at the deflector core | `RESULTS_v3.md` §1.1 — the checkerboard is not a rendering artifact (r = −0.109) |
| `undersampling_certify.py` | the same batch at `ss_ref=32`, ladder (1,2,4), plus per-sample `reference_self_error`, PSF-convention attribution, and ss=4-vs-ss=2 wall-clock | `RESULTS_v3.md` §1.1, §1.1n — the certified ladder behind `supersample = 4` |
| `map_deficit.py` | `diagnose.map_convergence_check` on `r1_s1l1` and `r2_s2l2`: recomputed `best_lp`, sampler log-posterior max/median, deficit in nats, `chisq_hist` tail slopes; and the SVI covariance condition numbers | `RESULTS_v3.md` §2, §2.1 |
| `arc_residual_anatomy.py` | first-order test: normalised residual projected on the model's own spatial gradient over 118 arc pixels — implied shift and fraction of arc `chi2` explained | `RESULTS_v3.md` §1.2 — the arc is not displaced |
| `arc_width_test.py` | second-order follow-up on the *same* 118 pixels: incremental R^2 of the second-derivative terms against a random-smooth-basis null | `RESULTS_v3.md` §1.2 — whether the arc is mis-shaped as opposed to clumpy |
| `foreground_calibration.py` | the interloper's centroid through the running code's WCS, its separation and PA, and `Ie_median` calibrated by prior-draw render against the object's own 0.82 e/s peak | `RESULTS_v3.md` §4(d) — the `ForegroundPrior` fields |
| `r3_diagnostics.py` | both round-3 runs end to end: `map_convergence_check`, R-hat/ESS on all params, `boundary_pinning_check`, `masked_chisq`, `theta_E`, the foreground component's nested-null distances, and the `r3_fg` SVI NaN anatomy | `RESULTS_v3.md` §5.1–5.3 |
| `mode_structure.py` | **frozen chains vs. multiple modes**: per-chain median and within-chain sd (ratio R), the interloper across chains, and k-means chain clustering with R-hat recomputed *within* each cluster | `RESULTS_v3.md` §5.4 — `r3_unmasked` frozen, `r3_fg` under-mixed along a ridge |

Four of these are worth reading before trusting a number rather than after.

**`mode_structure.py`'s ratio R is a reporting convention, not a test.** The frozen/mode-like
call at R > 10 is a label on a printed table, not a hypothesis test; the per-chain medians and
within-chain sds are printed in full precisely so the separation can be read directly. The
within-cluster R-hat in part C is the part that actually decides it — a partition under which
R-hat drops toward 1 means modes, and one under which bulk ESS stays equal to the cluster's
chain count means frozen. Its k-means is a plain 50-row Lloyd's with 10 restarts, inline
because a dependency was not worth it at that size.

**`undersampling_certify.py` is the one to read before quoting any rung.** A rung sitting one
doubling below an unconverged reference is not a measurement of that rung — it re-measures the
reference's own noise under a different label. The first ladder's ss=8 value (8.943 sigma)
equalled its reference's subgrid `reference_self_error` (8.9429) to four figures for exactly
that reason. Check `reference_converged`, and check how far below the reference the rung you
care about sits.

**`arc_width_test.py` is only meaningful with its null.** Five free parameters on 118
spatially correlated pixels will always explain something; the measured incremental R^2 means
nothing except against the distribution of what three arbitrary smooth regressors buy on the
same pixels.

**`map_deficit.py` reads `map.npz`, not `metadata.json`.** `r2_s2l2/metadata.json` reports
`map_steps: 350` (the `KNOWN_ISSUES.md` item-1 argparse-default corruption); the checkpoint's
own `chisq_hist` is 1500 entries long, which is the true value.
