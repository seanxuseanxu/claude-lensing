# DESI257 analysis scripts

Three short scripts, each producing numbers quoted in `../RESULTS_v2.md` or `../RESULTS_v3.md`.
They exist so those numbers are reproducible rather than asserted. All three go through
`claude/pipeline/diagnose.py`'s `rebuild_model` / `to_constrained` and gigalens' own
`ProbModel.log_prob` — nothing here reimplements a bijector, a likelihood or a normalisation
(`CLAUDE.md`).

Run from the repo root, one GPU is enough:

```bash
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python
CUDA_VISIBLE_DEVICES=0 $PY claude/DESI257/analysis/<script>.py
```

| script | what it measures | where the numbers land |
|---|---|---|
| `mode_seeding.py` | `gamma` at the MAP point, at the SVI surrogate mean, and at post-burn-in draw 0 of all 50 chains of `r2_s2l2` | `RESULTS_v2.md` §3.4(a) — the retraction of "dominant mode" |
| `mode_depth.py` | per-mode median log posterior, log prior and `chi2_nu` of `r2_s2l2`, split at `gamma = 2.66` | `RESULTS_v2.md` §3.4(b) — mode B is the deeper one |
| `unmasked_reference.py` | `r2_s2l2`'s draws scored under the **unmasked** likelihood, and the offset between the masked and unmasked likelihoods | `RESULTS_v3.md` — the floor round 3's MAP had to clear |
| `mode_split.py` | reproduces §3.2's 17-chain dominant-mode selection (`frac_below > 0.98` at `gamma = 2.66`) | shared by the two below |
| `caustic_membership.py` | at the dominant-mode median: each source centroid inside/outside the tangential caustic, its signed distance, and `LensSolver`'s point-source image count | `RESULTS_v2.md` §5.3 |
| `caustic_flux_fraction.py` | the same over 300 draws, plus **the fraction of each component's own flux inside the caustic** | `RESULTS_v2.md` §5.3 — the measurement that withdraws the "wrong topology" hold-back |

`unmasked_reference.py` is the one to read before comparing any log posterior across the
mask change. The two likelihoods differ by 403 pixels and their log posteriors differ by a
median of **-14346.7** nats on identical parameters, so a number from a masked run and a
number from an unmasked run are not on the same scale and must never be compared directly.

| `sw_region_chisq.py` | `chi2_nu`, mean residual and peak flux over the 403 pixels rounds 1-2 masked, **against equal-radius discs on the other lensed images** | `RESULTS_v3.md` §5 — the fourth-image test |

`sw_region_chisq.py`'s controls are the point of it. A `chi2_nu` inside the previously-masked
disc means nothing on its own; it means something only against the same statistic on the N
blob, the E arc and the deflector core, at the same parameters and the same error map. The
first version of this script reported only the disc and the field average, and that
comparison made an ordinary number look anomalous.
