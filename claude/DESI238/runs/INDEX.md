# DESI238 run index

One line per run, so you do not have to open 21 `metadata.json` files to find out which
one matters. **The headline run is `mclmc_long`.** Everything else is history, a trial, or a
test fixture.

Each run directory holds `metadata.json` (full settings, including the git commit it was
produced at), plus some subset of `params.csv`, `corner.png`, `residuals.png`,
`svi_loss.png`, and the `*.npz` checkpoints. Regenerate anything missing by re-running from
the recorded metadata.

**Checkpoint readability.** The scene-API port changed the flat-`z` column order and the
noise convention, so `diagnose.rebuild_model` **refuses every pre-port checkpoint** (rounds
1–4, the first three commits below). Those runs' numbers stand as recorded in the RESULTS
files; their `.npz` files cannot be re-analysed. Most `.npz` are gitignored anyway — six are
kept deliberately as `verify.py` test fixtures, marked below.

---

## Round 5 — scene API, commit `e6b7f57` — current

Checkpoints readable. See [`../RESULTS_v5.md`](../RESULTS_v5.md).

| run | purpose | result | quote it? |
|---|---|---|---|
| **`mclmc_long`** | **Headline.** MCLMC, 64 chains × 12,000 draws, 467 s | `theta_E = 1.478 ± 0.003"`, all 36 params R̂ ≤ 1.003, min ESS 18,692 | **yes — this one** |
| `prod_svi` | The surrogate all round-5 sampling starts from. MAP+SVI only, 4 GPUs, 12,000 SVI steps, 1,244 s | no posterior of its own; `map.npz`+`svi.npz` are copied into the sampling run | n/a — input, not a result |
| `trial_hmc` | Sampler comparison, HMC, 32 × 3,000 | worst R̂ 1.003, min ESS 10,928 | no — trial budget |
| `trial_mams` | Sampler comparison, MAMS, 32 × 3,000 | worst R̂ 1.004, min ESS 7,000 | no — trial budget |
| `trial_mams_s1` | MAMS at seed 1 | worst R̂ 1.011 — fails | no |
| `trial_mclmc` | MCLMC at trial budget, 81 s | worst R̂ 1.012 — fails, but 9× faster per draw than HMC; this is what motivated `mclmc_long` | no |
| `repl_seed1`, `repl_seed2`, `repl_seed3`, `repl_seed4` | Seed replication at the trial budget, HMC 32 × 3,000, seeds 1–4 | worst R̂ 1.019 / 1.009 / 1.093 / 1.019 respectively — **only 2 of 5 seeds clear 1.01** | no — these exist to show the trial budget is inadequate |
| `smoke2` | Pipeline smoke test, 22 params, 8 chains × 8 draws | none | no |
| `smoke_2src` | Pipeline smoke test, 36 params, 8 chains × 20 draws | none | no |

The four `repl_seed*` runs plus `trial_hmc` are the evidence for §6 of `RESULTS_v5.md`: at
3,000 draws, which side of R̂ = 1.01 a run lands on is partly luck. Do not cite a single
short run as converged.

---

## Round 4 — commit `8d6c7b1` — superseded

Accurate Sérsic `b_n`, source `n_sersic` floor lowered to 0.3. See
[`../RESULTS_v4.md`](../RESULTS_v4.md).

| run | model | result |
|---|---|---|
| `bn_rational_long` | 36 params, 50 chains × 3,000 | `theta_E = 1.4795 ± 0.0026"`, χ²ᵥ 0.988, worst R̂ 1.099, ESS 66,778 — first round with all 36 params under R̂ 1.1 |

Round 5 agrees with this to 0.002" from an independent re-fit, which is what validates the
port.

---

## Round 3 — commit `07631a6` — superseded

SVI divergence fixed; bulge+disk deflector introduced. See
[`../RESULTS_v3.md`](../RESULTS_v3.md).

| run | model | result |
|---|---|---|
| `bulgedisk_long` | 36 params, 4× draws | `theta_E = 1.480 ± 0.003"`, χ²ᵥ 0.989, worst R̂ 1.338, ESS 94,186 — round 3's quotable value |
| `bulgedisk` | 36 params, short | χ²ᵥ 0.991, worst R̂ 3.242 — short-budget version of the above |
| `twosource_v2` | 29 params, 2 source + 1 lens light | χ²ᵥ 1.277, worst R̂ 1.544 — the re-run of round 2's broken fit with SVI fixed. `svi.npz` is a **verify.py fixture** |

---

## Round 2 — commit `1df3d88` — superseded, nothing quotable

Ablation of two changes. Every run here failed its convergence diagnostic. See
[`../RESULTS_v2.md`](../RESULTS_v2.md).

| run | change | result |
|---|---|---|
| `nsersic10` | lens-light `n_sersic` ceiling 6 → 10, 1 source, 22 params | χ²ᵥ 1.760, worst R̂ 1.306. `svi.npz` is a **verify.py fixture** |
| `twosource` | above plus a second Sérsic source, 29 params | χ²ᵥ 1.302 (best yet) but worst R̂ 7.169, 29/29 params above 1.01 — the SVI stage had diverged. `svi.npz` is a **verify.py fixture**, and the one the guard must fire on |

---

## Round 1 — commit `1df3d88` — superseded, rejected by the data

22-parameter model, 1 source + 1 lens-light component. Rejected at ΔBIC ≈ −5294. See
[`../RESULTS.md`](../RESULTS.md).

| run | purpose | result |
|---|---|---|
| `seed0` | Four-image baseline | `theta_E = 1.516 ± 0.002"`, χ²ᵥ 1.837, worst R̂ 1.877. `svi.npz` is a **verify.py fixture** |
| `seed1` | Multimodality check, same priors at seed 1 | agrees with `seed0`. `svi.npz` is a **verify.py fixture** |
| `inner3` | Falsification test — inner three clumps only, `theta_E` prior recentred on 1.3" | did not falsify the four-image reading. `svi.npz` is a **verify.py fixture** |

---

## The six kept `.npz` files

`inner3`, `nsersic10`, `seed0`, `seed1`, `twosource`, `twosource_v2` each retain `svi.npz`
against the `*.npz` gitignore rule. They are 96 KB of fixed test fixture for `verify.py`
check 6, which replays their stored SVI covariances to assert the conditioning guard still
fires on the known-diverged run (`twosource`) and passes on the known-good ones. They are
keyed by `LensConfig.svi_guard_reference`. Do not delete them, and do not read them as
results — the statistic does not depend on the parameterisation, which is why they survive
the port when the rest of the pre-port checkpoints do not.
