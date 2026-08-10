# MAP step-count sweep (unmasked)

Not a fit. Three MAP-only runs of the round-3 model (`--stages map`, `--n-source 2
--n-lens-light 2 --n-samples 4096`, seed 0, two A100s), differing only in `--map-steps`,
made to choose round 3's step count by measurement instead of inheriting round 2's 1500.
`RESULTS_v3.md` section 3 holds the table and the reasoning; `sweep.log` is the raw stdout.

| file | `--map-steps` | wall | `best_lp` | final chi-squared | `chisq_hist` slope, last 500 steps |
|---|---|---|---|---|---|
| `map_1500.npz` | 1500 | 501.9 s | 16510.3 | 1.639 | -1.8e-4 /step |
| `map_5000.npz` | 5000 | 1640.4 s | 18035.3 | 1.163 | **-2.8e-6 /step** |
| *(no file)* | 20000 | ~110 min | **lost** | — | — |

**The 20,000-step point ran to completion and was then lost.** All three points wrote into a
session scratch directory under `/tmp`; the first two were copied here, and the node cleared
that directory before the third's `map.npz` or its stdout was ever read. So its `best_lp` was
never recorded and the row above is blank on purpose rather than pending. It was deliberately
not re-run: it was a confirmation of the plateau, not a gate on round 3, and the fit uses
`map_5000.npz` regardless. `sweep.log`'s third block is the command and the model banner with
no result after them — that is what the loss looks like, kept rather than trimmed.

Each is a self-contained optax schedule whose `transition_steps` is its own step count, so
**a longer run is not a continuation of a shorter one** and the three `best_lp` values are
not a single descent curve. The plateau judgement is made on one run's own `chisq_hist`
slope, not on the gap between two of them.

`map_5000.npz` is the checkpoint round 3 actually uses: it was copied to
`../r3_unmasked/map.npz` and `run_fit.py --stages svi` loaded it from there
(`run_fit.py:542` takes the MAP start from disk when `map` is not among the stages), so the
fit was not re-optimised and this file is bit-for-bit the start of the round-3 posterior.

The first point's stdout ends in a `FileNotFoundError` on a missing `svi.npz`. That is not a
failed run -- `map.npz` was already written -- it is `KNOWN_ISSUES.md` item 6, found during
this sweep and fixed while it ran: `--stages map` used to fall into the SVI-loading branch.
The later points ran under the fixed code.
