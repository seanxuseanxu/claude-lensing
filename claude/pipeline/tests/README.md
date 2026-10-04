# Pipeline unit tests

Standalone scripts, not pytest — pytest is not installed in `gigajax7.0`. Each is
self-checking: it prints PASS/FAIL per case and exits non-zero on any failure. Run from
the repo root with `PYTHONPATH=claude/pipeline` so each can `import` its sibling modules
by bare name.

| Test | Pins |
|---|---|
| `test_sersic_bn.py` | gigalens' Sersic `b_n` against the defining relation `gamma(2n, b_n) = Gamma(2n)/2` (via `scipy.special.gammaincinv`), not lenstronomy — see its docstring for why lenstronomy is not the reference here. |
| `test_mask.py` | `data.build_mask`: shape, exact disc exclusion, `mask_circles=None` -> `None`, off-image/all-masking raise. |
| `test_corner_check.py` | `verify.corner_report`'s four-corner companion check, especially the upper-left corner a lower-left-only check would miss. |
| `test_masked_chisq.py` | `diagnose.masked_chisq` / `diagnose.residual_figure` against `ProbModel`'s own `red_chi2` on a synthetic image with masked outliers — moved out of `verify.py` (was check 7) since it is a fact about pipeline code, not a per-lens preflight. |
| `test_make_data.py` | `make_cutout.py` (cut, centroid, provenance round-trip) and `make_psf.py` — *my* code around photutils/astropy, not those libraries themselves. |
| `test_diagnose_and_metadata.py` | `diagnose.map_hmc_verdict` (MAP-vs-HMC pass/fail, incl. the DESI257 r2_s2l2 deficit), `diagnose.boundary_pinning_check`'s TruncatedNormal coverage, `run_fit.merge_prior_metadata` (KNOWN_ISSUES.md item 1: a `--stages hmc` rerun must not overwrite prior stages' metadata with argparse defaults), and `diagnose.rebuild_model`'s `supersample` handling (KNOWN_ISSUES.md item 8: a run recorded at ss=2 must rebuild at 2 even when DESI257's config says 4 today, and a run predating the key falls back to config). The `supersample` tests, unlike the rest of this file, load DESI257's real data files and build a real model. |
| `test_foreground_roundtrip.py` | round 3b's foreground-interloper model (`priors.build_model(..., foreground=...)`, DESI257's `cfg.foreground_prior`): 36 -> 46 free params, `diagnose.rebuild_model` reconstructs identical `z_param_names` from a hand-written `metadata.json`, and `diagnose.column(..., "mass/0/theta_E", kind="mass")` still resolves to the host EPL rather than raising on the SIS's own `theta_E`. Also checks a run with no `"foreground"` key in metadata still rebuilds the plain 36-param model. |
| `test_ess_floor_and_plots.py` | `diagnose.ess_floor_report` (notably: it must fail 4 chains × 32 iid draws, whose R̂ is a clean 1.025 — a floor that only fired when R̂ also failed would add nothing), `ess_vs_draws_plot`'s slope verdict in both directions on synthetic iid and stuck chains, `rank_plot`'s chain cut keeping the one offset chain out of twelve, and `hmc_and_constrained`'s shape rejection and `need_con=False` path. Stub model throughout — no bijector, no FITS, no GPU. |

```bash
JAX_PLATFORMS=cpu PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_sersic_bn.py
PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_mask.py
PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_corner_check.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_masked_chisq.py --lens DESI238
PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_make_data.py
PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_diagnose_and_metadata.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_foreground_roundtrip.py
JAX_PLATFORMS=cpu PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_ess_floor_and_plots.py
```
