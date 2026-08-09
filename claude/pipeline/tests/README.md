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

```bash
JAX_PLATFORMS=cpu PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_sersic_bn.py
PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_mask.py
PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_corner_check.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_masked_chisq.py --lens DESI238
PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_make_data.py
```
