# DESI234, round 2: bulge+disk light, two source components

> **This round's `theta_E = 1.539 +0.003/-0.002"` is the system's current number**, with one
> standing caveat: the interloper is masked here, and masking removes an object's *light*
> from the likelihood while leaving its *deflection* unmodelled. That systematic is real and
> still unquantified.
>
> **The earlier version of this banner is RETRACTED (2026-08-10).** It said this interval was
> superseded because round 3 had found the interloper carries 0.370" of SIS mass at 12.2 sd
> from zero, shifting the main `theta_E` by 0.066". Both numbers come from `r3_fg`, which was
> then rejected as unphysical — its deflector is a 7:1 needle (`q = 0.143`) with
> `gamma = 2.951`. A sound statistic computed on a rejected model measures nothing, and the
> 0.066" is the gap between a sane mass model and a broken one, not the interloper's effect.
>
> Round 3's other fit, `r3_unmasked`, has a sane mass model and lands at `theta_E = 1.5386`,
> reproducing this round to 0.0007" despite frozen chains — support for this result, not a
> replacement. Round 4 will refit the foreground model with the deflector's ellipticity
> constrained, and only then can the interloper's mass be quantified. `RESULTS_v3.md` §7 has
> the systematic budget.

**Date:** 2026-08-08. First converged fit of this system. Round 1 (`r1_s1l1`) is documented
below for comparison only — it did not converge and none of its parameter values should be
quoted as measurements.

**Headline:** `theta_E = 1.539 -0.002/+0.003"`, all 36 parameters at R-hat <= 1.01 (worst
1.008). Reduced chi-squared 1.300 at the point estimate — the fit is not yet a good
description of the data, and section 4 says why.

---

## 1. What was fitted

| | |
|---|---|
| Data | HST WFC3/IR F140W, 80x80 crop at 0.065"/pix, EXPTIME 1197.699 s |
| Model | EPL + external shear; 2 Sersic source components; bulge+disk deflector light |
| Free parameters | 36 |
| Pipeline | `run_fit.py --lens DESI234 --n-source 2 --n-lens-light 2`, run label `r2_s2l2` |
| Code | `claude/pipeline/` at commit `b7d0178` |
| Sampler | MCLMC |

Round 1 (`r1_s1l1`) is the same pipeline with `--n-source 1 --n-lens-light 1`, 22 parameters.

---

## 2. The result

### 2.1 Staging

MAP (1500 steps, batch 4096, 503.7 s) -> SVI (12000 steps, `n_vi` 2048, peak LR 1e-3, 1954.3
s) -> MCLMC. MAP+SVI together cost 2458.0 s = 41.0 min (`r2_s2l2.log`).

The sampling stage was run three times off the *same* MAP+SVI checkpoints, at increasing
length. Only the last is the result; the first two are kept because together they show the
convergence trend (section 3):

| draws kept | chains | burn-in | wall (sampling) | worst R-hat |
|---|---|---|---|---|
| 750 | 50 | 250 | 26.8 s | 1.368 |
| 10,000 | 50 | 250 | 136.9 s | 1.022 |
| **40,000** | **50** | **5,000** | **454.7 s** | **1.008** |

(`r2_s2l2.log`, `r2_long.log`, `r2_long2.log`; the two short-run checkpoints are kept as
`hmc_short.npz.bak` / `hmc_10k.npz.bak` in the run directory, and were re-diagnosed for this
document by copying them alongside the shared `map.npz`/`svi.npz` into a scratch directory —
`diagnose.py` only ever reads the checkpoint it is pointed at.)

The SVI surrogate itself was already healthy before any sampling ran: covariance condition
7.4e+07 (healthy range 1e6-6e8), ELBO tail sd 23.5 over the last 300 steps at 36 parameters.
Convergence here came from chain length on top of an already-good preconditioner, not from
fixing a bad one — the opposite lesson from DESI238 round 5, where the surrogate was the
binding constraint. Both are true; they are different failure modes.

### 2.2 Headline run: `runs/r2_s2l2`, 50 chains x 40,000 draws, 5,000 burn-in, 464.5 s

Reduced chi-squared at the point estimate (HMC median): **1.300**, over 5959/6400 unmasked
pixels (441 pixels masked — see section 5).

**All 36 parameters R-hat <= 1.008; min bulk-ESS 7450, min tail-ESS 8461** (`diagnose.py`).
No boundary pinning: the closest any Uniform-prior parameter comes to its bound is
`source/light/src1/n_sersic` at 81.9% of `[0.3, 6]`, well short of the 2%/98% pinning
threshold.

#### Lens mass

| parameter | median | -1sig | +1sig | R-hat |
|---|---|---|---|---|
| `theta_E` | **1.5393** | 0.0024 | 0.0025 | 1.005 |
| `gamma` | 2.1989 | 0.0705 | 0.0568 | 1.008 |
| `e1` | -0.0069 | 0.0149 | 0.0149 | 1.003 |
| `e2` | -0.0505 | 0.0052 | 0.0049 | 1.005 |
| `center_x` | -0.0446 | 0.0032 | 0.0032 | 1.002 |
| `center_y` | -0.0793 | 0.0119 | 0.0123 | 1.004 |
| `shear/gamma1` | 0.1095 | 0.0084 | 0.0070 | 1.006 |
| `shear/gamma2` | -0.0469 | 0.0030 | 0.0028 | 1.003 |

#### Deflector light (bulge + disk)

| parameter | median | -1sig | +1sig | R-hat |
|---|---|---|---|---|
| `bulge/Ie` | 116.74 | 26.69 | 30.61 | 1.006 |
| `bulge/R_sersic` | 0.4280 | 0.0578 | 0.0742 | 1.006 |
| `bulge/n_sersic` | 5.2713 | 0.3643 | 0.4073 | 1.006 |
| `bulge/e1` | -0.1008 | 0.0037 | 0.0035 | 1.003 |
| `bulge/e2` | 0.0229 | 0.0056 | 0.0059 | 1.004 |
| `bulge/center_x` | -0.0041 | 0.0006 | 0.0006 | 1.003 |
| `bulge/center_y` | -0.0293 | 0.0006 | 0.0006 | 1.002 |
| `disk/Ie` | 8.4533 | 0.6786 | 0.7423 | 1.005 |
| `disk/R_sersic` | 1.7176 | 0.0503 | 0.0470 | 1.004 |
| `disk/n_sersic` | 1.4231 | 0.1176 | 0.1186 | 1.005 |
| `disk/e1` | 0.0126 | 0.0101 | 0.0118 | 1.005 |
| `disk/e2` | -0.1709 | 0.0169 | 0.0144 | 1.006 |
| `disk/center_x` | 0.0898 | 0.0143 | 0.0179 | 1.006 |
| `disk/center_y` | -0.0740 | 0.0114 | 0.0097 | 1.004 |

`bulge/n_sersic` sits at 61.0% of `[1, 8]`, `disk/n_sersic` at 46.2% of `[0.5, 2.5]` — neither
pinned.

#### Source

| parameter | median | -1sig | +1sig | R-hat |
|---|---|---|---|---|
| `src0/Ie` | 121.879 | 6.786 | 7.110 | 1.002 |
| `src0/R_sersic` | 0.0524 | 0.0032 | 0.0032 | 1.004 |
| `src0/n_sersic` | 1.7993 | 0.1256 | 0.1306 | 1.002 |
| `src0/e1` | 0.0797 | 0.0366 | 0.0348 | 1.002 |
| `src0/e2` | -0.0106 | 0.0410 | 0.0410 | 1.002 |
| `src0/center_x` | -0.0426 | 0.0036 | 0.0035 | 1.002 |
| `src0/center_y` | -0.4289 | 0.0174 | 0.0215 | 1.007 |
| `src1/Ie` | 16.528 | 10.366 | 28.669 | 1.003 |
| `src1/R_sersic` | 0.8449 | 0.2732 | 0.4200 | 1.004 |
| `src1/n_sersic` | 4.9673 | 1.2686 | 0.7365 | 1.003 |
| `src1/e1` | 0.3508 | 0.1076 | 0.0891 | 1.004 |
| `src1/e2` | -0.2682 | 0.0887 | 0.0870 | 1.004 |
| `src1/center_x` | 1.0097 | 0.4284 | 0.4314 | 1.004 |
| `src1/center_y` | 2.6302 | 0.1718 | 0.1826 | 1.004 |

**`src1` sits outside the caustic and is not multiply imaged — read section 6 before using it.**

No label switching between the two source components: `R_sersic` separates at 2.0 sigma
(`0.0524 +/- 0.0032` vs `0.8449 +/- 0.3950`), 0 crossings in 0/50 chains, so R-hat above is on
the raw chains. The separation itself is much weaker than DESI238's 20.2 sigma — consistent
with `src1` being marginal rather than a confidently resolved second component.

---

## 3. Comparison to round 1 (`r1_s1l1`, 22 parameters, never converged)

| | round 1 | round 2 |
|---|---|---|
| MAP chi-squared (final) | 1.87118 | 1.33393 |
| chi2_nu at point estimate | 1.560 | 1.300 |
| `theta_E` | 1.5324 +0.0019/-0.0016 (unreliable, see below) | 1.5393 +0.0025/-0.0024 |
| worst R-hat (36/22 params) | **1.652** | **1.008** |
| lens-light `n_sersic` | 9.1624 in `[2, 10]`, 89.5% — **PINNED** | no pinning (worst 81.9%) |

(MAP checkpoints: `runs/r1_s1l1/map.npz`, `runs/r2_s2l2/map.npz`, `chisq_hist` field.)

Round 1's chains never converged (worst R-hat 1.652, min bulk-ESS 81) and its single Sersic
lens-light component was pinned at 89.5% of its `[2, 10]` range — the prior, not the data, was
setting that value. **Round 1's `theta_E` is quoted above only to show the number, not as a
measurement; a parameter from an unconverged chain is not evidence of anything.** Round 2's
value is close to it (1.5393 vs 1.5324), which is a mild consistency check across model
variants, but round 1 cannot be leaned on to confirm it, since it never converged.

Adding 14 parameters (source count and lens-light split) bought a MAP chi-squared reduction
of `(1.87118 - 1.33393) x 5959 = 3202` over the 5959 unmasked pixels — far more than 14
degrees of freedom explain on their own, and consistent with the bulge+disk split and the
second source component both doing real work, not just absorbing noise.

---

## 4. Why chi2_nu = 1.300 is not the whole story

1.300 is a substantial improvement on round 1's 1.560, but it is still well above 1, and the
**0.003" statistical error on `theta_E` is not the total error budget.** Section 5 identifies
two distinct sources of unmodelled structure in the residual, either of which could move
`theta_E` by more than its quoted statistical error if addressed.

---

## 5. Residuals: two distinct unmodelled features

`runs/r2_s2l2/residuals.png` shows two features that two smooth Sersic source components and
a bulge+disk deflector cannot absorb:

**(a) A pixel-scale alternating +/- checkerboard at the deflector core**, reaching roughly
+/-5 sigma in the normalised residual. The same pattern, at the same location relative to the
deflector core, appears independently in `claude/DESI257/runs/r2_s2l2/residuals.png`. The
cross-cutting inference — that an alternating-sign, pixel-scale pattern common to two lenses
is a PSF mismatch rather than a per-lens light-profile failure, plus the recommended remedy —
is in `claude/pipeline/KNOWN_ISSUES.md` item 5.

**(b) Discrete +/-4 sigma knots along the arc**, at roughly crop (17-25, 12-25) and (50-60,
10-25) in the normalised-residual panel. Two smooth Sersic components cannot produce discrete
knots at fixed positions along an extended arc; this reads as genuine source substructure —
star-forming clumps — that would need a pixelated or shapelet source to capture.

Neither feature is new to this round; they are what remains after doubling both light
components. Recommended next step: rebuild the PSF (`claude/pipeline/KNOWN_ISSUES.md` item 5)
before adding more source freedom, since a PSF mismatch at the core could itself be inflating
the apparent need for arc-level structure by biasing the fit's flux allocation.

---

## 6. `src1` is not multiply imaged and should not be read as a second source component

`residuals.png`'s source-plane panel (bottom left) shows `src1` sitting at roughly
`(1.0, 2.6)"`, far outside both the caustic and the ray-traced cutout border, while `src0`
sits essentially on a caustic cusp. Checked numerically, not by eye, using
`diagnose.source_component_params` for the two centres and `LensSolver.critical_and_caustics`
for the caustic (`gigalens.jax.analysis.lens_solver`, the same machinery `residuals.png` is
built from):

| | centre (arcsec) | min distance to caustic | inside caustic | inside cutout border |
|---|---|---|---|---|
| `src0` | (-0.0426, -0.4289) | **0.023"** | no (sits on the cusp) | yes |
| `src1` | (1.0097, 2.6302) | **2.544"** | no | **no** |

`src0`'s 0.023" separation from the caustic is smaller than its own `R_sersic` (0.052"), i.e.
it sits on the cusp to within its own size — consistent with it being the component that
produces the arc-plus-counter-image configuration. `src1`'s 2.544" separation is roughly 50x
its own `R_sersic` (0.845") and puts it outside the region any point in the cutout maps back
from. **A source outside the caustic is not multiply imaged: `src1` is a singly-imaged blob,
not a contributor to the arc.**

This changes how `src1` should be read. `R_sersic` separates from `src0`'s by only **2.0
sigma** (`0.0524 +/- 0.0032"` vs `0.8449 +0.3950/-0.3950"` — note the asymmetric, very wide
interval on `src1`), and `src1/Ie` is **16.5, with a -10.4/+28.7 interval** — consistent with
zero flux is not far off. Combined with the caustic geometry above, the correct
characterisation is not "weakly detected second source" but: **`src1` is singly imaged, lies
outside the caustic, and is not well constrained — it improves chi-squared (section 3) without
being a credibly detected second source component.** Unlike DESI238's analogous second
component, which separated from the first at 20.2 sigma *and* sits within its system's
caustic, `src1` here should not be interpreted as a physically established second source. Do
not treat its parameter values as settled, and do not cite its presence as evidence of a
second lensed component either.

---

## 7. `theta_E` = 1.539" against the 1.43" geometric estimate

The posterior median sits **0.11" above** `config.py`'s geometric estimate of 1.43" (the mean
of the arc radius and the counter-image radius under the cusp-quad reading). Reported
honestly: this is a real offset, not a rounding matter, and it should be stated rather than
smoothed over.

It is not a sign of a misbehaving fit, for two reasons:

- **It sits well inside the prior's support.** The `theta_E` prior's 1-99% range is
  `[0.799, 2.558]"` (`diagnose.py`); 1.539" is not near either edge, so the value is
  data-driven, not an artifact of the prior dragging the posterior toward its own median.
- **It is geometrically sensible.** The S-arc clumps sit at 1.71-1.81" and the N
  counter-image at 1.11" (`config.py`'s `CLUMPS`); 1.539" falls between them, which is where
  the tangential critical curve of a cusp-quad lens is expected to sit.

This is not a resolution of the alternative reading in `config.py` (single fold pair vs. cusp
quad) — that remains round 5's falsification test — but the fitted value is consistent with
either.

---

## 8. Mask-aware reduced chi-squared

DESI234's round-1 chi2_nu was previously reported as 4.225, computed by averaging normalised
squared residuals over *every* pixel, including the masked interloper's. The true masked
value is 1.560. **Every chi-squared value in this document uses the corrected, mask-aware
definition** (`diagnose.masked_chisq`, matching `ProbModel.log_prob`'s own second return
value; `verify.py` check 7 guards it). The general bug and fix are recorded in
`claude/pipeline/KNOWN_ISSUES.md` item 3.

Also corrected: the mask covers **441** of 6400 pixels (6.9%), not 437 as an earlier note in
this README said. `data.build_mask` uses an inclusive `<=` radius test
(`(rows-row)**2 + (cols-col)**2 <= radius**2`), and 441 is what that test returns at radius 12
px around (64, 14) — confirmed directly from `run r2_s2l2`'s metadata (`n_masked_pixels: 5959`
of 6400, i.e. 441 masked) and from the residual figure's own title, which prints "441 px
masked."

---

## 9. Caveats, stated plainly

- **chi2_nu = 1.300 means unmodelled structure remains** — see section 5. The quoted 0.003"
  statistical error on `theta_E` does not include either the PSF-mismatch or arc-substructure
  systematic.
- **`src1` sits outside the caustic** (section 6): it is not multiply imaged and does not
  contribute to the arc. Its presence improves chi-squared but is not evidence of a second,
  physically established source component; do not treat its own parameter values as settled.
- **`theta_E` is 0.11" above the geometric estimate** (section 7): data-driven, within the
  prior's support, geometrically sensible, but not yet cross-checked against the alternative
  fold-pair reading.
- **The background RMS carries a +/-15% patch-to-patch spread** (`README.md`), which is
  roughly +/-30% on chi-squared and is not propagated into any quoted error here.
- **The interloper at 2.30" is masked, not modelled** (`README.md`): round 3 is where that
  gets addressed. Until then this fit carries an unmodelled interloper as an accepted
  systematic.
- **The R-hat trend runs (750, 10,000 draws) are diagnostic, not independent replications** —
  they share one MAP+SVI checkpoint and one seed with the 40,000-draw headline run. They show
  that chain length, not the surrogate, was the binding constraint here; they are not a
  multi-seed check.

---

## 10. Reproducing this

```bash
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python

# Preflight -- not yet run for DESI234; see README.md's status.
CUDA_VISIBLE_DEVICES=0 $PY claude/pipeline/verify.py --lens DESI234

# Surrogate: two GPUs, ~41 min.
CUDA_VISIBLE_DEVICES=2,3 $PY claude/pipeline/run_fit.py --lens DESI234 --label r2_s2l2 \
    --n-source 2 --n-lens-light 2 --stages map,svi \
    --n-samples 4096 --map-steps 1500 --n-vi 2048 --svi-steps 12000 --svi-peak-lr 1e-3

# Headline sample: ~8 min.
CUDA_VISIBLE_DEVICES=2,3 $PY claude/pipeline/run_fit.py --lens DESI234 --label r2_s2l2 \
    --n-source 2 --n-lens-light 2 --stages hmc --sampler mclmc \
    --n-hmc 50 --burnin 5000 --num-results 40000

CUDA_VISIBLE_DEVICES=2,3 $PY claude/pipeline/diagnose.py claude/DESI234/runs/r2_s2l2
```

Check, in this order: SVI condition number and ELBO tail sd; then the R-hat table; then the
pinning report; then the residual figure for the two features in section 5.
