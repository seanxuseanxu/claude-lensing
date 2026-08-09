# DESI257, round 2: bulge+disk light, two source components

**Date:** 2026-08-08. First fit of this system with data. Round 1 (`r1_s1l1`) is documented
below for comparison only — it did not converge and its `theta_E` (2.004") is quoted only to
show the number, not as a measurement.

**Headline, stated with its central caveat: `theta_E = 1.9811 (-0.0041,+0.0042)"` in the
dominant posterior mode (73.4% of draws, 17/50 chains, within-mode R-hat 1.0092). The full
50-chain posterior is multimodal and its combined R-hat fails outright (worst 1.461) — do not
quote a full-chain number as the result.** Section 3 is the central section of this document.

---

## 1. What was fitted

| | |
|---|---|
| Data | HST WFC3/IR F140W, 80x80 crop at 0.065"/pix, EXPTIME 1197.699 s |
| Model | EPL + external shear; 2 Sersic source components; bulge+disk deflector light |
| Free parameters | 36 |
| Pipeline | `run_fit.py --lens DESI257 --n-source 2 --n-lens-light 2`, run label `r2_s2l2` |
| Code | `claude/pipeline/` at commit `b7d0178` |
| Sampler | MCLMC, 50 chains x 10,000 draws, 2,000 burn-in |

Round 1 (`r1_s1l1`) is the same pipeline with `--n-source 1 --n-lens-light 1`, 22 parameters.

Staging: MAP (1500 steps, batch 4096, 504.4 s) -> SVI (12000 steps, peak LR 1e-3, 1954.0 s)
-> MCLMC (145.5 s) (`runs/r2_s2l2.log`, `runs/r2_long.log`). **`metadata.json` currently
misreports `svi_peak_lr` as `0.002`** — the log line above is the ground truth for what was
actually run. This is an instance of a pipeline bug (`run_fit.py` overwrites prior stages'
metadata with argparse defaults on a stages-only re-run); see
`claude/pipeline/KNOWN_ISSUES.md` item 1 for the general bug and its fix.

---

## 2. MAP and the resolution of round 1's pinning

MAP chi-squared, final value, read directly from `chisq_hist` in each run's `map.npz`:

| | round 1 (22 params) | round 2 (36 params) |
|---|---|---|
| MAP chi-squared | **2.81607** | **1.25299** |
| chi2_nu at point estimate | 1.533 | 1.075 |
| worst R-hat (of 22/36 params) | 1.878 | 1.461 |

Both chi2_nu values are over 5997/6400 unmasked pixels (403 masked, `README.md`).

**Round 1's single source Sersic was pinned at its prior ceiling:** `source/light/src0/n_sersic
= 6.0000` in `[0.3, 6]` — **100.0% of range, sd exactly 0.0000** (`diagnose.py`'s boundary-pinning
report, reproduced directly). A parameter with zero posterior spread sitting exactly on its
prior bound is not a measurement; it means one Sersic component was being asked to represent
light that needs two.

**Round 2 shows no such pinning.** With two source components, the closest any Uniform-prior
parameter comes to its bound is `lens/light/disk/n_sersic` at 51.4% of `[0.5, 2.5]` — nowhere
near the 2%/98% pinning threshold. The two source indices land at `src0/n_sersic = 3.0735`
(48.7% of `[0.3, 6]`) and `src1/n_sersic = 1.2504` (16.7%), both interior values with normal
posterior spread. **This is the round's main modelling conclusion: the pinning was one Sersic
being asked to do two components' work; given two, it resolves.**

The two source components do not label-switch: `R_sersic` separates at **4.53 sigma**
(`0.3244 +/- 0.0502"` vs `0.0904 +/- 0.0125"`), 0 crossings in 0/50 chains, so R-hat below is
reported on the raw chains.

---

## 3. Convergence: the central caveat

The full 50-chain, 10,000-draw posterior (`runs/r2_s2l2/hmc.npz`, restored from
`hmc_10k.npz.bak` after the failed long run in section 4) **does not converge as a whole**:

```
36/36 parameters have Rhat > 1.01 (worst 1.461)
```

(`diagnose.py`, reproduced twice independently in this session on separate GPU allocations,
bit-for-bit identical output both times.) The mass parameters are the worst offenders —
`gamma` R-hat 1.453, `theta_E` R-hat 1.417, host `e1` R-hat 1.456 — which is the signature of
chains disagreeing about *where* the posterior mass sits, not merely mixing slowly within one
mode.

### 3.1 Two modes, not slow mixing

Per-chain mean `gamma` splits cleanly at a threshold of 2.66: **32 of 50 chains cross this
threshold at least once** over their 10,000 draws (1-25 crossings per chain), 17 chains never
leave the `gamma < 2.66` side, and 1 chain sits almost entirely on the `gamma > 2.66` side.
This is a **permeable barrier**, not two disjoint absorbing states — which is exactly why
lengthening the chains 12x barely moved R-hat (1.647 at 750 draws -> 1.461 at 10,000 draws,
section 4's log): a chain that crosses even once contaminates its within-chain variance with
both modes' locations, and R-hat penalises that regardless of chain length.

### 3.2 Restricting to the 17 never-crossing chains

Selecting the 17 chains with `gamma < 2.66` in more than 98% of their draws
(`frac_below > 0.98`; kept-chain range 0.9918-1.0, rejected-chain range 0.0-0.9772 — a clean
separation, not a threshold sitting in the middle of a continuum):

| | value |
|---|---|
| max R-hat (36 params) | **1.0092** |
| min bulk-ESS | 1871 |
| min tail-ESS | 1480 |

**This R-hat is a single, unreplicated measurement.** A second agent independently rebuilt
the model and reproduced the same 17 chain indices and the same dominant/minority split from
the raw samples (spot-checked directly in this session: dominant-mode median `theta_E` =
1.9812, matching to four decimal places), but was blocked by transient ArviZ/pandas import
failures under filesystem contention before re-running `diagnose_convergence` itself. The
1.0092 figure is recorded here as measured once, not confirmed twice.

### 3.3 The two modes, by the numbers

| | mode A (dominant, 73.4% of draws, 17 chains) | mode B (minority, 26.6% of draws) |
|---|---|---|
| `theta_E` | **1.9811 (-0.0041, +0.0042)"** | 2.0017 (-0.0132, +0.0085)" |
| `gamma` | 2.5142 | 2.8364 |
| host `e1` | -0.2207 | -0.5914 |
| host `e2` | -0.2947 | -0.2573 |
| \|shear\| | 0.143 | 0.195 |

(Draw-fraction percentages are over all 50 chains' 500,000 draws, by per-draw `gamma < 2.66`
membership — a different accounting than the 17/50 *chain*-count above, since several
rejected chains spend part of their trajectory in mode A before crossing.)

**Mode B is not the known EPL `c -> 1` zero-deflection artefact**: `c = sqrt(e1^2+e2^2)` tops
out at 0.763 in mode B, well short of the 1.0 where the kernel silently zeroes its own
deflection (`priors.py`'s own runtime warning, reproduced verbatim in the fit log, documents
this failure mode explicitly). Mode B is a genuine, prior-disfavoured degeneracy in
`gamma`/ellipticity/shear: roughly 3 prior-sigma out on `gamma` and on host `e1`, with
`|shear| = 0.195` sitting right at the `0.2` plausibility flag `priors.py` itself raises at
model-build time (also reproduced in the fit log: `estimated mass 0.136 exceeds 0.05` for
`|shear| <= 0.2`).

**Conclusion, stated as directly as the data supports it:** quoting the dominant mode
(`theta_E = 1.981 +/- 0.004"`, within-mode R-hat 1.0092) with the multimodality flagged is
defensible. Quoting the full 50-chain median (`theta_E = 1.983 (-0.005,+0.015)"`, R-hat 1.417)
is not — that number averages over two physically distinct solutions and its own convergence
diagnostic says so. **Neither is a final answer until the modes mix honestly** (next steps,
section 8, item 2).

### 3.4 Critical-curve validation (full-chain point estimate)

Independent of the multimodality question, the point-estimate mass model is geometrically
self-consistent: `LensSolver`'s tangential critical-curve radius is **2.0046"** against the
same posterior's `theta_E = 1.9829"` — **agreement to 1.10%** (`diagnose.py`). This checks
that the EPL parameters and the solver's own root-finder agree with each other; it does not
by itself resolve which mode is correct, since both modes' `theta_E` are close enough that
this check would likely pass for either.

---

## 4. A failed run that must be recorded: the 80,000-draw resample froze

An 80,000-draw, 10,000-burnin MCLMC re-sample was attempted from the same MAP+SVI checkpoint,
hoping the longer run would let the modes mix. **It failed and is recorded here as a negative
result, not omitted.** Every chain froze (within-chain sd collapsed 135x while between-chain
sd stayed at the real posterior's scale; R-hat 7.86e15). This is a general MCLMC failure mode,
not specific to this lens's science — the full numbers, the frozen checkpoint's location, and
the unattributed cause are recorded in `claude/pipeline/KNOWN_ISSUES.md` item 2, along with the
one-variable test needed to isolate it.

The good 10,000-draw chains were restored to `hmc.npz` from `hmc_10k.npz.bak`, which is what
every number in this document (outside this paragraph) is computed from.

---

## 5. The SW object / fourth-image question: a blind prediction test

DESI257's open question (`README.md`) is whether the object 2.58" SW of the deflector, masked
entirely from this fit (403 px), is an interloper or a fourth lensed image. Because it is
masked, the fit never saw it — a genuine blind prediction test is possible: does the model,
fitted with that region excluded, predict any flux landing there anyway?

**Report this at exactly the confidence the evidence supports: suggestive, not conclusive.**

### 5.1 Method and its validation

Dominant-mode (17-chain) median parameters pushed through `gigalens.jax.analysis.lens_solver.
LensSolver`. Coordinate convention verified directly: `LensWCS.angle2pix(0,0)` returns pixel
`(row, col) = (39.5, 39.5)`, the array centre as expected, and `pix2angle` of the fit's own
`centroid_pix` reproduces `README.md`'s stated 0.017" centring offset. Array-space arcsec
frame throughout; crop pixel `(row, col)`, 0.065"/px.

**Control passes before the SW result is trusted** — position is the wrong test for a
resolved arc, flux is the right one, and both were run (azimuthal-median-subtracted aperture
photometry, radius 3 px, at each known clump):

| clump | data (az-med subtracted) | model | ratio |
|---|---|---|---|
| E arc S | 6.338 | 5.698 | 0.90 |
| E arc mid | 9.005 | 9.060 | 1.01 |
| E arc N | 4.619 | 3.601 | 0.78 |
| N blob | 19.932 | 19.660 | 0.99 |

Flux recovered at **78-101%** across all four known images, and the N blob's predicted
position lands within **0.082-0.090"** of measured (sub-pixel). The two az-median-subtracted
panels (data vs. model) are visually near-identical at all four known-image locations. This
validates the method: the solver reproduces known images well, so an SW non-detection or
detection is meaningful rather than an artefact of a broken pipeline.

### 5.2 The SW result

SW object flux centroid, measured directly from the data (background-subtracted, r <= 10 px
around the mask centre): crop **(row 6.68, col 16.15)**.

| predicted image | separation from SW centroid |
|---|---|
| src1 (weaker component, `n_sersic` 1.25) | **1.40 px = 0.091"** — inside one PSF FWHM (~0.15", from the PSF sidecar's size anchor 2.34 px) |
| src0 (stronger component, `n_sersic` 3.07) | **6.47 px = 0.421"** — a clear miss |

**The two source components disagree.** src1's prediction is a near-hit; src0's is not.

Component-by-component rendering confirms that whatever model flux sits near the SW position
is **lensed source light**: rendering plane 1 (source), light component 1 in isolation peaks
at crop (6, 17), 1.12 e/s, close to the SW centroid. The lens-light disk component peaks
elsewhere entirely, at crop (19, 24), roughly 30x fainter at the SW location than the source
component's own peak — so this is not lens-light wing contamination being mistaken for a
source prediction.

### 5.3 Hold-backs, stated plainly

- **Only a 2-image topology is found** at this point estimate for each source component — not
  the 4-image topology the fourth-image hypothesis requires.
- **The two source components disagree** on whether the SW object is predicted at all (0.09"
  vs 0.42").
- **This used a single median z with no posterior propagation.** It is not known whether
  0.091" is within 1-sigma of zero or 5-sigma — turning this into an uncertainty is next
  steps' item 3, and needs no new fit.

### 5.4 `theta_E` context — does not select either hypothesis

The interloper reading predicts `theta_E ~ 1.70"` (`README.md`, mean of the two secure image
radii); the fourth-image reading predicts `~2.1-2.2"`. The dominant-mode measurement,
**1.981 +/- 0.004"**, sits **between them** — 0.28" above the interloper prediction, roughly
0.12-0.22" below the fourth-image range. **State this carefully: `theta_E` is intermediate
and does not cleanly select either hypothesis.** Do not read the measured value as favouring
one reading over the other.

---

## 6. Also recorded this session

**The mask-aware chi-squared bug, fixed today.** DESI257's round-1 chi2_nu was previously
reported as 7.112, computed by averaging normalised squared residuals over *every* pixel
including the masked 131-sigma interloper. The true masked value is 1.533. **Every
chi-squared value in this document uses the corrected, mask-aware definition** (`ProbModel.
log_prob`'s own second return value; `verify.py` check 7 guards it). The general bug and fix
are recorded in `claude/pipeline/KNOWN_ISSUES.md` item 3.

**`residuals.png` is now the four-panel format** described in
`claude/pipeline/KNOWN_ISSUES.md` item 4: observed + critical curve, model + critical curve,
unlensed source + caustics, mask-aware normalised residual.

**The residual shows a pixel-scale alternating +/- checkerboard at the deflector core.** The
same pattern appears independently in DESI234's fit (`claude/DESI234/RESULTS_v2.md`, section
5a). The cross-cutting inference — that this is a PSF error common to the dataset, not a
per-lens light-profile failure — and the recommended remedy (rebuild the PSF, following
`claude/DESI238/PSF_REPLICATION.md`'s validated procedure) are in
`claude/pipeline/KNOWN_ISSUES.md` item 5.

**SVI health caution, worth stating even though it did not block a result**: the ELBO tail sd
over the last 300 steps is 1597.9, still visibly moving at the end of training (a healthy
22-parameter run settles near sd 10). The covariance condition number, 5.48e6, is itself
healthy (1e6-6e8 range), so the surrogate's *shape* is usable — but a still-moving ELBO means
the Gaussian ansatz may not have fully settled, and is worth revisiting alongside the
multimodality fix, since an SVI surrogate built around one mode could itself be part of why
MCLMC struggles to move between them.

---

## 7. Caveats, stated plainly

- **The headline number is a dominant-mode value, not a full-posterior value** — section 3.
  26.6% of the posterior's own draws sit in a mode this document does not quote as the answer.
- **The within-mode R-hat of 1.0092 is unreplicated** — section 3.2.
- **The SW blind-prediction test is suggestive, not conclusive** — section 5.3: wrong
  topology, disagreeing source components, no posterior propagation.
- **The 80,000-draw resample failed for an unattributed reason** — section 4. Whatever caused
  it has not been ruled out as a property of this system rather than of that one run.
- **The background RMS carries a +/-15% patch-to-patch spread** (`README.md`), propagated into
  no quoted error here.
- **`chi2_nu = 1.075` is a point-estimate number computed on a multimodal posterior's median**
  — treat it as descriptive, not as validating the noise model, independent of the
  multimodality question.
- **The SVI ELBO had not fully settled** (section 6) — a caveat on the surrogate underlying
  every sampler result in this document.

---

## 8. Next steps, ordered by value

1. **Rebuild the PSF.** Pipeline-level next step; see `claude/pipeline/KNOWN_ISSUES.md` item 5
   (the core checkerboard residual, section 6, is common to DESI234 and DESI257).
2. **Get the modes to mix honestly.** Blocked on the pipeline's MCLMC frozen-chain failure
   (section 4; the isolating test is pipeline-level, see `claude/pipeline/KNOWN_ISSUES.md`
   item 2). Excluding mode B by chain-selection, as this document does, understates the
   uncertainty — it holds ~27% of the posterior mass and cannot simply be discarded as the
   final word. This part is DESI257-specific: it is this system's own mass-model degeneracy
   that needs the modes to mix.
3. **Posterior-propagate the SW image-position test.** Push a few hundred draws from the
   17 dominant-mode chains through `LensSolver`, instead of the single median used here, to
   turn "0.091 inch away" into "0.091 +/- sigma away," and to test whether a 4-image topology
   appears anywhere in the posterior rather than only at one point. This needs no new fit.
4. **Fix `run_fit.py`'s metadata bug.** Pipeline-level; see
   `claude/pipeline/KNOWN_ISSUES.md` item 1.

---

## 9. Reproducing this

```bash
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python

# Surrogate: two GPUs.
CUDA_VISIBLE_DEVICES=0,1 $PY claude/pipeline/run_fit.py --lens DESI257 --label r2_s2l2 \
    --n-source 2 --n-lens-light 2 --stages map,svi \
    --n-samples 4096 --map-steps 1500 --n-vi 2048 --svi-steps 12000 --svi-peak-lr 1e-3

# Headline sample: the run this document is measured from.
CUDA_VISIBLE_DEVICES=0,1 $PY claude/pipeline/run_fit.py --lens DESI257 --label r2_s2l2 \
    --n-source 2 --n-lens-light 2 --stages hmc --sampler mclmc \
    --n-hmc 50 --burnin 2000 --num-results 10000

$PY claude/pipeline/diagnose.py claude/DESI257/runs/r2_s2l2
```

Check, in this order: SVI condition number and ELBO tail sd (section 6's caution); then the
R-hat table — expect it to fail (section 3), that is the correct and expected outcome for
this checkpoint, not a sign of a broken run; then the boundary-pinning report; then the
residual figure for the core checkerboard.

To reproduce the mode-splitting analysis (section 3), select chains by per-draw
`gamma < 2.66` membership exceeding 98% of draws, on the constrained samples from
`diagnose.rebuild_model` / `diagnose.to_constrained` — do not reimplement the bijector by
hand (`CLAUDE.md`).
