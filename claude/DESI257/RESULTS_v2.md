# DESI257, round 2: bulge+disk light, two source components

> # RETRACTED, 2026-08-09
>
> **The headline is withdrawn. There is no quotable `theta_E` from this run.**
>
> 1. **"Dominant mode" was not measured.** The 73.4% draw fraction is substantially a
>    measurement of where the chains were *seeded*: the SVI surrogate mean sits at
>    `gamma = 2.4188`, inside mode A, and **48 of 50 chains have `gamma < 2.66` at
>    post-burn-in draw 0**. The mode weights were never measured, and the mode that was
>    quoted is the *shallower* of the two. Section 3 is rewritten around this.
> 2. **The fit was conditioned on a mask that should not have been there.** The SW object was
>    masked out on the argument that a demagnified fourth image must be the faintest, not the
>    brightest. That argument is a physics error: demagnification needs a radial critical
>    curve, which needs `gamma < 2`, and both modes here have `gamma > 2`. The object is most
>    likely the fourth image, and this run was fitted without the field's most constraining
>    datum. See `README.md`, "The SW object", and round 3 (`r3_unmasked`).
>
> **Both points are now settled by measurement, not argument.** Round 3 (`RESULTS_v3.md`)
> refits this model with the SW object in the likelihood, one variable changed. The
> bimodality **closes**: R-hat <= 1.001 on all 36 parameters and all 50 chains with no chain
> selection, against round 2's 1.003 on a hand-picked 17. The model reproduces the SW
> object's peak flux to 1.2% with no component added for it, and `gamma` (2.54 -> 1.99),
> `|shear|` (0.148 -> 0.044) and `|e_host|` (0.380 -> 0.143) relax simultaneously. So the two
> modes were an artefact of the mask, and the mode-weight question below is moot rather than
> open. Quote `theta_E` from round 3, not from here.
>
> **What still stands, and is not retracted:** sections 3.1-3.3's and 5.1's measurements —
> the two-mode structure, the crossing counts, the per-mode parameter values, and the
> validation of the `LensSolver` method against the four known images. Section 2's conclusion
> about round 1's `n_sersic` pinning also stands. What is retracted is the *inference* drawn
> from those measurements: which mode to quote, and whether the SW object is an interloper.

**Date:** 2026-08-08, retracted 2026-08-09. First fit of this system with data. Round 1
(`r1_s1l1`) is documented below for comparison only — it did not converge and its `theta_E`
(2.004") is quoted only to show the number, not as a measurement.

**Superseded headline, kept so the record is legible: `theta_E = 1.9811 (-0.0041,+0.0042)"`
in the "dominant" posterior mode (73.4% of draws, 17/50 chains, within-mode R-hat 1.0092).**
Both the multimodality caveat that accompanied it and the reasons above apply. Section 3 is
the central section of this document.

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
-> MCLMC (145.5 s) (`runs/r2_s2l2.log`, `runs/r2_long.log`). This is an instance of a pipeline
bug (`run_fit.py` overwrote prior stages' metadata with argparse defaults on a stages-only
re-run); see `claude/pipeline/KNOWN_ISSUES.md` item 1 for the general bug and its fix.

**`runs/r2_s2l2/metadata.json` misreports five fields**, not the one originally recorded.
Ground truth is `runs/r2_s2l2.log`'s own printed lines and §9's reproduce command:

| field | metadata.json | actually used |
|---|---|---|
| `n_samples` | 1024 | 4096 |
| `map_steps` | 350 | 1500 |
| `n_vi` | 1000 | 2048 |
| `svi_steps` | 1500 | 12000 |
| `svi_peak_lr` | 0.002 | 1e-3 |

`n_devices: 2` is **correct** and is not part of this bug: `metadata.json` is byte-identical
to `metadata_10k.json.bak` (md5 `7e720b80…`), i.e. it describes the 10,000-draw run, which
`r2_long.log` confirms ran on two devices. The single-device run was the 80,000-draw one (§4),
whose metadata was replaced when the good checkpoint was restored.

**What this does *not* break**, so the blast radius is known rather than feared:
`diagnose.rebuild_model` reads only `api`, `lens`, `n_source`, `n_lens_light`,
`lens_light_n_max`, the two prior versions, `theta_e_center`/`theta_e_width`,
`mask_circles`/`n_masked_pixels`, and guards on `n_free_params` — all correct in that file. The
physical decoding of `r2_s2l2` was never in question; what was corrupted is the record of *how
the surrogate was optimised*, which is exactly what one needs to reproduce it or to diagnose a
bad seed (§3).

**The file is left as it stands**, wrong fields and all. Rewriting a finished run's provenance
record by hand is a worse habit than carrying a documented error, and this table is the
correction.

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

### 3.4 Why "dominant" was the wrong word, and why there is no quotable number

**Retraction, 2026-08-09.** The original conclusion here was that quoting mode A
(`theta_E = 1.981 +/- 0.004"`) with the multimodality flagged is defensible. It is not, for
two reasons that the measurements above do not support and a third measurement refutes.

**(a) The 73.4% is a seeding artefact, not a mode weight.** Measured directly from this run's
own checkpoints:

| | `gamma` |
|---|---|
| MAP best particle | 2.3427 |
| SVI surrogate mean | 2.4188 |
| 50 chains at post-burn-in draw 0 | min 2.3881, median 2.5036, max 2.8054 |
| chains starting below the 2.66 threshold | **48 / 50** |

Every chain but two was started inside mode A, by a surrogate that was itself centred there,
by a MAP that was itself centred there. A draw fraction measured from chains seeded that way
is a measurement of the seed. It is not evidence about how much posterior mass each mode
holds. That MAP is separately known not to have converged — 681.8 nats below this run's own
best draw — so the whole seeding chain is compromised at its root
(`claude/pipeline/KNOWN_ISSUES.md` item 6).

**(b) The mode that was quoted is the shallower one.** Scoring 20,000 thinned draws through
this run's own `ProbModel.log_prob`:

| | mode A (`gamma < 2.66`) | mode B (`gamma > 2.66`) |
|---|---|---|
| median log posterior | 17196.6 | **17252.9** |
| median log prior | -26.5 | -36.9 |
| median `chi2_nu` (5997 px) | 1.0544 | **1.0322** |

Mode B fits better by **Δchi² = 133.3** over 5997 pixels and sits **56.3 nats higher** in log
posterior, having already paid a prior penalty of 10.4 nats. The document quoted the mode
that fits the data *less* well, on a fraction that measured the seed.

**(c) But depth is not mass, and asserting mode B would repeat the error with the sign
flipped.** Those are densities at the median, not integrals. A shallower mode with a larger
volume can hold more probability than a deeper narrow one, and **no volume has been
measured.** So:

**The mode weights of this posterior are unmeasured, and no `theta_E` from this run is
quotable** — not mode A's 1.9811", not mode B's 2.0017", not the full-chain median
(1.983 (-0.005,+0.015)", R-hat 1.417, which averages two physically distinct solutions and
whose own diagnostic says so).

This may be moot. Section 3.3's mode B is exactly the contortion needed to suppress a fourth
image the model is not allowed to see — `|e| = 0.645` (q ~ 0.22), `gamma = 2.836`,
`|shear| = 0.195` — and round 3 removes the mask that forces it. If the bimodality closes
there, the weights never need measuring. If it does not, they have to be measured (Laplace
weight per mode from the existing draws, plus a resample with chains deliberately split
across both modes so occupancy becomes a measurement rather than an inheritance).

### 3.5 Critical-curve validation (full-chain point estimate)

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
result, not omitted.** Every chain froze:

| | good 10,000-draw run | frozen 80,000-draw run |
|---|---|---|
| burn-in / devices | 2,000 / 2 | 10,000 / **1** |
| mean within-chain sd | 0.08155 | **0.0006021** (135x smaller) |
| between-chain sd of chain means | — | 0.09816 (unchanged in scale) |
| all values finite | yes | yes |
| R-hat | — | **7.86e15** |
| bulk-ESS | — | 50 (= chain count; one effective sample per chain) |

All values finite, and the between-chain spread stayed at the real posterior's scale — the
signature of 50 independent point-samples with no internal exploration, not 50 chains
converging to a point. The frozen checkpoint is preserved, not deleted:
`runs/r2_s2l2/hmc_80k_FROZEN_do_not_use.npz`.

This is a general MCLMC failure mode, not specific to this lens's science, and its cause is
still unattributed — see `claude/pipeline/KNOWN_ISSUES.md` item 2, which also records that
DESI234's `r3_unmasked` later froze at burn-in 5,000 on two devices, ruling out both variables
this run changed.

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

### 5.3 Hold-backs, one of them withdrawn (2026-08-09)

- ~~**Only a 2-image topology is found** at this point estimate for each source component —
  not the 4-image topology the fourth-image hypothesis requires.~~ **WITHDRAWN. This was a
  misreading, and the error was in the inference, not the number.** `LensSolver` was being
  asked to solve for a **point source at each component's centroid**. A point outside the
  tangential caustic produces 2 images; that is the correct answer for such a point and says
  nothing about an extended source. Re-measured over 300 draws from the dominant-mode chains
  (`analysis/caustic_flux_fraction.py`):

  | | `src0` | `src1` |
  |---|---|---|
  | centroid inside the tangential caustic | 0/300 draws | 0/300 draws |
  | signed distance, centroid to caustic (median) | 0.338" outside | **0.081" outside** |
  | point-source images at the centroid | 2 | 2 |
  | **fraction of the component's own flux inside the caustic** | **0.081 [0.076, 0.087]** | **0.176 [0.168, 0.187]** |

  So the original observation reproduces exactly — both centroids outside, both giving 2
  images — and the conclusion drawn from it does not. **Part of the source is quadruply
  imaged**, at both components, in every draw examined: a Sersic is not a point, and 8% and
  18% of the two components' flux lies inside the caustic. The 4-image topology the
  fourth-image hypothesis requires is present.

  Method notes, so the numbers are readable at their true precision: geometry from
  `LensSolver.critical_and_caustics` and `.solve_images` only; the flux integral is a grid
  sum over the installed `SersicEllipse.light()`, converged to under 1% relative across a 3x
  change of window (k = 10, 18, 30); the point-in-polygon test is `matplotlib.path.Path`. The
  16-84 intervals are over 300 draws subsampled without an ESS correction, so they are
  credible intervals but not ESS-corrected ones. `gamma`'s median of 2.513 admits no radial
  critical curve, and exactly one critical curve was found in all 300 draws — the geometry is
  not numerically pathological.

- **The two source components disagree** on whether the SW object is predicted at all (0.09"
  vs 0.42"). This hold-back stands.
- ~~**This used a single median z with no posterior propagation.**~~ **Addressed** for the
  caustic-membership question above, which is now measured over 300 draws. The *image
  position* test (the 0.091" and 0.421" separations) has still not been propagated; that part
  of the hold-back stands.

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

**The residual shows a pixel-scale alternating +/- checkerboard at the deflector core**,
reaching roughly +/-5 sigma. The
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

- **The headline number is retracted** — see the banner and section 3.4. It is not that the
  uncertainty was understated; it is that the mode was chosen on a fraction that measures the
  seeding, and the deeper mode is the other one.
- **The whole run is conditioned on a mask that should not have been applied** — the SW
  object is most likely the fourth image (`README.md`). Every number here that depends on the
  mass model carries that as an assumption.
- **The within-mode R-hat of 1.0092 is unreplicated** — section 3.2.
- **The SW blind-prediction test is suggestive, not conclusive** — section 5.3, and one of
  its three hold-backs is itself withdrawn there.
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

Superseded by the retraction. The ordering below is what it became.

1. **Refit unmasked** (`r3_unmasked`), one variable changed against this run, with the MAP
   step count raised until it clears the best draw of the sample it seeds
   (`claude/pipeline/KNOWN_ISSUES.md` item 6). This is the run that matters: it tests the
   fourth-image reading and is the leading candidate cure for the bimodality.
2. **Fit the interloper hypothesis properly** (`r3b`), unmasked and with a foreground SIS
   deflector plus its own light at the SW position, and compare mask-aware `chi2_nu` over the
   identical pixel set. An interloper deflects whether or not its light is in the likelihood,
   so this — not masking — is the fair form of the alternative.
3. **Measure the mode weights, only if step 1 leaves the bimodality open** — Laplace weight
   per mode from the existing draws, plus a resample with chains deliberately split across
   both modes. Asserting either mode without this repeats the retracted error.
4. **Rebuild the PSF.** `claude/pipeline/KNOWN_ISSUES.md` item 5 (the core checkerboard
   residual, section 6, common to DESI234 and DESI257). Real, but a smaller effect than a
   missing image, which is why it moved down.

Done since: `run_fit.py`'s metadata bug (`KNOWN_ISSUES.md` item 1) is fixed, and the SW
image-position test's missing posterior propagation is addressed in section 5.3.

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
