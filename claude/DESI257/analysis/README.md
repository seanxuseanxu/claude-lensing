# DESI257 analysis scripts

Four short scripts, each producing numbers quoted in `../RESULTS_v2.md` or `../RESULTS_v3.md`.
They exist so those numbers are reproducible rather than asserted. All four go through
`claude/pipeline/diagnose.py`'s `rebuild_model` / `to_constrained` and gigalens' own
`ProbModel.log_prob` — nothing here reimplements a bijector, a likelihood or a normalisation
(`CLAUDE.md`).

Run from the repo root, one GPU is enough (`undersampling_check.py` is CPU-only, see below):

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

| `undersampling_check.py` | port of DESI234's `undersampling_check.py`: `diagnose_undersampling` ladder (1,2,4) on `r3_unmasked` at `reference_supersample=32`, 9-sample batch (MAP, HMC-median, 7 spread HMC draws), plus the same decisive test (Pearson r between the ss=2 predicted `delta/sigma` map and the observed residual in a 10x10 px box at `cfg.centroid_pix`) | `RESULTS_v3.md` — whether `supersample=2` (config.py:369) is costing the `theta_E = 1.991 [1.989, 1.994]"` interval anything |

`undersampling_check.py`'s reference (ss_ref=32) did **not** self-certify on DESI257 (subgrid
self-check 0.191 sigma at the batch max, vs the library's `tolerance/_REFERENCE_MARGIN` =
0.2/4.0 = 0.05 sigma threshold — every one of the 9 per-sample checks also exceeded it,
0.122–0.191 sigma subgrid), so the numbers below are reported but **not certified**, exactly
as the task that produced them required. Despite that, ss=2 fails the library's own 0.2
sigma/pixel tolerance outright: max|delta|/sigma = 3.688, delta_chi2 = -216 (mean over the
9-sample batch, range -294..-175) over 6400 unmasked pixels — the same order of magnitude as
DESI234's certified ss=2 failure (3.13 sigma, delta_chi2 = -340). The core-box decisive test
gives Pearson r = +0.107 between the ss=2 quadrature-error map and the observed residual —
weak/near-zero, i.e. the observed core residual is not (mainly) reproduced by the ss=2
quadrature error, the same qualitative finding as DESI234 (r = -0.109). Full numbers,
per-rung table, and the per-sample reference self-error breakdown are in
`undersampling_report.txt`; the ss=2 delta/sigma map is `undersampling_ss2.png` and
`undersampling_ss2_delta_over_sigma.npy`.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
    /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
    claude/DESI257/analysis/undersampling_check.py
```

Run CPU-only (login node, no SLURM/GPU): the batched ladder call took 28.6s and the 9
per-sample `reference_self_error` follow-up (only run because the batched self-check failed
and the projected cost, ~257s, was within the 30-minute budget) took 107.1s — both well
inside budget, so this did not require an allocation.

`undersampling_recheck.py` follows up with `reference_supersample=64` and an ss=8 rung, same
9-sample batch, appended to `undersampling_report.txt` (the ss_ref=32 section above is left
in place). ss_ref=64 also did **not** self-certify (subgrid self-check 0.072 sigma, still
above the 0.05 sigma threshold, though down from 0.191 sigma at ss_ref=32) — per the task,
this was not escalated further to ss_ref=128; whether the library's certify margin is
reachable at all on this system is a separate question. **ss=8 still FAILS** the 0.2 sigma/px
tolerance: max|delta|/sigma = 0.439, delta_chi2 mean -34.3 (down from ss=4's 1.497 / -65.0).
The ranking and magnitudes are stable across the two references (ss=2 max|d|/sigma 3.688 at
ss_ref=32 vs 3.697 at ss_ref=64; ss=4: 1.427 vs 1.497 — both changed by <5% when the
reference's own self-error was cut by ~2.7x), which is why the ranking is treated as
trustworthy in `RESULTS_v3.md` despite neither reference certifying outright.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
    /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
    claude/DESI257/analysis/undersampling_recheck.py
```

Wall-clock: 85.7s for the batched ladder=(1,2,4,8) at ss_ref=64 (4 rungs + 2 PSF-convention
references at higher supersample, vs 3 rungs at ss_ref=32, plus the intrinsic ss^2 cost of a
deeper reference — both inside budget).

| `arc_residual_anatomy.py` | port of DESI234's script of the same name: normalised residual projected on the model's own error-normalised spatial gradient (row/col translation, and a tangential/radial decomposition about the deflector centre), fit separately on THREE regions where `residuals.png` shows paired opposite-sign lobes — the E arc, the N blob, and the SW object | `RESULTS_v3.md` — whether the dipoles are a displaced/mis-shaped model arc or genuine source-plane structure |

`arc_residual_anatomy.py` tests the paired +/-4 sigma residual lobes seen in `runs/r3_unmasked/residuals.png`
at the E arc (r 1.76-1.92", array-space PA 40-110), the N blob (r 1.40-1.70", PA 320-20, wrapping
0), and the SW object / fourth image (r 2.40-2.76", PA 195-240) — all array-space angles, **not**
sky PA (see README.md's "Lens configuration" warning). The deflector-core checkerboard
(`KNOWN_ISSUES.md` item 5) is excluded from every region and is not this script's subject.

Region sizes and peak significance (sanity check that each box actually contains its bright
feature): E arc 84 px (peak 26.8 sigma), N blob 118 px (peak 36.8 sigma), SW object 169 px
(peak 34.2 sigma).

**Result: all three regions read SUBSTRUCTURE**, not displacement. The (row, col) translation
fit removes 14.6% of the E arc's local chi-squared (implied shift 0.30 px = 0.020"), 0.2% of
the N blob's (0.015 px = 0.001"), and 7.8% of the SW object's (0.122 px = 0.008") — all well
under the 50% threshold the DESI234 script uses for a DISPLACEMENT verdict, and the E arc and
SW object sit below the 20% "BOTH" threshold too, though not deep in SUBSTRUCTURE territory the
way DESI234's S-arc was (1%). The tangential/radial decomposition agrees with each region's
row/col verdict to within 1 percentage point in every case.

The fitted (row, col) shift vectors do **not** agree across regions: N blob's implied shift
(0.015 px) is an order of magnitude smaller than the E arc's (0.30 px) or SW object's (0.12
px), and the E-arc/N-blob and N-blob/SW-object shift vectors have negative cosine similarity
(-0.38, -0.72) — inconsistent with a single global astrometric or PSF-centring offset, which
would produce the same shift vector everywhere. (E arc and SW object alone have cosine
similarity +0.92, but neither individually clears even the 20% partial-explanation threshold,
so this is a coincidence of two weak, mostly-unexplained fits rather than evidence for a shared
displacement.) Read together with the "no global shift" finding, the per-region SUBSTRUCTURE
verdicts stand: the paired lobes are consistent with source-plane structure the current
two-component Sersic source cannot reproduce, not with the model arc/blob/counter-image sitting
in the wrong place. This should be weighed before any future round adds source freedom (e.g. a
pixelated source), which would absorb such structure without diagnosing it further.

Full per-region numbers (pixel counts, fit coefficients, both chi2_nu before/after, both
fractions explained, and the cross-region cosine-similarity table) are in
`arc_residual_anatomy.txt`; the five-panel-per-region figure (data, model, normalised residual,
best-fit shift model, leftover) is `arc_residual_anatomy.png`.

```bash
CUDA_VISIBLE_DEVICES="" /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
    claude/DESI257/analysis/arc_residual_anatomy.py
```

| `adaptive_supersample_check.py` | validates gigalens' EXPERIMENTAL `gigalens.jax.experimental.adaptive_supersample` (per-pixel factor map driven by observed SNR) on `r3_unmasked`, as a MEASUREMENT only — not wired into the pipeline, no fit run. Same 9-sample batch as `undersampling_check.py`. Uses the library's own `compare_to_reference` at `reference_supersample=64`, both `psf_mode`s, over 3 `snr_levels` settings; produces `plot_factor_map` per setting | whether the adaptive grid is a viable cheaper alternative to uniform `supersample=16` for round 4 |

`adaptive_supersample_check.py` is a validation exercise, not a production change — **nothing outside this directory was edited, no fit was run.** It reuses the ss_ref=64 reference self-check already measured by `undersampling_recheck.py` (native 0.018 sigma, subgrid 0.072 sigma vs `undersampling_report.txt`) rather than re-deriving it: NOT CONVERGED (subgrid, vs the library's 0.05 sigma pass margin), so every number below is reported, not certified, exactly as for the uniform ladder.

Three `snr_levels` settings were swept, each producing an `AdaptiveGrid` and both `compare_to_reference` PSF modes:

| setting | n_points | uniform-equivalent ss | bin_first max\|d\|/sigma | subgrid max\|d\|/sigma | bin_first delta_chi2 (mean) | render wall |
|---|---|---|---|---|---|---|
| `library_default` (`DEFAULT_SNR_LEVELS` as shipped) | 65,500 | 3.20 | 0.778 FAIL | 4.034 FAIL | +16.8 | 0.49s |
| `factor8_opt_in` (default + factor-8 above SNR 14, the module's own suggested opt-in) | 187,324 | 5.41 | **0.134 PASS** | 3.700 FAIL | +1.1 | 0.59s |
| `aggressive_core` (lower thresholds, wider `dilate`, pushing more of the compact core to factor 8) | 394,576 | 7.85 | 0.134 PASS | 3.700 FAIL | +1.3 | 0.91s |

The as-shipped defaults (calibrated on the bundled demo, not DESI257) fail outright — expected, and exactly why the module's own docs say to re-derive `snr_levels` with `compare_to_reference` per system. Opting in to factor-8 refinement (`factor8_opt_in`) reaches the 0.2 sigma/px tolerance in the module's **own** `bin_first` PSF convention at 187k evaluation points — 2.2x cheaper than the uniform `supersample=8` rung's 409,600 points (which itself FAILS at 0.439 sigma in `subgrid` convention, `undersampling_report.txt`), and 8.75x cheaper than the untested uniform `supersample=16` candidate. Pushing further (`aggressive_core`) buys no additional accuracy at 2.1x more points — `factor8_opt_in` is the efficient setting of the three tried.

**But `bin_first` is not the tolerance that matters here.** The adaptive module always PSF-convolves bin-first (scatter to native grid, convolve with the native-resolution kernel) — it has no subgrid-convolution path, unlike the uniform pipeline which switches to `subgrid_kernel` convolution at `supersample>=2` (the convention `r3_unmasked` was fitted under). Measured against that `subgrid` reference, **every setting fails outright** (3.70–4.03 sigma, `delta_chi2` -83 to -200) — and the number barely moves between the loosest and tightest `snr_levels` tried (4.03 -> 3.70), because it is not measuring the factor map's quadrature error at all: it reproduces almost exactly the 3.75 sigma bin-first-vs-subgrid **PSF convention gap** `undersampling_report.txt` already measured independently on the uniform ladder ("NOT reducible by supersampling" — and, this script now confirms, not reducible by adaptive refinement either, since both are bin-first). Adopting the adaptive grid therefore is not a drop-in fix for `r3_unmasked` as fitted; it requires first deciding, and separately validating, a switch of the whole model's PSF convolution convention from subgrid to bin-first — matching how DESI257's PSF was actually measured (module docstring: bin-first assumes an already pixel-integrated, e.g. star-stacked empirical PSF) — a modeling decision, not an engineering one.

`plot_factor_map` (`adaptive_factor_map_<setting>.png`) shows the grid refining the entire visible lens system (the extended arc/lens-light footprint occupying most of the 80x80 cutout) up to factor 8 in `factor8_opt_in`/`aggressive_core`, native (factor 1) only in the narrow sky margin around the edges — physically sensible (it refines where there is flux, not blank sky), though notably **no pixel in this cutout is ever demoted below native** (no factor <1 subsampling fires) at any setting tried: the 80x80 cutout is small enough relative to the lens system that no region reads as safely-dark under these `snr_levels`.

Verdict: the adaptive supersampler's own quadrature is accurate and cheap on DESI257 — `factor8_opt_in` clears the library's tolerance at a fraction of uniform `supersample=16`'s cost. It is **not yet a ready production path for round 4**: (1) it is EXPERIMENTAL with an explicitly less-stable interface; (2) its bin-first-only PSF convolution does not match `r3_unmasked`'s subgrid convention, and the gap between them (~3.7 sigma) swamps any accuracy gained by grid refinement — closing it requires a separate, validated convention decision, not a `snr_levels` tweak; (3) the map is frozen from the DATA at construction, so a burn-in proposal or unseen model feature more than `dilate` pixels from where the data shows flux is silently rendered at the coarse tier — a real constraint for HMC exploration, not just initial calibration. Full per-setting numbers and the module's self-stated limitations are in `adaptive_supersample_report.txt`.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
    /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
    claude/DESI257/analysis/adaptive_supersample_check.py
```

| `theta_e_systematic.py` | first-order (linear-response) estimate of how far the ss=2 quadrature systematic shifts `r3_unmasked`'s posterior mean, via `diagnose_undersampling(ladder=[2], reference_supersample=64)` on N=60 HMC draws spread across all 50 chains/10000 steps (+MAP +HMC-median as extra points), a standardized 6-parameter (host mass `theta_E, gamma, e1, e2, center_x, center_y`) OLS regression of `delta(theta) = delta_chi2(theta)/2` against the parameters, and `Delta_theta = Sigma @ g` | whether the quoted `theta_E = 1.991 [1.989, 1.994]"` interval survives the already-measured ss=2 quadrature error, or must be widened |

`theta_e_systematic.py` reuses the same `reference_supersample=64` depth as `undersampling_recheck.py` (justified there: ss=2's number moved <5% between an ss_ref=32 and ss_ref=64 reference) and confirms the same subgrid-only self-check failure at this draw batch (max subgrid self-error 0.091 sigma vs the library's 0.05 sigma pass margin — **not certified**, reported anyway). Both the ss=2 rung and the ss=64 reference render under the confirmed-identical `subgrid` PSF convention, so this isolates the quadrature (grid-refinement) error and does **not** include the ~3.7 sigma bin-first-vs-subgrid PSF-convention gap measured separately.

Sanity checks passed: mean(`delta`) = -108.36 nats over the 60 draws vs the already-measured -221.1/2 = -110.55 nats (2.0% difference — consistent, same setup); the per-draw spread (std 29.2 nats in `delta_chi2` units) is the same order of magnitude as the previously-reported ~17-nat spread that already ruled out full importance reweighting (weight collapse). The regression itself is weak: **R² = 0.197** on 60 draws / 6 predictors — delta is *not* well-described as linear in these parameters over the posterior bulk. The point estimate it produces anyway is `Delta theta_E = +0.0106"` (+4.12 posterior sd, 5.3x the quoted ±0.002" half-width) and `Delta gamma = -0.0090` (-0.43 sd); the sign and rough magnitude are stable under a split-half check (first 30 vs last 30 draws: +0.0176" vs +0.0127", ratio 1.39, same sign), but a shift of >4 posterior sd is not "small" in the sense the first-order/linear-response expansion requires, so the low R² and the large shift are two independent reasons the same conclusion holds:

**this measurement does not resolve the question to a defensible number.** It rules out "no effect" (the sign is stable and the magnitude is consistently large across independent checks) but cannot certify *how much* theta_E moves — only that a real, currently-unquantified shift, likely larger than the quoted ±0.002" interval, is plausible. The interval should not be quoted as final without either a larger-N repeat of this regression or an actual refit at a finer supersample. Full regression table, both sanity checks, and the split-half table are in `theta_e_systematic.txt`; `theta_e_systematic.png` is the `delta` vs `theta_E` scatter with the (weak, R²=0.09) marginal OLS line — the honest picture of how little the univariate view alone would explain.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
    /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
    claude/DESI257/analysis/theta_e_systematic.py
```

Wall-clock: 345.6s (7 chunks of <=10 samples each, ~5.5s/sample) — chunked rather than one 62-sample batch call because a single call at `reference_supersample=64` risked exhausting memory on the shared login node (an 80x80 cutout renders at an effective 5120x5120 grid per sample at ss=64); chunking reused the exact batch size (10) already validated by `undersampling_check.py`'s timing check. Well inside the 30-minute budget; N was not reduced and the reference was not dropped to ss_ref=32.

**Correction to the paragraph above, 2026-08-11 (added after review, not by the script).** The
entry says the measurement "rules out 'no effect'". That is too strong, and the regression's
own overall significance test says so. At `R^2 = 0.1967` with n = 60 and 6 predictors,
`F(6,53) = 2.167`, **p = 0.061** — below the conventional 5% threshold, with adjusted
`R^2 = 0.106`, and `theta_E`'s own standardized coefficient at `t = +1.92` (also p ~ 0.06
two-sided). The fitted gradient is therefore **not statistically distinguishable from zero**
at 5%. The split-half sign agreement is weak corroboration on its own: two halves agreeing on
a sign is a coin flip a priori.

What survives is the weaker and still useful statement, and it is the one to quote:

> The ss=2 quadrature error implies a systematic on `theta_E` of **order 0.01"** — roughly 5x
> the +/-0.002" statistical half-width — with the shift's sign suggestive (positive) but
> marginal at p ~ 0.06. It is a systematic to state, not a correction to apply.

That is enough to settle the reporting question without a refit: round 3's interval is a
**statistical** interval, it is much narrower than the grid systematic, and it must not be
quoted as a total error bar. Nothing here licenses moving the central value.

---

| `core_checkerboard_extent.py` | radial profile of the deflector-core residual excess on all four latest+round-2 fits across DESI257 and DESI234, fixing the centre-mask radius by the radius enclosing 95% of the excess chi-squared above a **local** background, plus a lag-1 autocorrelation test of whether the excess actually alternates in sign | what radius the centre-masked `chi2_nu` (`KNOWN_ISSUES.md` item 5's last open piece) should use, and whether that radius is a property of the instrument or of the lens |

**The radius: 8 px = 0.520".** DESI257 `r3_unmasked` gives 7 px, DESI234 `r3_unmasked` gives
8 px, and the answer is insensitive to the one free choice (90% of the excess -> 7 / 6 px,
99% -> 8 / 8 px). The two systems agree to one pixel and all three systems are HST WFC3/IR
F140W at 0.065"/px, so an **instrument-level constant in arcsec is justified** rather than a
per-lens config field — the script tests that fork explicitly rather than assuming it. Take
the larger of the two so neither system is under-masked; 0.520" is ~3.7x the F140W PSF FWHM,
the right order for a PSF-scale artifact. Total excess chi-squared inside the disc: 659
(DESI257), 410 (DESI234).

**"Checkerboard" is the wrong name, and this is a correction to item 5's wording.** Item 5
describes a "pixel-scale alternating +/- sigma checkerboard". A checkerboard has *negative*
lag-1 autocorrelation between neighbouring pixels. Measured, it is positive in every case:

| run | supersample | core lag-1 rho | field lag-1 rho |
|---|---|---|---|
| DESI257 `r3_unmasked` | 2 | **+0.198** | +0.713 |
| DESI234 `r3_unmasked` | 4 | **+0.345** | +0.778 |
| DESI257 `r2_s2l2` | 2 | **+0.207** | +0.722 |
| DESI234 `r2_s2l2` | 2 | **+0.357** | +0.757 |

The round-2 runs are included precisely because item 5's description came from them
(each system's `RESULTS_v2.md`), and DESI234 changed supersample 2 -> 4 between the rounds —
so the pair tests both "was the description loose?" and "did something between the rounds
remove the alternation?". Neither: no run alternates, at either round or either supersample.
The core *is* measurably rougher than the field (rho falls from ~+0.75 to +0.20-0.35, so there
is real excess pixel-scale power there), but it is positively correlated — a smooth compact
excess, not a sign-flipping pattern. **Item 5's physical conclusion is untouched** (a PSF
artifact of Sersic deflector modelling; mask when reporting, not when modelling — which came
from field experience, not from this statistic). Only the name and the round-2 amplitude
description should stop being requoted as measured.

**Two criteria were tried and rejected before this one; both are recorded in the script's
docstring, because each failed for a reason worth not repeating.** (1) A 3-sigma test on each
annulus's mean squared residual is powerless — squared residuals are heavy-tailed, so a
9-pixel annulus reads `8.56 +/- 4.13` and the test rejected an excess plainly visible in the
profile, collapsing the answer to 0 px. (2) A *far-field* background annulus (34-46 px =
2.2-3.0") reads mean `(r/sigma)^2 = 1.16` on DESI257, essentially the whole-field
`chi2_nu = 1.154`, because it is a fair sample of a field that includes the arcs' outskirts;
meanwhile the 7-19 px shoulder reads 0.4-0.9, genuinely *quieter* than the field average. Against
that baseline the cumulative excess peaked at r=6 and then fell back, so "the radius enclosing
95% of the excess" had no plateau to be a fraction of. A local background (8-20 px) fixes it and
is the right reference for a localised artifact.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=claude/pipeline \
    /global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python \
    claude/DESI257/analysis/core_checkerboard_extent.py \
    > claude/DESI257/analysis/core_checkerboard_extent.txt 2>&1
```
