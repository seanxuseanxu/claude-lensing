# DESI257

HST WFC3/IR F140W imaging of DESI-257.4348+31.9046. This file records the data and the
conditioning decisions, so that when a number does appear it is clear what it was
conditioned on. **For results, see `RESULTS_v3.md`** (round 3); `RESULTS_v2.md` is superseded
and carries a retraction banner.

> **Current result — round 3 (`r3_unmasked`), unimodal, unmasked, 2026-08-09:**
>
> | | |
> |---|---|
> | `theta_E` | **1.991 [1.989, 1.994]" (statistical)**, with a **grid systematic of order 0.01"** — see the third caveat |
> | `gamma` | 1.986 [1.965, 2.007] (statistical) |
> | \|shear\| | 0.044 [0.040, 0.048] (statistical) |
> | convergence | R-hat <= 1.001 on **all 36 params, all 50 chains, no chain selection**; min bulk-ESS 70,858 |
> | `chi2_nu` | **1.065** centre-masked (6198 px, core disc of 0.520" excluded) — the number to quote; 1.154 over all 6400 px |
>
> Round 2's masked, bimodal fit is **retracted** — its "dominant" mode's 73.4% draw fraction
> measured where the chains were seeded, and the mask rested on a physics error. Unmasking the
> SW object closed the bimodality outright and relaxed `gamma` (2.54 -> 1.99), `|shear|`
> (0.148 -> 0.044) and `|e_host|` (0.380 -> 0.143) simultaneously.
>
> **Three caveats before quoting `theta_E`.** The MAP still finishes 18.1 nats below the best
> HMC draw, so `diagnose.map_convergence_check` is red (down from round 2's 681.8). And the
> PSF is unrebuilt: the deflector core is the worst-fitting region in the image
> (`chi2_nu = 2.16`), which is `KNOWN_ISSUES.md` item 5. This is a result, not yet a
> publishable number.
>
> **The third caveat is the largest, and it is new as of 2026-08-11: the quoted interval is a
> statistical interval much narrower than a measured numerical systematic.** This fit renders
> at `supersample = 2`, which **fails** the library's 0.2 sigma/px quadrature tolerance by 18x
> (max 3.697 sigma/px, mean `delta_chi2` = -216 over 6400 px;
> `analysis/undersampling_report.txt`). A first-order estimate of what that costs `theta_E`
> gives a shift of **order 0.01"** — about 5x the +/-0.002" half-width above — though at
> `p = 0.061` it is statistically marginal, so it is a systematic to **state, not a correction
> to apply** (`analysis/theta_e_systematic.txt`, and the review correction appended to
> `analysis/README.md`). **Quote `theta_E = 1.99"`; the third decimal is not defensible.**
>
> The grid was nonetheless left at ss=2 deliberately, and the alternatives were measured
> rather than assumed: ss=1 is worse (5.290 sigma/px) *and* crosses the PSF-convolution
> convention boundary; ss=4 (1.497) and ss=8 (0.439) both still fail, so raising it is not a
> fix; and the experimental adaptive supersampler reaches tolerance cheaply but only in a
> bin-first PSF convention, whose ~3.7 sigma gap against this fit's subgrid convention is a
> separate modelling decision needing its own validation. ss=2 is the most-tested pathway and
> the one this result stands on. See `RESULTS_v3.md` section 8.

Round 1 (`r1_s1l1`, one source component) did not converge (worst R-hat 1.878) and pinned its
single source Sersic `n_sersic` at its prior ceiling (6.0, sd exactly 0.0); round 2's second
source component resolves that pinning entirely. That conclusion stands — it is a measurement
about the source light, and does not depend on the mask.

**Read "The SW object" below before quoting any theta_E from this system.** Its
counter-image identification was the least secure of the three lenses modelled here; round 3
settled it in favour of the fourth-image reading, and that section records both the retracted
argument and the measurement that retracted it.

**Independently confirmed, 2026-08-10, by the repository owner:** the SW object is a
counter-image — it is at the expected position for one, is about the same brightness as the
other images, and the model predicts it. That is three lines of agreement with round 3's
measurement, arrived at without reference to it. This identification is no longer the weak
point of this system and should not be re-litigated: the standing advice is *don't fight a good
model*. Where a model reproduces an object it was not fitted to, that is the model earning
trust, not a coincidence to be explained away.

Everything below is measured from this system's own files. Nothing is inherited from DESI238
except the *structure* of `config.py` and two prior widths that are explicitly flagged as
unmeasured (`SOURCE_PRIOR`'s scales, and the bulge/disk split).

## The data

| | |
|---|---|
| Parent frame | `final_99_drz.fits` (1997 x 2258, HDU 1 = SCI, HDU 2 = WHT) |
| Filter | F140W |
| EXPTIME | 1197.699462 s, NCOMBINE 3 |
| Pixel scale | 0.065000 "/px, from `sqrt(\|det CD\|) * 3600` on the SCI CD matrix |
| Cutout | `cutout257.npy`, 120 x 120, parent slice `[1008:1128, 1170:1290]` |
| Crop used for fitting | `[20:100, 20:100]` -> 80 x 80 |
| PSF | `psf257.fits`, 27 x 27, stacked from 21 stars (`psf257.json`) |
| Peak | 8.4677 e/s (the deflector core, at crop (39, 39)) |

`cutout257.json` and `psf257.json` record the parent frame's sha256
(`554dceca...6cfedd5d`), the exact slice, and every threshold used, so both products are
reproducible from `claude/pipeline/make_cutout.py` and `make_psf.py`.

### Why the cutout is centred on light, not on the catalogue position

The deflector centroid is 3.9 px from where `all_world2pix` puts the catalogue RA/Dec.
`make_cutout.py` centres on the measured centroid (cutout (59.397, 59.236)) instead. The crop
then places the deflector at (39.40, 39.24) against an array centre of (39.5, 39.5) -- 0.10 px
in row, 0.26 px in column, i.e. 0.017". That is a third of `MASS_PRIOR`'s `center_sigma`, and
better centred than DESI234. Shifting the data to do better would correlate neighbouring
pixels and invalidate the independent-noise likelihood.

### Exposure map

The WHT extension is `BUNIT = UNITLESS`, so it is a weight, not an exposure map. Over the
81x81 box around the lens its median is 99.80% of its maximum, so coverage is uniform there
and a scalar `exp_time` is right to first order. That is a measurement, not an assumption.

## Conditioning decisions

### Background RMS: 0.0073 e/s

200 random 40x40 patches in the *parent* frame, rejecting any patch containing a pixel above
0.15 e/s or whose WHT median is below half the frame's. A single seed's median is not a
measurement, so the estimator was run over five:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| median | 0.00732 | 0.00719 | 0.00736 | 0.00725 | 0.00736 |

The 16-84th percentile *within* a seed is 0.0067-0.0093. So the patch-to-patch spread
(+/-15%) dwarfs the seed-to-seed spread (+/-1%), and 0.0073 is the frame's robust median.

The rule is deliberately **self-contained per frame** -- no cross-frame rescaling -- so that
DESI234 and DESI257 are conditioned identically and neither inherits a judgement made on
DESI238. The same estimator returns 0.0076 on DESI234's frame and 0.0078 on DESI238's,
against the 0.0075 adopted there by hand.

What that costs, stated rather than hidden: +/-15% in sigma is +/-30% in chi-squared, larger
than any difference between the frames. **A reduced chi-squared near 1 is not by itself
evidence that the noise model is right.**

`sigma_clipped_stats` over the cutout is the documented trap: the cutout is mostly galaxy,
the clip never lands on sky, and the answer comes out several times too large -- which would
down-weight the arc into irrelevance.

### No sky subtraction

The azimuthal median at r = 45-55 px reads 0.043 e/s. That is the deflector's own Sersic
wings, not a sky pedestal -- true blank sky in this frame is consistent with zero.
Subtracting it would remove real galaxy light and bias the lens-light `R_sersic` and
`n_sersic` low. The Sersic profile is there to model those wings.

### The SW object at 2.58", PA 218: masked in rounds 1-2, unmasked from round 3

A smooth, centrally-concentrated ellipse at crop (8, 15) peaks at 0.957 e/s -- 131 sigma --
and dominates the lower-left corner. It is the brightest thing in the field after the
deflector.

Rounds 1 and 2 masked it with a 12 px disc, excluding 403 of 6400 pixels (6.3%). **That mask
is removed from round 3 onward**, because the argument that carried it was wrong. The
retraction and the evidence are below, under "The SW object". Masking a real image discards
the field's most constraining datum, so this is not a small change: it is the difference
between asking the model to reproduce an exotic 3-image configuration and asking it to
reproduce the generic 4-image one.

The masked rounds stay readable. `run_fit.py` records `mask_circles` and `n_masked_pixels`
per run and `diagnose.rebuild_model` rebuilds each run's mask from its own record, not from
today's config (`claude/pipeline/KNOWN_ISSUES.md` item 8), so `r1_s1l1` and `r2_s2l2` are
still diagnosed against the likelihood they were actually fitted with -- and `diagnose.py`
prints a `NOTE:` saying so.

The hazard that motivated the mask is real: the source components' priors put no constraint
on where a component may sit, so a free Sersic can go and sit on a 131-sigma blob without
lensing anything. It is now checked rather than avoided. What distinguishes the two cases is
whether the SW object is produced by the **same** source components and the **same** mass
model that produce the E arc and the N blob -- which is what a four-image configuration means
-- or by a component that has wandered off to sit on it alone. The residual over the
previously-masked disc, and the fitted source-plane geometry, answer that directly.

## Lens configuration

Measured on the cutout after subtracting a point-symmetric reflection of the image about the
deflector centroid. (Not an azimuthal median: this deflector is genuinely elongated,
|e| = 0.137 at array-space major-axis PA -51 deg, so a circular median would leave the light's
own major axis behind as a spurious residual.)

| clump | r (") | PA (deg, array-space) | peak (sigma) | \|e\| | major axis vs tangential |
|---|---|---|---|---|---|
| E arc S | 1.90 | 60 | 63 | 0.25-0.35 | 1-2 deg |
| E arc mid | 1.80 | 76 | 87 | 0.25-0.35 | 1-2 deg |
| E arc N | 1.86 | 104 | 56 | 0.25-0.35 | 1-2 deg |
| N blob | 1.55 | 350 | 149 | 0.125 | 22 deg |

**These PAs are array-space angles**, measured from the +row axis toward +col in the crop --
not sky PA. Converting them through the WCS the standard way (North through East) puts the
markers on empty sky. This is the single most misreadable field in `config.py`.

### What is secure

The E arc. Continuous from PA 40 to PA 110 at a radius that stays within 1.76-1.92" the whole
way, elongated |e| = 0.25-0.35 with its major axis within 1-2 degrees of tangential
everywhere along it. That is what lensing does.

The N blob at PA 350 is a **separate image**, not the same arc continued: the azimuthal trace
between them drops to 3-7 sigma over PA 15-35 -- a genuine flux gap, not a dip. It is also
rounder (|e| = 0.125) and 22 degrees off tangential, as a merging pair near the minor axis
should be.

Two images at 1.85" and 1.55" is already a lens. Taking theta_E as the mean gives **1.70"**.

### The SW object: the interloper reading, retracted

Three images invite a fourth, and PA 218 -- where the SW object sits -- is close to where a
quad would put it. Rounds 1 and 2 read it as an interloper on three grounds. Taking them in
order of how much weight they were carrying:

- **Brightness ordering — RETRACTED. This is a physics error, not a judgement call.** The
  argument ran: "at 131 sigma it is 2x brighter than the brightest arc pixel while sitting
  1.4x farther out; a demagnified fourth image is the *faintest*, not the brightest." A
  demagnified image requires a **radial** critical curve, and a radially demagnified image
  forms *near that curve* — close to the deflector centre, not 2.58" out. Round 3 fits
  `gamma = 1.986 [1.965, 2.007]`, and `LensSolver` does find a radial critical curve, at
  **0.0278"**. So this lens does have one, and it is buried in the deflector core; nothing
  at 2.58" can be the demagnified image it produces. All four images of a quad lie near the
  **tangential** critical curve and are magnified. The brightness ordering does not
  discriminate between the two readings in either direction.

  *(An earlier form of this retraction argued that `gamma > 2` in both round-2 modes means
  there is no radial critical curve at all. Round 3's unmasked fit has `gamma < 2` with
  probability 0.75, so that premise is false for the fit that matters, and the retraction is
  restated above on the grounds that survive it. See `RESULTS_v3.md` §5.2.)*
- **Orientation — downgraded.** Its major axis is 18 degrees off tangential, where the E
  arc's is within 1-2 degrees. But strong tangential elongation is a property of images near
  a fold *of an extended arc*; a counter-image of a compact knot need not show it. This is
  weak evidence at best.
- **Morphology — all that survives.** A smooth, centrally-concentrated ellipse with a bright
  core. A compact image of a compact source also looks compact, so this discriminates only
  weakly.

Against them, four pieces of positive evidence for the fourth-image reading, three of them
visible in `runs/r2_s2l2/residuals.png`:

1. **The round-2 model predicts flux at the SW position while blind to it.** Those 403 pixels
   were masked out of the likelihood, and panel 2 still renders a bright SW blob comparable
   to the N blob. `RESULTS_v2.md` §5.2's component-by-component rendering identifies it as
   *lensed source light* (plane 1, component 1, peaking at crop (6, 17) at 1.12 e/s), not
   lens-light wings: the disk component is roughly 30x fainter there.
2. **Both source components straddle the tangential caustic, so part of the source is
   quadruply imaged.** Measured, not eyeballed (`analysis/caustic_flux_fraction.py`, 300
   draws from the dominant mode): **8.1% [7.6, 8.7] of `src0`'s flux and 17.6%
   [16.8, 18.7] of `src1`'s flux lie inside the tangential caustic**. Both *centroids* lie
   outside it in 300/300 draws — `src1`'s by only 0.081", `src0`'s by 0.338" — which is
   why a point-source solve at either centroid returns 2 images. A Sersic is not a point.
3. **The tangential critical curve passes right by the SW object's position** (panel 1).
4. **A 3-image tangential configuration is exotic**; 4 is the generic case for a source
   straddling the caustic, which this one measurably does.

**A fifth piece of evidence, from round 3 itself: the unmasked fit reproduces the object.**
With those 403 pixels in the likelihood and no component added for them, the model puts
**0.9688 e/s** where the data has **0.9572 e/s** — 1.2% on the peak of a 131-sigma object —
with a mean normalised residual of `-0.0014` sigma across the disc, and `chi2_nu` there (2.05)
indistinguishable from the E arc's (1.96) and better than the deflector core's (2.16).
`analysis/sw_region_chisq.py`, `RESULTS_v3.md` §5.1.

**`theta_E` was cited here as evidence and is not.** The argument ran: the interloper reading
predicts `theta_E ~ 1.70"`, the fourth-image reading `~2.1-2.2"`, and the masked fit's
1.98-2.00" is nearer the latter. Round 3, unmasked, gives **1.991 [1.989, 1.994]"** — it moved
by 0.008", less than round 2's own interval, and it is not 2.1-2.2" either. `theta_E` is
robust to the mask, which is good for the number and means it never discriminated between the
readings. Do not cite it as support. (`RESULTS_v3.md` §5.2.)

This is also the leading explanation for round 2's bimodality. Suppressing an image that
wants to exist is expensive, and mode B's parameters are what it costs: host mass
`e1 = -0.591`, `e2 = -0.257` (|e| = 0.645, q ~ 0.22), `gamma = 2.836`, `|shear| = 0.195` --
three extremes at once, the last sitting exactly on the plausibility flag `priors.py` raises
at model-build time. The mask is a better candidate for the cause than the PSF.

`MASS_PRIOR`'s `theta_E_logsigma = 0.35` -- DESI238's, wider than DESI234's 0.25 -- puts the
prior's 1-99% range at 0.75-3.85", so the support covers both readings and the data is
allowed to choose. That decision was made before any of this and it is what makes the refit
possible without touching a bound.

**Round 3 refits unmasked**, one variable changed against `r2_s2l2`. Written down before the
fit, the fourth-image hypothesis predicts: the SW object is reproduced by the *existing*
source components, with no new component and no new mass, at good chi-squared over the
newly-unmasked 403 pixels; `gamma`, host ellipticity and `|shear|` relax toward plausible
values; and the bimodality weakens or closes. If instead the chi-squared over those pixels is
poor, or the mass parameters go *more* extreme, the object is a foreground galaxy and the
mask was right. **Round 3b then fits the interloper hypothesis properly** -- unmasked, with a
foreground SIS deflector and its own light at the SW position -- because an interloper
deflects whether or not its light is in the likelihood, and masking was never a neutral
choice between the two readings.

### Blind-prediction test result (round 2, suggestive, not conclusive)

Round 2's fit never saw the SW object -- it is masked out entirely -- so its dominant-mode
posterior can be used to *predict* whether any flux should land there, without that prediction
having been able to influence the fit. `RESULTS_v2.md` section 5 runs this test through
`gigalens`'s `LensSolver`, validated first against the four known images (flux recovered at
78-101%, positions to <0.1"), then applied to the SW object:

- The weaker source component's predicted image lands **1.40 px = 0.091"** from the SW
  object's measured flux centroid -- inside one PSF FWHM (~0.15").
- The stronger source component's predicted image misses by **6.47 px = 0.421"**.
- **The two source components disagree.**
- ~~only a 2-image topology is found, not the 4-image topology the fourth-image hypothesis
  requires~~ — **withdrawn.** That was a point-source solve at each component's *centroid*,
  and both centroids lie outside the caustic, so 2 images is the correct answer to the wrong
  question. Integrating each component's own light over the caustic interior gives 8.1% and
  17.6% of its flux inside, in 300/300 draws: part of the source is quadruply imaged.
  (`analysis/caustic_flux_fraction.py`, `RESULTS_v2.md` §5.3.)
- ~~`theta_E = 1.981 +/- 0.004"` sits between the interloper (~1.70") and fourth-image
  (~2.1-2.2") predictions~~ — **withdrawn.** The unmasked fit gives 1.991", moving 0.008".
  `theta_E` does not discriminate between the readings. (`RESULTS_v3.md` §5.2.)

**Superseded by round 3.** This test was the best available while the object was masked, and
it was rightly called suggestive rather than conclusive. Round 3 fits those pixels directly
and answers the question: see the current-result banner at the top of this file.

## Flux scale

The simulator multiplies its output by `delta_pix**2 = 0.004225`, so `Ie` is of order 10-100,
not order 1. A value copied from another lens can be 50x wrong, so both are calibrated here
by rendering 200 draws from this system's own prior and comparing the median rendered peak
with the data (measured 2026-08-08, `n_source=1, n_lens_light=1`):

| | median rendered peak | 16-84th pct | data | scale | adopted |
|---|---|---|---|---|---|
| lens light | 8.231 e/s | 3.81-21.6 | 8.47 e/s (core) | 1.029 | `LENS_IE = 82` |
| lensed source | 0.5425 e/s | 0.110-2.62 | 0.64 e/s (arc) | 1.18 | `SOURCE_IE = 24` |

**The median is a weak anchor.** The LogNormal `Ie` prior's 16-84th percentile spans nearly
two orders of magnitude, so this fixes the prior's *centre* to within a factor of a few, not
better. That the provisional placeholders (80/20) came out within 20% of the calibrated
values is coincidence, not a validation of guessing.

## Status

- [x] Cutout and PSF built, both with reproducible sidecars
- [x] `config.py` written, every number measured on this system
- [x] `LENS_IE = 82` / `SOURCE_IE = 24` calibrated by prior-draw render (see below)
- [x] `verify.py --lens DESI257` re-confirmed 2026-08-11: **4/5 passed, 1 skipped.** Checks 1-4
      (data conditioning, parameter count/ordering, simulator flux scale, short end-to-end
      MAP->SVI->HMC) pass; check 5, the SVI covariance guard, reports itself **skipped**
      because this lens config names no `svi_guard_reference` pair. The suite used to print
      this as "5/5 passed" -- it counted the skip as a pass. Fixed the same day
      (`verify.summarize`, tested in `tests/test_verify_summary.py`): a skipped check has
      established nothing and is now reported as its own category.
- [x] Round 1 fit (`r1_s1l1`, 22 params) -- did not converge (worst R-hat 1.878); single
      source Sersic pinned at its prior ceiling. Quoted only for comparison; see `RESULTS_v2.md`.
- [x] Round 2 fit (`r2_s2l2`, 36 params) -- **multimodal, and its headline is retracted.** The
      "dominant mode" holding 73.4% of draws measures where the chains were seeded, not where
      the posterior mass is: 48 of 50 chains start inside it, because the SVI surrogate mean
      does (`analysis/mode_seeding.py`). The deeper mode is the other one. See `RESULTS_v2.md`
      section 3.4. **Nothing from this run is quotable**; round 3 supersedes it.
- [ ] Modes mixing honestly -- the 80,000-draw attempt to force this **failed** (every chain
      froze; cause unattributed). See `RESULTS_v2.md` section 4.
- [x] The MAP step count fixed by measurement rather than inherited -- round 2's MAP finished
      681.8 nats below the best draw of the chain it seeded, which is what put every chain in
      one mode. `RESULTS_v3.md` section 3.
- [x] Round 3 (`r3_unmasked`, 36 params) -- the SW object unmasked and fitted as the fourth
      image, one variable against `r2_s2l2`, with the predictions recorded in advance in
      `RESULTS_v3.md` section 5. **Unimodal, R-hat <= 1.001 on all 50 chains with no
      selection.** The SW object is reproduced by the existing source components to 1.2% on
      its peak flux, with no component added for it, and `gamma`, `|e_host|` and `|shear|`
      relax together. `RESULTS_v3.md`.
- [~] Round 3b (`r3b`) -- **built, tested (`--foreground`, 46 params), deliberately not run.**
      It was justified by the SW disc's `chi2_nu`; measured against equal discs on the other
      images that `chi2_nu` is unremarkable, so there is no outcome in doubt for it to
      resolve. It would still give an upper bound on any foreground deflector's mass, which is
      worth having but is not a test. `RESULTS_v3.md` section 6.
- [x] PSF rebuild (`KNOWN_ISSUES.md` item 5) -- **withdrawn, not done.** The deflector core is
      the worst-fitting region in the image (`chi2_nu` 2.16 against the field's 1.15) and shows
      an excess residual there, which was read here as the leading defect. It is not:
      a pixel-scale core residual is an ordinary artifact of Sersic deflector modelling, and
      the field's convention is to mask the centre when *reporting* chi-squared and not when
      modelling. DESI234 rebuilt its PSF and the excess survived it.
      (This entry used to call the feature a "checkerboard". Measured, it does **not** alternate
      in sign -- core lag-1 autocorrelation is +0.20 here and positive on all four fits tested.
      `KNOWN_ISSUES.md` item 5's naming correction.)
- [x] Centre-masked `chi2_nu`, which was what item 5 owed instead -- **done 2026-08-11.**
      `chi2_nu` = **1.065** over 6198 px with a 0.520" core disc excluded, against 1.154 over all
      6400. The radius was measured, not chosen (`analysis/core_checkerboard_extent.py`); the core
      pixels remain in the likelihood. `RESULTS_v3.md` section 8.6.
- [ ] MAP exactly at the mode -- round 3's MAP is 18.1 nats below its best HMC draw
      (round 2's was 681.8). The 20,000-step sweep point that would have tested whether a
      longer optimisation closes that gap **completed and was then lost** with the `/tmp`
      scratch it wrote to, and was not re-run: it is a diagnostic, not a gate, and round 3
      uses the 5,000-step checkpoint either way. `runs/mapsweep/README.md`.

`verify.py`'s SVI guard (**check 5** as the suite is numbered today — this file said "check 6",
which was stale) will report itself **skipped** until this lens is given its own known-good and
known-diverged runs in `config.py`'s `svi_guard_reference`. A skipped check is honest; it must
not be weakened to pass. As of 2026-08-11 the suite's summary line reports skips separately
instead of folding them into the pass count, so this one is visible rather than implied.
