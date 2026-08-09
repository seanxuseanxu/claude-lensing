# DESI257

HST WFC3/IR F140W imaging of DESI-257.4348+31.9046. This file records the data and the
conditioning decisions, so that when a number does appear it is clear what it was
conditioned on. **For results, see `RESULTS_v2.md`.**

**Current result (round 2, `r2_s2l2`): `theta_E = 1.9811 (-0.0041,+0.0042)"` in the dominant
posterior mode (73.4% of draws, within-mode R-hat 1.0092, unreplicated).** The full 50-chain
posterior is multimodal and its combined R-hat fails outright (worst 1.461) — see
`RESULTS_v2.md` section 3 before quoting any number from this system. Round 1 (`r1_s1l1`, one
source component) did not converge (worst R-hat 1.878) and pinned its single source Sersic
`n_sersic` at its prior ceiling (6.0, sd exactly 0.0); round 2's second source component
resolves that pinning entirely.

**Read "The open question" below before quoting any theta_E from this system.** Its
counter-image identification is the least secure of the three lenses modelled here, and the
ambiguity is documented rather than resolved. `RESULTS_v2.md` section 5 reports a blind
prediction test on the masked SW object — suggestive, not conclusive — that bears on this
question without resolving it.

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

### One masked object, at 2.58", PA 218

A smooth, centrally-concentrated ellipse at crop (8, 15) peaks at 0.957 e/s -- 131 sigma --
and dominates the lower-left corner. It is the brightest thing in the field after the
deflector.

It is masked with a 12 px disc: its residual profile falls through 1 sigma between r = 10 and
12 px, the disc excludes 403 of 6400 pixels (6.3%), and the nearest catalogued clump (the E
arc's northern end) is 45 px from the disc edge. Masking takes the lower-left corner from
0.957 to 0.079 e/s.

Leaving it unmasked is not an option under the interloper reading: the source components'
priors put no constraint on where a component may sit, so a free Sersic will go and sit on a
131-sigma blob. That is the cheapest way for the model to buy chi-squared.

**Whether masking it is the right call is the open question below.**

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

### The open question

Three images invite a fourth, and PA 218 -- where the masked SW object sits -- is close to
where a quad would put it. I read it as an interloper, on three grounds:

- **Morphology.** A smooth, centrally-concentrated ellipse with a bright core, which is what
  an ordinary galaxy looks like and not what a lensed knot looks like.
- **Brightness ordering.** At 131 sigma it is 2x brighter than the brightest arc pixel while
  sitting 1.4x farther out. A demagnified fourth image is the *faintest*, not the brightest.
- **Orientation.** Its major axis is 18 degrees off tangential, where the E arc's is within
  1-2 degrees.

**None of those is decisive on its own.** If the reading is wrong, theta_E is biased *low*:
fitting the SW object as an image would demand a critical curve out near 2.1-2.2" rather than
1.70".

Two things follow, and both are already in `config.py`:

1. `theta_E_logsigma = 0.35` -- DESI238's, and wider than DESI234's 0.25 -- putting the 1-99%
   range at 0.75-3.85" specifically so the prior's support covers both readings and the data
   is allowed to prefer the other one.
2. **Round 5 fits the SW object as a fourth image, unmasked, and compares.** That fit has not
   been run. Until it exists, every theta_E from this system carries the interloper reading as
   an assumption and must be reported with it.

### Blind-prediction test result (round 2, suggestive, not conclusive)

Round 2's fit never saw the SW object -- it is masked out entirely -- so its dominant-mode
posterior can be used to *predict* whether any flux should land there, without that prediction
having been able to influence the fit. `RESULTS_v2.md` section 5 runs this test through
`gigalens`'s `LensSolver`, validated first against the four known images (flux recovered at
78-101%, positions to <0.1"), then applied to the SW object:

- The weaker source component's predicted image lands **1.40 px = 0.091"** from the SW
  object's measured flux centroid -- inside one PSF FWHM (~0.15").
- The stronger source component's predicted image misses by **6.47 px = 0.421"**.
- **The two source components disagree**, and only a 2-image topology is found at this point
  estimate, not the 4-image topology the fourth-image hypothesis requires.
- `theta_E = 1.981 +/- 0.004"` sits *between* the interloper prediction (~1.70") and the
  fourth-image prediction (~2.1-2.2") and does not cleanly select either reading.

**Read together: near-miss on one source component, clear miss on the other, wrong image
topology, no posterior propagation yet.** This nudges the question open rather than closing
it -- it is not evidence strong enough to unmask the SW object and call it resolved, but it is
not nothing either. Next step: posterior-propagate the test (draws, not one median) before
treating 0.091" as anything more than a point number.

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
- [ ] `verify.py --lens DESI257` passing -- not (re-)confirmed this round
- [x] Round 1 fit (`r1_s1l1`, 22 params) -- did not converge (worst R-hat 1.878); single
      source Sersic pinned at its prior ceiling. Quoted only for comparison; see `RESULTS_v2.md`.
- [x] Round 2 fit (`r2_s2l2`, 36 params) -- **multimodal**; dominant-mode result reported with
      its convergence caveat prominent. See `RESULTS_v2.md`.
- [ ] Modes mixing honestly -- the 80,000-draw attempt to force this **failed** (every chain
      froze; cause unattributed). See `RESULTS_v2.md` section 4.
- [ ] Round 5 falsification run on the SW object (fit as a fourth image, unmasked) -- not run.
      A posterior-based blind-prediction test was run instead, using the existing masked fit;
      see above and `RESULTS_v2.md` section 5.

`verify.py`'s check 6 (the SVI guard) will report itself **skipped** until round 1 gives this
lens its own known-good and known-diverged runs. A skipped check is honest; it must not be
weakened to pass.
