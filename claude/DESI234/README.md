# DESI234

HST WFC3/IR F140W imaging of DESI-234.4783+14.7232. **No fit has been run yet.** This file
records the data and the conditioning decisions, so that when a number does appear it is
clear what it was conditioned on.

Everything below is measured from this system's own files. Nothing is inherited from
DESI238 except the *structure* of `config.py` and two prior widths that are explicitly
flagged as unmeasured (`SOURCE_PRIOR`'s scales, and the bulge/disk split).

## The data

| | |
|---|---|
| Parent frame | `final_92_drz.fits` (1997 x 2258, HDU 1 = SCI, HDU 2 = WHT) |
| Filter | F140W |
| EXPTIME | 1197.699 s, NCOMBINE 3 |
| Pixel scale | 0.06500 "/px, from `sqrt(|det CD|) * 3600` on the SCI CD matrix |
| Cutout | `cutout234.npy`, 120 x 120, parent slice `[1008:1128, 1170:1290]`-equivalent (see `cutout234.json`) |
| Crop used for fitting | `[20:100, 20:100]` -> 80 x 80 |
| PSF | `psf234.fits`, 27 x 27, stacked from 16 stars (`psf234.json`) |
| Peak | 6.63 e/s (the deflector core) |

`cutout234.json` and `psf234.json` record the parent frame's sha256, the exact slice, and
every threshold used, so both products are reproducible from `claude/pipeline/make_cutout.py`
and `make_psf.py`.

### Why the cutout is centred on light, not on the catalogue position

The deflector centroid is 3.4 px from where `all_world2pix` puts the catalogue RA/Dec.
`make_cutout.py` centres on the measured centroid instead. The crop then places the
deflector at (39.13, 39.47) against an array centre of (39.5, 39.5) -- 0.37 px in row,
0.03 px in column. The row residual is 0.024", half of `MASS_PRIOR`'s `center_sigma`, so the
deflector-centre prior still covers the truth comfortably. No integer crop does better, and
shifting the data to do so would correlate neighbouring pixels and invalidate the
independent-noise likelihood.

## Conditioning decisions

### Background RMS: 0.0075 e/s

200 random 40x40 patches in the *parent* frame, rejecting any patch containing a pixel above
0.15 e/s or whose WHT median is below half the frame's. A single seed's median is not a
measurement, so the estimator was run over five:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| median | 0.00756 | 0.00743 | 0.00746 | 0.00752 | 0.00767 |

The 16-84th percentile *within* a seed is 0.0069-0.0100. So the patch-to-patch spread
(+/-15%) dwarfs the seed-to-seed spread (+/-1.5%), and 0.0075 is the frame's robust median.

The rule is deliberately **self-contained per frame** -- no cross-frame rescaling -- so that
DESI234 and DESI257 are conditioned identically and neither inherits a judgement made on
DESI238. What that costs: the same estimator returns 0.0078 on DESI238's frame against the
0.0075 adopted there by hand, so it reads ~4% high, and rescaling would give 0.0072 here.
4% in sigma is 8% in chi-squared, smaller than the +/-15% patch-to-patch spread the
measurement carries anyway. **A reduced chi-squared near 1 is not by itself evidence that
the noise model is right.**

`sigma_clipped_stats` over the cutout is the documented trap: the cutout is mostly galaxy,
the clip never lands on sky, and the answer comes out several times too large -- which would
down-weight the arc into irrelevance.

### No sky subtraction

The cutout's clean corners read 0.004-0.015 e/s. That is the deflector's own Sersic wings
plus its neighbours, not a sky pedestal -- true blank sky in this frame is consistent with
zero. Subtracting it would remove real galaxy light and bias the lens-light `R_sersic` and
`n_sersic` low. The Sersic profile is there to model those wings.

### One masked interloper, at 2.30", PA 314

A compact source at crop (64, 14) peaks at 0.82 e/s -- 110 sigma above the subtracted
deflector light -- and is the brightest thing in the field after the deflector.

**No square crop removes it without clipping the arc.** For a box of half-width h centred on
the deflector, the interloper is inside whenever h > 25.5 px, while the arc runs out to
h = 26.5 px. The two requirements do not overlap, so per `LENS_MODELING_GUIDE.md` the options
are mask or model, and cropping is not among them.

Rounds 1-2 mask it, radius 12 px: its residual profile falls through 1 sigma between r = 11
and 12 px, and the nearest arc pixel is 48 px away. That costs 441 of 6400 pixels (6.9%) and,
more importantly, declines to use the interloper's own light. **Round 3 replaces the mask
with a modelled Sersic and compares.** Until then the fit carries an unmodelled interloper as
an accepted systematic and must be reported as carrying one.

Leaving it unmasked is not an option: the source components' priors put no constraint on
where a component may sit, so a free Sersic will go and sit on a 110-sigma blob. That is the
cheapest way for the model to buy chi-squared.

## Lens configuration

Read as a **cusp quad**. After subtracting an elliptical-isophote median about the deflector:

| clump | r (") | PA (deg, array-space) | peak (sigma) | \|e\| | major axis vs tangential |
|---|---|---|---|---|---|
| S arc, E end | 1.81 | 145 | 85 | 0.31 | 2.5 deg |
| S arc, mid | 1.71 | 184 | 62 | 0.39 | 2.9 deg |
| S arc, W end | 1.74 | 215 | 57 | 0.40 | 3.7 deg |
| N counter | 1.11 | 358 | 37 | 0.05 | -- (round) |

**These PAs are array-space angles**, measured from the +row axis toward +col in the crop --
not sky PA. Converting them through the WCS the standard way (North through East) puts the
markers on empty sky. This is the single most misreadable field in `config.py`.

The southern structure is continuous from PA 125 to PA 235 -- 110 degrees -- at a radius that
stays within 1.66-1.82" the whole way, with three local maxima and no gap between them, each
elongated to within 4 degrees of tangential. A chance alignment of three galaxies does not
produce a constant-radius, tangentially-elongated, unbroken ridge over 110 degrees.

Three merging images along one arc with a fourth opposite is a cusp configuration, and the
fourth is where a cusp quad puts it: the N clump at PA 358, opposite the arc's midpoint to
within 5 degrees, at the smaller radius 1.11", and the faintest of the four. Taking theta_E
as the mean of the two radii gives **1.43"**, with `theta_E_logsigma = 0.25` (1-99% range
0.80-2.55").

The competing reading -- that the southern structure is a single fold pair and the three
maxima are source structure -- is not excluded by the image and does not move theta_E much,
since both put the tangential critical curve between 1.11" and 1.75". It changes the *source*
model, which is what round 2's ladder tests.

The interloper at 2.30" is deliberately **not** in the clump list, and that is a judgement.
Against it being a fifth image: it is round (|e| = 0.036) with its major axis 66 degrees from
tangential; at 3.5x the counter-image's brightness where a fifth image should be faintest;
and at PA 314 it sits 44 degrees from the counter-image on the same side, which no standard
quad geometry produces. If that is wrong, theta_E is biased low. **Round 5 is the
falsification run.**

## Flux scale

The simulator multiplies its output by `delta_pix**2 = 0.004225`, so `Ie` is of order 10-100,
not order 1. A value copied from another lens can be 50x wrong, so both are calibrated here
by rendering 200 draws from this system's own prior and comparing the median rendered peak
with the data (measured 2026-08-08, `n_source=1, n_lens_light=1`):

| | median rendered peak | 16-84th pct | data | scale | adopted |
|---|---|---|---|---|---|
| lens light | 7.375 e/s | 3.44-19.5 | 6.63 e/s (core) | 0.899 | `LENS_IE = 72` |
| lensed source | 0.6222 e/s | 0.112-2.95 | 0.64 e/s (arc) | 1.029 | `SOURCE_IE = 21` |

**The median is a weak anchor.** The LogNormal `Ie` prior's 16-84th percentile spans nearly
two orders of magnitude, so this fixes the prior's *centre* to within a factor of a few, not
better. That the provisional placeholders (80/20) came out within 10-20% of the calibrated
values is coincidence, not a validation of guessing.

## Status

- [x] Cutout and PSF built, both with reproducible sidecars
- [x] `config.py` written, every number measured on this system
- [x] `LENS_IE = 72` / `SOURCE_IE = 21` calibrated by prior-draw render (see below)
- [x] `verify.py --lens DESI234` passing — 5/5, check 5 honestly skipped (`RESULTS_v3.md` §3)
- [x] Round 1 fit (`r1_s1l1`, 22 params) — **did not converge**, worst R-hat 1.652;
      lens-light `n_sersic` pinned at 89.5% of its range. Not a measurement; see `RESULTS_v2.md` section 3.
- [x] Round 2 fit (`r2_s2l2`, 36 params, bulge+disk light + 2 source components) —
      **converged**, all 36 parameters R-hat <= 1.008. `theta_E = 1.539 +0.003/-0.002"`.
      See `RESULTS_v2.md` for the full result, the residual-structure caveats (PSF mismatch
      at the core, unmodelled arc knots), and section 6's finding that the second source
      component sits outside the caustic and is not multiply imaged.
      **Its `theta_E` is superseded as a measurement by round 3** — see the banner on
      `RESULTS_v2.md`.
- [x] Round 3 fits (`r3_unmasked` 36 params, `r3_fg` 46 params; both unmasked, `supersample = 4`)
      — **neither converged**, max R-hat 13.94 and 1.92. No `theta_E` from this round.
      What it did settle: the interloper carries **0.370" of SIS mass, 12.2 sd from zero**,
      and modelling it moves the main `theta_E` by **0.066"**. `RESULTS_v3.md`.
- [ ] Round 4: a converged `r3_fg`-class run. `RESULTS_v3.md` §5.4 shows the fix is sampling,
      not modelling — `r3_fg` is under-mixed along a `theta_E` ridge, `r3_unmasked` is frozen.

`verify.py`'s check 6 (the SVI guard) will report itself **skipped** until round 1 gives this
lens its own known-good and known-diverged runs. A skipped check is honest; it must not be
weakened to pass.

## Results

| round | run | params | converged | theta_E | notes |
|---|---|---|---|---|---|
| 1 | `r1_s1l1` | 22 | no (worst R-hat 1.652) | 1.532" (unreliable) | lens-light `n_sersic` pinned at bound |
| 2 | `r2_s2l2` | 36 | yes (worst R-hat 1.008) | 1.539 +0.003/-0.002" — **superseded, see below** | `RESULTS_v2.md` — converged, but with the interloper's mass unmodelled; round 3 shows that mass shifts `theta_E` by 0.066" |
| 3 | `r3_unmasked` | 36 | no (worst R-hat 13.94) | — | `RESULTS_v3.md` — chains **frozen** (§5.4); chi2_nu 1.662, the worse model by Δchi² ≈ 3700 |
| 3 | `r3_fg` | 46 | no (worst R-hat 1.92) | — | `RESULTS_v3.md` — chains **under-mixed along a `theta_E` ridge**, not multimodal; chi2_nu **1.082**, the best of the round. Foreground SIS `theta_E = 0.370"` at 12.2 sd from zero |

**No `theta_E` is currently quotable for this system.** Round 2's interval is statistically
sound but on a misspecified model (interloper mass absent); round 3 fixes the model and fails
to converge. `RESULTS_v3.md` §7 gives the systematic budget and states what round 4 needs.
