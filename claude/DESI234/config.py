"""Configuration for DESI234 (HST WFC3/IR, F140W).

Structure copied from DESI238's config; **no number is inherited from it**. Every value
below was measured from this system's own files, and carries the measurement in a comment.

Bounds are deliberately absent. A prior's *support* determines how a stored unconstrained
sample maps back to a physical value, so supports are versioned in
``claude/pipeline/priors.py`` and a config may not override one. See
``claude/pipeline/lenses.py`` for the full argument.

Provenance of the data products this file points at:
  cutout234.npy  -- claude/pipeline/make_cutout.py, sidecar cutout234.json
  psf234.fits    -- claude/pipeline/make_psf.py,    sidecar psf234.json (16 stars)
Both sidecars record the parent frame's sha256, the exact slice, and the thresholds used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from lenses import ForegroundPrior, LensConfig, MassPrior, SersicPrior  # noqa: E402

# ---------------------------------------------------------------------------------------
# Data conditioning
# ---------------------------------------------------------------------------------------

# Background noise sigma, in ELECTRONS/S, from blank patches in the *parent* frame
# (final_92_drz.fits, HDU 1). 200 random 40x40 patches, rejecting any patch with a pixel
# above 0.15 e/s or with a WHT median below half the frame's. Run over five seeds, since a
# single seed's median is not a measurement:
#
#     seed 0  0.00756     seed 3  0.00752
#     seed 1  0.00743     seed 4  0.00767
#     seed 2  0.00746
#
# median across seeds 0.0075; the 16-84th percentile *within* a seed is 0.0069-0.0100, so the
# patch-to-patch spread (+/-15%) dwarfs the seed-to-seed spread (+/-1.5%).
#
# 0.0075 is this frame's own robust median, rounded. The rule is deliberately
# self-contained per frame -- no cross-frame rescaling -- so that the two new lenses are
# conditioned identically and neither inherits a judgement made on a third system.
#
# What that rule costs, stated rather than hidden: the same estimator run on DESI238's
# frame returns 0.00780 against the 0.0075 adopted there by hand, so it reads about 4%
# high, and rescaling by that factor would give 0.0072 here instead. 4% in sigma is 8% in
# chi-squared, which is smaller than the +/-15% patch-to-patch spread (itself +/-30% on
# chi-squared) that this measurement carries anyway. **A reduced chi-squared near 1 is
# therefore not by itself evidence that the noise model is right.**
#
# `sigma_clipped_stats` over the cutout is the documented trap: the cutout is mostly
# galaxy, so the clip never lands on sky and the result is several times too large, which
# would down-weight the arc into irrelevance.
BACKGROUND_RMS = 0.0075

# Crop bounds into the 120x120 cutout.
#
# The deflector centroid sits at cutout pixel (row 59.13, col 59.47) -- make_cutout.py
# measured it, and it is 3.4 px from the WCS position of the catalogue RA/Dec, which is why
# the cutout is centred on light rather than on `all_world2pix`. [20:100, 20:100] places it
# at (39.13, 39.47) against an array centre of (39.5, 39.5): centred to 0.03 px in column
# and 0.37 px in row. The row residual is 0.024", half of MASS_PRIOR's center_sigma, so the
# deflector-centre prior still covers the truth comfortably; no integer crop does better,
# and shifting the data to do so would correlate neighbouring pixels and invalidate the
# independent-noise likelihood.
#
# The crop excludes the bright extended galaxy at cutout ~(117, 103) (4.7" away) and the
# compact source at ~(76, 110) (3.5"). It does NOT exclude the interloper at ~(84, 34);
# see MASK_CIRCLES.
CROP = (slice(20, 100), slice(20, 100))

# The interloper that cannot be cropped, and what is done about it instead.
#
# A compact source sits at cutout (84, 34) -- crop (64, 14) -- 2.30" from the deflector at
# array-space PA 314, peaking at 0.82 e/s (110 sigma) above the subtracted deflector light.
# It is the brightest thing in the field after the deflector itself.
#
# **No square crop can remove it without clipping the arc.** For a box of half-width h
# centred on the deflector, the interloper is inside whenever h > 25.5 px, while the arc
# runs out to h = 26.5 px. The two requirements do not overlap. Per LENS_MODELING_GUIDE.md
# the options are then mask or model, and cropping is not among them.
#
# Rounds 1-2 mask it. Radius 12 px: its residual profile falls through 1 sigma between
# r = 11 and 12 px and is consistent with zero beyond, and the nearest arc pixel is 48 px
# away, so nothing lensed is inside the disc. Masking is not free -- it discards 437 pixels
# and, more importantly, it declines to use the interloper's own light -- so round 3 is to
# replace the mask with a modelled Sersic and compare. Until that is done, **the fit
# carries an unmodelled interloper as an accepted systematic**, and it is reported as one.
#
# Why not simply leave it unmasked: the source components' priors put no constraint on
# where a component may sit, so a free Sersic can and will go and sit on a 110-sigma blob.
# That is not a hypothetical; it is the cheapest way for the model to buy chi-squared.
#
# **Round 3 unmasks it** (2026-08-09). Both round-3 fits use the whole crop:
# `r3_unmasked` leaves the object to be reproduced (or not) by the same mass model and the
# same source components that make the arc, and `r3_fg` adds the nested SIS+Sersic pair
# below (FOREGROUND) at its measured position. The paragraph above is the reason the two
# must be run as a PAIR rather than unmasking alone: an unconstrained free Sersic sitting
# on the blob is a real failure mode, and `r3_fg`'s dedicated, position-tied component is
# what distinguishes "the model reproduces this object" from "some component went and sat
# on it". DESI257 is the precedent for the unmasking itself -- it masked its SW object on
# an argument that turned out to be a physics error, and unmasking in its round 3 closed a
# bimodality and moved gamma 2.54 -> 1.99.
#
# Rounds 1-2 remain diagnosable: `run_fit.py` records `mask_circles` per run and
# `diagnose.rebuild_model` reads back the RECORDED value, not this one
# (KNOWN_ISSUES.md item 8). It prints a NOTE when the two differ, and raises if the
# rebuilt mask's kept-pixel count disagrees with the run's recorded `n_masked_pixels`.
# Note also that log posteriors are NOT comparable across this change: DESI257 measured a
# 14,347-nat offset between its masked and unmasked likelihoods on identical parameters.
MASK_CIRCLES = ()

# No sky is subtracted. The cutout's clean corners read 0.004-0.015 e/s, which is the
# deflector's own Sersic wings plus its neighbours, not a sky pedestal: true blank sky in
# this frame is consistent with zero (see BACKGROUND_RMS). Subtracting it would remove real
# galaxy light and bias the lens-light R_sersic and n_sersic low. The Sersic profile is
# there to model those wings.

# ---------------------------------------------------------------------------------------
# Flux scale
# ---------------------------------------------------------------------------------------

# The simulator multiplies its output by conversion_factor = delta_pix^2 = 0.004225, so Ie
# is of order 10-100, not order 1. These are calibrated against this image by rendering
# prior draws and comparing the median peak with the data peak of 6.63 e/s -- never copied
# from another lens or from the gigalens demo, whose values are tuned to a synthetic image
# with a different flux scale.
#
# Measured 2026-08-08: 200 prior draws rendered with SceneSimulator(sees=...) to isolate
# lens-light-only and lensed-source-only components (n_source=1, n_lens_light=1). Median
# lens-light peak 7.375 e/s vs data core peak 6.63 e/s -> scale 0.899 -> LENS_IE = 80.0 *
# 0.899 = 71.9. Median lensed-source peak 0.6222 e/s vs data arc peak 0.64 e/s -> scale
# 1.029 -> SOURCE_IE = 20.0 * 1.029 = 20.6. Both spreads are wide (16-84th pct
# 3.44-19.5 e/s lens, 0.112-2.95 e/s source), consistent with a LogNormal prior.
LENS_IE = 72.0
SOURCE_IE = 21.0

# ---------------------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------------------

# The counter-image inventory, measured on the cutout after subtracting an
# elliptical-isophote median about the deflector (the light's own second moments inside
# r < 8 px give |e| = 0.036 at array-space PA 7.6 deg -- this deflector is nearly round, so
# a circular median would have served almost as well).
#
#     clump         r (")   PA (deg)   peak (sigma)   |e|    major axis vs tangential
#     S arc (E end)  1.81     145          85         0.31        2.5 deg
#     S arc (mid)    1.71     184          62         0.39        2.9 deg
#     S arc (W end)  1.74     215          57         0.40        3.7 deg
#     N counter      1.11     358          37         0.05         --  (round)
#
# **These PAs are array-space angles**, measured from the +row axis toward +col in the crop
# -- NOT sky PA. Converting them through the WCS the standard way (North through East)
# places the markers on empty sky. This is the trap DESI238's config documents and it is
# repeated here because it is the config field most likely to be misread.
#
# The identification. Tracing the maximum of the subtracted image in 5-degree azimuthal
# bins, the southern structure is continuous from PA 125 to PA 235 -- 110 degrees -- at a
# radius that stays within 1.66-1.82" the whole way, with three local maxima at PA 145
# (83 sigma), PA 183 (55 sigma) and PA 215 (55 sigma) and no gap between them. The measured
# major axis at each of the three lies within 4 degrees of the tangential direction there.
# That is what lensing does; a chance alignment of three galaxies does not produce a
# constant-radius, tangentially-elongated, unbroken ridge over 110 degrees.
#
# Three merging images along one arc with a fourth on the opposite side is a **cusp
# configuration**, and the fourth is where a cusp quad puts it: the northern clump at
# PA 358, opposite the arc's midpoint at PA 183 to within 5 degrees, at the smaller radius
# 1.11", and the faintest of the four. I read this as a cusp quad. Taking theta_E as the
# mean of the arc radius and the counter-image radius gives 1.43".
#
# The alternative reading -- that the southern structure is a single fold pair and the three
# maxima are surface-brightness structure in the source rather than separate images -- is
# not excluded by the image alone and does not change theta_E much, since both readings put
# the tangential critical curve between 1.11" and 1.75". It would change the source model,
# which is what round 2's ladder is for.
#
# The interloper at 2.30", PA 314, is deliberately NOT in this list, and that is a judgement
# rather than a measurement. Against it being a fifth image: it is round (|e| = 0.036) with
# its major axis 66 degrees away from tangential; it is 3.5x brighter than the counter-image,
# where a fifth image should be the faintest; and at PA 314 it is 44 degrees from the
# counter-image on the same side of the lens, which no standard quad geometry produces.
# If this is wrong, the consequence is a theta_E biased low, since fitting it would demand
# images at both 1.11" and 2.30". Round 5's falsification run is where that gets tested.
CLUMPS = {
    "S arc E": (1.81, 145.0),
    "S arc mid": (1.71, 184.0),
    "S arc W": (1.74, 215.0),
    "N counter": (1.11, 358.0),
}

MASS_PRIOR = MassPrior(
    # 1.43" is the mean of the arc radius and the counter-image radius, which is theta_E
    # exactly for an isothermal sphere and approximately for anything close to one.
    # logsigma 0.25 puts the 1-99% range at 0.80-2.55", covering both the possibility that
    # the arc is a merged pair at a fold (theta_E nearer the arc radius) and that the
    # counter-image is misidentified.
    theta_E_median=1.43,
    theta_E_logsigma=0.25,
    gamma_center=2.0,
    gamma_sigma=0.25,
    # The deflector light is nearly round (|e| = 0.036), but light ellipticity is a weak
    # predictor of mass ellipticity, and the 1.75"/1.11" radial asymmetry between the arc
    # and its counter-image has to come from somewhere -- ellipticity, shear, or a source
    # offset. 0.15 lets the data choose rather than forcing the answer onto the shear.
    ellip_sigma=0.15,
    # Arcsec relative to the crop centre, which the crop puts the deflector at to within
    # 0.37 px = 0.024" in row and 0.03 px in column. 0.05" is 0.77 px.
    center_sigma=0.05,
    shear_width=0.1,
)

# Source components share this prior, which makes them formally exchangeable: the data, not
# the prior, decides whether the source is two separated compact clumps or a compact knot
# inside an extended host. A component the data does not want can go faint and switch itself
# off, so the two-component model nests the one-component model and comparing them is fair.
#
# Scales taken from DESI238's, which were themselves widened after a round-1 underfit; the
# arc here is comparably compact in the radial direction (2-3 px FWHM across) and there is
# no measurement on this system yet that would justify a different width. This is the one
# prior below whose scales are not independently measured, and it is flagged as such.
SOURCE_PRIOR = SersicPrior(
    R_sersic_median=0.4,
    R_sersic_logsigma=0.6,
    ellip_sigma=0.2,
    center_sigma=0.5,
    Ie_median=SOURCE_IE,
    Ie_logsigma=1.2,
)

# Single-component deflector light. R_sersic_median from this system's own curve of growth:
# the azimuthal-median profile about the centroid, with the sky reference taken at
# r = 45-55 px, reaches half its enclosed flux within r < 40 px at r = 10 px = 0.65".
# Light second moments inside r < 8 px give |e| = 0.036, so the prior is centred on round
# and 0.1 covers the measurement several times over.
LENS_LIGHT_SINGLE = SersicPrior(
    R_sersic_median=0.65,
    R_sersic_logsigma=0.3,
    ellip_sigma=0.1,
    center_sigma=0.05,
    Ie_median=LENS_IE,
    Ie_logsigma=0.5,
)

# Bulge and disk, for round 2's ladder. Unlike the source components these are deliberately
# NOT identical: they mean distinct physical things, so identical priors would only add a
# label-switching mode. The split brackets the measured half-light radius of 0.65" by about
# a factor of 2.5 either way, with 0.5 dex of width so the data can overturn the split
# entirely. **No structured core residual has been measured on this system yet** -- unlike
# DESI238, where the split was motivated by a measured S-curve in the single-Sersic
# residual. Round 2 is where that evidence gets collected; if the residual does not ask for
# two components, one is the answer.
LENS_LIGHT_BULGE = SersicPrior(
    R_sersic_median=0.25,
    R_sersic_logsigma=0.5,
    ellip_sigma=0.1,
    center_sigma=0.05,
    Ie_median=LENS_IE * 2.0,
    Ie_logsigma=0.8,
)

LENS_LIGHT_DISK = SersicPrior(
    R_sersic_median=1.0,
    R_sersic_logsigma=0.5,
    ellip_sigma=0.1,
    center_sigma=0.05,
    Ie_median=LENS_IE * 0.25,
    Ie_logsigma=0.8,
)

# Round 3: the masked interloper (MASK_CIRCLES above) modelled instead as a candidate
# foreground galaxy -- one SIS + its own Sersic light, nested against round 2's masked
# model so that SIS theta_E -> 0 and light Ie -> 0 recover it exactly (see priors.py's
# build_model docstring for the nesting argument). This block is INERT unless
# `run_fit.py --foreground` is passed -- r1_s1l1 and r2_s2l2 built their models with
# foreground=None and are unaffected by anything below.
#
# Every number here comes from claude/DESI234/analysis/foreground_calibration.py, whose
# full output is saved alongside it as foreground_calibration.txt. None is copied from
# DESI257's FOREGROUND block or from this file's own LENS_IE/SOURCE_IE.
#
# Position, measured from the r2_s2l2 residual (data minus the round-2 point-estimate
# model, rebuilt via diagnose.rebuild_model): the interloper was MASKED out of round 2's
# likelihood, so the round-2 model contains no attempt to fit it, and the residual in its
# disc IS the object (plus noise), not a model-mismatch artefact on some other component.
#
#   Background: local median over an annulus 12-20 px around the mask centre (635 px,
#   outside the mask radius where round 2 says the profile is consistent with zero) is
#   0.00054 e/s -- negligible next to the object's own signal, but subtracted anyway.
#   Flux centroid (background-subtracted, negative pixels clipped to 0, r <= 10 px around
#   the mask centre (64.0, 14.0)): crop (row 63.900, col 14.344). The peak residual pixel
#   in that disc is 0.8206 e/s at crop (64, 14), matching MASK_CIRCLES's stated 0.82 e/s.
#
#   Which pixel axis feeds center_x and which feeds center_y was independently
#   RE-VERIFIED by rendering on THIS system's own grid (80x80/0.065", supersample=1,
#   delta-function PSF for pixel-exact placement) -- not merely cited from DESI257's
#   config: a Sersic placed at center_x=0.5, center_y=-1.0 peaks at array (row, col) =
#   (24, 47), matching the predicted ROW shift from center_y=-1.0 (-1.0/0.065 = -15.385 px
#   from the array centre 39.5 -> row 24.115) and the predicted COLUMN shift from
#   center_x=0.5 (0.5/0.065 = +7.692 px -> col 47.192). So **center_x is the
#   COLUMN-derived coordinate, center_y is the ROW-derived one** -- same convention as
#   DESI257's FOREGROUND block, confirmed fresh rather than assumed.
#
#   gigalens.simulator.LensWCS(n=80, supersample=1, pix_scale=0.065) is the same WCS
#   SceneSimulator builds internally; its angle2pix(0, 0) returns pixel (col, row) =
#   (39.5, 39.5), the array centre. wcs.pix2angle(col=14.344, row=63.900) gives
#   (center_x, center_y) = (-1.63513, 1.58600) arcsec.
#
#   Cross-check: separation from the deflector centroid (cfg.centroid_pix, crop
#   (row 39.130, col 39.470), pushed through the same WCS) is 2.2934" at array-space PA
#   314.6 deg -- within 0.3% and 0.6 deg of MASK_CIRCLES's stated 2.30" at PA 314. No
#   transposition or sign error.
FOREGROUND_CENTER_X = -1.63513
FOREGROUND_CENTER_Y = 1.58600

# Ie calibrated by the same render-and-match procedure LENS_IE/SOURCE_IE above used:
# build the model with a provisional ForegroundPrior, render prior draws of the
# foreground LIGHT component ALONE (SceneSimulator(sees=...) isolating it, real PSF, on
# this system's own grid num_pix=80/delta_pix=0.065/supersample=2 -- the data peak below
# is already PSF-convolved, so the render must be too), and tune Ie_median until the
# median rendered peak matches the data peak the interloper actually reaches, 0.82 e/s
# (MASK_CIRCLES comment above; reproduced by this calibration's own peak-residual check,
# 0.8206 e/s).
#
# 500 draws per seed, 5 seeds, at R_sersic_median=0.2, R_sersic_logsigma=0.5,
# ellip_sigma=0.2, Ie_logsigma=0.8, center_sigma=0.1 (provisional for this search --
# Ie_median is the only number the search determines):
#
#   Ie_median=50  seed 0 0.566  seed 1 0.688  seed 2 0.637  seed 3 0.612  seed 4 0.542  -> median 0.6121
#   Ie_median=60  seed 0 0.679  seed 1 0.826  seed 2 0.765  seed 3 0.735  seed 4 0.650  -> median 0.7346
#   Ie_median=70  seed 0 0.792  seed 1 0.964  seed 2 0.892  seed 3 0.857  seed 4 0.759  -> median 0.8570
#   Ie_median=80  seed 0 0.905  seed 1 1.101  seed 2 1.020  seed 3 0.979  seed 4 0.867  -> median 0.9794
#
# Linear interpolation against the 0.82 target picks Ie_median=67; a dedicated
# confirmation run there gives per-seed medians 0.758/0.922/0.854/0.820/0.726 (median
# across seeds 0.8203, mean 0.8162) against the 0.82 target -- within 0.1%. The spread
# within one seed's 500 draws is wide (16-84th pct roughly 0.23-3.1 e/s at this median),
# consistent with a LogNormal prior and with LENS_IE/SOURCE_IE's own spreads.
FOREGROUND_IE = 67.0

FOREGROUND = ForegroundPrior(
    center_x=FOREGROUND_CENTER_X,
    center_y=FOREGROUND_CENTER_Y,
    # Looser than every other center_sigma in this file (0.05, e.g. MASS_PRIOR's and
    # LENS_LIGHT_SINGLE's) by design: this object is fainter relative to its surroundings
    # than the deflector is, and its centroid comes from a background-subtracted residual
    # rather than a direct curve-of-growth measurement on bright, isolated light. 0.1" =
    # 1.5 px keeps the component tied to the object while giving the data more room to
    # correct a centroid measured this way.
    center_sigma=0.1,
    # SIS theta_E: HalfNormal(scale=0.3), NOT a LogNormal -- see priors.py's
    # _foreground_mass_priors_v1 for why this family is required (support must reach
    # exactly 0). scipy.stats.halfnorm(scale=0.3): median 0.2023", 1-99% range
    # [0.0038", 0.7727"] -- of order the object's own measured size (R_sersic_median
    # below), reaching up to a plausible-but-not-enormous foreground galaxy's Einstein
    # radius without a floor that would break the nesting round 3 depends on. A fitted
    # theta_E near 0 favours the fourth-image/pure-interloper-light reading; a fitted
    # theta_E of order 0.2-0.5" together with non-negligible light Ie favours the
    # foreground-deflector reading.
    theta_E_scale=0.3,
    # From this object's own curve of growth: background-subtracted flux (same map used
    # for the centroid above), summed in growing apertures about the measured centroid
    # out to r=10 px (beyond which MASK_CIRCLES records the profile is consistent with
    # zero). Half of the r<=10 px total (23.674) is enclosed at r=3.067 px = 0.1993",
    # rounded to 0.2".
    R_sersic_median=0.2,
    R_sersic_logsigma=0.5,
    # Measured, not assumed: second moments of the same background-subtracted disc
    # (r <= 10 px about the measured centroid, negative pixels clipped) give
    # e1=0.120, e2=-0.087, |e|=0.148 -- notably rounder than "very flattened" but not
    # round either. 0.3 is roughly double that measurement, wider than this file's other
    # ellip_sigma values (0.1-0.2, set against measurements an order of magnitude
    # smaller) because a 317-pixel, background-subtracted second-moment estimate on a
    # faint object is noisier than the curve-of-growth and light-profile measurements
    # behind those other values, and the width should say so rather than pretend to the
    # same precision.
    ellip_sigma=0.3,
    Ie_median=FOREGROUND_IE,
    Ie_logsigma=0.8,
)

CONFIG = LensConfig(
    name="DESI234",
    cutout_file="cutout234.npy",
    psf_file="psf234.fits",
    parent_frame="final_92_drz.fits",
    # arcsec/pixel, from the SCI CD matrix: sqrt(|det CD|) * 3600 = 0.06500. The cutout
    # carries no WCS of its own, so this comes from its provenance as a crop of
    # final_92_drz.fits, recorded in cutout234.json as `delta_pix`.
    delta_pix=0.065,
    num_pix=80,
    # Round 3 raises this from 2 to 4. `analysis/undersampling_certify.py` measured the
    # quadrature error against a supersample=32 reference on a 9-sample batch (MAP,
    # HMC-median, 7 spread draws): max |delta|/sigma falls 3.132 -> 0.997 and the fraction
    # of pixels above the 0.2 sigma/pixel tolerance falls 8.17% -> 1.97%. ss=2 and ss=4 are
    # BOTH the `subgrid` PSF convention, so this refines the quadrature only and does not
    # move the PSF model -- unlike 2 -> 1, which crosses into `native` and changes the PSF
    # by 5.02 sigma, a gap supersampling cannot close. Measured wall cost of 2 -> 4 on this
    # 80x80 image: 1.05x at batch 50, 0.90x at batch 1 -- kernel-launch overhead dominates
    # the O(ss^2) compute, so the naive 4x is not what this costs.
    # Neither rung reaches tolerance and the ss=32 reference is itself uncertified
    # (self-check 0.45 sigma against a 0.05 sigma bar), so ss=4 is the better of two
    # imperfect renders, not a solved problem. See RESULTS_v3.md section 1.1.
    supersample=4,
    # EXPTIME header keyword, NCOMBINE 3. The WHT extension is BUNIT = UNITLESS, so it is a
    # weight and not an exposure map; over the 81x81 box around the lens its median equals
    # its maximum to 0.2%, so coverage is uniform there and a scalar exposure time is right
    # to first order. That is a measurement, not the assumption it was for DESI238.
    exp_time=1197.699,
    background_rms=BACKGROUND_RMS,
    crop=CROP,
    centroid_pix=(59.13 - 20, 59.47 - 20),
    mask_circles=MASK_CIRCLES,
    mass_prior=MASS_PRIOR,
    source_prior=SOURCE_PRIOR,
    lens_light_priors={
        "single": LENS_LIGHT_SINGLE,
        "bulge": LENS_LIGHT_BULGE,
        "disk": LENS_LIGHT_DISK,
    },
    clumps=CLUMPS,
    # Round 3's candidate foreground interloper (see FOREGROUND above). Inert unless a run
    # is built with `priors.build_model(..., foreground=cfg.foreground_prior)` --
    # `run_fit.py --foreground` -- so r1_s1l1 and r2_s2l2 are unaffected.
    foreground_prior=FOREGROUND,
    # Upper bound on the brightest *unmasked* corner pixel. `verify.py`'s corner_report
    # checks all four corners -- DESI238's check looked only at the lower-left, where this
    # system's interloper would have been invisible -- and it checks them AFTER the mask,
    # so this bound moves with MASK_CIRCLES.
    #
    # Rounds 1-2 (masked) used 0.30. With the interloper masked the worst of the four
    # corners was the lower-right at 0.241 e/s, at crop (19, 60) -- the innermost pixel of
    # that quadrant, one PSF width from the arc's eastern tip, and meant to be there;
    # masking took the upper-left corner from 0.859 e/s down to 0.041.
    #
    # Round 3 unmasks (MASK_CIRCLES above), which puts the interloper's own 0.859 e/s back
    # into the upper-left corner and would fail a 0.30 bound. Raised to 0.95: about 11%
    # headroom over the 0.859 this field's one known corner object actually reaches, so the
    # check still fails on anything NEW appearing in any corner while admitting the object
    # round 3 deliberately fits. This is exactly what DESI257 did when it unmasked its own
    # SW object -- 1.05 against that object's 0.957 e/s peak, the same ~10% margin.
    #
    # This bound is therefore no longer a statement that the field is clean; it is a
    # statement that the field contains one known bright corner object, which the model now
    # carries explicitly. If MASK_CIRCLES is ever restored, restore 0.30 with it.
    companion_max=0.95,
    # Populated after round 1 gives this lens its own known-good and known-diverged runs.
    # Until then verify.py's check 6 reports itself skipped, which is honest.
    svi_guard_reference={},
    notes=(
        "HST WFC3/IR F140W, EXPTIME 1197.699 s, NCOMBINE 3, TARGNAME DESI-234.4783+14.7232. "
        "Fold arc at 1.75\" plus a counter-image at 1.11\"; one masked interloper at 2.30\". "
        "See README.md."
    ),
)
