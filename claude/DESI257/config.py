"""Configuration for DESI257 (HST WFC3/IR, F140W).

Structure copied from DESI238's config; **no number is inherited from it**. Every value
below was measured from this system's own files, and carries the measurement in a comment.

Bounds are deliberately absent. A prior's *support* determines how a stored unconstrained
sample maps back to a physical value, so supports are versioned in
``claude/pipeline/priors.py`` and a config may not override one. See
``claude/pipeline/lenses.py`` for the full argument.

Provenance of the data products this file points at:
  cutout257.npy  -- claude/pipeline/make_cutout.py, sidecar cutout257.json
  psf257.fits    -- claude/pipeline/make_psf.py,    sidecar psf257.json (21 stars)
Both sidecars record the parent frame's sha256, the exact slice, and the thresholds used.

This system's counter-image identification is the least secure of the three modelled here,
and the ambiguity is stated in CLUMPS rather than resolved by fiat. Read that comment
before using any number that depends on theta_E.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from lenses import LensConfig, MassPrior, SersicPrior  # noqa: E402

# ---------------------------------------------------------------------------------------
# Data conditioning
# ---------------------------------------------------------------------------------------

# Background noise sigma, in ELECTRONS/S, from blank patches in the *parent* frame
# (final_99_drz.fits, HDU 1). 200 random 40x40 patches, rejecting any patch with a pixel
# above 0.15 e/s or with a WHT median below half the frame's. Run over five seeds, since a
# single seed's median is not a measurement:
#
#     seed 0  0.00732     seed 3  0.00725
#     seed 1  0.00719     seed 4  0.00736
#     seed 2  0.00736
#
# median across seeds 0.0073; the 16-84th percentile *within* a seed is 0.0067-0.0093, so
# the patch-to-patch spread (+/-15%) dwarfs the seed-to-seed spread (+/-1%). 0.0073 is this
# frame's own robust median, rounded.
#
# The rule is deliberately self-contained per frame -- no cross-frame rescaling -- so that
# DESI234 and DESI257 are conditioned identically and neither inherits a judgement made on a
# third system. The same estimator returns 0.0076 on DESI234's frame and 0.0078 on
# DESI238's, against the 0.0075 adopted there by hand.
#
# What that costs, stated rather than hidden: +/-15% in sigma is +/-30% in chi-squared,
# which is larger than any difference between the two frames. **A reduced chi-squared near 1
# is therefore not by itself evidence that the noise model is right.**
#
# `sigma_clipped_stats` over the cutout is the documented trap: the cutout is mostly galaxy,
# so the clip never lands on sky and the result is several times too large, which would
# down-weight the arc into irrelevance.
BACKGROUND_RMS = 0.0073

# Crop bounds into the 120x120 cutout.
#
# The deflector centroid sits at cutout pixel (row 59.397, col 59.236) -- make_cutout.py
# measured it, and it is 3.9 px from the WCS position of the catalogue RA/Dec, which is why
# the cutout is centred on light rather than on `all_world2pix`. [20:100, 20:100] places it
# at (39.40, 39.24) against an array centre of (39.5, 39.5): centred to 0.10 px in row and
# 0.26 px in column, i.e. 0.017" -- better centred than DESI234 and a third of MASS_PRIOR's
# center_sigma. Shifting the data to do better would correlate neighbouring pixels and
# invalidate the independent-noise likelihood.
#
# The crop does NOT exclude the SW object at cutout (28, 35); see MASK_CIRCLES.
CROP = (slice(20, 100), slice(20, 100))

# The SW object, and what is done about it.
#
# A smooth, centrally-concentrated ellipse sits at crop (8, 15) -- 2.58" from the deflector
# at array-space PA 218 -- peaking at 0.957 e/s, 131 sigma. It is the brightest thing in the
# field after the deflector itself, and it dominates the lower-left corner.
#
# It is masked because it is read as an INTERLOPER, not as a fourth image. That reading is a
# judgement; the evidence for and against it is set out in CLUMPS below, and round 5 is a
# falsification run that fits it as a fourth image instead.
#
# Radius 12 px: its residual profile falls through 1 sigma between r = 10 and 12 px. The
# disc excludes 403 of 6400 pixels (6.3%), and the nearest catalogued clump -- the E arc's
# northern end -- is 45 px from the disc edge, so nothing lensed is anywhere near it. With
# the mask applied the lower-left corner falls from 0.957 to 0.079 e/s.
#
# Why not leave it unmasked: the source components' priors put no constraint on where a
# component may sit, so a free Sersic can and will go and sit on a 131-sigma blob. That is
# not a hypothetical; it is the cheapest way for the model to buy chi-squared.
MASK_CIRCLES = ((8.0, 15.0, 12.0),)

# No sky is subtracted. The azimuthal median at r = 45-55 px reads 0.043 e/s, which is the
# deflector's own Sersic wings, not a sky pedestal: true blank sky in this frame is
# consistent with zero (see BACKGROUND_RMS). Subtracting it would remove real galaxy light
# and bias the lens-light R_sersic and n_sersic low. The Sersic profile models those wings.

# ---------------------------------------------------------------------------------------
# Flux scale
# ---------------------------------------------------------------------------------------

# The simulator multiplies its output by conversion_factor = delta_pix^2 = 0.004225, so Ie
# is of order 10-100, not order 1. These are calibrated against this image by rendering
# prior draws and comparing the median peak with the data peak of 8.47 e/s -- never copied
# from another lens or from the gigalens demo, whose values are tuned to a synthetic image
# with a different flux scale.
#
# Measured 2026-08-08: 200 prior draws rendered with SceneSimulator(sees=...) to isolate
# lens-light-only and lensed-source-only components (n_source=1, n_lens_light=1). Median
# lens-light peak 8.231 e/s vs data core peak 8.47 e/s -> scale 1.029 -> LENS_IE = 80.0 *
# 1.029 = 82.3. Median lensed-source peak 0.5425 e/s vs data arc peak 0.64 e/s -> scale
# 1.18 -> SOURCE_IE = 20.0 * 1.18 = 23.6. Both spreads are wide (16-84th pct
# 3.81-21.6 e/s lens, 0.110-2.62 e/s source), consistent with a LogNormal prior.
LENS_IE = 82.0
SOURCE_IE = 24.0

# ---------------------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------------------

# The counter-image inventory, measured on the cutout after subtracting a point-symmetric
# reflection of the image about the deflector centroid (this deflector is genuinely
# elongated -- |e| = 0.137 -- so a circular median would leave the light's own major axis
# behind as a spurious residual).
#
#     clump        r (")   PA (deg)   peak (sigma)   |e|     major axis vs tangential
#     E arc S      1.90      60           63         0.25-0.35     1-2 deg
#     E arc mid    1.80      76           87         0.25-0.35     1-2 deg
#     E arc N      1.86     104           56         0.25-0.35     1-2 deg
#     N blob       1.55     350          149         0.125        22 deg
#
# **These PAs are array-space angles**, measured from the +row axis toward +col in the crop
# -- NOT sky PA. Converting them through the WCS the standard way (North through East)
# places the markers on empty sky. This is the trap DESI238's config documents and it is
# repeated here because it is the config field most likely to be misread.
#
# The identification, and its weakest link.
#
# The E arc is secure: continuous from PA 40 to PA 110 at a radius that stays within
# 1.76-1.92" the whole way, elongated |e| = 0.25-0.35 with its major axis within 1-2 deg of
# tangential everywhere along it. That is what lensing does.
#
# The N blob at PA 350 is a SEPARATE image, not the same arc continued: the azimuthal trace
# between them drops to 3-7 sigma over PA 15-35, a genuine flux gap, not a dip. It is also
# rounder (|e| = 0.125) and 22 deg off tangential, as a merging pair near the minor axis
# should be. Two images at 1.85" and 1.55" on opposite-ish sides is already a lens; taking
# theta_E as the mean of the two gives 1.70".
#
# **The open question is the SW object at 2.58", PA 218.** Three images invite a fourth, and
# PA 218 is close to where a quad would put it. I read it as an interloper and mask it
# (MASK_CIRCLES), on three grounds:
#   - morphology: a smooth, centrally-concentrated ellipse with a bright core, which is what
#     an ordinary galaxy looks like and not what a lensed knot looks like;
#   - brightness ordering: at 131 sigma it is 2x brighter than the brightest arc pixel while
#     sitting 1.4x farther out. A demagnified fourth image is the FAINTEST, not the
#     brightest;
#   - orientation: its major axis is 18 deg off tangential, where the E arc's is within
#     1-2 deg.
#
# None of those is decisive on its own. If the reading is wrong, the consequence is a
# theta_E biased LOW, since fitting the SW object as an image would demand a critical curve
# out near 2.1-2.2" rather than 1.70". MASS_PRIOR's theta_E_logsigma is therefore set to
# 0.35 -- DESI238's, and wider than DESI234's 0.25 -- specifically so the prior's support
# covers both readings and the data is allowed to prefer the other one. Round 5 fits the SW
# object as a fourth image, unmasked, and compares. **Until that run exists, every theta_E
# from this system carries the interloper reading as an assumption, and must be reported
# with it.**
CLUMPS = {
    "E arc S": (1.90, 60.0),
    "E arc mid": (1.80, 76.0),
    "E arc N": (1.86, 104.0),
    "N blob": (1.55, 350.0),
}

MASS_PRIOR = MassPrior(
    # 1.70" is the mean of the arc radius (1.85") and the counter-image radius (1.55"),
    # which is theta_E exactly for an isothermal sphere and approximately for anything close
    # to one.
    theta_E_median=1.70,
    # 0.35 puts the 1-99% range at 0.75-3.85". This is wider than DESI234's 0.25 on purpose:
    # see the CLUMPS comment. The SW-object-as-fourth-image reading wants ~2.1-2.2", and the
    # prior must not exclude it.
    theta_E_logsigma=0.35,
    gamma_center=2.0,
    gamma_sigma=0.25,
    # The deflector light is genuinely elongated: second moments inside r < 8 px give
    # |e| = 0.137 at array-space major-axis PA -51 deg (q = 0.76 under the (1-e)/(1+e)
    # convention gigalens' e1/e2 use). Light ellipticity is a weak predictor of mass
    # ellipticity, but here it at least points somewhere, so 0.2 covers the measurement with
    # room for the mass to be rounder or more elongated than the light.
    ellip_sigma=0.2,
    # Arcsec relative to the crop centre, which the crop puts the deflector at to within
    # 0.26 px = 0.017". 0.05" is 0.77 px.
    center_sigma=0.05,
    shear_width=0.1,
)

# Source components share this prior, which makes them formally exchangeable: the data, not
# the prior, decides whether the source is two separated compact clumps or a compact knot
# inside an extended host. A component the data does not want can go faint and switch itself
# off, so the two-component model nests the one-component model and comparing them is fair.
#
# Scales taken from DESI238's, which were themselves widened after a round-1 underfit. This
# is the one prior below whose scales are not independently measured on this system, and it
# is flagged as such. The E arc is 2-3 px FWHM across radially, comparable to DESI238's, so
# there is no measurement here that would justify a different width yet.
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
# r = 45-55 px, reaches half its enclosed flux within r < 40 px at r = 11 px = 0.72".
# ellip_sigma 0.2 rather than DESI234's 0.1, because this deflector's light really is
# elongated (|e| = 0.137) where DESI234's was round (|e| = 0.036); a 0.1 prior would be
# fighting the measurement.
LENS_LIGHT_SINGLE = SersicPrior(
    R_sersic_median=0.72,
    R_sersic_logsigma=0.3,
    ellip_sigma=0.2,
    center_sigma=0.05,
    Ie_median=LENS_IE,
    Ie_logsigma=0.5,
)

# Bulge and disk, for round 2's ladder. Unlike the source components these are deliberately
# NOT identical: they mean distinct physical things, so identical priors would only add a
# label-switching mode. The split brackets the measured half-light radius of 0.72" by about
# a factor of 2.5 either way, with 0.5 dex of width so the data can overturn the split
# entirely. **No structured core residual has been measured on this system yet** -- unlike
# DESI238, where the split was motivated by a measured S-curve in the single-Sersic
# residual. Round 2 is where that evidence gets collected; if the residual does not ask for
# two components, one is the answer.
LENS_LIGHT_BULGE = SersicPrior(
    R_sersic_median=0.3,
    R_sersic_logsigma=0.5,
    ellip_sigma=0.2,
    center_sigma=0.05,
    Ie_median=LENS_IE * 2.0,
    Ie_logsigma=0.8,
)

LENS_LIGHT_DISK = SersicPrior(
    R_sersic_median=1.1,
    R_sersic_logsigma=0.5,
    ellip_sigma=0.2,
    center_sigma=0.05,
    Ie_median=LENS_IE * 0.25,
    Ie_logsigma=0.8,
)

CONFIG = LensConfig(
    name="DESI257",
    cutout_file="cutout257.npy",
    psf_file="psf257.fits",
    parent_frame="final_99_drz.fits",
    # arcsec/pixel, from the SCI CD matrix: sqrt(|det CD|) * 3600 = 0.065000. The cutout
    # carries no WCS of its own, so this comes from its provenance as a crop of
    # final_99_drz.fits, recorded in cutout257.json as `delta_pix`.
    delta_pix=0.065,
    num_pix=80,
    supersample=2,
    # EXPTIME header keyword, NCOMBINE 3. The WHT extension is BUNIT = UNITLESS, so it is a
    # weight and not an exposure map; over the 81x81 box around the lens its median is
    # 99.80% of its maximum, so coverage is uniform there and a scalar exposure time is
    # right to first order. That is a measurement, not an assumption.
    exp_time=1197.699462,
    background_rms=BACKGROUND_RMS,
    crop=CROP,
    centroid_pix=(59.397 - 20, 59.236 - 20),
    mask_circles=MASK_CIRCLES,
    mass_prior=MASS_PRIOR,
    source_prior=SOURCE_PRIOR,
    lens_light_priors={
        "single": LENS_LIGHT_SINGLE,
        "bulge": LENS_LIGHT_BULGE,
        "disk": LENS_LIGHT_DISK,
    },
    clumps=CLUMPS,
    # Upper bound on the brightest *unmasked* corner pixel. The SW object puts 0.957 e/s in
    # the lower-left corner and is masked; with the mask applied the worst of the four
    # corners is the upper-left at 0.1906 e/s, at crop (60, 19). That pixel is the innermost
    # pixel of its quadrant, 1.88" from the deflector, and its mirror across the centre --
    # crop (19, 61), 1.94" out -- reads 0.1906 too. Two pixels equal to four figures at
    # opposite PAs is the signature of smooth elliptical galaxy light, not of a companion.
    # Masking takes the lower-left corner from 0.957 to 0.079 e/s. 0.30 fails on anything
    # new without failing on that.
    companion_max=0.30,
    # Populated after round 1 gives this lens its own known-good and known-diverged runs.
    # Until then verify.py's check 6 reports itself skipped, which is honest.
    svi_guard_reference={},
    notes=(
        "HST WFC3/IR F140W, EXPTIME 1197.699 s, NCOMBINE 3, TARGNAME DESI-257.4348+31.9046. "
        "E arc at 1.85\" plus an N counter-image at 1.55\"; one masked object at 2.58\" whose "
        "interloper-vs-fourth-image status is unresolved and is round 5's falsification "
        "test. See README.md."
    ),
)
