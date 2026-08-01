"""Configuration for DESI238 (HST WFC3/IR, F140W).

Every number here was measured from the files, not taken from documentation. Where a value
contradicts an earlier write-up, the discrepancy is explained in a comment.

Bounds are deliberately absent. A prior's *support* determines how a stored unconstrained
sample maps back to a physical value, so supports are versioned in
``claude/pipeline/priors.py`` and a config may not override one. See
``claude/pipeline/lenses.py`` for the full argument.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from lenses import LensConfig, MassPrior, SersicPrior  # noqa: E402

# ---------------------------------------------------------------------------------------
# Data conditioning
# ---------------------------------------------------------------------------------------

# Background noise sigma, in ELECTRONS/S.
#
# An earlier note quoted 0.0315 from sigma_clipped_stats over the whole 120x120 cutout.
# That number is contaminated: the cutout is mostly galaxy, so the clip never lands on sky,
# and using it down-weights the arcs into irrelevance. Measured instead from six widely
# separated 60x60 blank patches in the parent drizzle frame (final_94_drz.fits, HDU 1):
#
#   patch(300,300)   mean=-0.00198  std=0.00731
#   patch(1800,400)  mean=-0.00046  std=0.00768
#   patch(400,1600)  mean=+0.00137  std=0.00915
#   patch(1700,1500) mean=-0.00170  std=0.00731
#   patch(900,200)   mean=-0.00061  std=0.00794
#   patch(1400,1800) mean=-0.00146  std=0.00701
#
# Consistent at 0.0070-0.0092. Take 0.0075. Note that the +/-10% spread here is +/-20% on
# chi-squared, so a reduced chi-squared near 1 is not by itself evidence the noise model is
# right.
BACKGROUND_RMS = 0.0075

# Crop bounds into the 120x120 cutout.
#
# The deflector centroid sits at cutout pixel (row 57.48, col 57.64). [18:98, 18:98] places
# it at (39.48, 39.64) against an array centre of (39.5, 39.5) -- centred to better than a
# tenth of a pixel in both axes. It also excludes the companion galaxy at ~(11, 11), which
# matters because gigalens has no pixel masking: `grep -rn "mask" gigalens/src` returns
# nothing, so the likelihood runs over every pixel unconditionally. A companion inside the
# frame would be fit as if it were part of the lens system.
CROP = (slice(18, 98), slice(18, 98))

# No sky is subtracted. The crop's outer ring reads ~0.035 e/s, but that is the deflector's
# own Sersic wings, not sky: a radial profile in the parent frame falls smoothly 0.057
# (2.3") -> 0.0054 (5.9") -> 0.0007 (24"), and true blank sky is consistent with zero (see
# BACKGROUND_RMS above). Subtracting 0.035 would remove real galaxy light and bias the
# lens-light R_sersic and n_sersic low. The Sersic profile is there to model those wings.

# ---------------------------------------------------------------------------------------
# Flux scale
# ---------------------------------------------------------------------------------------

# LensSimulator multiplies its output by conversion_factor = det(transform_pix2angle),
# which defaults to delta_pix^2 = 0.065^2 = 0.004225. A unit-Ie lens+source Sersic pair
# renders to a peak of 0.083 e/s -- so Ie must be of order 100, not order 1, to reach this
# image's 7.2 e/s peak.
#
# Measured: with LENS_IE=80 and SOURCE_IE=20, the median peak over 64 prior draws is 5.26
# e/s against a data peak of 7.18. The gigalens demo notebook's values (2.0 and 0.5) are
# tuned to a synthetic image with a completely different flux scale and give 0.14 here --
# 50x too faint. Do not copy prior scales between images.
LENS_IE = 80.0
SOURCE_IE = 20.0

# ---------------------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------------------

# Four clumps measured in the azimuthal-median-subtracted cutout. The repository owner's
# decision is that all four are counter-images:
#
#     clump    r (")   PA (deg)   residual flux (e/s)
#     SW arc   1.06     224        18.8
#     E blob   2.11      88        24.0
#     W clump  1.64     278        12.5
#     N arc    1.25     339         8.7
#
# **These PAs are array-space angles**, measured from the +row axis toward +col in the crop
# -- NOT sky PA. Converting them through the image's WCS the standard way (North through
# East) places the markers on empty sky.
#
# Requiring one source to produce images at both 1.06" and 2.11" is a strong demand on a
# single elliptical deflector, and it is what widens theta_E, e1/e2 and the shear below
# relative to the gigalens demo. See MODELING_PLAN.md, "Consequences of the four-image
# choice".
CLUMPS = {
    "SW arc": (1.06, 224.0),
    "E blob": (2.11, 88.0),
    "W clump": (1.64, 278.0),
    "N arc": (1.25, 339.0),
}

MASS_PRIOR = MassPrior(
    # 1.5 sits between the inner clumps and the E blob, with logsigma 0.35 putting real
    # prior mass across 1.0-2.2" as the four-image reading requires. For the falsification
    # run that fits the inner three clumps only, pass --theta-e-center 1.3 --theta-e-width
    # 0.2 on the command line rather than editing this.
    theta_E_median=1.5,
    theta_E_logsigma=0.35,
    gamma_center=2.0,
    gamma_sigma=0.25,
    # Widened from the demo's 0.1: a 2:1 radial spread between images generally needs
    # either high ellipticity or strong shear.
    ellip_sigma=0.15,
    # Arcsec relative to the crop centre, which is the deflector to within 0.1 px. 0.05" is
    # ~0.8 px.
    center_sigma=0.05,
    # Widened from the demo's 0.05, for the same reason as ellip_sigma.
    shear_width=0.1,
)

# Source components share this prior, which makes them formally exchangeable: the data, not
# the prior, decides whether the source is two separated compact clumps or a compact knot
# inside an extended host.
#
# At 1-99%, R_sersic spans 0.10-1.62" -- wide enough for a compact knot at one end and an
# extended host at the other. The original prior's 0.157-0.398" could only ever have been a
# knot, which is why round 1 underfit the N arc by 2x.
#
# Ie spans 1.2-326 at 1-99%. The low end matters structurally: a component the data does
# not want can go faint and effectively switch itself off, so the two-component model
# *nests* the one-component model and a chi-squared comparison between them is fair rather
# than a free-parameter giveaway. If a component does switch off, its remaining six
# parameters become unconstrained and their Rhat will be poor *correctly* -- the model
# reporting an unidentified parameter, not a sampler failure.
SOURCE_PRIOR = SersicPrior(
    R_sersic_median=0.4,
    R_sersic_logsigma=0.6,
    ellip_sigma=0.2,
    center_sigma=0.5,
    Ie_median=SOURCE_IE,
    Ie_logsigma=1.2,
)

# The single-component deflector prior, as fitted in runs seed0 / seed1 / inner3 /
# nsersic10 / twosource. Light second moments of the deflector give e1 = -0.017,
# e2 = +0.036 -- nearly round, |e| = 0.04. The prior is centred on round and covers it
# comfortably.
LENS_LIGHT_SINGLE = SersicPrior(
    R_sersic_median=0.8,
    R_sersic_logsigma=0.3,
    ellip_sigma=0.1,
    center_sigma=0.05,
    Ie_median=LENS_IE,
    Ie_logsigma=0.5,
)

# Bulge and disk. Unlike the source components these are deliberately NOT identical: they
# mean distinct physical things, so identical priors would only add a label-switching mode.
#
# The motivation for splitting at all is measured, not aesthetic. The single-Sersic residual
# about the lens-light centre runs -2.7 sigma at r < 1 px, +0.4 to +0.8 sigma over
# r = 1.5-5.5 px, and -0.2 to -0.8 sigma over r = 6.5-11.5 px. The amplitude is only 1-3%,
# but it is coherent over hundreds of pixels, and it is what drove a core chi-squared of
# 7.10 against a global 1.302. A single Sersic is a one-shape family and cannot produce that
# S-curve.
#
# The split is centred on a 1D indicator fit to the inner-arcsecond data profile (compact
# Re = 0.216", n = 0.72; extended Re = 1.150", n = 1.13), with sigmas wide enough -- 0.5 dex
# on R_sersic, 0.8 on Ie -- that the data can overturn it. The two R_sersic priors overlap
# at about 1 sigma, which separates the components without forbidding them from converging
# on one shape. The Ie centres carry the indicator fit's 1.965 : 0.240 flux ratio rescaled
# to this model's LENS_IE convention.
LENS_LIGHT_BULGE = SersicPrior(
    R_sersic_median=0.25,
    R_sersic_logsigma=0.5,
    ellip_sigma=0.1,
    center_sigma=0.05,
    Ie_median=LENS_IE * 2.0,
    Ie_logsigma=0.8,
)

LENS_LIGHT_DISK = SersicPrior(
    R_sersic_median=1.2,
    R_sersic_logsigma=0.5,
    ellip_sigma=0.1,
    center_sigma=0.05,
    Ie_median=LENS_IE * 0.25,
    Ie_logsigma=0.8,
)

# The ORIGINAL source prior, as fitted in runs seed0 / seed1 / inner3. Kept verbatim so
# those runs stay readable: their checkpoints hold unconstrained samples, and the
# constrained value depends on the bijector. Reading seed0 through the current prior
# silently reports source.n_sersic = 0.6146 where the fit actually said 0.5729 -- no error,
# just a wrong number. Do not delete this to tidy up.
SOURCE_PRIOR_V1 = SersicPrior(
    R_sersic_median=0.25,
    R_sersic_logsigma=0.2,
    ellip_sigma=0.15,
    center_sigma=0.3,
    Ie_median=SOURCE_IE,
    Ie_logsigma=0.6,
)

CONFIG = LensConfig(
    name="DESI238",
    cutout_file="cutout238.npy",
    psf_file="psf238.fits",
    parent_frame="final_94_drz.fits",
    # arcsec/pixel, from the SCI CD matrix: sqrt(|det CD|) * 3600. The cutout carries no
    # WCS of its own, so this comes from its provenance as a crop of final_94_drz.fits.
    delta_pix=0.065,
    num_pix=80,
    supersample=2,
    exp_time=1197.699,  # EXPTIME header keyword
    background_rms=BACKGROUND_RMS,
    crop=CROP,
    centroid_pix=(57.48 - 18, 57.64 - 18),
    mass_prior=MASS_PRIOR,
    source_prior=SOURCE_PRIOR,
    lens_light_priors={
        "single": LENS_LIGHT_SINGLE,
        "bulge": LENS_LIGHT_BULGE,
        "disk": LENS_LIGHT_DISK,
    },
    clumps=CLUMPS,
    legacy_source_prior=SOURCE_PRIOR_V1,
    legacy_lens_light_prior=LENS_LIGHT_SINGLE,
    # The companion galaxy sits at cutout ~(11, 11), which CROP excludes. The corner of
    # the crop should therefore be sky plus deflector wings only.
    companion_max=0.15,
    # Runs whose SVI conditioning is known-good or known-diverged. twosource is the run
    # whose covariance came out at 8.8e+11, whose float32 inverse was not a valid
    # precision matrix, and whose 29 parameters therefore all failed Rhat at once.
    svi_guard_reference={
        "seed0": True,
        "seed1": True,
        "inner3": True,
        "nsersic10": True,
        "twosource": False,
    },
    notes="HST WFC3/IR F140W, EXPTIME 1197.699 s, NCOMBINE 3. See README.md.",
)
