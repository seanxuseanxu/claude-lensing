"""Load and condition the DESI238 cutout and PSF.

Every number here was measured from the files, not taken from documentation. Where a
value contradicts ``CLAUDE.md``, the discrepancy is explained in a comment.
"""

import os

import numpy as np
from astropy.io import fits

from gigalens.simulator import SimulatorConfig

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DELTA_PIX = 0.065  # arcsec/pixel, from the SCI CD matrix: sqrt(|det CD|) * 3600
NUM_PIX = 80
SUPERSAMPLE = 2
EXP_TIME = 1197.699  # EXPTIME header keyword

# Background noise sigma, in ELECTRONS/S.
#
# CLAUDE.md quotes 0.0315 from sigma_clipped_stats over the whole 120x120 cutout. That
# number is contaminated: the cutout is mostly galaxy, so the clip never lands on sky.
# Measured instead from six widely separated 60x60 blank patches in the parent drizzle
# frame (final_94_drz.fits, HDU 1):
#
#   patch(300,300)   mean=-0.00198  std=0.00731
#   patch(1800,400)  mean=-0.00046  std=0.00768
#   patch(400,1600)  mean=+0.00137  std=0.00915
#   patch(1700,1500) mean=-0.00170  std=0.00731
#   patch(900,200)   mean=-0.00061  std=0.00794
#   patch(1400,1800) mean=-0.00146  std=0.00701
#
# Consistent at 0.0070-0.0092. Take 0.0075.
BACKGROUND_RMS = 0.0075

# Crop bounds into the 120x120 cutout.
#
# The deflector centroid sits at cutout pixel (row 57.48, col 57.64). [18:98, 18:98]
# places it at (39.48, 39.64) against an array centre of (39.5, 39.5) -- centred to
# better than a tenth of a pixel in both axes. It also excludes the companion galaxy at
# ~(11, 11), which matters because gigalens has no pixel masking: `grep -rn "mask"
# gigalens/src` returns nothing, so the likelihood runs over every pixel unconditionally.
# A companion inside the frame would be fit as if it were part of the lens system.
CROP = (slice(18, 98), slice(18, 98))

# Deflector centroid within the crop, in pixels (row, col).
CENTROID_PIX = (57.48 - 18, 57.64 - 18)


def load_image():
    """Return the 80x80 science crop as native-endian float32, in ELECTRONS/S.

    No sky is subtracted. The crop's outer ring reads ~0.035 e/s, but that is the
    deflector's own Sersic wings, not sky: a radial profile in the parent frame falls
    smoothly 0.057 (2.3") -> 0.0054 (5.9") -> 0.0007 (24"), and true blank sky is
    consistent with zero (see BACKGROUND_RMS above). Subtracting 0.035 would remove real
    galaxy light and bias the lens-light R_sersic and n_sersic low. The Sersic profile is
    there to model those wings; let it.
    """
    path = os.path.join(DATA_DIR, "cutout238.npy")
    # The file is dtype '>f4'. JAX rejects big-endian arrays outright
    # (TypeError: Error interpreting argument as a JAX value), so cast before slicing.
    arr = np.load(path).astype(np.float32)
    return np.ascontiguousarray(arr[CROP])


def load_psf():
    """Return the 27x27 PSF, native-endian float32, normalised to unit sum.

    The file sums to 1.1445, so it must be divided by its sum or every simulated image is
    scaled up by 14%. It is odd-sized and already centred (flux centroid within 0.12 px of
    pixel (13, 13)), so no recentring is needed.

    The PSF carries no WCS or scale keywords. It is at the same 0.065"/pix as the science
    image -- confirmed by the repository owner, not derivable from the file. It is
    therefore not oversampled relative to the drizzled grid, and lenstronomy's
    subgrid_kernel (called by LensSimulator when supersample > 1) handles the rest.
    """
    path = os.path.join(DATA_DIR, "psf238.fits")
    psf = fits.getdata(path).astype(np.float32)
    return np.ascontiguousarray(psf / psf.sum())


def load():
    """Return ``(image, psf, sim_config)`` ready for LensSimulator and ForwardProbModel."""
    image = load_image()
    psf = load_psf()
    sim_config = SimulatorConfig(
        delta_pix=DELTA_PIX,
        num_pix=NUM_PIX,
        supersample=SUPERSAMPLE,
        kernel=psf,
    )
    return image, psf, sim_config
