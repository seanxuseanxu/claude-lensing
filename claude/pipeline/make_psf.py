"""Build an empirical PSF by stacking field stars from a parent drizzled frame.

Lens-independent: the lens is named only by ``--lens``.

The heavy lifting is photutils'. ``DAOStarFinder`` detects, ``extract_stars`` cuts, and
``EPSFBuilder`` does the iterative effective-PSF construction -- sub-pixel recentring,
oversampled reconstruction, sigma-clipped combination. Reimplementing any of that would
only produce an unvalidated copy of a well-tested algorithm.

What is written here is the part photutils does not do:

1. **Star selection**, which is the judgement. A detection is kept only if it is bright
   enough to matter, faint enough to be linear, compact (``sharpness``), round
   (``roundness``), far from the frame edge, on a well-covered part of the drizzle
   (``WHT``), and free of any neighbour above a few per cent of its own peak in the
   surrounding annulus. Every threshold is an argument and every rejection is counted, so
   the selection is inspectable rather than a matter of taste.
2. **Post-processing to the pipeline's contract** -- block-sum the oversampled ePSF back to
   the science grid, crop to an odd size, check the centroid really does sit on the central
   pixel, normalise to unit sum. ``data.load_psf`` normalises too, so that part is
   belt-and-braces; the centring is not. A PSF whose centroid is off-centre is an
   astrometric bias the fit will silently absorb into the deflector centre.
3. **Reporting** -- star count, positions, peaks, FWHM, encircled energy, and a QA figure.
   The star count and positions go into the FITS header, so a PSF's own signal-to-noise is
   inspectable from the file rather than implied by prose.

Two practical notes, both measured on this filesystem:

- **Importing photutils takes ~240 s on a cold cache.** Background anything that touches
  this module rather than running it under a short timeout.
- **photutils 3.0's ``DAOStarFinder`` returns ``x_centroid`` / ``y_centroid``**, with
  underscores; the pre-3.0 ``xcentroid`` / ``ycentroid`` raise ``KeyError``. Its
  ``sharplo`` / ``sharphi`` / ``roundlo`` / ``roundhi`` arguments are deprecated in favour
  of ``sharpness_range`` / ``roundness_range``, which is what this module passes.

Usage::

    python claude/pipeline/make_psf.py --lens DESI234 --frame final_92_drz.fits \\
        --out psf234.fits --size 27

    # Compare two PSFs (FWHM, radial profile, encircled energy).
    python claude/pipeline/make_psf.py --compare claude/DESI234/psf234.fits \\
        claude/DESI238/psf238.fits
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import lenses
from make_cutout import first_image_hdu

# ------------------------------------------------------------------------------------
# Selection defaults
#
# These are the thresholds the star selection runs at unless overridden. They are named
# constants rather than bare numbers in a signature so that the report can print them and
# a change is visible in a diff.
# ------------------------------------------------------------------------------------

# Detection threshold, in units of the frame's own sigma-clipped sky sigma.
DETECT_NSIGMA = 5.0
# Starting FWHM guess for the finder, in science pixels. WFC3/IR F140W drizzled to
# 0.065"/pix has a PSF FWHM of roughly 0.14" = 2.2 px.
DETECT_FWHM = 2.2
# Peak flux window. The upper bound is in the frame's units (ELECTRONS/S) and rejects
# saturated and non-linear stars.
PEAK_MAX = 50.0
# The lower bound is a multiple of the frame's own sigma-clipped sky sigma, not an absolute
# flux, because a threshold in e/s means nothing on a frame with a different exposure time.
# Fifty: EPSFBuilder normalises each star by its flux before combining, so a faint star's
# wing -- which is where its local background sits -- is amplified by the same factor that
# scales its core. At 50 sigma a wing pixel at 1% of the peak is still a half-sigma
# measurement; at 15 sigma it is a sixth of a sigma, and what gets stacked there is the
# star's local background rather than the PSF.
#
# This is measured, not assumed. On the DESI238 frame the full 51-star stack, reaching down
# to 21 sigma, put the kernel's flux-weighted centroid 0.24 px off centre in column while
# its core stayed centred to 0.06 px -- an asymmetric outer wing, exactly the predicted
# failure. Cutting at 50 sigma (37 stars) brought the offset to 0.09 px and reproduced the
# delivered psf238.fits to 0.001 px in FWHM. See claude/DESI238/PSF_REPLICATION.md.
PEAK_NSIGMA = 50.0
# DAOStarFinder shape cuts. Extended objects (galaxies) have low sharpness; cosmic rays
# and hot pixels have high sharpness. Elongated objects fail roundness.
SHARPNESS_RANGE = (0.3, 0.9)
ROUNDNESS_RANGE = (-0.4, 0.4)
# Neighbour rejection: in an annulus from the star's cutout half-width outward, no pixel
# may exceed this fraction of the star's own peak.
NEIGHBOUR_FRAC = 0.10
NEIGHBOUR_ANNULUS = (8, 20)  # inner, outer radius in science pixels
# Distance from the frame edge, in science pixels, below which a detection is dropped.
EDGE_MARGIN = 60
# Minimum WHT in the star's box, as a fraction of the frame's median WHT. Drizzle mosaic
# edges and chip gaps have low weight and correspondingly inflated noise.
WHT_FRAC = 0.5
# Oversampling for the ePSF reconstruction. Must be **odd**: block-summing an oversampled
# grid back to the science grid only aligns the central science pixel with the ePSF centre
# when each block has a middle element.
OVERSAMPLING = 3
# Size cut: nothing in the field is more compact than a point source, so stars sit at the
# small-size edge of the size distribution, not somewhere in its middle. The cut anchors on
# that edge and keeps only sources within SIZE_TOL of it, rather than trying to fit a
# bimodal population or pick an absolute size in pixels that would not travel to another
# frame's PSF. sharpness and roundness alone do not do this job: measured on a DESI238
# frame, they let 21 of 39 selected objects through that were actually compact galaxies,
# stacking to a 3.82 px FWHM kernel with 40% encircled energy at r=3 against the true
# stars' 2.83 px and 70%. SIZE_TOL = 1.10 is a 10% tolerance above the anchor; on that same
# frame it cleanly separated a stellar group measuring 2.25-2.48 px from an extended group
# at 2.6-2.9 px, with a gap between them.
SIZE_TOL = 1.10
# Radius, in science pixels, within which the size is measured. Kept small deliberately: a
# neighbour just outside the core must not be able to inflate a star's measured size.
SIZE_RMAX = 5.0
# Annulus, in science pixels, for measuring the background under an individual star. A
# single sigma-clipped median over the whole mosaic is one number standing in for a
# background that varies across a 2258x1997 frame; each star needs its own. The inner
# radius is the tension: too small and the annulus samples real PSF halo, which then gets
# subtracted off as though it were sky and biases the wings low; too large and it stops
# describing the background where the star actually sits. 15 px is roughly five times the
# PSF FWHM on these frames, where the halo has fallen to the 1e-4 level of the peak, so the
# annulus is sky, not wing.
BKG_ANNULUS = (15.0, 25.0)


def sky_stats(image):
    """Sigma-clipped ``(median, sigma)`` of a frame, for the detection threshold."""
    from astropy.stats import sigma_clipped_stats

    mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    return float(median), float(std)


def detect(image, threshold, fwhm=DETECT_FWHM,
           sharpness_range=SHARPNESS_RANGE, roundness_range=ROUNDNESS_RANGE):
    """Run ``DAOStarFinder`` and return its table.

    Columns (photutils 3.0.0): ``id, x_centroid, y_centroid, sharpness, roundness1,
    roundness2, n_pixels, peak, flux, mag, daofind_mag``. ``sharpness``, ``roundness1/2``
    and ``peak`` are exactly the star-selection cuts, so they come free from the finder and
    need no separate measurement.
    """
    from photutils.detection import DAOStarFinder

    finder = DAOStarFinder(
        threshold=threshold,
        fwhm=fwhm,
        sharpness_range=sharpness_range,
        roundness_range=roundness_range,
        exclude_border=True,
    )
    return finder(image)


def source_size(image, x, y, rmax=SIZE_RMAX, sky=0.0):
    """Flux-weighted rms radius of a source, in science pixels.

    Measured on positive flux inside ``rmax`` only. This is a size, not a fit: it makes no
    assumption about the profile, which is the point -- it has to rank a point source
    against a galaxy without a model for either.
    """
    r = int(np.ceil(rmax))
    row, col = int(round(y)), int(round(x))
    box = np.asarray(image, dtype=np.float64)[row - r:row + r + 1, col - r:col + r + 1]
    if box.shape != (2 * r + 1, 2 * r + 1):
        return np.nan
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rr2 = xx ** 2 + yy ** 2
    w = np.clip(box - sky, 0.0, None) * (rr2 < rmax ** 2)
    total = w.sum()
    if total <= 0:
        return np.nan
    return float(np.sqrt((w * rr2).sum() / total))


def local_background(image, x, y, rin=BKG_ANNULUS[0], rout=BKG_ANNULUS[1]):
    """Median background in an annulus around a source, in the image's own units.

    A median rather than a mean, so a single neighbour caught in the annulus -- which the
    star's own neighbour cut permits, since that cut only reaches out to
    ``NEIGHBOUR_ANNULUS`` and only at the bright end -- pulls the answer only as far as the
    next ring pixel, not toward its own flux.

    Returns ``nan`` if the annulus does not fit inside the frame or is empty.
    """
    r = int(np.ceil(rout))
    row, col = int(round(y)), int(round(x))
    box = np.asarray(image, dtype=np.float64)[row - r:row + r + 1, col - r:col + r + 1]
    if box.shape != (2 * r + 1, 2 * r + 1):
        return np.nan
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(yy, xx)
    ring = box[(d >= rin) & (d <= rout)]
    ring = ring[np.isfinite(ring)]
    return float(np.median(ring)) if ring.size else np.nan


class _Rejections(dict):
    """A rejection-reason-to-count dict that also carries the size cut's anchor.

    Compares and prints exactly like a plain dict -- ``dict.__eq__`` only looks at entries,
    not instance attributes -- so nothing that treats ``select_stars``'s second return value
    as a dict needs to change.
    """
    size_anchor_px = None


def select_stars(image, sources, wht=None, size=27,
                 peak_min=0.0, peak_max=PEAK_MAX,
                 neighbour_frac=NEIGHBOUR_FRAC, annulus=NEIGHBOUR_ANNULUS,
                 edge_margin=EDGE_MARGIN, wht_frac=WHT_FRAC, exclude=(),
                 size_tol=SIZE_TOL, sky=0.0):
    """Filter a detection table down to PSF stars, and say why each one was dropped.

    Args:
        image: 2-D parent frame.
        sources: table from :func:`detect`, or any table with ``x_centroid``,
            ``y_centroid`` and ``peak`` columns.
        wht: the drizzle weight extension, or ``None`` to skip the coverage cut.
        size: the science-pixel cutout size the stars will be extracted at. Used to keep
            the whole cutout inside the frame.
        peak_min, peak_max: retained peak-flux window, in the frame's units. ``peak_min``
            defaults to no cut at all here; :func:`build` is what turns ``PEAK_NSIGMA``
            into a flux, because only it has measured the frame's sky sigma.
        neighbour_frac: reject if any pixel in ``annulus`` exceeds this fraction of the
            star's peak.
        annulus: ``(r_in, r_out)`` in science pixels for the neighbour search.
        edge_margin: minimum distance from the frame edge, in science pixels.
        wht_frac: minimum median WHT in the star box as a fraction of the frame median.
        exclude: iterable of ``(row, col, radius)`` keep-out zones -- the lens itself, for
            one, which must never enter its own PSF.
        size_tol: keep survivors with :func:`source_size` no larger than this multiple of
            the population's own small-size anchor. See ``SIZE_TOL`` for why.
        sky: sky level to subtract before measuring size, passed to :func:`source_size`.

    Returns:
        ``(kept, rejections)``: a list of dicts, one per accepted star, and a dict mapping
        rejection reason to count. Each dict in ``kept`` also carries ``"size"`` and
        ``"background"`` keys, the latter possibly ``nan`` if the star's background annulus
        does not fit inside the frame -- :func:`build_epsf` is what falls back to the
        global median for those. ``rejections`` is a plain dict for equality and printing,
        but it is actually a
        :class:`_Rejections`, which additionally carries the size cut's anchor as the
        attribute ``size_anchor_px`` (``None`` if no survivor reached the size cut) -- a
        place to hang one extra number without changing the return arity every caller
        already unpacks.
    """
    ny, nx = image.shape
    half = size // 2
    r_in, r_out = annulus
    wht_floor = None
    if wht is not None:
        med = float(np.median(wht))
        wht_floor = wht_frac * med

    kept, rejections = [], _Rejections()

    def drop(reason):
        rejections[reason] = rejections.get(reason, 0) + 1

    for row in sources:
        x = float(row["x_centroid"])
        y = float(row["y_centroid"])
        peak = float(row["peak"])
        r, c = int(round(y)), int(round(x))

        if not (edge_margin <= c < nx - edge_margin and edge_margin <= r < ny - edge_margin):
            drop("edge")
            continue
        if r - half < 0 or r + half + 1 > ny or c - half < 0 or c + half + 1 > nx:
            drop("cutout off frame")
            continue
        if peak > peak_max:
            drop("saturated / non-linear")
            continue
        if peak < peak_min:
            drop("too faint")
            continue
        if any((r - er) ** 2 + (c - ec) ** 2 < rad ** 2 for er, ec, rad in exclude):
            drop("in keep-out zone")
            continue

        if wht_floor is not None:
            wbox = wht[r - half:r + half + 1, c - half:c + half + 1]
            if float(np.median(wbox)) < wht_floor:
                drop("low WHT coverage")
                continue

        # Neighbour test. Anything bright in the annulus would be inside the extracted
        # cutout's outer region or just beyond it, contaminating the wings -- which is
        # precisely the part of the PSF a faint-star stack is least able to defend.
        a0r, a1r = max(r - r_out, 0), min(r + r_out + 1, ny)
        a0c, a1c = max(c - r_out, 0), min(c + r_out + 1, nx)
        patch = np.asarray(image[a0r:a1r, a0c:a1c], dtype=np.float64)
        rr, cc = np.mgrid[a0r:a1r, a0c:a1c]
        rad2 = (rr - y) ** 2 + (cc - x) ** 2
        ring = (rad2 >= r_in ** 2) & (rad2 <= r_out ** 2)
        if ring.any() and float(patch[ring].max()) > neighbour_frac * peak:
            drop("neighbour in annulus")
            continue

        kept.append({
            "x": x, "y": y, "peak": peak,
            "sharpness": float(row["sharpness"]) if "sharpness" in row.colnames else None,
            "roundness1": float(row["roundness1"]) if "roundness1" in row.colnames else None,
            "flux": float(row["flux"]) if "flux" in row.colnames else None,
        })

    # Size cut. Unlike every cut above, this one is relative to the surviving population
    # rather than to a fixed threshold, so it cannot be folded into the per-source loop --
    # it needs every survivor's size in hand before it can place the anchor.
    for star in kept:
        star["size"] = source_size(image, star["x"], star["y"], sky=sky)
        star["background"] = local_background(image, star["x"], star["y"])

    measurable = [s for s in kept if np.isfinite(s["size"])]
    for _ in range(len(kept) - len(measurable)):
        drop("size unmeasurable")
    kept = measurable

    if kept:
        sizes = [s["size"] for s in kept]
        anchor = float(np.percentile(sizes, 10)) if len(sizes) >= 10 else float(min(sizes))
        rejections.size_anchor_px = anchor
        survivors = [s for s in kept if s["size"] <= size_tol * anchor]
        for _ in range(len(kept) - len(survivors)):
            drop("extended (not point-like)")
        kept = survivors

    return kept, rejections


def epsf_grid_shape(size, oversampling):
    """Side of the oversampled ePSF grid to ask ``EPSFBuilder`` for.

    Two constraints, from opposite directions. photutils requires at least
    ``oversampling * size + 1`` -- one oversampled row and column beyond the star cutout,
    so the sub-pixel recentring shifts have somewhere to go; ``oversampling * size`` on its
    own is rejected outright. :func:`to_science_grid` requires an odd side, because it
    locates the centre as ``n // 2`` and an even grid has no middle element to put the
    central science pixel on. So: the smallest odd value that clears the minimum. The extra
    margin is trimmed off in :func:`to_science_grid`, so it costs nothing downstream.
    """
    minimum = size * oversampling + 1
    return minimum + 1 if minimum % 2 == 0 else minimum


def build_epsf(image, stars, size=27, oversampling=OVERSAMPLING, maxiters=8,
               sky_median=0.0):
    """Extract the selected stars and run ``EPSFBuilder``.

    Returns ``(epsf_data, n_used)``. ``epsf_data`` is on the **oversampled** grid; use
    :func:`to_science_grid` to bring it back.

    Each star's own ``"background"`` (from :func:`local_background`, via
    :func:`select_stars`) is subtracted from its own box, not one global sky value for the
    whole frame. On a large drizzled mosaic one sigma-clipped median does not describe the
    background under every star; the stars sitting on locally low background get
    over-subtracted and the rest under-subtracted, and because ``EPSFBuilder`` normalises
    each star by its own flux before stacking, those pedestals survive into the combined
    kernel as spurious flux in the wings. A star with no usable local measurement (its
    annulus ran off the frame) falls back to ``sky_median``.

    The subtraction is done on a copy of the frame, box by box, rather than by constructing
    ``EPSFStar`` objects directly -- that lower-level photutils API is not one this module
    has validated against a reference. Two adjacent stars' boxes can overlap, in which case
    the overlap gets its background subtracted twice; this is accepted rather than handled,
    because the neighbour cut in :func:`select_stars` has already removed any star with a
    bright neighbour nearby, so an overlap here is between two faint boxes and the second
    subtraction is a small correction on top of a small correction.
    """
    from astropy.nddata import NDData
    from astropy.table import Table
    from photutils.psf import EPSFBuilder, extract_stars

    if oversampling % 2 == 0:
        raise ValueError(
            f"oversampling must be odd (got {oversampling}); an even factor gives no "
            f"middle element to align the central science pixel on"
        )
    arr = np.asarray(image, dtype=np.float64).copy()
    half = size // 2
    for s in stars:
        b = s.get("background")
        if b is None or not np.isfinite(b):
            b = sky_median
        r0, c0 = int(round(s["y"])) - half, int(round(s["x"])) - half
        arr[r0:r0 + size, c0:c0 + size] -= b
    data = NDData(data=arr)
    cat = Table({"x": [s["x"] for s in stars], "y": [s["y"] for s in stars]})
    extracted = extract_stars(data, cat, size=size)

    builder = EPSFBuilder(
        oversampling=oversampling,
        shape=epsf_grid_shape(size, oversampling),
        maxiters=maxiters,
        progress_bar=False,
    )
    epsf, fitted = builder(extracted)
    return np.asarray(epsf.data, dtype=np.float64), len(extracted)


def recentre_epsf(epsf, oversampling=OVERSAMPLING):
    """Shift an oversampled ePSF onto its own centroid, to the nearest oversampled pixel.

    ``EPSFBuilder`` does not guarantee its output is centred on the grid's central element
    -- measured on a real frame, an 83x83 ePSF's flux-weighted centroid sat at (40.68,
    39.91) against a central element of (41, 41). :func:`to_science_grid` lays its blocks
    out from that central element, so any such offset passes straight through the block-sum
    into the delivered kernel, and it is exactly the astrometric bias
    :func:`check_contract` exists to catch.

    The correction is an integer reindex, not a resample: ``np.roll`` shifts the array by a
    whole number of oversampled pixels, which moves flux without interpolating it, so it is
    exact and flux-conserving. ``np.roll`` wraps rather than zero-fills, though, and a wrap
    would carry wing flux around to the opposite edge of the kernel -- worse than the offset
    it is meant to fix -- so the wrapped-in edge is zeroed after the roll.

    Returns ``(shifted, (drow, dcol))`` with the shift applied, in oversampled pixels.
    """
    epsf = np.asarray(epsf, dtype=np.float64)
    ny, nx = epsf.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    row, col = centroid_of(epsf)
    drow = int(round(cy - row))
    dcol = int(round(cx - col))
    shifted = np.roll(epsf, (drow, dcol), axis=(0, 1))
    if drow > 0:
        shifted[:drow, :] = 0.0
    elif drow < 0:
        shifted[drow:, :] = 0.0
    if dcol > 0:
        shifted[:, :dcol] = 0.0
    elif dcol < 0:
        shifted[:, dcol:] = 0.0
    return shifted, (drow, dcol)


def to_science_grid(epsf, oversampling=OVERSAMPLING, size=27, recentre=True):
    """Block-sum an oversampled ePSF back onto the science grid, ``size`` odd.

    Summing rather than interpolating: a science pixel *integrates* the flux that the
    oversampled grid resolves, so a sum is the flux-conserving operation and introduces no
    smoothing of its own. Blocks are laid out from the ePSF's central element outward, so
    the centre of the central science pixel coincides with the ePSF centre exactly -- which
    is the whole reason ``oversampling`` is required to be odd. That coincidence is only as
    good as the assumption that the ePSF itself is centred on its central element, which
    ``EPSFBuilder`` does not guarantee; ``recentre=True`` (the default) corrects that first
    with :func:`recentre_epsf`. Callers that already know their input is centred -- a test
    fixture built that way on purpose -- can pass ``recentre=False`` to skip it; the shift
    would be a no-op there in any case.
    """
    if size % 2 == 0:
        raise ValueError(f"size must be odd (got {size}); an even kernel has no centre")
    ny, nx = epsf.shape
    if ny % 2 == 0 or nx % 2 == 0:
        raise ValueError(f"ePSF grid {epsf.shape} must be odd on both axes")
    if recentre:
        epsf, _ = recentre_epsf(epsf, oversampling=oversampling)
    cy, cx = ny // 2, nx // 2
    half, o2 = size // 2, oversampling // 2
    need = half * oversampling + o2
    if cy - need < 0 or cy + need + 1 > ny or cx - need < 0 or cx + need + 1 > nx:
        raise ValueError(
            f"ePSF grid {epsf.shape} is too small for a {size}x{size} science kernel at "
            f"oversampling {oversampling}: needs at least {2 * need + 1} on a side"
        )
    trimmed = epsf[cy - need:cy + need + 1, cx - need:cx + need + 1]
    out = trimmed.reshape(size, oversampling, size, oversampling).sum(axis=(1, 3))
    return np.ascontiguousarray(out)


def centroid_of(psf):
    """Flux-weighted ``(row, col)`` centroid of a kernel, in its own pixel coordinates."""
    w = np.clip(np.asarray(psf, dtype=np.float64), 0.0, None)
    rr, cc = np.mgrid[0:psf.shape[0], 0:psf.shape[1]]
    t = w.sum()
    return float((w * rr).sum() / t), float((w * cc).sum() / t)


def radial_profile(psf, binwidth=0.5):
    """Azimuthally averaged profile: ``(radius, mean_value)``, radius in pixels.

    Each bin reports the **mean radius of the pixels in it**, not the bin's nominal
    centre. That distinction is not cosmetic. Pixel radii are quantised as
    0, 1, sqrt(2), 2, sqrt(5), ..., so a bin's occupants sit well off its nominal centre;
    pairing a bin's mean value with its nominal radius biases the profile, and hence the
    FWHM, several per cent low on a kernel this narrow.
    """
    psf = np.asarray(psf, dtype=np.float64)
    cy, cx = (psf.shape[0] - 1) / 2.0, (psf.shape[1] - 1) / 2.0
    rr, cc = np.mgrid[0:psf.shape[0], 0:psf.shape[1]]
    r = np.hypot(rr - cy, cc - cx).ravel()
    v = psf.ravel()
    rmax = min(cy, cx)
    keep = r <= rmax
    r, v = r[keep], v[keep]
    nbins = int(np.ceil(rmax / binwidth)) + 1
    idx = np.clip((r / binwidth + 0.5).astype(int), 0, nbins - 1)
    n = np.bincount(idx, minlength=nbins)
    s = np.bincount(idx, weights=v, minlength=nbins)
    rs = np.bincount(idx, weights=r, minlength=nbins)
    ok = n > 0
    return rs[ok] / n[ok], s[ok] / n[ok]


def fwhm_of(psf):
    """FWHM in pixels, from the azimuthally averaged profile by linear interpolation.

    Measured on the profile rather than by fitting a Gaussian: a WFC3/IR PSF is not a
    Gaussian, and a Gaussian fit's FWHM depends on how much of the wing is included.
    """
    r, p = radial_profile(psf)
    half = p[0] / 2.0
    below = np.nonzero(p < half)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    r0, r1, p0, p1 = r[i - 1], r[i], p[i - 1], p[i]
    return float(2.0 * (r0 + (p0 - half) * (r1 - r0) / (p0 - p1)))


def encircled_energy(psf, radii=(3, 5, 10)):
    """Fraction of the (unit-normalised) kernel's flux within each radius, in pixels."""
    a = np.asarray(psf, dtype=np.float64)
    a = a / a.sum()
    cy, cx = (a.shape[0] - 1) / 2.0, (a.shape[1] - 1) / 2.0
    rr, cc = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    r = np.hypot(rr - cy, cc - cx)
    return {int(rad): float(a[r <= rad].sum()) for rad in radii}


def describe(psf):
    """``{fwhm_pix, ee, centroid, centroid_offset}`` for one kernel."""
    cy, cx = centroid_of(psf)
    centre = ((psf.shape[0] - 1) / 2.0, (psf.shape[1] - 1) / 2.0)
    return {
        "shape": list(psf.shape),
        "sum": float(np.sum(psf)),
        "fwhm_pix": fwhm_of(psf),
        "ee": encircled_energy(psf),
        "centroid_rowcol": [cy, cx],
        "centroid_offset_rowcol": [cy - centre[0], cx - centre[1]],
    }


def compare(psf_a, psf_b):
    """Compare two kernels after normalising both to unit sum and matching their sizes.

    Returns a dict with each kernel's description, the peak and RMS fractional difference,
    and the difference in FWHM and encircled energy. This is the form the DESI238
    replication test and the cross-lens PSF check both report.
    """
    a = np.asarray(psf_a, dtype=np.float64)
    b = np.asarray(psf_b, dtype=np.float64)
    a = a / a.sum()
    b = b / b.sum()
    n = min(a.shape[0], b.shape[0])
    if n % 2 == 0:
        n -= 1

    def centre_crop(k):
        h = k.shape[0] // 2
        m = n // 2
        return k[h - m:h + m + 1, h - m:h + m + 1]

    ac, bc = centre_crop(a), centre_crop(b)
    # Renormalise after cropping so the comparison is of shape, not of how much wing each
    # kernel happens to carry outside the common footprint.
    ac, bc = ac / ac.sum(), bc / bc.sum()
    diff = ac - bc
    da, db = describe(a), describe(b)
    return {
        "a": da,
        "b": db,
        "compared_size": int(n),
        "max_abs_frac_diff": float(np.max(np.abs(diff)) / bc.max()),
        "rms_frac_diff": float(np.sqrt(np.mean(diff ** 2)) / bc.max()),
        "fwhm_diff_pix": da["fwhm_pix"] - db["fwhm_pix"],
        "ee_diff": {k: da["ee"][k] - db["ee"][k] for k in da["ee"]},
    }


# ------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------


def read_frame(frame_path, ext=None, wht_ext=None):
    """Return ``(sci, wht, header)``. ``wht`` is ``None`` if the frame has no WHT."""
    from astropy.io import fits

    with fits.open(frame_path) as hdul:
        if ext is None:
            ext = first_image_hdu(hdul)
        sci = np.asarray(hdul[ext].data, dtype=np.float64)
        header = hdul[ext].header
        wht = None
        if wht_ext is None:
            for i, h in enumerate(hdul):
                if h.name.upper() == "WHT" and h.data is not None:
                    wht_ext = i
                    break
        if wht_ext is not None:
            wht = np.asarray(hdul[wht_ext].data, dtype=np.float64)
    return sci, wht, header


def build(frame_path, size=27, oversampling=OVERSAMPLING, exclude=(), ext=None,
          detect_nsigma=DETECT_NSIGMA, peak_nsigma=PEAK_NSIGMA, **select_kw):
    """Detect, select, stack. Returns ``(psf, report)``.

    ``psf`` is on the science grid, odd-sized, unit sum. ``report`` carries everything the
    selection did, so the PSF's provenance travels with it.
    """
    sci, wht, header = read_frame(frame_path, ext=ext)
    sky_median, sky_sigma = sky_stats(sci)
    sources = detect(sci, threshold=detect_nsigma * sky_sigma)
    n_detected = 0 if sources is None else len(sources)
    if n_detected == 0:
        raise RuntimeError(f"no detections at {detect_nsigma} sigma in {frame_path}")

    # The faint cut is set by the frame's own noise unless the caller names a flux.
    if select_kw.get("peak_min") is None:
        select_kw["peak_min"] = peak_nsigma * sky_sigma

    stars, rejections = select_stars(sci, sources, wht=wht, size=size,
                                     exclude=exclude, sky=sky_median, **select_kw)
    if len(stars) < 3:
        raise RuntimeError(
            f"only {len(stars)} stars survived selection in {Path(frame_path).name} "
            f"(rejections: {rejections}); loosen a threshold deliberately or use another "
            f"frame's PSF, but do not stack two stars and call it a PSF"
        )

    epsf, n_used = build_epsf(sci, stars, size=size, oversampling=oversampling,
                              sky_median=sky_median)
    # Recentred here, explicitly, rather than left to to_science_grid's own default, so the
    # shift that was applied is in hand to record in the report -- a silent correction is
    # not something the report should have to reconstruct after the fact.
    epsf, epsf_shift = recentre_epsf(epsf, oversampling=oversampling)
    psf = to_science_grid(epsf, oversampling=oversampling, size=size, recentre=False)
    psf = psf / psf.sum()

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame": Path(frame_path).name,
        "sky_median": sky_median,
        "sky_sigma": sky_sigma,
        "n_detected": int(n_detected),
        "n_selected": len(stars),
        "n_stacked": int(n_used),
        "rejections": rejections,
        "thresholds": {
            "detect_nsigma": detect_nsigma,
            "peak_nsigma": peak_nsigma,
            "peak_min": select_kw["peak_min"],
            "peak_max": select_kw.get("peak_max", PEAK_MAX),
            "sharpness_range": list(SHARPNESS_RANGE),
            "roundness_range": list(ROUNDNESS_RANGE),
            "neighbour_frac": select_kw.get("neighbour_frac", NEIGHBOUR_FRAC),
            "neighbour_annulus": list(select_kw.get("annulus", NEIGHBOUR_ANNULUS)),
            "edge_margin": select_kw.get("edge_margin", EDGE_MARGIN),
            "wht_frac": select_kw.get("wht_frac", WHT_FRAC),
            "size_tol": select_kw.get("size_tol", SIZE_TOL),
            "size_anchor_px": rejections.size_anchor_px,
            "bkg_annulus": list(BKG_ANNULUS),
            "epsf_recentre_oversampled_px": list(epsf_shift),
            "size": size,
            "oversampling": oversampling,
        },
        "stars": stars,
        "psf": describe(psf),
    }
    return psf, report


def write_psf(out_path, psf, report):
    """Write the kernel to FITS, recording the star inventory in its own header.

    The star count, their pixel positions and their peak fluxes go in the header so that a
    PSF file carries its own signal-to-noise evidence. A PSF stacked from four faint stars
    and one stacked from forty bright ones are different objects, and nothing else in the
    file distinguishes them.
    """
    from astropy.io import fits

    out_path = Path(out_path)
    hdu = fits.PrimaryHDU(np.asarray(psf, dtype=np.float64))
    h = hdu.header
    h["ORIGIN"] = ("claude/pipeline/make_psf.py", "builder")
    h["FRAME"] = (report["frame"], "parent drizzled frame")
    h["NSTARS"] = (report["n_stacked"], "stars in the stack")
    h["NDETECT"] = (report["n_detected"], "DAOStarFinder detections")
    h["OVERSAMP"] = (report["thresholds"]["oversampling"], "ePSF oversampling")
    h["SKYSIG"] = (report["sky_sigma"], "sigma-clipped sky sigma of the frame")
    h["FWHMPIX"] = (report["psf"]["fwhm_pix"], "FWHM from the radial profile, pixels")
    h["EE3"] = (report["psf"]["ee"][3], "encircled energy within 3 pix")
    h["EE5"] = (report["psf"]["ee"][5], "encircled energy within 5 pix")
    h["EE10"] = (report["psf"]["ee"][10], "encircled energy within 10 pix")
    h["CENOFFR"] = (report["psf"]["centroid_offset_rowcol"][0], "centroid offset, rows")
    h["CENOFFC"] = (report["psf"]["centroid_offset_rowcol"][1], "centroid offset, cols")
    h["DATE"] = (report["created_utc"], "build time, UTC")
    for i, s in enumerate(report["stars"], start=1):
        h[f"STARX{i:02d}"] = (round(s["x"], 2), f"star {i} x, parent frame")
        h[f"STARY{i:02d}"] = (round(s["y"], 2), f"star {i} y, parent frame")
        h[f"STARP{i:02d}"] = (round(s["peak"], 4), f"star {i} peak, e/s")
    hdu.writeto(out_path, overwrite=True)

    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(report, indent=2) + "\n")
    return sidecar


def check_contract(psf, tol=0.15):
    """Assert the kernel meets what ``data.load_psf`` assumes. Raises on violation.

    ``data.load_psf`` checks only that the kernel is odd-sized -- centring and
    normalisation it takes on trust, because neither is checkable from a bare PSF file.
    This is where that trust is earned.
    """
    psf = np.asarray(psf, dtype=np.float64)
    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        raise ValueError(f"PSF is {psf.shape}: an even-sized kernel has no centre pixel")
    if not np.isfinite(psf).all():
        raise ValueError("PSF contains non-finite values")
    total = psf.sum()
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"PSF sums to {total!r}, not 1")
    dr, dc = describe(psf)["centroid_offset_rowcol"]
    if abs(dr) > tol or abs(dc) > tol:
        raise ValueError(
            f"PSF centroid is ({dr:+.3f}, {dc:+.3f}) px from the central pixel, beyond "
            f"{tol} px. An off-centre kernel is an astrometric bias the fit absorbs "
            f"silently into the deflector centre."
        )
    return dr, dc


def qa_figure(psf, report, sci, out_path, size=27):
    """Individual stars, the stack, and the radial profile, in one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    stars = report["stars"]
    ncol = min(8, max(1, len(stars)))
    nrow = 1 + int(np.ceil(len(stars) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.2 * nrow))
    axes = np.atleast_2d(axes)
    half = size // 2
    for ax in axes.ravel():
        ax.axis("off")
    for i, s in enumerate(stars):
        ax = axes.ravel()[i]
        r, c = int(round(s["y"])), int(round(s["x"]))
        cut = sci[r - half:r + half + 1, c - half:c + half + 1]
        ax.imshow(cut, origin="lower", cmap="magma")
        ax.set_title(f"{s['peak']:.2f} e/s", fontsize=7)
        ax.axis("off")

    ax = axes.ravel()[ncol * (nrow - 1)]
    ax.axis("on")
    ax.imshow(psf, origin="lower", cmap="magma",
              norm=LogNorm(vmin=max(psf.max() * 1e-4, psf[psf > 0].min()), vmax=psf.max()))
    ax.set_title(f"stack, {report['n_stacked']} stars", fontsize=8)
    ax.set_xticks([]), ax.set_yticks([])

    ax = axes.ravel()[ncol * (nrow - 1) + 1] if ncol > 1 else ax
    ax.axis("on")
    r, p = radial_profile(psf)
    ax.semilogy(r, np.clip(p, 1e-8, None), "o-", ms=3)
    ax.axvline(report["psf"]["fwhm_pix"] / 2, ls=":", c="k", lw=0.8)
    ax.set_xlabel("r (pix)", fontsize=7)
    ax.set_ylabel("mean", fontsize=7)
    ax.set_title(f"FWHM {report['psf']['fwhm_pix']:.2f} px", fontsize=8)
    ax.tick_params(labelsize=6)

    fig.suptitle(f"{report['frame']}: PSF from {report['n_stacked']} stars", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _print_report(report):
    t = report["thresholds"]
    print(f"frame       {report['frame']}")
    print(f"sky         median {report['sky_median']:+.5f}  sigma {report['sky_sigma']:.5f}")
    print(f"detections  {report['n_detected']} at {t['detect_nsigma']:g} sigma")
    print(f"faint cut   peak >= {t['peak_nsigma']:g} sigma = {t['peak_min']:.3f} e/s")
    if t["size_anchor_px"] is not None:
        print(f"size cut    locus {t['size_anchor_px']:.3f} px, keep <= "
              f"{t['size_tol'] * t['size_anchor_px']:.3f} px (tol {t['size_tol']:.2f})")
    for reason, n in sorted(report["rejections"].items(), key=lambda kv: -kv[1]):
        print(f"  rejected  {n:5d}  {reason}")
    print(f"selected    {report['n_selected']}   stacked {report['n_stacked']}")
    peaks = [s["peak"] for s in report["stars"]]
    if peaks:
        print(f"peaks       {min(peaks):.2f} - {max(peaks):.2f} e/s "
              f"(median {np.median(peaks):.2f})")
    bkgs = [s["background"] for s in report["stars"]
            if np.isfinite(s.get("background", np.nan))]
    if bkgs:
        lo, hi = np.percentile(bkgs, [16, 84])
        print(f"background  per-star median {np.median(bkgs):+.5f}, 16-84 pct "
              f"{lo:+.5f} to {hi:+.5f} (global {report['sky_median']:+.5f})")
    dr_over, dc_over = t["epsf_recentre_oversampled_px"]
    o = t["oversampling"]
    print(f"recentre    ePSF shifted by ({dr_over:+d}, {dc_over:+d}) oversampled px = "
          f"({dr_over / o:+.3f}, {dc_over / o:+.3f}) science px")
    d = report["psf"]
    print(f"PSF         {d['shape'][0]}x{d['shape'][1]}  FWHM {d['fwhm_pix']:.2f} px  "
          f"EE(3/5/10) {d['ee'][3]:.3f} / {d['ee'][5]:.3f} / {d['ee'][10]:.3f}")
    print(f"centroid    offset ({d['centroid_offset_rowcol'][0]:+.3f}, "
          f"{d['centroid_offset_rowcol'][1]:+.3f}) px from centre")


def _print_comparison(cmp, name_a="A", name_b="B"):
    print(f"{'':12s}{name_a:>16s}{name_b:>16s}{'diff':>12s}")
    print(f"{'FWHM (pix)':12s}{cmp['a']['fwhm_pix']:16.3f}{cmp['b']['fwhm_pix']:16.3f}"
          f"{cmp['fwhm_diff_pix']:+12.3f}")
    for rad in sorted(cmp["a"]["ee"]):
        print(f"{'EE(' + str(rad) + ' pix)':12s}{cmp['a']['ee'][rad]:16.4f}"
              f"{cmp['b']['ee'][rad]:16.4f}{cmp['ee_diff'][rad]:+12.4f}")
    print(f"compared over the central {cmp['compared_size']}x{cmp['compared_size']}: "
          f"max |diff| = {cmp['max_abs_frac_diff'] * 100:.1f}% of peak, "
          f"rms = {cmp['rms_frac_diff'] * 100:.2f}% of peak")


def main():
    from astropy.io import fits

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    lenses.add_argument(ap, required=False)
    ap.add_argument("--dir", default=None, help="lens directory, if --lens has no config")
    ap.add_argument("--frame", help="parent FITS frame, relative to the lens directory")
    ap.add_argument("--out", default=None, help="output FITS, relative to the lens directory")
    ap.add_argument("--size", type=int, default=27, help="kernel side, odd, science pixels")
    ap.add_argument("--oversampling", type=int, default=OVERSAMPLING)
    ap.add_argument("--detect-nsigma", type=float, default=DETECT_NSIGMA)
    ap.add_argument("--peak-nsigma", type=float, default=PEAK_NSIGMA,
                    help="faint cut on peak flux, in sky sigma")
    ap.add_argument("--peak-min", type=float, default=None,
                    help="faint cut as an absolute flux, overriding --peak-nsigma")
    ap.add_argument("--peak-max", type=float, default=PEAK_MAX)
    ap.add_argument("--neighbour-frac", type=float, default=NEIGHBOUR_FRAC)
    ap.add_argument("--edge-margin", type=int, default=EDGE_MARGIN)
    ap.add_argument("--exclude", default=None,
                    help="keep-out zones as 'row,col,radius;row,col,radius' in parent pixels")
    ap.add_argument("--qa", default=None, help="write a QA PNG here")
    ap.add_argument("--compare", nargs=2, default=None,
                    help="compare two existing PSF files and exit")
    args = ap.parse_args()

    if args.compare:
        a, b = (np.asarray(fits.getdata(p), dtype=np.float64) for p in args.compare)
        _print_comparison(compare(a, b), Path(args.compare[0]).name,
                          Path(args.compare[1]).name)
        return

    if args.dir is not None:
        root = Path(args.dir)
    elif args.lens is not None:
        root = lenses.CLAUDE_DIR / args.lens
    else:
        ap.error("one of --lens, --dir or --compare is required")
    if args.frame is None:
        ap.error("--frame is required")

    exclude = ()
    if args.exclude:
        exclude = tuple(
            tuple(float(v) for v in z.split(","))
            for z in args.exclude.split(";") if z.strip()
        )

    frame = Path(args.frame)
    if not frame.is_absolute():
        frame = root / frame

    psf, report = build(frame, size=args.size, oversampling=args.oversampling,
                        exclude=exclude, detect_nsigma=args.detect_nsigma,
                        peak_nsigma=args.peak_nsigma,
                        peak_min=args.peak_min, peak_max=args.peak_max,
                        neighbour_frac=args.neighbour_frac, edge_margin=args.edge_margin)
    _print_report(report)
    check_contract(psf)
    print("contract    odd-sized, unit sum, centred")

    if args.out:
        sidecar = write_psf(root / args.out, psf, report)
        print(f"wrote       {args.out} and {sidecar.name}")
    if args.qa:
        sci, _, _ = read_frame(frame)
        qa_figure(psf, report, sci, root / args.qa, size=args.size)
        print(f"wrote       {args.qa}")


if __name__ == "__main__":
    main()
