"""Cut a science window out of a parent drizzled frame, with provenance.

Lens-independent, like everything else in this directory: the lens is named only by
``--lens``, and the sky position by ``--ra``/``--dec``.

Two decisions are baked in, and both are deliberate.

**The window is centred on the measured light centroid, not on the WCS position.** For
all three frames in this repository the catalogue position and the centroid of the light
differ by 3-5 pixels, which is 0.2-0.3 arcsec. A window centred on the WCS position puts
the deflector visibly off-centre, and the deflector-centre prior in ``priors.py`` is a
Normal of sigma ~0.05 arcsec about the crop centre -- so a 0.3 arcsec offset is a six-sigma
prior violation that the fit would have to fight.

**Nothing is interpolated.** Shifting the data by a sub-pixel amount to centre the galaxy
exactly would correlate neighbouring pixels, and the likelihood in
``gigalens.jax.scene_prob_model`` assumes independent per-pixel noise. So the window is cut
on an integer pixel and the residual sub-pixel offset is *recorded*, to be spent where it
belongs: in ``LensConfig.centroid_pix`` and hence in the deflector-centre prior.

The sidecar JSON written next to the ``.npy`` makes the provenance checkable rather than
asserted. It records the parent file, its SHA-256, the exact slice, and the centroid, so
the claim "this cutout is ``SCI[r0:r1, c0:c1]`` of that file" can be re-verified at any
later date by ``verify_cutout``.

Usage::

    python claude/pipeline/make_cutout.py --lens DESI234 \\
        --frame final_92_drz.fits --ra 234.4783 --dec 14.7232 --size 120

    # Re-check an existing cutout against its parent frame.
    python claude/pipeline/make_cutout.py --lens DESI234 --verify cutout234.npy
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import lenses

# Default half-width, in pixels, of the box the centroid is measured in. 12 px at
# 0.065"/pix is 0.78", which comfortably contains a deflector's bright core without
# reaching the arcs -- large enough to be stable against a single hot pixel, small enough
# that a companion a few arcsec away cannot drag it.
CENTROID_BOX = 12


def file_sha256(path):
    """SHA-256 of a file, so a recorded provenance can be re-checked, not just believed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pixel_scale(header):
    """Arcsec/pixel from the CD matrix, as ``sqrt(|det CD|) * 3600``.

    Derived from the matrix rather than read from a keyword: ``CDELT``-style keywords are
    absent or stale in drizzle products often enough that trusting one is a real risk, and
    the pixel scale sets the arcsec value of every fitted radius.
    """
    wcs = WCS(header)
    cd = wcs.pixel_scale_matrix  # degrees/pixel
    return float(np.sqrt(np.abs(np.linalg.det(cd))) * 3600.0)


def light_centroid(image, row, col, box=CENTROID_BOX, ):
    """Flux-weighted centroid in a ``(2*box+1)``-square window about ``(row, col)``.

    The window median is subtracted before weighting and negative residuals are clipped to
    zero. Without that subtraction the centroid of a positive-background patch is pulled
    towards the geometric centre of the window, which would make the routine partly
    self-fulfilling; with it, only light above the local sky votes.

    Args:
        image: 2-D array, the parent frame's science extension.
        row, col: starting guess, in array (0-based) pixel coordinates.
        box: half-width of the window in pixels.

    Returns:
        ``(row, col)`` of the centroid in the parent frame's pixel coordinates, as floats.
    """
    r0, r1 = int(round(row)) - box, int(round(row)) + box + 1
    c0, c1 = int(round(col)) - box, int(round(col)) + box + 1
    if r0 < 0 or c0 < 0 or r1 > image.shape[0] or c1 > image.shape[1]:
        raise ValueError(
            f"centroid box [{r0}:{r1}, {c0}:{c1}] falls outside the frame {image.shape}"
        )
    sub = np.asarray(image[r0:r1, c0:c1], dtype=np.float64)
    w = np.clip(sub - np.median(sub), 0.0, None)
    total = w.sum()
    if not total > 0:
        raise ValueError("no positive flux in the centroid box; wrong position?")
    rr, cc = np.mgrid[r0:r1, c0:c1]
    return float((w * rr).sum() / total), float((w * cc).sum() / total)


def cut(image, centroid, size):
    """Return ``(window, slices, offset)`` for an integer-pixel window about ``centroid``.

    The window is placed so that the centroid lands as close to the array centre as an
    integer shift allows. ``offset`` is what is left over: the centroid's position within
    the window minus the window's own centre, in pixels, and it is always within half a
    pixel of zero. For an ``N``-pixel axis the centre is at index ``(N - 1) / 2``, which is
    a half-integer when ``N`` is even -- so the origin is chosen as
    ``round(centroid - centre)`` rather than ``round(centroid) - N // 2``. The latter looks
    equivalent and is not: for even ``N`` it leaves a systematic half-pixel bias.

    Nothing is resampled -- ``window`` is a plain slice of ``image``.
    """
    if size < 1:
        raise ValueError("size must be positive")
    rc, cc = centroid
    centre = (size - 1) / 2.0
    # round(), not int(): we want the nearest placement, and int() truncates toward zero.
    r0 = int(round(rc - centre))
    c0 = int(round(cc - centre))
    if r0 < 0 or c0 < 0 or r0 + size > image.shape[0] or c0 + size > image.shape[1]:
        raise ValueError(
            f"window [{r0}:{r0 + size}, {c0}:{c0 + size}] falls outside {image.shape}"
        )
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    offset = (rc - r0 - centre, cc - c0 - centre)
    return image[sl], sl, offset


def make(frame_path, ra, dec, size=120, box=CENTROID_BOX, ext=None):
    """Locate the target, cut the window, and return ``(window, provenance_dict)``.

    ``ext`` names the science extension; ``None`` picks the first HDU that has 2-D data,
    which is HDU 1 (``SCI``) for these drizzle products and HDU 0 for a plain image.
    """
    frame_path = Path(frame_path)
    with fits.open(frame_path) as hdul:
        if ext is None:
            ext = next(
                i for i, h in enumerate(hdul)
                if h.data is not None and np.ndim(h.data) == 2
            )
        hdu = hdul[ext]
        image = np.asarray(hdu.data, dtype=np.float64)
        header = hdu.header
        wcs = WCS(header)
        primary = hdul[0].header
        exptime = primary.get("EXPTIME", header.get("EXPTIME"))
        targname = primary.get("TARGNAME", header.get("TARGNAME"))
        filt = primary.get("FILTER", header.get("FILTER"))
        ncombine = primary.get("NCOMBINE", header.get("NCOMBINE"))

    x_wcs, y_wcs = wcs.all_world2pix(ra, dec, 0)
    x_wcs, y_wcs = float(x_wcs), float(y_wcs)
    # WCS is (x, y) = (column, row); array indexing is (row, column).
    cen_row, cen_col = light_centroid(image, y_wcs, x_wcs, box=box)

    window, sl, offset = cut(image, (cen_row, cen_col), size)

    prov = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent_frame": frame_path.name,
        "parent_sha256": file_sha256(frame_path),
        "parent_ext": int(ext),
        "parent_shape": [int(n) for n in image.shape],
        "targname": targname,
        "filter": filt,
        "exptime": float(exptime) if exptime is not None else None,
        "ncombine": int(ncombine) if ncombine is not None else None,
        "delta_pix": pixel_scale(header),
        "requested_radec": [float(ra), float(dec)],
        "wcs_pixel_xy": [x_wcs, y_wcs],
        "centroid_box_halfwidth": int(box),
        "centroid_pixel_rowcol": [cen_row, cen_col],
        "wcs_to_centroid_offset_pix": [cen_row - y_wcs, cen_col - x_wcs],
        # The load-bearing line: what slice of the parent this array is.
        "slice": [
            [int(sl[0].start), int(sl[0].stop)],
            [int(sl[1].start), int(sl[1].stop)],
        ],
        "size": int(size),
        # Centroid within the cutout, in cutout pixels. This is what feeds
        # LensConfig.centroid_pix (after any further crop is subtracted off).
        "centroid_in_cutout_rowcol": [cen_row - sl[0].start, cen_col - sl[1].start],
        # What integer-pixel placement could not remove. Not corrected by resampling: see
        # this module's docstring.
        "subpixel_offset_rowcol": [offset[0], offset[1]],
        "peak": float(np.max(window)),
    }
    return np.ascontiguousarray(window), prov


def write(out_path, window, prov):
    """Write ``<out>.npy`` and its ``<out>.json`` sidecar."""
    out_path = Path(out_path)
    np.save(out_path, window)
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(prov, indent=2) + "\n")
    return sidecar


def verify_cutout(cutout_path, frame_path=None):
    """Re-cut from the parent and assert the stored array is bit-for-bit identical.

    This is what turns the provenance from a claim into a check. Raises on any mismatch;
    returns the provenance dict on success.
    """
    cutout_path = Path(cutout_path)
    prov = json.loads(cutout_path.with_suffix(".json").read_text())
    if frame_path is None:
        frame_path = cutout_path.parent / prov["parent_frame"]
    frame_path = Path(frame_path)

    digest = file_sha256(frame_path)
    if digest != prov["parent_sha256"]:
        raise ValueError(
            f"{frame_path.name} has SHA-256 {digest}, but the cutout was made from "
            f"{prov['parent_sha256']}. The parent frame has changed."
        )
    (r0, r1), (c0, c1) = prov["slice"]
    parent = fits.getdata(frame_path, ext=prov["parent_ext"])
    expect = np.asarray(parent[r0:r1, c0:c1], dtype=np.float64)
    got = np.asarray(np.load(cutout_path), dtype=np.float64)
    if got.shape != expect.shape:
        raise ValueError(f"cutout is {got.shape}, slice gives {expect.shape}")
    if not np.array_equal(got, expect):
        n = int((got != expect).sum())
        raise ValueError(f"cutout differs from the parent slice in {n} pixels")
    return prov


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    lenses.add_argument(ap, required=False)
    ap.add_argument("--frame", help="parent FITS frame, relative to claude/<LENS>/")
    ap.add_argument("--ra", type=float, help="target RA, degrees")
    ap.add_argument("--dec", type=float, help="target Dec, degrees")
    ap.add_argument("--size", type=int, default=120, help="window side, pixels")
    ap.add_argument("--box", type=int, default=CENTROID_BOX,
                    help="centroid box half-width, pixels")
    ap.add_argument("--ext", type=int, default=None, help="science extension index")
    ap.add_argument("--out", default=None,
                    help="output .npy, relative to claude/<LENS>/ (default cutout<NNN>.npy)")
    ap.add_argument("--dir", default=None,
                    help="lens directory, if --lens has no config.py yet")
    ap.add_argument("--verify", default=None,
                    help="instead of cutting, re-check this existing .npy against its parent")
    args = ap.parse_args()

    if args.dir is not None:
        root = Path(args.dir)
    elif args.lens is not None:
        root = lenses.CLAUDE_DIR / args.lens
    else:
        ap.error("one of --lens or --dir is required")

    if args.verify:
        prov = verify_cutout(root / args.verify)
        print(f"{args.verify}: bit-for-bit identical to "
              f"{prov['parent_frame']}[{prov['parent_ext']}]"
              f"[{prov['slice'][0][0]}:{prov['slice'][0][1]}, "
              f"{prov['slice'][1][0]}:{prov['slice'][1][1]}]")
        return

    for required in ("frame", "ra", "dec"):
        if getattr(args, required) is None:
            ap.error(f"--{required} is required unless --verify is given")

    frame = Path(args.frame)
    if not frame.is_absolute():
        frame = root / frame
    window, prov = make(frame, args.ra, args.dec,
                        size=args.size, box=args.box, ext=args.ext)
    name = args.out or f"cutout{(args.lens or 'X')[-3:]}.npy"
    sidecar = write(root / name, window, prov)

    print(f"TARGNAME  {prov['targname']}  {prov['filter']}  "
          f"EXPTIME {prov['exptime']}  NCOMBINE {prov['ncombine']}")
    print(f"scale     {prov['delta_pix']:.5f} arcsec/pix")
    print(f"WCS px    x={prov['wcs_pixel_xy'][0]:.1f} y={prov['wcs_pixel_xy'][1]:.1f}")
    print(f"centroid  row={prov['centroid_pixel_rowcol'][0]:.2f} "
          f"col={prov['centroid_pixel_rowcol'][1]:.2f}  "
          f"(offset from WCS {prov['wcs_to_centroid_offset_pix'][0]:+.2f}, "
          f"{prov['wcs_to_centroid_offset_pix'][1]:+.2f} px)")
    print(f"slice     [{prov['slice'][0][0]}:{prov['slice'][0][1]}, "
          f"{prov['slice'][1][0]}:{prov['slice'][1][1]}]  peak {prov['peak']:.2f}")
    print(f"in cutout row={prov['centroid_in_cutout_rowcol'][0]:.2f} "
          f"col={prov['centroid_in_cutout_rowcol'][1]:.2f}  "
          f"(sub-pixel {prov['subpixel_offset_rowcol'][0]:+.2f}, "
          f"{prov['subpixel_offset_rowcol'][1]:+.2f})")
    print(f"wrote     {name} and {sidecar.name}")

    verify_cutout(root / name, frame_path=frame)
    print("verified  re-cut from the parent is bit-for-bit identical")


if __name__ == "__main__":
    main()
