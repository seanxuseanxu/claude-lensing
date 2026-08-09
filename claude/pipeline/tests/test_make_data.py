"""Validate ``make_cutout.py`` and ``make_psf.py``.

Run directly (pytest is not installed in this env, as ``test_sersic_bn.py`` also notes)::

    PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_make_data.py

These test *my* code, not the libraries'. photutils' ePSF construction and astropy's WCS
are taken as trustworthy and are not re-verified; what is verified is the cutout's
bit-for-bit provenance, the kernel post-processing contract, and the star selection --
the three places where a mistake of mine produces a plausible, wrong answer rather than
an exception.

Nothing here imports photutils, so the suite runs in seconds rather than paying the ~240 s
cold-cache import. The star selection is exercised against a synthetic frame whose answer
is known by construction: one clean star, plus the three planted failures the modelling
plan names -- a saturated star, a blended pair, and a source on the frame edge.
"""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from astropy.io import fits

import make_cutout
import make_psf


# ------------------------------------------------------------------------------------
# Minimal harness
# ------------------------------------------------------------------------------------

@contextmanager
def raises(exc, match=None):
    """Assert the block raises ``exc``, and that ``match`` appears in the message."""
    try:
        yield
    except exc as e:
        if match is not None and match not in str(e):
            raise AssertionError(
                f"raised {type(e).__name__} but {match!r} not in {str(e)!r}"
            ) from None
        return
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def close(a, b, abs_tol=None, rel_tol=None, what=""):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    tol = abs_tol if abs_tol is not None else 0.0
    if rel_tol is not None:
        tol = max(tol, rel_tol * np.max(np.abs(b)))
    if not np.all(np.abs(a - b) <= tol):
        raise AssertionError(f"{what}: {a} != {b} within {tol}")


def gaussian(shape, row, col, peak, sigma=1.0):
    """A round Gaussian on a zero frame. Stands in for a star."""
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    return peak * np.exp(-((rr - row) ** 2 + (cc - col) ** 2) / (2 * sigma ** 2))


class _Row(dict):
    """One row of a stand-in detection table.

    ``select_stars`` reads only ``x_centroid``, ``y_centroid``, ``peak`` and ``colnames``,
    so the selection can be tested without running a detector.
    """

    colnames = ["x_centroid", "y_centroid", "peak"]


def row(x, y, peak):
    return _Row({"x_centroid": x, "y_centroid": y, "peak": peak})


def synthetic_frame():
    """400x400 of sky noise, one clean star, and the three planted failures."""
    rng = np.random.default_rng(0)
    img = rng.normal(0.0, 0.005, (400, 400))
    stars = {
        "clean": (200.0, 200.0, 1.0),
        "saturated": (300.0, 120.0, 500.0),
        "blend_primary": (150.0, 150.0, 1.0),
        "blend_neighbour": (150.0, 162.0, 0.5),  # 12 px away, inside the annulus
        "edge": (8.0, 200.0, 1.0),
    }
    for r, c, peak in stars.values():
        img += gaussian(img.shape, r, c, peak, sigma=0.95)
    return img, stars


# ------------------------------------------------------------------------------------
# make_cutout
# ------------------------------------------------------------------------------------

def test_cut_is_a_plain_slice_and_centres_within_half_a_pixel():
    """No interpolation, and the residual offset is always sub-pixel.

    The half-pixel bound is the point. An integer-pixel placement cannot do better, so
    anything worse means the placement arithmetic is wrong -- and this is exactly the kind
    of error that never raises. An earlier version of ``cut`` used
    ``round(c) - size // 2``, which is systematically half a pixel off for even sizes.
    """
    img = np.arange(400 * 400, dtype=np.float64).reshape(400, 400)
    for size in (120, 121, 80, 81):
        for centroid in [(200.0, 200.0), (200.49, 199.51), (137.2, 288.8)]:
            win, sl, off = make_cutout.cut(img, centroid, size)
            assert win.shape == (size, size)
            assert np.array_equal(win, img[sl]), "cut must be a plain slice"
            assert abs(off[0]) <= 0.5 + 1e-9 and abs(off[1]) <= 0.5 + 1e-9, off
            centre = (size - 1) / 2.0
            close(centroid[0], sl[0].start + centre + off[0], abs_tol=1e-9, what="row")
            close(centroid[1], sl[1].start + centre + off[1], abs_tol=1e-9, what="col")
    print("cut: plain slice, centred to <= 0.5 px, offset locates the centroid  OK")
    return True


def test_cut_refuses_a_window_off_the_frame():
    with raises(ValueError, "outside"):
        make_cutout.cut(np.zeros((100, 100)), (5.0, 5.0), 40)
    print("cut: refuses a window that falls off the frame  OK")
    return True


def test_light_centroid_ignores_a_pedestal():
    """The median subtraction is what stops a background pulling the centroid.

    Without it, a positive pedestal drags the answer toward the window's geometric centre
    -- making the routine partly self-fulfilling, since it would then report roughly the
    position it was asked about.
    """
    img = np.full((200, 200), 3.0)  # a large pedestal
    img += gaussian(img.shape, 103.3, 96.7, 10.0, sigma=1.5)
    r, c = make_cutout.light_centroid(img, 100, 100, box=12)
    close(r, 103.3, abs_tol=0.05, what="centroid row")
    close(c, 96.7, abs_tol=0.05, what="centroid col")
    print(f"light_centroid: found (103.3, 96.7) under a 3.0 pedestal as "
          f"({r:.2f}, {c:.2f})  OK")
    return True


def test_write_and_verify_round_trip(tmp):
    """``verify_cutout`` re-cuts from the parent and compares every pixel."""
    rng = np.random.default_rng(1)
    img = rng.normal(0, 1, (300, 300))
    frame_path = tmp / "parent.fits"
    fits.PrimaryHDU(img).writeto(frame_path, overwrite=True)

    win, sl, _ = make_cutout.cut(img, (150.3, 149.8), 60)
    prov = {
        "parent_frame": frame_path.name,
        "parent_sha256": make_cutout.file_sha256(frame_path),
        "parent_ext": 0,
        "slice": [[sl[0].start, sl[0].stop], [sl[1].start, sl[1].stop]],
    }
    out = tmp / "cut.npy"
    make_cutout.write(out, win, prov)
    assert make_cutout.verify_cutout(out)["parent_ext"] == 0

    # One altered pixel, at the smallest amount float64 will carry, must be caught.
    bad = np.load(out)
    bad[10, 10] = np.nextafter(bad[10, 10], np.inf)
    np.save(out, bad)
    with raises(ValueError, "differs from the parent slice"):
        make_cutout.verify_cutout(out)
    print("verify_cutout: bit-for-bit round trip, and one ULP is caught  OK")
    return True


def test_verify_cutout_catches_a_changed_parent(tmp):
    """A frame swapped underneath a cutout invalidates the provenance, loudly."""
    rng = np.random.default_rng(2)
    frame_path = tmp / "parent2.fits"
    fits.PrimaryHDU(rng.normal(0, 1, (200, 200))).writeto(frame_path, overwrite=True)
    win, sl, _ = make_cutout.cut(fits.getdata(frame_path), (100.0, 100.0), 40)
    out = tmp / "cut2.npy"
    make_cutout.write(out, win, {
        "parent_frame": frame_path.name,
        "parent_sha256": make_cutout.file_sha256(frame_path),
        "parent_ext": 0,
        "slice": [[sl[0].start, sl[0].stop], [sl[1].start, sl[1].stop]],
    })
    frame_path.unlink()
    fits.PrimaryHDU(rng.normal(0, 1, (200, 200))).writeto(frame_path, overwrite=True)
    with raises(ValueError, "has changed"):
        make_cutout.verify_cutout(out)
    print("verify_cutout: a changed parent frame is caught by checksum  OK")
    return True


# ------------------------------------------------------------------------------------
# make_psf: star selection
# ------------------------------------------------------------------------------------

def test_selection_rejects_saturated_blended_and_edge_sources():
    """The three planted failures, each rejected for its own stated reason.

    This is the test that matters most. A saturated star flattens the core, a blend adds a
    second peak in the wings, and an edge source is cut off -- and all three produce
    something that still looks like a PSF.
    """
    img, stars = synthetic_frame()
    cand = [row(stars[k][1], stars[k][0], stars[k][2])
            for k in ("clean", "saturated", "blend_primary", "edge")]
    kept, rejections = make_psf.select_stars(img, cand, size=27, edge_margin=20)
    assert len(kept) == 1, f"expected 1 survivor, got {len(kept)}"
    close(kept[0]["x"], 200.0, abs_tol=1e-9, what="survivor x")
    for reason in ("saturated / non-linear", "neighbour in annulus", "edge"):
        assert rejections.get(reason) == 1, f"{reason}: {rejections}"
    print(f"select_stars: 1 of 4 kept; rejected {dict(rejections)}  OK")
    return True


def test_selection_rejects_faint_detections():
    """The faint cut, and the fact that ``select_stars`` no longer owns its threshold.

    ``peak_min`` is a flux, but the number that matters is a signal-to-noise ratio, and
    only :func:`make_psf.build` has measured the frame's sky sigma. So ``select_stars``
    defaults to no faint cut and must be told; passing nothing must not silently drop
    stars against a stale absolute default.
    """
    img, _ = synthetic_frame()
    faint = [row(200.0, 200.0, 0.01)]
    kept, rej = make_psf.select_stars(img, faint, size=27, edge_margin=20, peak_min=0.15)
    assert kept == [] and rej["too faint"] == 1

    # neighbour_frac is a fraction of this star's own peak, and at a peak of 0.01 every
    # other star in the synthetic frame clears it -- so that cut has to be lifted here to
    # leave the faint cut as the only thing under test.
    kept, rej = make_psf.select_stars(img, faint, size=27, edge_margin=20,
                                      neighbour_frac=1e9)
    assert len(kept) == 1 and rej == {}, (kept, rej)
    print("select_stars: below peak_min is dropped; with no peak_min, no faint cut  OK")
    return True


def test_selection_honours_the_keep_out_zone():
    """The lens itself must never enter its own PSF."""
    img, _ = synthetic_frame()
    kept, rej = make_psf.select_stars(img, [row(200.0, 200.0, 1.0)], size=27,
                                      edge_margin=20, exclude=[(200, 200, 30)])
    assert kept == [] and rej["in keep-out zone"] == 1
    print("select_stars: a keep-out zone excludes the lens itself  OK")
    return True


def test_selection_rejects_low_weight_regions():
    """A drizzle chip gap has low WHT and correspondingly inflated noise."""
    img, _ = synthetic_frame()
    wht = np.full(img.shape, 300.0)
    wht[180:220, 180:220] = 10.0
    kept, rej = make_psf.select_stars(img, [row(200.0, 200.0, 1.0)], wht=wht,
                                      size=27, edge_margin=20)
    assert kept == [] and rej["low WHT coverage"] == 1
    print("select_stars: a low-WHT region is dropped  OK")
    return True


def test_source_size_ranks_compact_below_extended():
    """A narrower Gaussian must measure smaller than a broader one, and a source too close
    to the frame edge for its measurement box to fit must return NaN rather than a number
    computed from a truncated box.
    """
    shape = (60, 60)
    img = (gaussian(shape, 20, 20, 1.0, sigma=0.8)
           + gaussian(shape, 40, 40, 1.0, sigma=3.0))
    narrow = make_psf.source_size(img, 20, 20)
    broad = make_psf.source_size(img, 40, 40)
    assert narrow < broad, (narrow, broad)

    edge = make_psf.source_size(img, 2, 2)
    assert np.isnan(edge), edge
    print(f"source_size: narrow {narrow:.3f} px < broad {broad:.3f} px; a source too close "
          f"to the edge is NaN  OK")
    return True


def test_size_cut_rejects_extended_sources_even_outnumbered_four_to_one():
    """The size cut anchors on the small edge of the distribution, not its middle.

    12 extended sources against 4 point-like ones: if the anchor were dragged toward the
    population's centre -- a mean or median, say -- the majority-extended population would
    pull it into the gap and let some galaxies through. Anchoring on the 10th percentile of
    a population that is 75% extended does not do that: the narrow group still occupies the
    small tail the percentile lands in.
    """
    shape = (300, 300)
    rng = np.random.default_rng(3)
    img = rng.normal(0.0, 0.001, shape)
    positions = [(40 + 60 * i, 40 + 60 * j) for i in range(4) for j in range(4)]
    narrow_pos, broad_pos = positions[:4], positions[4:]
    for r, c in narrow_pos:
        img += gaussian(shape, r, c, 1.0, sigma=0.8)
    for r, c in broad_pos:
        img += gaussian(shape, r, c, 1.0, sigma=3.0)

    cand = ([row(c, r, 1.0) for r, c in narrow_pos]
            + [row(c, r, 1.0) for r, c in broad_pos])
    kept, rej = make_psf.select_stars(img, cand, size=27, edge_margin=20,
                                      neighbour_frac=1e9)

    assert len(kept) == 4, f"expected 4 survivors, got {len(kept)}"
    kept_xy = {(round(s["x"]), round(s["y"])) for s in kept}
    expect_xy = {(c, r) for r, c in narrow_pos}
    assert kept_xy == expect_xy, (kept_xy, expect_xy)
    assert rej["extended (not point-like)"] == 12, rej
    print(f"select_stars: size cut keeps the 4 point-like sources and drops all 12 "
          f"extended ones, anchor {rej.size_anchor_px:.3f} px  OK")
    return True


def test_local_background_recovers_pedestal_ignoring_the_source():
    """The annulus sits well outside the source's own light, so a bright source at the
    centre must not move the answer, and a source too close to the edge for the annulus to
    fit must return NaN.
    """
    shape = (80, 80)
    pedestal = 2.5
    img = np.full(shape, pedestal, dtype=np.float64)
    img += gaussian(shape, 40, 40, 500.0, sigma=2.0)  # bright; its halo is gone by r=15
    bg = make_psf.local_background(img, 40, 40)
    close(bg, pedestal, abs_tol=1e-9, what="background under a bright source")

    edge = make_psf.local_background(img, 5, 5)
    assert np.isnan(edge), edge
    print(f"local_background: recovered pedestal {pedestal} under a bright source as "
          f"{bg:.5f}; an annulus off the frame is NaN  OK")
    return True


def test_local_background_median_ignores_a_neighbour_in_the_annulus():
    """A median, not a mean, is the point: one bright neighbour caught in the annulus
    should move only a handful of ring pixels, not the aggregate the answer is drawn from.
    """
    shape = (80, 80)
    pedestal = 1.0
    img = np.full(shape, pedestal, dtype=np.float64)
    img += gaussian(shape, 40, 40, 500.0, sigma=2.0)   # the star itself
    img += gaussian(shape, 40, 60, 300.0, sigma=1.5)   # neighbour, 20 px away -- inside
                                                        # BKG_ANNULUS = (15, 25)
    bg = make_psf.local_background(img, 40, 40)
    close(bg, pedestal, abs_tol=0.05,
          what="background with a neighbour sitting inside the annulus")
    print(f"local_background: a neighbour inside the annulus barely moves the median "
          f"({bg:.5f} vs pedestal {pedestal}) -- a mean would not survive this  OK")
    return True


# ------------------------------------------------------------------------------------
# make_psf: kernel post-processing
# ------------------------------------------------------------------------------------

def test_to_science_grid_conserves_flux_and_stays_centred():
    """Block-summing is flux-conserving; interpolation would not be.

    Odd oversampling is what makes the central science pixel land on the ePSF centre, so
    this pins that too.
    """
    o, size = 3, 27
    n = size * o
    over = gaussian((n, n), (n - 1) / 2, (n - 1) / 2, 1.0, sigma=3.0)
    sci = make_psf.to_science_grid(over, oversampling=o, size=size)
    assert sci.shape == (size, size)
    close(sci.sum(), over.sum(), rel_tol=1e-9, what="flux")
    dr, dc = make_psf.describe(sci)["centroid_offset_rowcol"]
    assert abs(dr) < 1e-6 and abs(dc) < 1e-6, (dr, dc)
    print(f"to_science_grid: 81x81 -> 27x27, flux conserved to 1e-9, centroid off by "
          f"({dr:.1e}, {dc:.1e}) px  OK")
    return True


def test_recentre_epsf_fixes_a_known_integer_offset():
    """A whole-oversampled-pixel displacement is exactly what ``recentre_epsf`` is for: an
    integer reindex undoes it exactly, with no interpolation and, for a source this
    compact, no flux lost off either edge.
    """
    o, size = 3, 27
    n = size * o
    cy = (n - 1) / 2.0
    # Displaced from the grid centre by a whole number of oversampled pixels, sigma small
    # enough that nothing meaningful reaches the edge before or after the shift.
    over = gaussian((n, n), cy + 2, cy - 3, 1.0, sigma=3.0)

    recentred, shift = make_psf.recentre_epsf(over, oversampling=o)
    assert shift == (-2, 3), shift
    close(recentred.sum(), over.sum(), rel_tol=1e-9, what="flux after recentring")

    sci_fixed = make_psf.to_science_grid(recentred, oversampling=o, size=size,
                                         recentre=False)
    dr, dc = make_psf.describe(sci_fixed)["centroid_offset_rowcol"]
    assert abs(dr) < 1e-6 and abs(dc) < 1e-6, (dr, dc)

    # Without the correction, the same displacement passes straight through the block-sum.
    sci_raw = make_psf.to_science_grid(over, oversampling=o, size=size, recentre=False)
    dr0, dc0 = make_psf.describe(sci_raw)["centroid_offset_rowcol"]
    assert abs(dr0) > 0.3 and abs(dc0) > 0.3, (dr0, dc0)
    print(f"recentre_epsf: a ({shift[0]:+d}, {shift[1]:+d}) oversampled-px offset is fixed "
          f"to ({dr:.1e}, {dc:.1e}) px, against ({dr0:+.3f}, {dc0:+.3f}) px uncorrected  OK")
    return True


def test_recentre_epsf_residual_is_bounded_by_half_an_oversampled_pixel():
    """An integer-pixel correction can only close a displacement to the nearest whole
    oversampled pixel, so a residual of up to half an oversampled pixel survives by
    construction -- that granularity, not a bug, is why ``check_contract``'s centring
    tolerance has to be set loose enough to admit it. At oversampling 3, half an
    oversampled pixel is 1/6 = 0.167 science px.

    Swept offsets are deliberately not multiples of 1/oversampling, so the residual is
    generically non-zero rather than accidentally exact.
    """
    o, size = 3, 27
    n = size * o
    cy = (n - 1) / 2.0
    bound = 1.0 / (2 * o)
    tol = 0.02  # numerical slack: a sampled Gaussian's own centroid isn't exact either
    worst = 0.0
    for frac in (-0.9, -0.61, -0.34, -0.11, 0.13, 0.37, 0.58, 0.83):
        over = gaussian((n, n), cy + frac * o, cy - 0.2 * o, 1.0, sigma=3.0)
        recentred, _ = make_psf.recentre_epsf(over, oversampling=o)
        sci = make_psf.to_science_grid(recentred, oversampling=o, size=size,
                                       recentre=False)
        dr, dc = make_psf.describe(sci)["centroid_offset_rowcol"]
        assert abs(dr) <= bound + tol, (frac, dr)
        assert abs(dc) <= bound + tol, (frac, dc)
        worst = max(worst, abs(dr), abs(dc))
    print(f"recentre_epsf: worst residual over the sweep was {worst:.4f} px, within the "
          f"{bound:.3f} px (+{tol}) bound  OK")
    return True


def test_to_science_grid_and_build_epsf_reject_bad_geometry():
    with raises(ValueError, "size must be odd"):
        make_psf.to_science_grid(np.ones((81, 81)), oversampling=3, size=26)
    with raises(ValueError, "too small"):
        make_psf.to_science_grid(np.ones((81, 81)), oversampling=3, size=41)
    with raises(ValueError, "must be odd"):
        make_psf.build_epsf(np.zeros((50, 50)), [], size=27, oversampling=4)
    print("geometry: even kernel size, undersized grid and even oversampling all "
          "refused  OK")
    return True


def test_epsf_grid_shape_clears_photutils_minimum_and_stays_odd():
    """The grid photutils is asked for must be big enough for it and odd for us.

    Asking for ``size * oversampling`` exactly is what photutils rejects, with
    ``Requested ePSF shape [81 81] is incompatible with star dimensions and oversampling``
    -- a runtime failure four minutes into a build. The rule is pinned here so it fails in
    seconds instead.
    """
    for size in (17, 21, 27, 41):
        for o in (1, 3, 5):
            n = make_psf.epsf_grid_shape(size, o)
            assert n % 2 == 1, (size, o, n)
            assert n >= size * o + 1, (size, o, n)
            assert n <= size * o + 2, f"wasteful grid {n} for {size}x{o}"
            # And the grid it returns must survive the trip back to the science grid.
            make_psf.to_science_grid(np.ones((n, n)), oversampling=o, size=size)
    print("epsf_grid_shape: odd, >= oversampling*size+1, and always trimmable  OK")
    return True


def test_check_contract_catches_the_three_ways_a_kernel_is_wrong():
    good = gaussian((27, 27), 13, 13, 1.0, sigma=1.5)
    good = good / good.sum()
    make_psf.check_contract(good)

    with raises(ValueError, "no centre pixel"):
        make_psf.check_contract(np.ones((26, 26)) / 676)
    with raises(ValueError, "sums to"):
        make_psf.check_contract(good * 2)
    # Shifted by a pixel and a half: normalised, odd-sized, and astrometrically wrong.
    off = gaussian((27, 27), 14.5, 13, 1.0, sigma=1.5)
    with raises(ValueError, "centroid"):
        make_psf.check_contract(off / off.sum())
    print("check_contract: even size, wrong sum and off-centre kernel all rejected  OK")
    return True


def test_fwhm_and_encircled_energy_on_a_known_gaussian():
    """A Gaussian is the one case where FWHM and encircled energy have closed forms."""
    sigma = 2.0
    g = gaussian((41, 41), 20, 20, 1.0, sigma=sigma)
    g = g / g.sum()
    expect = 2 * np.sqrt(2 * np.log(2)) * sigma
    got = make_psf.fwhm_of(g)
    close(got, expect, rel_tol=0.03, what="FWHM")
    ee = make_psf.encircled_energy(g, radii=(3, 5, 10))
    for r in (3, 5):
        close(ee[r], 1 - np.exp(-r ** 2 / (2 * sigma ** 2)), abs_tol=0.02, what=f"EE({r})")
    assert ee[3] < ee[5] <= ee[10] <= 1.0
    print(f"fwhm_of: {got:.3f} px against the analytic {expect:.3f}; EE(3/5/10) = "
          f"{ee[3]:.3f}/{ee[5]:.3f}/{ee[10]:.3f}  OK")
    return True


def test_compare_is_scale_free_and_zero_on_identity():
    g = gaussian((27, 27), 13, 13, 1.0, sigma=1.5)
    c = make_psf.compare(g, g * 7.0)  # normalisation must not matter
    close(c["max_abs_frac_diff"], 0.0, abs_tol=1e-12, what="identity diff")
    close(c["fwhm_diff_pix"], 0.0, abs_tol=1e-12, what="identity FWHM diff")

    wider = gaussian((27, 27), 13, 13, 1.0, sigma=2.0)
    c2 = make_psf.compare(g, wider)
    assert c2["fwhm_diff_pix"] < -0.3, c2["fwhm_diff_pix"]  # g is the narrower kernel
    assert c2["ee_diff"][3] > 0.0, c2["ee_diff"]            # and the more concentrated
    print(f"compare: identical kernels differ by 0 at any scale; sigma 1.5 vs 2.0 gives "
          f"dFWHM {c2['fwhm_diff_pix']:+.2f} px  OK")
    return True


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        tests = [
            test_cut_is_a_plain_slice_and_centres_within_half_a_pixel,
            test_cut_refuses_a_window_off_the_frame,
            test_light_centroid_ignores_a_pedestal,
            lambda: test_write_and_verify_round_trip(tmp),
            lambda: test_verify_cutout_catches_a_changed_parent(tmp),
            test_selection_rejects_saturated_blended_and_edge_sources,
            test_selection_rejects_faint_detections,
            test_selection_honours_the_keep_out_zone,
            test_selection_rejects_low_weight_regions,
            test_source_size_ranks_compact_below_extended,
            test_size_cut_rejects_extended_sources_even_outnumbered_four_to_one,
            test_local_background_recovers_pedestal_ignoring_the_source,
            test_local_background_median_ignores_a_neighbour_in_the_annulus,
            test_to_science_grid_conserves_flux_and_stays_centred,
            test_recentre_epsf_fixes_a_known_integer_offset,
            test_recentre_epsf_residual_is_bounded_by_half_an_oversampled_pixel,
            test_to_science_grid_and_build_epsf_reject_bad_geometry,
            test_epsf_grid_shape_clears_photutils_minimum_and_stays_odd,
            test_check_contract_catches_the_three_ways_a_kernel_is_wrong,
            test_fwhm_and_encircled_energy_on_a_known_gaussian,
            test_compare_is_scale_free_and_zero_on_identity,
        ]
        results = []
        for t in tests:
            try:
                results.append(bool(t()))
            except AssertionError as e:
                print(f"FAIL  {e}")
                results.append(False)

    n_pass = sum(results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
