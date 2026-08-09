"""Unit tests for ``data.build_mask``.

Run directly (pytest is not installed in this env):

    python claude/pipeline/test_mask.py

Covers: mask shape matches the image shape; a disc excludes exactly the pixels within its
radius and no others; ``mask_circles=None`` yields ``None`` (the unchanged-default-behaviour
case every existing config relies on); an off-image disc raises; a config whose circles mask
every pixel raises.
"""

import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import data


@dataclass
class FakeConfig:
    """Just enough of ``LensConfig`` for ``build_mask`` -- ``name`` and ``mask_circles``."""

    name: str
    mask_circles: Optional[Tuple[Tuple[float, float, float], ...]] = None


def test_mask_none_when_no_circles():
    cfg = FakeConfig(name="test", mask_circles=None)
    mask = data.build_mask(cfg, (20, 20))
    assert mask is None, "mask_circles=None must yield None, not an all-True array"


def test_mask_shape_matches_image():
    cfg = FakeConfig(name="test", mask_circles=((10, 10, 3),))
    shape = (20, 20)
    mask = data.build_mask(cfg, shape)
    assert mask.shape == shape, f"got {mask.shape}, expected {shape}"
    assert mask.dtype == bool, f"got dtype {mask.dtype}, expected bool"


def test_disc_excludes_exactly_pixels_within_radius():
    shape = (25, 25)
    row, col, radius = 12, 12, 4.0
    cfg = FakeConfig(name="test", mask_circles=((row, col, radius),))
    mask = data.build_mask(cfg, shape)

    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    dist2 = (rows - row) ** 2 + (cols - col) ** 2
    expect_excluded = dist2 <= radius**2

    # mask is True = keep, so it must be the exact complement of "within radius".
    assert np.array_equal(~mask, expect_excluded), (
        "mask does not exclude exactly the disc of pixels within radius"
    )
    # Sanity: at least one pixel excluded, at least one kept.
    assert expect_excluded.any() and (~expect_excluded).any()


def test_multiple_discs_union():
    shape = (30, 30)
    cfg = FakeConfig(
        name="test", mask_circles=((5, 5, 2.0), (25, 25, 2.0))
    )
    mask = data.build_mask(cfg, shape)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    excluded = ((rows - 5) ** 2 + (cols - 5) ** 2 <= 4.0) | (
        (rows - 25) ** 2 + (cols - 25) ** 2 <= 4.0
    )
    assert np.array_equal(~mask, excluded)


def test_off_image_disc_raises():
    cfg = FakeConfig(name="test", mask_circles=((1000, 1000, 3.0),))
    try:
        data.build_mask(cfg, (20, 20))
        raised = False
    except ValueError:
        raised = True
    assert raised, "a disc entirely outside the image must raise, not silently no-op"


def test_all_masking_config_raises():
    # A disc big enough to cover the whole 10x10 image from its centre.
    cfg = FakeConfig(name="test", mask_circles=((5, 5, 100.0),))
    try:
        data.build_mask(cfg, (10, 10))
        raised = False
    except ValueError:
        raised = True
    assert raised, "a mask that keeps no pixels must raise"


def main():
    tests = [
        test_mask_none_when_no_circles,
        test_mask_shape_matches_image,
        test_disc_excludes_exactly_pixels_within_radius,
        test_multiple_discs_union,
        test_off_image_disc_raises,
        test_all_masking_config_raises,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
