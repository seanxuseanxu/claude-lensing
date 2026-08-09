"""Unit tests for ``verify.corner_report``, the four-corner companion check.

Run directly (pytest is not installed in this env):

    PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_corner_check.py

Covers the upper-left corner specifically, since a single-corner (lower-left-only) check
would be blind to a companion sitting there:

  (a) A synthetic image whose only bright pixel is in the upper-left corner, with
      ``companion_max`` below that value, raises ``AssertionError`` naming "upper-left".
  (b) The same image, with ``mask_circles`` covering that bright pixel via
      ``data.build_mask``, passes -- the masked maximum falls below ``companion_max``.
  (c) ``mask_circles=None`` (mask=None) on a clean image is unaffected: the check passes and
      reports the raw maxima.

Does not import or run ``verify.build()`` -- that needs real FITS data and gigalens. This
tests the ``corner_report`` helper directly against numpy arrays.
"""

import numpy as np

import data
import verify


def _corner_image(num_pix, bright_value=100.0, background=1.0):
    """A flat image with a single bright pixel in the upper-left corner patch."""
    image = np.full((num_pix, num_pix), background, dtype=np.float32)
    q = num_pix // 4
    # Upper-left corner patch is image[-q:, :q]; put the bright pixel at its centre.
    image[-q:, :q][q // 2, q // 2] = bright_value
    return image


def test_upper_left_corner_raises():
    num_pix = 40
    image = _corner_image(num_pix, bright_value=100.0, background=1.0)
    companion_max = 50.0  # below the bright pixel's value

    try:
        verify.corner_report(image, num_pix, companion_max, mask=None)
        raised = False
        msg = ""
    except AssertionError as exc:
        raised = True
        msg = str(exc)

    assert raised, "expected AssertionError for a companion in the upper-left corner"
    assert "upper-left" in msg, f"error message does not name upper-left: {msg!r}"


def test_masking_the_corner_suppresses_it():
    num_pix = 40
    image = _corner_image(num_pix, bright_value=100.0, background=1.0)
    companion_max = 50.0

    q = num_pix // 4
    # The bright pixel sits at row (num_pix - q + q//2), col (q // 2) in full-image coords.
    row = num_pix - q + q // 2
    col = q // 2

    class FakeConfig:
        name = "test"
        mask_circles = ((row, col, 3.0),)

    mask = data.build_mask(FakeConfig(), image.shape)
    assert mask is not None
    assert not mask[row, col], "sanity: the bright pixel must actually be masked out"

    # Should NOT raise now -- the masked maximum in that corner falls below companion_max.
    worst_name, corner_max, note = verify.corner_report(image, num_pix, companion_max, mask)
    assert corner_max < companion_max, (
        f"masked corner max {corner_max} still exceeds companion_max {companion_max}"
    )


def test_mask_none_unaffected_on_clean_image():
    num_pix = 40
    image = np.full((num_pix, num_pix), 1.0, dtype=np.float32)
    companion_max = 50.0

    # Should not raise: no bright pixels anywhere, mask=None is a pure pass-through.
    worst_name, corner_max, note = verify.corner_report(image, num_pix, companion_max, mask=None)
    assert corner_max == 1.0, f"expected raw background max 1.0, got {corner_max}"
    assert "unmasked" in note


def main():
    tests = [
        test_upper_left_corner_raises,
        test_masking_the_corner_suppresses_it,
        test_mask_none_unaffected_on_clean_image,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
