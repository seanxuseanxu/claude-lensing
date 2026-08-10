"""Unit tests for ``diagnose.check_theta_E``'s ``extra_mass`` parameter (added 2026-08-09).

Run directly (pytest is not installed in this env), CPU-only like ``test_mask.py`` --
nothing here builds a real scene or runs inference, so no ``--lens`` and no GPU:

    PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_check_theta_E.py

Background: the tangential critical curve traces the TOTAL lens-plane convergence, while
``theta_E`` passed to ``check_theta_E`` is the main EPL's alone. With a second deflector on
the lens plane (e.g. the foreground SIS from ``--foreground``) those are different
quantities, so the old unconditional 10%-disagreement check fired "VALIDATION FAILED
LOUDLY" on DESI234's ``r3_fg`` -- a run whose chi2_nu (1.08) was the best of the round.
``extra_mass``, when non-empty, switches the message to "Validation N/A (multi-mass lens
plane)" reporting the excess, instead of the failure. It must still fail loudly when
``extra_mass`` is empty (the ordinary, single-deflector case) -- that is the regression
these tests guard against in both directions.
"""

import io
import sys
from contextlib import redirect_stdout

import numpy as np

import diagnose


def circle(radius, center=(0.0, 0.0), n=200):
    """A synthetic critical curve: ``n`` points on a circle of ``radius`` about ``center``,
    in the ``(xs, ys)`` form ``LensSolver.critical_and_caustics`` returns and
    ``diagnose.curve_radius`` consumes -- not an ``(N, 2)`` array. ``curve_radius`` is the
    mean vertex distance from ``center``, which is exactly ``radius`` for a circle, so this
    gives ``check_theta_E`` an exact, known tangential radius to compare against.
    """
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return (center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta))


# -- extra_mass=None (single deflector): the loud check must survive unchanged -----------

def test_no_extra_mass_agreement_prints_validation_ok():
    theta_E = 1.5
    curves = [circle(1.55, center=(0.0, 0.0))]  # 3.3% off -- within the 10% bound

    buf = io.StringIO()
    with redirect_stdout(buf):
        diagnose.check_theta_E(curves, (0.0, 0.0), theta_E, extra_mass=None)
    out = buf.getvalue()

    assert "Validation OK" in out
    assert "VALIDATION FAILED" not in out
    assert "Validation N/A" not in out


def test_no_extra_mass_disagreement_still_fails_loudly():
    # This is the check that must NOT have been broken by the extra_mass fix: with no
    # second deflector, a >10% disagreement between the tangential curve and theta_E is
    # still a genuine red flag and must still fail loudly.
    theta_E = 1.4726
    curves = [circle(1.9500, center=(0.0, 0.0))]  # ~32%, well past 10%

    buf = io.StringIO()
    with redirect_stdout(buf):
        diagnose.check_theta_E(curves, (0.0, 0.0), theta_E, extra_mass=None)
    out = buf.getvalue()

    assert "VALIDATION FAILED LOUDLY" in out
    assert "Validation N/A" not in out


# -- extra_mass non-empty: DESI234 r3_fg's real numbers -----------------------------------

def test_extra_mass_reports_na_instead_of_failure():
    # DESI234 r3_fg (2026-08-09): tangential critical-curve radius 1.7610, main EPL
    # theta_E 1.4726, foreground SIS theta_E 0.3702 -- a 19.6% "disagreement" that is not a
    # disagreement at all, but the SIS's own convergence pushing the tangential curve out.
    # chi2_nu for this run was 1.08, the best of the round -- the fit is not bad.
    tangential = 1.7610
    theta_E = 1.4726
    sis_theta_E = 0.3702
    curves = [circle(tangential, center=(0.0, 0.0))]
    extra_mass = [{"theta_E": sis_theta_E, "_profile": "SIS"}]

    buf = io.StringIO()
    with redirect_stdout(buf):
        diagnose.check_theta_E(curves, (0.0, 0.0), theta_E, extra_mass=extra_mass)
    out = buf.getvalue()

    assert "Validation N/A (multi-mass lens plane)" in out
    assert "VALIDATION FAILED" not in out
    # Excess = tangential - theta_E = 1.7610 - 1.4726 = +0.2884", ~19.6%.
    assert "+0.2884" in out, out
    frac = abs(tangential - theta_E) / theta_E
    assert abs(frac - 0.19585) < 1e-3
    assert "19.6%" in out, out


# -- the SHEAR discrimination in the caller -----------------------------------------------
#
# check_theta_E itself does not filter extra_mass -- it trusts whatever list it is handed.
# The filter lives in the CALLER, diagnose.residual_figure (around line 610), inline in a
# list comprehension keyed on "theta_E" in d, not on component index or type. It exists
# because SHEAR is also a lens-plane mass component but carries only gamma1/gamma2 and adds
# no convergence, so it must not be treated as a second deflector the way a foreground SIS
# is. residual_figure needs a full model/prob/params to invoke directly, so there is no
# smaller function seam to call. This test instead pins the filter EXPRESSION verbatim, as
# it appears in diagnose.py's source (see the grep below), against synthetic component
# dicts of the shape residual_figure builds them (plain floats keyed by parameter name).
# It is therefore a test of the same logic, not an exercise of residual_figure's own code
# path -- flagged here rather than silently claimed as full coverage.

def test_shear_only_component_is_excluded_by_the_callers_filter():
    import inspect
    src = inspect.getsource(diagnose.residual_figure)
    assert 'if "theta_E" in d' in src, (
        "the caller's extra_mass filter expression has changed shape; "
        "update this test to match diagnose.py's residual_figure"
    )

    lens_plane_mass = {
        0: {"theta_E": 1.4726, "gamma": 2.05, "center_x": 0.0, "center_y": 0.0},  # main EPL
        1: {"gamma1": 0.03, "gamma2": -0.01},  # SHEAR: no theta_E, no convergence
    }
    # The exact expression from diagnose.py's residual_figure (extra_mass = [...]).
    extra_mass = [
        d for d in (
            dict(lens_plane_mass[i]) for i in sorted(lens_plane_mass)[1:]
        )
        if "theta_E" in d
    ]

    assert extra_mass == [], "a SHEAR-only component (gamma1/gamma2, no theta_E) must not count as extra_mass"


def test_second_deflector_with_theta_E_is_kept_by_the_callers_filter():
    lens_plane_mass = {
        0: {"theta_E": 1.4726, "gamma": 2.05, "center_x": 0.0, "center_y": 0.0},
        1: {"theta_E": 0.3702, "center_x": 1.0, "center_y": -1.0},  # foreground SIS
    }
    extra_mass = [
        d for d in (
            dict(lens_plane_mass[i]) for i in sorted(lens_plane_mass)[1:]
        )
        if "theta_E" in d
    ]

    assert len(extra_mass) == 1
    assert extra_mass[0]["theta_E"] == 0.3702


def main():
    tests = [
        test_no_extra_mass_agreement_prints_validation_ok,
        test_no_extra_mass_disagreement_still_fails_loudly,
        test_extra_mass_reports_na_instead_of_failure,
        test_shear_only_component_is_excluded_by_the_callers_filter,
        test_second_deflector_with_theta_E_is_kept_by_the_callers_filter,
    ]
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"FAIL  {t.__name__}  {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - n_fail}/{len(tests)} tests passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
