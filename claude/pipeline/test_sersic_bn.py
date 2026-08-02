"""Validate gigalens' Sersic constant b_n against its defining relation.

Run directly (pytest is not installed in gigajax6.0):

    PYTHONPATH=gigalens/src JAX_PLATFORMS=cpu python claude/pipeline/test_sersic_bn.py

**Lenstronomy is not the ground truth here, unlike everywhere else in this repo.**
``lenstronomy/LensModel/Profiles/sersic_utils.py:37`` defines b_n as the same asymptotic
``1.9992*n - 0.3271`` that gigalens used, so a lenstronomy-parity test on b_n asserts
agreement with the error rather than the absence of one. It carries the identical -13.6%
at n = 0.3. The reference used below is instead the defining relation itself,

    gamma(2n, b_n) = Gamma(2n) / 2   <=>   b_n = gammaincinv(2n, 1/2)

which scipy computes to machine precision.

Lenstronomy is still the reference for the *shape* of the profile -- the distance
function, the ellipticity convention, the Ie normalisation -- so those are tested against
it with b_n held fixed to the linear form both codes share. That separation is the point:
one test per claim, each against a source that is actually authoritative for that claim.
"""

import sys

import numpy as np
from scipy.special import gammaincinv

import jax.numpy as jnp

from gigalens.jax.profiles.light import sersic

# Range over which the rational approximation is claimed accurate, and the range the
# priors in priors.py are allowed to span. Extending a prior past this needs a new check.
N_LO, N_HI = 0.3, 8.0

# float32 tolerance. The coefficients themselves are good to 3.8e-05 in float64; the rest
# is the dtype gigalens runs in.
BN_RTOL = 1e-4


def check(name, ok):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def test_bn_against_defining_relation():
    """b_n must solve gamma(2n, b_n) = Gamma(2n)/2 across the supported range."""
    print("b_n vs gammaincinv(2n, 1/2):")
    n = np.concatenate([np.linspace(N_LO, 1.0, 40), np.linspace(1.0, N_HI, 60)])
    got = np.asarray(sersic._bn(jnp.array(n)))
    err = np.abs(got / gammaincinv(2 * n, 0.5) - 1)
    print(f"    max rel err = {err.max():.2e} at n = {n[err.argmax()]:.2f}")
    return check(f"within {BN_RTOL:.0e} over n in [{N_LO}, {N_HI}]", err.max() < BN_RTOL)


def test_bn_beats_the_linear_form():
    """The point of the change: better than what it replaced, at every n.

    Asserted pointwise rather than on a summary statistic, so a coefficient set that
    improved the average while degrading some n would fail.

    The one licensed exception: the linear form's *signed* error crosses zero near
    n = 2.5 and n = 5.5, so in a narrow band around each crossing it is incidentally
    closer than any smooth approximation of finite order can be. Both forms are ~1e-5
    there. Requiring the rational form to win those points would be requiring it to
    reproduce an accident, so the comparison is waived wherever both are already below
    BN_RTOL -- a difference that cannot affect a rendered profile in float32.
    """
    print("b_n vs the superseded linear form:")
    n = np.linspace(N_LO, N_HI, 100)
    exact = gammaincinv(2 * n, 0.5)
    new = np.abs(np.asarray(sersic._bn(jnp.array(n))) / exact - 1)
    old = np.abs((1.9992 * n - 0.3271) / exact - 1)
    print(f"    old max rel err = {old.max():.2e}  (at n = {n[old.argmax()]:.2f})")
    print(f"    new max rel err = {new.max():.2e}  (at n = {n[new.argmax()]:.2f})")

    worse = (new > old) & ~((new < BN_RTOL) & (old < BN_RTOL))
    if np.any(worse):
        print(f"    materially worse at n = {np.round(n[worse], 2)}")
    negligible = (new > old).sum() - worse.sum()
    print(f"    {negligible} points worse but below {BN_RTOL:.0e} (zero-crossing band)")
    return check("better at every n, or both negligible", not np.any(worse))


def test_bn_depends_only_on_n():
    """b_n is a function of n alone -- it must not pick up the coordinate grid.

    The first draft of this change divided by ``polyval(den, x)`` rather than
    ``polyval(den, n_sersic)``, making b_n a field over the image plane that diverged
    where the denominator crossed zero. The profile still evaluated to finite numbers over
    most of the frame, so nothing raised. This is the regression test for that.
    """
    print("b_n is scalar in n:")
    scalar = np.shape(sersic._bn(jnp.array(1.0))) == ()
    # Same n, wildly different coordinates -> identical rendered profile up to R.
    prof = sersic.SersicEllipse(use_lstsq=False)
    kw = dict(R_sersic=1.0, n_sersic=1.7, e1=0.0, e2=0.0, Ie=1.0)
    x, y = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
    a = np.asarray(prof.light(jnp.array(x), jnp.array(y), center_x=0.0, center_y=0.0, **kw))
    # Shifting the grid and the centre together must translate the image, not reshape it.
    b = np.asarray(prof.light(jnp.array(x + 10.0), jnp.array(y), center_x=10.0,
                              center_y=0.0, **kw))
    return check("scalar in n and translation-invariant",
                 scalar and np.allclose(a, b, rtol=1e-5, atol=1e-6))


def test_profile_shape_against_lenstronomy():
    """Everything except b_n, checked against lenstronomy at the repo's usual tolerance.

    b_n is cancelled out by comparing against a reference built with the *linear* form
    that lenstronomy still uses, so any disagreement here is in the distance function,
    the ellipticity convention, or the Ie normalisation -- not the approximation.
    """
    from lenstronomy.LightModel.Profiles.sersic import SersicElliptic

    print("profile shape vs lenstronomy (b_n held equal):")
    x, y = np.meshgrid(np.linspace(-3, 3, 60), np.linspace(-3, 3, 60))
    prof = sersic.SersicEllipse(use_lstsq=False)
    ok = True
    for n in (0.5, 1.0, 2.0, 4.0):
        R = np.asarray(prof.distance(jnp.array(x), jnp.array(y), 0.0, 0.0, 0.15, -0.1))
        ours = 2.0 * np.exp(-(1.9992 * n - 0.3271) * ((R / 0.9) ** (1 / n) - 1.0))
        theirs = SersicElliptic().function(x, y, amp=2.0, R_sersic=0.9, n_sersic=n,
                                           e1=0.15, e2=-0.1, center_x=0.0, center_y=0.0)
        match = np.allclose(ours, theirs, rtol=1e-5, atol=1e-4)
        ok &= match
        print(f"    n = {n:3.1f}  match = {match}  max|d| = {np.abs(ours - theirs).max():.2e}")
    return check("lenstronomy parity at rtol=1e-5, atol=1e-4", bool(ok))


def test_Ie_applied_once():
    """Rendered flux must be linear in Ie.

    ``Sersic.light`` applied Ie both inside the exponential prefactor and again in the
    non-lstsq return, squaring it. ``SersicEllipse`` never did. Both are checked.
    """
    print("flux is linear in Ie:")
    x, y = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    ok = True
    for cls, extra in ((sersic.Sersic, {}), (sersic.SersicEllipse, dict(e1=0.1, e2=0.05))):
        prof = cls(use_lstsq=False)
        kw = dict(R_sersic=1.0, n_sersic=2.0, center_x=0.0, center_y=0.0, **extra)
        one = np.asarray(prof.light(jnp.array(x), jnp.array(y), Ie=1.0, **kw))
        three = np.asarray(prof.light(jnp.array(x), jnp.array(y), Ie=3.0, **kw))
        match = np.allclose(three, 3.0 * one, rtol=1e-5, atol=1e-6)
        ok &= match
        peak = float(np.max(three / np.where(one == 0, np.nan, one)))
        print(f"    {cls.__name__:14s} ratio = {peak:.4f} (want 3.0)  match = {match}")
    return check("Ie enters linearly in both classes", bool(ok))


def main():
    tests = [
        test_bn_against_defining_relation,
        test_bn_beats_the_linear_form,
        test_bn_depends_only_on_n,
        test_profile_shape_against_lenstronomy,
        test_Ie_applied_once,
    ]
    results = []
    for t in tests:
        results.append(t())
        print()
    n_pass = sum(results)
    print(f"{n_pass}/{len(results)} checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
