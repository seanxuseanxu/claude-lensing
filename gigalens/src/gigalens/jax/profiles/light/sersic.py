import functools

import jax.numpy as jnp
from jax import jit

import gigalens.profile

# Rational (3,2) approximation to the Sersic constant b_n, the solution of
# gamma(2n, b_n) = Gamma(2n)/2. Numerator coefficients first, then denominator, both in
# descending powers of n as jnp.polyval expects. This replaces the asymptotic linear form
# bn = 1.9992*n - 0.3271, which is accurate to 0.02% for n >= 2 but degrades badly at low
# n: -3.0% at n = 0.5 and -13.6% at n = 0.3. The rational form holds to 0.005% over
# n in [0.3, 8], which removes the low-n floor that form imposed on source priors.
_BN_COEFFS = jnp.array([
    9.76405972e+01, -1.06311838e+01, 3.50444477e-01, -1.99301509e-03,
    4.88184193e+01, 2.82927599e+00, 4.03956637e-01,
])


def _bn(n_sersic):
    """Sersic constant b_n. Accurate to 0.005% over n in [0.3, 8]; see _BN_COEFFS."""
    return jnp.polyval(_BN_COEFFS[:4], n_sersic) / jnp.polyval(_BN_COEFFS[4:], n_sersic)


class Sersic(gigalens.profile.LightProfile):
    _name = "SERSIC"
    _params = ["R_sersic", "n_sersic", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, center_x, center_y, Ie=None):
        R = self.distance(x, y, center_x, center_y)
        bn = _bn(n_sersic)
        # Ie is applied once, in the return, matching SersicEllipse below. The original
        # applied it here AND in the non-lstsq return, squaring it -- verified against
        # lenstronomy, whose Sersic is correct on this point even though its b_n is not.
        ret = jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)

    @functools.partial(jit, static_argnums=(0,))
    def distance(self, x, y, cx, cy, e1=None, e2=None):
        if e1 is None:
            e1 = jnp.zeros_like(cx)
        if e2 is None:
            e2 = jnp.zeros_like(cx)
        phi = jnp.arctan2(e2, e1) / 2
        c = jnp.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        dx, dy = x - cx, y - cy
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        xt1 = (cos_phi * dx + sin_phi * dy) * jnp.sqrt(q)
        xt2 = (-sin_phi * dx + cos_phi * dy) / jnp.sqrt(q)
        return jnp.sqrt(xt1 ** 2 + xt2 ** 2)


class SersicEllipse(Sersic):
    _name = "SERSIC_ELLIPSE"
    _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = _bn(n_sersic)
        ret = jnp.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)


class CoreSersic(Sersic):
    _name = "CORE_SERSIC"
    _params = [
        "R_sersic",
        "n_sersic",
        "Rb",
        "alpha",
        "gamma",
        "e1",
        "e2",
        "center_x",
        "center_y",
    ]

    @functools.partial(jit, static_argnums=(0,))
    def light(
            self,
            x,
            y,
            R_sersic,
            n_sersic,
            Rb,
            alpha,
            gamma,
            e1,
            e2,
            center_x,
            center_y,
            Ie=None,
    ):
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = _bn(n_sersic)
        ret = ((1 + (Rb / R) ** alpha) ** (gamma / alpha) * jnp.exp(-bn * (
                (R ** alpha + Rb ** alpha)
                / R_sersic ** alpha ** 1.0
                / (alpha * n_sersic)
        ) - 1.0))
        return ret[jnp.newaxis, ...] if self.use_lstsq else (Ie * ret)
