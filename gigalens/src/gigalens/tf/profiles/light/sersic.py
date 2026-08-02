import tensorflow as tf
import tensorflow_probability as tfp

import gigalens.profile

tfd = tfp.distributions

# Rational (3,2) approximation to the Sersic constant b_n, the solution of
# gamma(2n, b_n) = Gamma(2n)/2. Numerator coefficients first, then denominator, both in
# descending powers of n. This replaces the asymptotic linear form
# bn = 1.9992*n - 0.3271, which is accurate to 0.02% for n >= 2 but degrades badly at low
# n: -3.0% at n = 0.5 and -13.6% at n = 0.3. The rational form holds to 0.005% over
# n in [0.3, 8]. Mirrors gigalens/jax/profiles/light/sersic.py; validated only in the JAX
# backend, since TensorFlow is not installed in this environment.
_BN_NUM = [9.76405972e+01, -1.06311838e+01, 3.50444477e-01, -1.99301509e-03]
_BN_DEN = [4.88184193e+01, 2.82927599e+00, 4.03956637e-01]


def _bn(n_sersic):
    """Sersic constant b_n. Accurate to 0.005% over n in [0.3, 8]; see _BN_NUM."""
    return tf.math.polyval(_BN_NUM, n_sersic) / tf.math.polyval(_BN_DEN, n_sersic)


class Sersic(gigalens.profile.LightProfile):
    """A spherically symmetric Sersic light profile.

    .. math::
        I(x,y) = I_e \\exp\\left(-b_n \\left(\\left(\\frac{D(x,y)}{R_s}\\right)^{1/n} - 1\\right)\\right)

    where :math:`D(x,y)` is the distance function (as defined in :func:`~gigalens.tf.profiles.light.sersic.Sersic.distance`).
    In the simplest case, it is just Euclidean distance from the center, and when ellipticity is non-zero, the
    coordinate axes are translated, rotated and scaled to match the ellipse defined by the complex ellipticities
    ``(e1,e2)`` with center ``(center_x, center_y)`` then the Euclidean distance from the center is calculated.
    If least squares is not being used, the amplitude :math:`I_e` is set to be 1.
    """

    _name = "SERSIC"
    _params = ["R_sersic", "n_sersic", "center_x", "center_y"]

    @tf.function
    def light(self, x, y, R_sersic, n_sersic, center_x, center_y, Ie=None):
        Ie = tf.ones_like(R_sersic) if self.use_lstsq else Ie
        R = self.distance(x, y, center_x, center_y)
        bn = _bn(n_sersic)
        ret = Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        return ret[tf.newaxis, ...] if self.use_lstsq else ret

    @tf.function
    def distance(self, x, y, cx, cy, e1=None, e2=None):
        """

        Args:
            x: The :math:`x` coordinates to evaluate the distance function at
            y: The :math:`y` coordinates to evaluate the distance function at
            cx: The :math:`x` coordinate of the center of the Sersic light component
            cy: The :math:`y` coordinate of the center of the Sersic light component
            e1: Complex ellipticity component. If unspecified, it is assumed to be zero.
            e2: Complex ellipticity component. If unspecified, it is assumed to be zero.

        Returns:
            The distance function evaluated at ``(x,y)``
        """
        if e1 is None:
            e1 = tf.zeros_like(cx)
        if e2 is None:
            e2 = tf.zeros_like(cx)
        phi = tf.atan2(e2, e1) / 2
        c = tf.math.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        dx, dy = x - cx, y - cy
        cos_phi, sin_phi = tf.math.cos(phi), tf.math.sin(phi)
        xt1 = (cos_phi * dx + sin_phi * dy) * tf.math.sqrt(q)
        xt2 = (-sin_phi * dx + cos_phi * dy) / tf.math.sqrt(q)
        return tf.sqrt(xt1 ** 2 + xt2 ** 2)


class SersicEllipse(Sersic):
    _name = "SERSIC_ELLIPSE"
    _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]

    @tf.function
    def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
        Ie = tf.ones_like(R_sersic) if self.use_lstsq else Ie
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = _bn(n_sersic)
        ret = Ie * tf.math.exp(-bn * ((R / R_sersic) ** (1 / n_sersic) - 1.0))
        return ret[tf.newaxis, ...] if self.use_lstsq else ret


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

    @tf.function
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
        Ie = tf.ones_like(R_sersic) if self.use_lstsq else Ie
        R = self.distance(x, y, center_x, center_y, e1, e2)
        bn = _bn(n_sersic)
        ret = (
                Ie
                * (1 + (Rb / R) ** alpha) ** (gamma / alpha)
                * tf.math.exp(
            -bn
            * (
                    (R ** alpha + Rb ** alpha)
                    / R_sersic ** alpha ** 1.0
                    / (alpha * n_sersic)
            )
            - 1.0
        )
        )
        return ret[tf.newaxis, ...] if self.use_lstsq else ret
