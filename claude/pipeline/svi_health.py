"""The SVI-covariance conditioning guard.

This is the one piece of ``compat.py`` that survived the port to the GIGA-Lens scene
API. Every JAX-version shim in that file is fixed upstream; this is not, because it
never was a compatibility shim. ``gigalens.jax.inference.HMC`` still builds its momentum
distribution from ``jnp.linalg.inv(q_z.covariance())`` (``inference/HMC.py:71-74``), and
the library carries no conditioning check of its own, so the failure mode below is still
live.

Read ``check_svi_covariance``'s docstring before changing the threshold: the number is
set by measurement across stored runs, not by taste.
"""

import numpy as np

#: Condition number above which the SVI covariance is rejected outright.
#:
#: The threshold is set by measurement, not by taste. Across the five runs to date the two
#: populations are separated by three orders of magnitude:
#:
#:     nsersic10  9.4e+05   inner3  1.1e+07   seed1  4.8e+07   seed0  5.9e+08   (22 par, ok)
#:     twosource  8.8e+11                                             (29 par, diverged)
#:
#: 1e10 sits in the gap with ~17x margin below and ~90x above. Those runs predate the
#: scene-API port, but a covariance's conditioning does not depend on z-column order, so
#: the calibration survives it -- ``verify.py`` check 6 still replays them.
COND_MAX = 1e10


def check_svi_covariance(cov=None, scale_tril=None, max_cond=COND_MAX):
    """Raise unless the SVI covariance is usable as an HMC preconditioner.

    Pass ``scale_tril`` when you have it -- see the note on precision at the end. Returns
    ``(min_eig, cond)``.

    ``gigalens.jax.inference.HMC`` builds the momentum distribution from
    ``inv(q_z.covariance())``, in float32 (the pipeline runs
    ``likelihood_precision="float32"`` with JAX x64 off). What kills a run is the
    *conditioning* of that inverse, not the sign of the covariance's eigenvalues:

        run         cov condition   max|inv(C) @ C - I|
        nsersic10      9.4e+05            0.07
        seed0          5.9e+08            0.98
        twosource      8.8e+11           55.3

    At 8.8e+11 the float32 inverse is not an inverse at all. Its raw spectrum carries five
    negative eigenvalues reaching -3.2e+07, so HMC sampled against an indefinite mass
    matrix, and all 29 of that run's parameters failed Rhat. That was read at the time as
    slow mixing; it was not.

    A positive-definiteness test does NOT catch this, which is worth stating plainly
    because the obvious version of this guard was written first and silently passed the one
    run it existed for. ``twosource``'s covariance is positive definite: min eigenvalue
    +5.7e-13 as TFP computes it, +1.5e-13 in float64. The -1.8e-12 originally attributed to
    it came from forming ``L @ L.T`` in float32 during the audit -- rounding noise in a
    matrix whose smallest eigenvalues sit twelve orders below its largest, not a property
    of the fit. Conditioning is the real discriminant, so conditioning is the test.

    Both failures are hard errors. A repaired or floored covariance would yield an HMC run
    that looks fine downstream and is not, which costs more than a failed stage.

    Eigenvalues come from an SVD of the Cholesky factor when ``scale_tril`` is given
    (squaring the singular values), because forming ``L @ L.T`` loses half the available
    precision exactly where this matrix has none to spare. Falling back to ``cov`` is
    supported but strictly worse conditioned; it is what produced the spurious negative
    eigenvalue described above.
    """
    if (cov is None) == (scale_tril is None):
        raise TypeError("pass exactly one of cov= or scale_tril=")

    if scale_tril is not None:
        L = np.asarray(scale_tril, dtype=np.float64)
        if not np.isfinite(L).all():
            raise RuntimeError(
                "SVI scale_tril contains non-finite entries; the SVI stage diverged."
            )
        s = np.linalg.svd(L, compute_uv=False)
        # A Cholesky factor's singular values are the square roots of the covariance's
        # eigenvalues, and are non-negative by construction -- so a zero here is a
        # rank-deficient (singular) covariance, reported below as a non-PD failure.
        min_eig, max_eig = float(s.min() ** 2), float(s.max() ** 2)
    else:
        c = np.asarray(cov, dtype=np.float64)
        c = 0.5 * (c + c.T)
        if not np.isfinite(c).all():
            raise RuntimeError(
                "SVI covariance contains non-finite entries; the SVI stage diverged."
            )
        w = np.linalg.eigvalsh(c)
        min_eig, max_eig = float(w.min()), float(w.max())

    cond = abs(max_eig / min_eig) if min_eig > 0 else np.inf
    if min_eig <= 0:
        raise RuntimeError(
            f"SVI covariance is not positive definite (min eigenvalue {min_eig:.3e}, "
            f"condition {cond:.3e}). The SVI stage diverged; HMC preconditioned on this "
            f"would produce meaningless Rhat. Re-run SVI, lowering --svi-peak-lr."
        )
    if cond > max_cond:
        raise RuntimeError(
            f"SVI covariance condition number {cond:.3e} exceeds the limit {max_cond:.0e} "
            f"(min eigenvalue {min_eig:.3e}). Its float32 inverse will not be a valid "
            f"precision matrix, so HMC would sample against an effectively indefinite "
            f"mass matrix and every Rhat would be meaningless -- this is what happened to "
            f"the twosource run. Healthy runs here sit at 1e6-6e8. Re-run SVI, lowering "
            f"--svi-peak-lr."
        )
    return min_eig, cond
