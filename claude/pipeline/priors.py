"""The physical model and its priors, for any lens system.

Under the GIGA-Lens scene API a model is one object. ``LensModel`` holds ``Plane``s of
``Component``s, each ``Component`` a profile bundled with a dict of per-parameter priors,
and the joint prior, the unconstraining bijector, the flat-``z`` column order and the
constants are all *derived* from it. There is no separate prior to keep in step with a
separate ``PhysicalModel``, which retires the old zip-mismatch hazard: a profile and its
prior can no longer disagree about how many source components there are, because they are
the same object.

A model here is built from two things: a **version**, which fixes the functional form --
which distribution family each parameter gets and, critically, what its *support* is --
and a **``LensConfig``**, which supplies the location and scale within that form.

That division is load-bearing, not stylistic. Checkpoints under ``runs/`` store
*unconstrained* samples. A stored value maps back to a physical one through the model's
bijector, and for a ``Uniform`` that bijector is ``low + (high - low) * sigmoid(z)`` --
purely a function of the support. Change a bound and every old run silently reports
different numbers, with no error raised. So:

- **Supports live in this file, keyed by version.** They appear as literals in the
  ``_source_priors_*`` and ``_lens_light_priors_*`` functions below. A config cannot
  reach them.
- **Locations and scales come from the config.** Moving a ``LogNormal``'s median or a
  ``Normal``'s sigma redistributes prior mass without touching the bijector, so an old
  checkpoint still decodes to the same physical value.

If a new lens needs a different bound, add a version -- do not edit one in place, and do
not add a config field for it. Keep the superseded builder verbatim so the runs fitted
with it stay readable. ``run_fit.py`` records the versions in ``metadata.json`` and
``diagnose.py`` reads them back.

**Version numbering restarts at 1 with the scene API.** The port changed the flat-z column
order (path-keyed sorted order, so lens *light* now precedes lens *mass* within the lens
plane) and the noise convention, which makes pre-port checkpoints unreadable by this code
regardless of prior version. ``metadata.json`` records ``"api": "scene"``, and
``diagnose.rebuild_model`` refuses a run without it rather than decoding old samples
through the new bijector. From here on the versioning invariant governs every change as
before.

A note on the Sersic index floor, which recurs across systems. gigalens *used* to hardcode
the Sersic constant as the asymptotic ``bn = 1.9992 * n - 0.3271``, which is accurate at
large n and badly wrong at small n -- against the exact ``gammaincinv(2n, 0.5)``:

    n      0.3      0.5      1.0      2.0      6.0     10.0
    err  -13.6%    -3.0%   -0.37%   -0.02%   +0.00%   -0.01%

That is why source ``n_sersic`` used to carry a 0.5 floor: below it, a fitted index was
reporting a profile-accuracy artefact rather than a morphology. The installed gigalens uses
a rational (3,2) approximation accurate to 5.5e-05 over n in [0.3, 8]
(``claude/pipeline/test_sersic_bn.py`` validates it against the defining relation, not
against lenstronomy -- lenstronomy carries the same linear form and the same -13.6%). The
floor is therefore no longer required by the profile, and the support below runs down to
0.3, the lowest n where the approximation is verified. Do not go below 0.3 without
extending that test.
"""

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.sis import SIS
from gigalens.jax.scene import Component, LensModel, Plane

# Version of the source-component prior that build_model produces by default. Bump this
# whenever the source prior's *support* changes (a Uniform's bounds, a TruncatedNormal's
# limits). See the module docstring.
SOURCE_PRIOR_VERSION = 1

# Same discipline for the lens-light prior. v1 is the single-component prior; v2 is the
# bulge+disk split, whose bulge ``n_sersic`` support is Uniform(1, 8) rather than
# Uniform(2, lens_light_n_max). That is a *support* change, so it must be versioned.
#
# The version is a function of the component count, not a free choice: v1 describes one
# component and v2 describes two, so ``lens_light_prior_version_for(n_lens_light)``
# resolves it. Use that helper rather than a bare constant.
LENS_LIGHT_PRIOR_VERSIONS = {1: 1, 2: 2}


def lens_light_prior_version_for(n_lens_light):
    """Prior version implied by the lens-light component count."""
    try:
        return LENS_LIGHT_PRIOR_VERSIONS[n_lens_light]
    except KeyError:
        raise ValueError(
            f"no lens-light prior version defined for n_lens_light={n_lens_light}; "
            f"known counts are {sorted(LENS_LIGHT_PRIOR_VERSIONS)}"
        ) from None


def _source_priors_v1(p):
    """One Sersic source component's priors (v1, current). Support: n in (0.3, 6), |e| < 0.5.

    Returns a fresh dict of fresh distributions on every call. Both freshnesses matter:
    ``LensModel`` raises if the *same* ``tfd.Distribution`` object appears at two sites
    (reuse means sharing, and must be spelled ``shared(...)``), so two source components
    built from one dict would be rejected rather than silently linked.

    Components share one *config* prior, which makes them formally exchangeable:
    relabelling them leaves the posterior unchanged, so the data rather than the prior
    decides the morphology. In practice all HMC chains start from ``q_z.sample()``, a single
    unimodal Gaussian around one MAP mode, so they should stay in one labelling --
    ``diagnose.py`` measures that rather than assuming it (see ``label_switching_check``).
    """
    return dict(
        R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
        n_sersic=tfd.Uniform(0.3, 6.0),
        e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
        center_x=tfd.Normal(0.0, p.center_sigma),
        center_y=tfd.Normal(0.0, p.center_sigma),
        Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
    )


# Every source prior version that has ever been written under the scene API, so a run
# fitted with any of them stays readable. Add an entry when adding a version; never
# repoint an existing one.
_SOURCE_PRIOR_BUILDERS = {1: _source_priors_v1}


def _lens_light_priors_v1(p, lens_light_n_max=10.0):
    """The single-component deflector-light priors (v1). Support: n in (2, n_max).

    ``lens_light_n_max`` is an argument rather than a config field because it is a *bound*:
    it is exposed on the command line so a run records its own value in ``metadata.json``,
    and diagnostics replay that recorded value rather than whatever a config says today.
    """
    return dict(
        R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
        n_sersic=tfd.Uniform(2.0, lens_light_n_max),
        e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
        center_x=tfd.Normal(0.0, p.center_sigma),
        center_y=tfd.Normal(0.0, p.center_sigma),
        Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
    )


# n_sersic support per v2 lens-light component. Both stay inside the range where the
# Sersic bn approximation is accurate to well under a percent.
_LENS_LIGHT_V2_N_BOUNDS = {"bulge": (1.0, 8.0), "disk": (0.5, 2.5)}


def _lens_light_priors_v2(p, which):
    """One deflector light component's priors (v2): ``which`` is "bulge" or "disk".

    Deliberately NOT identical across components, which is the opposite of the choice made
    for the source. Source components are exchangeable on purpose. Here the two components
    mean distinct physical things, and giving them identical priors would only add a
    label-switching mode to the posterior for no benefit.
    """
    try:
        n_lo, n_hi = _LENS_LIGHT_V2_N_BOUNDS[which]
    except KeyError:
        raise ValueError(
            f"unknown lens-light component {which!r}; expected "
            f"{' or '.join(sorted(_LENS_LIGHT_V2_N_BOUNDS))}"
        ) from None
    return dict(
        R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
        n_sersic=tfd.Uniform(n_lo, n_hi),
        e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
        center_x=tfd.Normal(0.0, p.center_sigma),
        center_y=tfd.Normal(0.0, p.center_sigma),
        Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
    )


def _lens_light_prior_for(cfg, which):
    """Look up a lens-light component's location-scale parameters in the config."""
    try:
        return cfg.lens_light_priors[which]
    except KeyError:
        raise KeyError(
            f"{cfg.name}: config defines no lens_light_priors[{which!r}]; it has "
            f"{sorted(cfg.lens_light_priors)}"
        ) from None


# Version of the host deflector's EPL + external-shear prior. Added 2026-10-03, when
# bounding the EPL ellipticity became necessary (v2 below); v1 is what rounds 1-3 of every
# system were fitted under and is kept verbatim so those checkpoints stay readable.
#
# The default stays 1 deliberately. Every run in ``runs/`` was fitted at v1, and a new
# default would silently change what a run gets when nobody asked for it; v2 is opted into
# per run, recorded in ``metadata.json``, and replayed from there by
# ``diagnose.rebuild_model``.
MASS_PRIOR_VERSION = 1


def _mass_priors_v1(m, theta_E_center, theta_E_width):
    """Host EPL priors (v1). Support: gamma in (1, 3), e1/e2 UNBOUNDED.

    What "unbounded ellipticity" costs, learned on DESI234 and recorded here rather than
    in one lens's notes: ``e1``/``e2`` are ``Normal``s, so ``boundary_pinning_check``
    cannot see them by construction -- it only inspects ``Uniform`` and
    ``TruncatedNormal`` (see its docstring). DESI234's ``r3_fg`` fitted
    ``c = sqrt(e1^2+e2^2) = 0.750``, i.e. an axis ratio ``q = (1-c)/(1+c) = 0.143`` -- a
    7:1 needle -- and nothing in the pinning check reported it. Only
    ``prior_posterior_tension`` caught the associated ``gamma`` excursion.

    Kept verbatim. Do not "fix" this function: rounds 1-3 on DESI234, DESI238 and DESI257
    were all fitted under it, and editing it in place would silently redefine their stored
    unconstrained samples.
    """
    return dict(
        theta_E=tfd.LogNormal(jnp.log(theta_E_center), theta_E_width),
        gamma=tfd.TruncatedNormal(m.gamma_center, m.gamma_sigma, 1.0, 3.0),
        e1=tfd.Normal(0.0, m.ellip_sigma),
        e2=tfd.Normal(0.0, m.ellip_sigma),
        # Arcsec relative to the crop centre, which the config's crop is chosen to put
        # the deflector at.
        center_x=tfd.Normal(0.0, m.center_sigma),
        center_y=tfd.Normal(0.0, m.center_sigma),
    )


# Per-component ellipticity truncation for the v2 host mass prior, and the axis ratio it
# is derived from.
#
# gigalens publishes the (e1, e2) -> q map and a plausibility floor on q, so this bound is
# taken from the library rather than invented here:
# ``gigalens.physicality.axis_ratio_constraint`` (physicality.py:177-191) states that
# across every gigalens mass profile ``c = sqrt(e1^2 + e2^2)`` and ``q = (1-c)/(1+c)``
# (verified uniform 2026-07-13), and the EPL kernel itself computes exactly that
# (``profiles/mass/epl.py:57-58``). The library registers ``q >= 0.2`` as a SOFT
# plausibility band -- "a very flattened mass profile (small q) is physically valid but
# rare".
#
# q >= 0.2 is therefore c <= (1-0.2)/(1+0.2) = 0.6667. A per-component ``TruncatedNormal``
# on e1 and e2 is a SQUARE, not that disc, so the square must be inscribed in it:
# |e_i| <= c_max / sqrt(2) = 0.4714 guarantees c <= 0.6667 everywhere in the box. The
# inscribed square is the conservative choice -- it cannot admit a q below the floor,
# whereas the circumscribed square could.
#
# Why a box and not the disc itself: the disc is a JOINT constraint on two parameters, and
# this pipeline's priors are per-parameter ``tfd`` distributions composed into a
# ``JointDistributionNamed``. A joint truncation would need a custom bijector, whose
# inverse every stored checkpoint would then depend on. The box is expressible in the
# existing machinery, is strictly inside the published floor, and keeps
# ``boundary_pinning_check`` able to SEE the bound -- which is precisely what failed at v1,
# since that check reads ``TruncatedNormal`` bounds and ignores ``Normal``.
#
# Cost in prior mass, at DESI234's ellip_sigma = 0.15 (iid Normal, 4e6 draws): the box
# removes 3.35e-3 of the prior, and the true disc c > 0.6667 holds only 5.15e-5. So this
# excludes a region the prior barely used, while r3_fg's c = 0.7498 sits at 5.0 Rayleigh
# sigma -- a place the prior assigned 3e-6 mass and the likelihood went anyway. The bound
# removes almost no prior mass and all of the pathology.
MASS_V2_Q_MIN = 0.2
MASS_V2_ELLIP_ABS_MAX = 0.4714  # = ((1 - q_min)/(1 + q_min)) / sqrt(2), q_min = 0.2


def _mass_priors_v2(m, theta_E_center, theta_E_width):
    """Host EPL priors (v2). Support: gamma in (1, 3), |e1|, |e2| <= MASS_V2_ELLIP_ABS_MAX.

    Identical to v1 except that ``e1`` and ``e2`` are ``TruncatedNormal`` rather than
    ``Normal``. The scale still comes from the config's ``ellip_sigma``; only the support
    changes, which is exactly why this needs its own version.

    The truncation is the inscribed square of gigalens' own soft ``q >= 0.2`` plausibility
    band -- see ``MASS_V2_ELLIP_ABS_MAX`` above for the derivation and for the measured
    prior mass this costs (3.35e-3 at ellip_sigma = 0.15).

    This does not assert that the deflector IS round. q = 0.2 is a 5:1 axis ratio, which is
    extreme for a galaxy-scale deflector and far beyond anything the other runs on these
    systems fit (DESI234 round 2: q = 0.900; r3_unmasked: q = 0.943). It asserts only that
    the fit may not go outside the range the library calls physically plausible in order to
    buy chi-squared -- which is what r3_fg did.
    """
    return dict(
        theta_E=tfd.LogNormal(jnp.log(theta_E_center), theta_E_width),
        gamma=tfd.TruncatedNormal(m.gamma_center, m.gamma_sigma, 1.0, 3.0),
        e1=tfd.TruncatedNormal(
            0.0, m.ellip_sigma, -MASS_V2_ELLIP_ABS_MAX, MASS_V2_ELLIP_ABS_MAX
        ),
        e2=tfd.TruncatedNormal(
            0.0, m.ellip_sigma, -MASS_V2_ELLIP_ABS_MAX, MASS_V2_ELLIP_ABS_MAX
        ),
        center_x=tfd.Normal(0.0, m.center_sigma),
        center_y=tfd.Normal(0.0, m.center_sigma),
    )


# Every host-mass prior version ever written, so a run fitted with any of them stays
# readable. Add an entry when adding a version; never repoint an existing one.
_MASS_PRIOR_BUILDERS = {1: _mass_priors_v1, 2: _mass_priors_v2}


# Version of the foreground-interloper prior (SIS mass + Sersic light) that build_model
# adds when given a ``foreground``. Same versioning discipline as SOURCE_PRIOR_VERSION:
# bump on a support change, never repoint an existing entry.
FOREGROUND_PRIOR_VERSION = 1


def _foreground_mass_priors_v1(p):
    """SIS mass-component priors for a candidate foreground interloper (v1). Support:
    theta_E in [0, inf).

    theta_E is a **HalfNormal**, not the LogNormal every other Einstein radius in this
    codebase uses. That is deliberate, not an oversight: this component exists so a fit
    can choose between "the SW object has no mass" (theta_E -> 0) and "it deflects like a
    foreground galaxy" (theta_E of order the object's own scale), and a LogNormal's support
    is bounded away from 0 by construction -- it would rule the first reading out before
    the data gets a vote. HalfNormal(scale=p.theta_E_scale) has mode and lower bound at
    exactly 0, and its scale is chosen (see the config that sets it) to put real mass out
    to ~1", a plausible foreground galaxy's Einstein radius. There is deliberately no upper
    floor either: see build_model's foreground docstring for the nesting argument this
    preserves -- a bound here would be the same mistake as a floor on the light Ie below.

    A fitted theta_E near 0 says the interloper reading is disfavoured relative to the
    fourth-image reading (round 3's model, which this model nests); a fitted theta_E well
    away from 0, with the light component's Ie also away from 0, says the SW object is
    doing double duty as a plausible foreground galaxy -- both deflecting and emitting at a
    self-consistent mass-to-light ratio for its type.
    """
    return dict(
        theta_E=tfd.HalfNormal(p.theta_E_scale),
        center_x=tfd.Normal(p.center_x, p.center_sigma),
        center_y=tfd.Normal(p.center_y, p.center_sigma),
    )


def _foreground_light_priors_v1(p):
    """Sersic light-component priors for the same candidate foreground interloper (v1).
    Support: n in (0.3, 6), |e| < 0.5 -- identical to a source Sersic component's, because
    there is no reason this object's light should carry a different shape prior than any
    other free Sersic component in this pipeline.

    Ie's LogNormal support runs down to arbitrarily faint (no floor), which is the light
    side of the same nesting argument the mass prior's docstring states: Ie -> 0 switches
    this component off, reducing the model to round 3's (the SW object as pure fourth
    image, no separate foreground light). Do not add a floor here for the same reason a
    floor on theta_E above would break the nesting.
    """
    return dict(
        R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
        n_sersic=tfd.Uniform(0.3, 6.0),
        e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
        center_x=tfd.Normal(p.center_x, p.center_sigma),
        center_y=tfd.Normal(p.center_y, p.center_sigma),
        Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
    )


# Every foreground prior version that has ever been written, so a run fitted with any of
# them stays readable. Each entry is (mass_priors_fn, light_priors_fn); add an entry when
# adding a version, never repoint an existing one.
_FOREGROUND_PRIOR_BUILDERS = {1: (_foreground_mass_priors_v1, _foreground_light_priors_v1)}


def build_model(
    cfg,
    theta_E_center=None,
    theta_E_width=None,
    shear_width=None,
    n_source=1,
    n_lens_light=1,
    lens_light_n_max=10.0,
    source_prior_version=SOURCE_PRIOR_VERSION,
    lens_light_prior_version=None,
    mass_prior_version=MASS_PRIOR_VERSION,
    foreground=None,
    foreground_prior_version=FOREGROUND_PRIOR_VERSION,
):
    """Build the ``LensModel``: EPL + external shear, ``n_lens_light`` Sersic deflector
    light components on the lens plane, ``n_source`` Sersic sources behind it, and
    optionally one more SIS + Sersic pair on the lens plane standing in for a specific
    foreground interloper.

    The profile *choice* is a modelling decision shared across systems and lives here; the
    numbers that distinguish one lens from another come from ``cfg``. If a lens needs a
    different profile family, that is a new function in this file, not a config field.

    Free-parameter count: 22 at ``n_source=1, n_lens_light=1``; 29 with a second source;
    36 with both; 46 with both plus ``foreground`` (10 more: SIS's 3, Sersic's 7).
    ``model.z_param_names`` is the authoritative column->name map for the flat ``z``; never
    reconstruct it from declaration order.

    ``use_lstsq=False`` throughout, which pairs with ``ProbModel(..., mode="forward")``.
    The escalation path -- shapelets with ``use_lstsq=True`` -- needs ``mode="lstsq"``
    instead, and ``ImageLikelihoodTerm`` now raises on a mismatch rather than dying in a
    rank error deep in the first trace.

    Every ``Component`` and every distribution is a fresh object. Under the scene API
    identity *is* sharing: reusing a Component would make two sites the same physical
    component, and reusing a distribution raises. Names ("host", "bulge", "src0") are
    display labels only -- they feed ``model.z_labels`` and plot titles, and deliberately
    do not enter the parameter path, so renaming one can never permute a saved chain.

    **Why ``foreground`` is a nested model, not a competing one.** A masked fit and an
    unmasked fourth-image fit are not a fair comparison of "interloper" against
    "fourth image": masking always removes the SW object's light from the likelihood, so
    it silently assumes the interloper reading (an interloper deflects whether or not its
    light is fitted) rather than testing it. The fair comparison adds a component that
    *can* act like a foreground galaxy -- it deflects (SIS) and it emits (Sersic) -- and
    lets the fit decide how much of either it wants. With the SIS ``theta_E -> 0`` and the
    light's ``Ie -> 0``, this model IS the model without ``foreground``: the extra galaxy
    contributes no deflection and no light, so it exactly reduces to the fourth-image
    model. That nesting is why the two priors below carry no floor near zero (see their own
    docstrings) -- a floor would make the reduction impossible and turn a nested comparison
    into two disjoint models that cannot be compared by a chi-squared difference over the
    10 extra parameters. The fitted SIS ``theta_E`` is therefore itself the discriminating
    measurement: pinned near 0 favours the fourth-image reading, well away from 0 (with
    nonzero ``Ie``) favours the foreground reading.

    Args:
        cfg: the ``LensConfig`` supplying every location and scale.
        theta_E_center: override the config's Einstein-radius median, in arcsec. Used by
            falsification runs that deliberately centre the prior elsewhere. ``None`` takes
            the config value.
        theta_E_width: override the config's Einstein-radius sigma, in log space.
        shear_width: override the config's external-shear sigma.
        n_source: number of Sersic source components. Components get identical priors.
        n_lens_light: number of Sersic deflector-light components, 1 or 2. At 2 the
            components get *different* (bulge / disk) priors.
        lens_light_n_max: upper bound of the v1 lens-light ``n_sersic`` Uniform. A bound,
            so it is a command-line argument recorded per run, not a config field. Applies
            to v1 only; the v2 bulge/disk priors carry their own bounds.
        source_prior_version: currently only 1. **Reading an old run's checkpoints requires
            its own version** -- the stored samples are unconstrained, and a different
            support maps the same value to a different number without raising.
            ``diagnose.py`` selects this from ``metadata.json``.
        lens_light_prior_version: 1 for the single-component prior, 2 for the bulge/disk
            split. Defaults to whatever ``n_lens_light`` implies, which is the only
            consistent pairing -- the versions differ in component count, so they are not
            independently selectable. Same checkpoint hazard as ``source_prior_version``.
        mass_prior_version: which versioned support the host EPL uses. v1 (the default,
            and what every run fitted before 2026-10-03 used) leaves ``e1``/``e2``
            unbounded; v2 truncates them to the inscribed square of gigalens' soft
            ``q >= 0.2`` plausibility floor. Same checkpoint hazard as
            ``source_prior_version``: the support fixes the bijector, so a run must be
            read back at the version it was fitted under.
        foreground: a ``lenses.ForegroundPrior`` (typically ``cfg.foreground_prior``), or
            ``None`` (the default) to build the model without the extra SIS + Sersic pair.
            Adding this changes the flat-``z`` column count and order, so it is recorded in
            ``metadata.json`` exactly like ``n_source``/``n_lens_light`` -- see
            ``run_fit.py`` and ``diagnose.rebuild_model``.
        foreground_prior_version: which versioned support the foreground SIS + Sersic pair
            uses. Same checkpoint hazard as ``source_prior_version``. Ignored when
            ``foreground`` is ``None``.
    """
    if n_source < 1:
        raise ValueError(f"n_source must be >= 1, got {n_source}")
    if source_prior_version not in _SOURCE_PRIOR_BUILDERS:
        raise ValueError(
            f"source_prior_version must be one of "
            f"{sorted(_SOURCE_PRIOR_BUILDERS)}, got {source_prior_version}"
        )
    if n_lens_light not in (1, 2):
        raise ValueError(f"n_lens_light must be 1 or 2, got {n_lens_light}")
    expected_llv = lens_light_prior_version_for(n_lens_light)
    if lens_light_prior_version is None:
        lens_light_prior_version = expected_llv
    elif lens_light_prior_version != expected_llv:
        raise ValueError(
            f"lens_light_prior_version={lens_light_prior_version} does not describe "
            f"n_lens_light={n_lens_light} (which implies v{expected_llv}). The versions "
            f"differ in component count, so they cannot be mixed."
        )
    if mass_prior_version not in _MASS_PRIOR_BUILDERS:
        raise ValueError(
            f"mass_prior_version must be one of "
            f"{sorted(_MASS_PRIOR_BUILDERS)}, got {mass_prior_version}"
        )
    if foreground is not None and foreground_prior_version not in _FOREGROUND_PRIOR_BUILDERS:
        raise ValueError(
            f"foreground_prior_version must be one of "
            f"{sorted(_FOREGROUND_PRIOR_BUILDERS)}, got {foreground_prior_version}"
        )

    m = cfg.mass_prior
    theta_E_center = m.theta_E_median if theta_E_center is None else theta_E_center
    theta_E_width = m.theta_E_logsigma if theta_E_width is None else theta_E_width
    shear_width = m.shear_width if shear_width is None else shear_width

    host = Component(
        EPL(50),
        _MASS_PRIOR_BUILDERS[mass_prior_version](m, theta_E_center, theta_E_width),
        name="host",
    )
    ext_shear = Component(
        Shear(),
        dict(
            gamma1=tfd.Normal(0.0, shear_width),
            gamma2=tfd.Normal(0.0, shear_width),
        ),
        name="shear",
    )

    # One component per entry, each a fresh profile and a fresh prior dict. Unlike the
    # source, the components are NOT interchangeable at v2 -- bulge and disk carry
    # distinct priors on purpose.
    if lens_light_prior_version == 1:
        lens_light = [
            Component(
                SersicEllipse(use_lstsq=False),
                _lens_light_priors_v1(
                    _lens_light_prior_for(cfg, "single"), lens_light_n_max
                ),
                name="single",
            )
        ]
    else:
        lens_light = [
            Component(
                SersicEllipse(use_lstsq=False),
                _lens_light_priors_v2(_lens_light_prior_for(cfg, w), w),
                name=w,
            )
            for w in ("bulge", "disk")
        ]

    # Identical priors across source components: the data chooses the morphology, not the
    # prior.
    make_priors = _SOURCE_PRIOR_BUILDERS[source_prior_version]
    source_light = [
        Component(
            SersicEllipse(use_lstsq=False),
            make_priors(cfg.source_prior),
            name=f"src{i}",
        )
        for i in range(n_source)
    ]

    # The candidate foreground interloper: unlensed (it sits on the SAME lens plane as the
    # host, not behind it), so its SIS goes in the lens plane's mass list -- ADDED AFTER
    # host, so host stays mass index 0 and diagnose.column("mass/0/theta_E") keeps picking
    # the host's Einstein radius unambiguously even though the SIS now has one too -- and
    # its Sersic light goes in the lens plane's light list, alongside the deflector's own.
    lens_mass = [host, ext_shear]
    if foreground is not None:
        mass_priors_fn, light_priors_fn = _FOREGROUND_PRIOR_BUILDERS[foreground_prior_version]
        lens_mass.append(
            Component(SIS(), mass_priors_fn(foreground), name="foreground")
        )
        lens_light = lens_light + [
            Component(
                SersicEllipse(use_lstsq=False), light_priors_fn(foreground), name="foreground"
            )
        ]

    return LensModel(
        [
            Plane(name="lens", mass=lens_mass, light=lens_light),
            # A lone source plane may take the documented deflection_ratio=1.0 default,
            # but state it: an implicit geometry is the sort of thing that stops being
            # true the moment a second source plane is added.
            Plane(name="source", deflection_ratio=1.0, light=source_light),
        ]
    )


def source_component_spans(model):
    """Flat-z index spans for each source component, as a list of ``(start, stop)``.

    Derived from ``model.z_param_names`` by grouping on the ``planes/<i>/light/<j>/``
    prefix of the *lensed* planes, rather than from hardcoded offsets: the columns are
    contiguous per component today (22-28 and 29-35 in the 36-parameter model), but a
    change in component counts must not be able to silently misalign the spans that
    ``diagnose.label_switching_check`` swaps whole blocks of.

    Returns ``[]`` for a model whose z-columns are not one contiguous run per component,
    which is the honest answer -- the block-swap logic is only valid when they are.
    """
    names = list(model.z_param_names or [])
    # Which planes carry lensed (source) light: any plane with mass in front of it.
    lensed = [
        i
        for i in range(len(model.planes))
        if model.planes[i].has_light
        and any(model.planes[j].has_mass for j in range(i))
    ]
    spans = []
    for i in lensed:
        for j in range(len(model.planes[i].light)):
            prefix = f"planes/{i}/light/{j}/"
            idx = [k for k, n in enumerate(names) if n.startswith(prefix)]
            if not idx:
                continue
            if idx != list(range(idx[0], idx[-1] + 1)):
                return []  # non-contiguous: block swapping would corrupt the samples
            spans.append((idx[0], idx[-1] + 1))
    return spans
