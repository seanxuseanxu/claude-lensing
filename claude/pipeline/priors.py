"""Prior construction and the physical model, for any lens system.

A prior here is built from two things: a **version**, which fixes the functional form --
which distribution family each parameter gets and, critically, what its *support* is --
and a **``LensConfig``**, which supplies the location and scale within that form.

That division is load-bearing, not stylistic. Checkpoints under ``runs/`` store
*unconstrained* samples. A stored value maps back to a physical one through the prior's
bijector, and for a ``Uniform`` that bijector is ``low + (high - low) * sigmoid(z)`` --
purely a function of the support. Change a bound and every old run silently reports
different numbers, with no error raised. So:

- **Supports live in this file, keyed by version.** They appear as literals in the
  ``_source_component_*`` and ``_lens_light_component_*`` functions below. A config cannot
  reach them.
- **Locations and scales come from the config.** Moving a ``LogNormal``'s median or a
  ``Normal``'s sigma redistributes prior mass without touching the bijector, so an old
  checkpoint still decodes to the same physical value.

If a new lens needs a different bound, add a version -- do not edit one in place, and do
not add a config field for it. Keep the superseded builder verbatim so the runs fitted
with it stay readable. ``run_fit.py`` records the versions in ``metadata.json`` and
``diagnose.py`` reads them back.

A note on the Sersic index floor, which recurs across systems: gigalens hardcodes the
Sersic constant as ``bn = 1.9992 * n - 0.3271``
(``gigalens/jax/profiles/light/sersic.py:15``), an asymptotic fit valid at large n. Against
the exact ``gammaincinv(2n, 0.5)``:

    n      0.3      0.5      1.0      2.0      6.0     10.0
    err  -13.6%    -3.0%   -0.37%   -0.02%   +0.00%   -0.01%

The 0.5 floor on source ``n_sersic`` is therefore deliberate. If a fit pins against it,
dropping the bound would trade a prior artefact for a profile-accuracy artefact; the fix
is a different profile, not a lower bound.
"""

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

# Version of the source-component prior that build_prior produces by default. Bump this
# whenever the source prior's *support* changes (a Uniform's bounds, a TruncatedNormal's
# limits). See the module docstring.
SOURCE_PRIOR_VERSION = 2

# Same discipline for the lens-light prior. v1 is the single-component prior; v2 is the
# bulge+disk split, whose bulge ``n_sersic`` support is Uniform(1, 8) rather than
# Uniform(2, lens_light_n_max). That is a *support* change, so it must be versioned.
#
# The version is a function of the component count, not a free choice: v1 describes one
# component and v2 describes two, so ``lens_light_prior_version_for(n_lens_light)``
# resolves it and a single-component run keeps the exact prior the early runs were fitted
# with. Use that helper rather than a bare constant.
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


def _source_component_v1(p):
    """The ORIGINAL single-source prior (v1). Supports: n in (0.5, 4), |e| < 0.5.

    Kept verbatim so runs fitted with it stay readable. Reading such a run through the v2
    prior below silently reports ``source.n_sersic = 0.6146`` where the fit actually said
    ``0.5729`` -- no error, just a wrong number, because ``Uniform(0.5, 4)`` and
    ``Uniform(0.5, 6)`` map the same unconstrained value to different physical ones. Do not
    delete this function to tidy up.
    """
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
            n_sersic=tfd.Uniform(0.5, 4.0),
            e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
            center_x=tfd.Normal(0.0, p.center_sigma),
            center_y=tfd.Normal(0.0, p.center_sigma),
            Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
        )
    )


def _source_component(p):
    """One Sersic source component's prior (v2, current). Support: n in (0.5, 6), |e| < 0.5.

    Components share one config prior, which makes them formally exchangeable: relabelling
    them leaves the posterior unchanged, so the data rather than the prior decides the
    morphology. In practice all HMC chains start from ``q_z.sample()``, a single unimodal
    Gaussian around one MAP mode, so they should stay in one labelling -- ``diagnose.py``
    measures that rather than assuming it (see ``label_switching_check``).
    """
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
            n_sersic=tfd.Uniform(0.5, 6.0),
            e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.5, 0.5),
            center_x=tfd.Normal(0.0, p.center_sigma),
            center_y=tfd.Normal(0.0, p.center_sigma),
            Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
        )
    )


def _lens_light_component_v1(p, lens_light_n_max=10.0):
    """The ORIGINAL single-component deflector prior (v1). Support: n in (2, n_max).

    Kept verbatim for the same reason ``_source_component_v1`` is. ``lens_light_n_max`` is
    an argument rather than a config field because it is a *bound*: it is exposed on the
    command line so a run records its own value in ``metadata.json``, and diagnostics
    replay that recorded value rather than whatever a config happens to say today.
    """
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
            n_sersic=tfd.Uniform(2.0, lens_light_n_max),
            e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
            center_x=tfd.Normal(0.0, p.center_sigma),
            center_y=tfd.Normal(0.0, p.center_sigma),
            Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
        )
    )


# n_sersic support per v2 lens-light component. Both stay inside the range where gigalens'
# bn approximation is accurate: the error is -0.37% at n = 1 and below 0.02% for n >= 2.
_LENS_LIGHT_V2_N_BOUNDS = {"bulge": (1.0, 8.0), "disk": (0.5, 2.5)}


def _lens_light_component(p, which):
    """One deflector light component's prior (v2): ``which`` is "bulge" or "disk".

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
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(p.R_sersic_median), p.R_sersic_logsigma),
            n_sersic=tfd.Uniform(n_lo, n_hi),
            e1=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0.0, p.ellip_sigma, -0.3, 0.3),
            center_x=tfd.Normal(0.0, p.center_sigma),
            center_y=tfd.Normal(0.0, p.center_sigma),
            Ie=tfd.LogNormal(jnp.log(p.Ie_median), p.Ie_logsigma),
        )
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


def build_prior(
    cfg,
    theta_E_center=None,
    theta_E_width=None,
    shear_width=None,
    n_source=1,
    n_lens_light=1,
    lens_light_n_max=10.0,
    source_prior_version=SOURCE_PRIOR_VERSION,
    lens_light_prior_version=None,
):
    """Build the nested prior over ``[lens, lens_light, source_light]``.

    Returns a ``tfd.JointDistributionSequential`` of ``JointDistributionNamed`` dicts,
    matching the structure ``ProbabilisticModel.pack_bij`` expects. 22 free parameters at
    ``n_source=1, n_lens_light=1``; 29 with a second source; 36 with both.

    Args:
        cfg: the ``LensConfig`` supplying every location and scale.
        theta_E_center: override the config's Einstein-radius median, in arcsec. Used by
            falsification runs that deliberately centre the prior elsewhere. ``None`` takes
            the config value.
        theta_E_width: override the config's Einstein-radius sigma, in log space.
        shear_width: override the config's external-shear sigma.
        n_source: number of Sersic source components. Must match ``build_phys_model``'s
            ``n_source``, or ``LensSimulator``'s ``zip`` over profiles and parameters
            silently drops the extra one. Components get identical priors.
        n_lens_light: number of Sersic deflector-light components, 1 or 2. Must match
            ``build_phys_model``'s ``n_lens_light`` for the same zip reason. At 2 the
            components get *different* (bulge / disk) priors.
        lens_light_n_max: upper bound of the v1 lens-light ``n_sersic`` Uniform. A bound,
            so it is a command-line argument recorded per run, not a config field. Applies
            to v1 only; the v2 bulge/disk priors carry their own bounds.
        source_prior_version: 2 (default) for the current wide priors; 1 for the narrow
            originals. **Reading an old run's checkpoints requires its own version** -- the
            stored samples are unconstrained, and a different support maps the same value
            to a different number without raising. ``diagnose.py`` selects this from
            ``metadata.json``.
        lens_light_prior_version: 1 for the single-component prior, 2 for the bulge/disk
            split. Defaults to whatever ``n_lens_light`` implies, which is the only
            consistent pairing -- the versions differ in component count, so they are not
            independently selectable. Same checkpoint hazard as ``source_prior_version``.
    """
    if n_source < 1:
        raise ValueError(f"n_source must be >= 1, got {n_source}")
    if source_prior_version not in (1, 2):
        raise ValueError(
            f"source_prior_version must be 1 or 2, got {source_prior_version}"
        )
    if source_prior_version == 1 and n_source != 1:
        raise ValueError("source_prior_version=1 only ever described a single component")
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

    m = cfg.mass_prior
    theta_E_center = m.theta_E_median if theta_E_center is None else theta_E_center
    theta_E_width = m.theta_E_logsigma if theta_E_width is None else theta_E_width
    shear_width = m.shear_width if shear_width is None else shear_width

    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(theta_E_center), theta_E_width),
                    gamma=tfd.TruncatedNormal(m.gamma_center, m.gamma_sigma, 1.0, 3.0),
                    e1=tfd.Normal(0.0, m.ellip_sigma),
                    e2=tfd.Normal(0.0, m.ellip_sigma),
                    # Arcsec relative to the crop centre, which the config's crop is
                    # chosen to put the deflector at.
                    center_x=tfd.Normal(0.0, m.center_sigma),
                    center_y=tfd.Normal(0.0, m.center_sigma),
                )
            ),
            tfd.JointDistributionNamed(
                dict(
                    gamma1=tfd.Normal(0.0, shear_width),
                    gamma2=tfd.Normal(0.0, shear_width),
                )
            ),
        ]
    )

    # One component per entry, each a fresh object. Unlike the source, the components are
    # NOT interchangeable at v2 -- bulge and disk carry distinct priors on purpose.
    if lens_light_prior_version == 1:
        p = cfg.legacy_lens_light_prior or _lens_light_prior_for(cfg, "single")
        lens_light_parts = [_lens_light_component_v1(p, lens_light_n_max)]
    else:
        lens_light_parts = [
            _lens_light_component(_lens_light_prior_for(cfg, w), w)
            for w in ("bulge", "disk")
        ]
    lens_light_prior = tfd.JointDistributionSequential(lens_light_parts)

    # One component per entry, each a fresh object. Identical priors at v2: the data
    # chooses the morphology, not the prior.
    if source_prior_version == 1:
        if cfg.legacy_source_prior is None:
            raise ValueError(
                f"{cfg.name}: source_prior_version=1 requested but the config defines no "
                f"legacy_source_prior. Old runs cannot be read back without it."
            )
        source_p, make_one = cfg.legacy_source_prior, _source_component_v1
    else:
        source_p, make_one = cfg.source_prior, _source_component
    source_light_prior = tfd.JointDistributionSequential(
        [make_one(source_p) for _ in range(n_source)]
    )

    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )


def build_phys_model(n_source=1, n_lens_light=1):
    """EPL + shear deflector, ``n_lens_light`` Sersic lens lights, ``n_source`` sources.

    Lens-independent: the profile *choice* is a modelling decision shared across systems,
    while the numbers that distinguish them live in the prior. If a lens needs a different
    profile family, that is a new function here, not a config field.

    ``use_lstsq=False`` throughout, which pairs with ``ForwardProbModel``. The escalation
    path -- shapelets with ``use_lstsq=True`` -- requires ``BackwardProbModel`` instead.

    ``LensSimulator.simulate`` sums source components with
    ``zip(self.phys_model.source_light, source_light_params)``, so nothing in gigalens
    needs changing to support more than one -- but the zip also means a mismatch between
    this ``n_source`` and ``build_prior``'s silently drops the extra component instead of
    raising. Always pass the same value to both.

    Each component is a **fresh** SersicEllipse instance rather than one object repeated.

    Note an upstream gigalens landmine while doing this. ``Parameterized.__init__`` does
    ``self.params = self._params`` -- binding the *class* list, not copying it -- and
    ``LightProfile.__init__`` then does ``self.params.append("amp")`` when
    ``use_lstsq=True``. Constructing a single lstsq profile therefore mutates the class
    attribute and retroactively corrupts every other instance of that class, including
    ones already built:

        a = SersicEllipse(use_lstsq=False)   # a.params has 6 entries
        b = SersicEllipse(use_lstsq=True)    # appends "amp" to the CLASS list
        a.params                             # now has 7 entries -- silently wrong

    Harmless here because every profile in this model is ``use_lstsq=False``. It will bite
    the shapelets escalation path, which mixes lstsq and non-lstsq profiles in one model.
    The fix belongs upstream in gigalens/src; this file only records the hazard.
    """
    if n_source < 1:
        raise ValueError(f"n_source must be >= 1, got {n_source}")
    if n_lens_light not in (1, 2):
        raise ValueError(f"n_lens_light must be 1 or 2, got {n_lens_light}")
    return PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False) for _ in range(n_lens_light)],
        [sersic.SersicEllipse(use_lstsq=False) for _ in range(n_source)],
    )


def _structure(prior):
    """Return the prior's nested key structure as ``[[sorted_keys, ...], ...]``.

    One outer entry per top-level block (lens, lens_light, source), one inner entry per
    sub-block (profile component), each a list of keys in the sorted order
    ``tfb.pack_sequence_as`` flattens them in.
    """
    import jax

    sample = prior.sample(1, seed=jax.random.PRNGKey(0))
    return [[sorted(sub.keys()) for sub in block] for block in sample]


def n_params(prior):
    """Total number of free parameters in a prior."""
    return sum(len(keys) for block in _structure(prior) for keys in block)


def param_names(prior):
    """Flat parameter names in the order ``bij.inverse`` produces them.

    Derived from the prior rather than hardcoded, because the ordering is *not* the
    declaration order of the dicts above. ``tfb.pack_sequence_as`` flattens each
    ``JointDistributionNamed`` by sorted key, so the lens block comes out as
    ``center_x, center_y, e1, e2, gamma, theta_E`` -- alphabetical, not as written.
    Hardcoding the declaration order would mislabel every diagnostic plot without raising
    an error. ``verify.py`` cross-checks this against a perturbation probe.

    Naming rule for repeated components: a sub-block index is appended to the group name
    **only when two sub-blocks in that group share a key**. So the lens block stays
    ``lens.theta_E`` / ``lens.gamma1`` (EPL and Shear have disjoint keys) and a single
    source stays ``source.Ie``, while two source components become ``source0.Ie`` and
    ``source1.Ie``. Without this the two components' names would collide and every
    per-parameter diagnostic would be ambiguous.
    """
    groups = ["lens", "lens_light", "source"]
    structure = _structure(prior)
    if len(structure) > len(groups):
        raise ValueError(
            f"prior has {len(structure)} top-level blocks; expected at most "
            f"{len(groups)} ({groups})"
        )

    names = []
    for group, block in zip(groups, structure):
        seen = [k for keys in block for k in keys]
        collides = len(seen) != len(set(seen))
        for si, keys in enumerate(block):
            prefix = f"{group}{si}" if collides else group
            names.extend(f"{prefix}.{key}" for key in keys)
    return names


def source_component_indices(prior):
    """Flat-index slices for each source component, as a list of ``(start, stop)``.

    The source block's components occupy contiguous runs of the flat parameter vector
    (verified by perturbation probe: with two components they are 15-21 and 22-28), which
    is what lets ``diagnose.py`` swap whole blocks when checking label switching.
    """
    structure = _structure(prior)
    offset = sum(len(keys) for block in structure[:-1] for keys in block)
    spans = []
    for keys in structure[-1]:
        spans.append((offset, offset + len(keys)))
        offset += len(keys)
    return spans
