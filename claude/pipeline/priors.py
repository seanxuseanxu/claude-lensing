"""Priors and physical model for DESI238.

The demo notebook's priors are tuned to a synthetic image with a completely different flux
scale (Ie ~ 500, sky 0.2, peak of order 100). DESI238 peaks at 7.2 e/s. Copying the demo
priors puts the prior mass two orders of magnitude below the data, so every Ie here is set
from a measured render, not copied.

Four clumps were measured in the azimuthal-median-subtracted cutout, and the repository
owner's decision is that all four are counter-images:

    clump    r (")   PA (deg)   residual flux (e/s)
    SW arc   1.06     224        18.8
    E blob   2.11      88        24.0
    W clump  1.64     278        12.5
    N arc    1.25     339         8.7

Requiring one source to produce images at both 1.06" and 2.11" is a strong demand on a
single elliptical deflector, and it is what widens theta_E, e1/e2 and the shear priors
below relative to the demo. See MODELING_PLAN.md, "Consequences of the four-image choice".

Revisions after the first production run (runs/seed0, RESULTS.md)
-----------------------------------------------------------------

The first fit pinned *two* parameters against prior boundaries, not one:

    lens_light.n_sersic   5.9896   against Uniform(2, 6)   -- upper bound
    source.n_sersic       0.5729   against Uniform(0.5, 4) -- lower bound

Only the first is recorded in RESULTS.md. Both matter, because HMC mixes badly along a
boundary and that is what drove 18/22 parameters past Rhat > 1.01.

- ``lens_light.n_sersic`` is now ``Uniform(2, lens_light_n_max)`` with a default ceiling
  of 10. If the posterior re-pins at 10, no bound will fix it and the deflector light
  needs a different profile.

- ``source.n_sersic`` keeps its 0.5 floor deliberately. gigalens hardcodes the Sersic
  constant as ``bn = 1.9992 * n - 0.3271`` (jax/profiles/light/sersic.py:15), which is an
  asymptotic fit valid at large n. Against the exact ``gammaincinv(2n, 0.5)``:

      n      0.3      0.5      1.0      2.0      6.0     10.0
      err  -13.6%    -3.0%   -0.37%   -0.02%   +0.00%   -0.01%

  Dropping the floor below 0.5 would trade a prior artefact for a profile-accuracy
  artefact. The pinning is documented instead of papered over; a second source component
  may relieve the pressure on n by itself.

- The source ``R_sersic`` prior was ``LogNormal(log 0.25, 0.20)``, spanning only
  0.157-0.398" at 1-99%. That is too tight to represent an extended host at all, so it
  could not have found one. It is widened below.

Multiple source components
--------------------------

``n_source=2`` gives the source two Sersic components with **identical** priors, so the
data -- not the prior -- decides whether it wants two separated compact clumps or a
compact knot inside an extended host. The wide ``Ie`` prior lets a component switch itself
off by going faint, which means the 2-component model *nests* the 1-component model and a
chi-squared comparison against the 1-component fit is fair rather than a free-parameter
giveaway.

Identical priors make the two components formally exchangeable: relabelling them leaves
the posterior unchanged. In practice all HMC chains start from ``q_z.sample()``, a single
unimodal Gaussian around one MAP mode, so they should stay in one labelling. diagnose.py
measures this rather than assuming it (see ``label_switching_check``).

Multiple lens-light components
------------------------------

``n_lens_light=2`` splits the deflector into bulge and disk. The motivation is measured,
not aesthetic: the single-Sersic residual about the lens-light centre runs -2.7 sigma at
r < 1 px, +0.4 to +0.8 sigma over r = 1.5-5.5 px, and -0.2 to -0.8 sigma over r = 6.5-11.5
px. The amplitude is only 1-3%, but it is coherent over hundreds of pixels, and it is what
drives a core chi2 of 7.10 against a global 1.302. A single Sersic is a one-shape family
and cannot produce that S-curve.

Unlike the source components, the two lens-light components get **different** priors. Bulge
and disk are distinguished by the physics, so identical priors would introduce a
label-switching mode for no benefit. The split is centred on a 1D indicator fit to the
inner-arcsecond data profile (compact Re = 0.216", n = 0.72; extended Re = 1.150",
n = 1.13), with sigmas of 0.5-0.8 dex so the data can overturn it -- the two ``R_sersic``
priors overlap at about 1 sigma, which separates the components without forbidding them
from converging on one shape.
"""

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens.model import PhysicalModel
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

# Radial span of the four clumps, in arcsec. Used by diagnose.py to sanity-check the
# theta_E posterior against the data that motivated the prior.
CLUMP_RADII = {"SW arc": 1.06, "N arc": 1.25, "W clump": 1.64, "E blob": 2.11}

# Parameter count for the DEFAULT (single-source) model. Kept as a module constant for
# backwards compatibility with the seed0/seed1/inner3 runs, but callers that support
# n_source > 1 must use n_params(prior) instead -- two source components give 29.
N_PARAMS = 22

# Version of the source-component prior that build_prior produces by default. Bump this
# whenever the source prior's *support* changes (a Uniform's bounds, a TruncatedNormal's
# limits), because that changes how stored unconstrained samples map back to physical
# values. run_fit.py records it in metadata.json; diagnose.py reads it back.
SOURCE_PRIOR_VERSION = 2

# Same discipline for the lens-light prior. v1 is the single-component prior that
# seed0 / seed1 / inner3 / nsersic10 / twosource were fitted with; v2 is the bulge+disk
# split, whose bulge ``n_sersic`` support is Uniform(1, 8) rather than
# Uniform(2, lens_light_n_max). That is a *support* change, so it must be versioned.
#
# The version is a function of the component count, not a free choice: v1 describes one
# component and v2 describes two, so ``lens_light_prior_version_for(n_lens_light)``
# resolves it and a single-component run keeps the exact prior seed0 was fitted with. Use
# that helper rather than a bare constant.
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

# Ie scale.
#
# LensSimulator multiplies its output by conversion_factor = det(transform_pix2angle),
# which defaults to delta_pix^2 = 0.065^2 = 0.004225. A unit-Ie lens+source Sersic pair
# renders to a peak of 0.083 e/s -- so Ie must be of order 100, not order 1, to reach the
# data's 7.2 e/s.
#
# Measured: with LENS_IE=80 and SOURCE_IE=20, the median peak over 64 prior draws is
# 5.26 e/s against a data peak of 7.18. With the values in the original plan (2.0 and
# 0.5) it was 0.14 -- 50x too faint.
LENS_IE = 80.0
SOURCE_IE = 20.0


def _source_component_v1():
    """The ORIGINAL single-source prior, as fitted in runs seed0 / seed1 / inner3.

    Kept verbatim so those runs stay readable. Their checkpoints hold *unconstrained*
    samples, and the constrained value depends on the bijector, which for a Uniform is
    ``low + (high - low) * sigmoid(z)``. Reading seed0 through the v2 prior below silently
    reports ``source.n_sersic = 0.6146`` where the fit actually said ``0.5729`` -- no
    error, just a wrong number, because ``Uniform(0.5, 4)`` and ``Uniform(0.5, 6)`` map the
    same z to different values. Do not delete this function to tidy up.
    """
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(0.25), 0.2),
            n_sersic=tfd.Uniform(0.5, 4.0),
            e1=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.15, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.3),
            center_y=tfd.Normal(0.0, 0.3),
            Ie=tfd.LogNormal(jnp.log(SOURCE_IE), 0.6),
        )
    )


def _source_component():
    """One Sersic source component's prior (v2, current).

    Deliberately wide and identical across components, so the fit is free to choose its
    own morphology. At 1-99%, ``R_sersic`` spans 0.10-1.62" -- wide enough for a compact
    knot at one end and an extended host at the other -- against the original prior's
    0.157-0.398", which could only ever have been a knot.

    ``Ie`` spans 1.2-326 at 1-99%. The low end matters structurally: a component that the
    data does not want can go faint and effectively switch itself off, so two components
    nest one. If that happens, that component's remaining six parameters become
    unconstrained and their Rhat will be poor *correctly* -- the model reporting an
    unidentified parameter, not a sampler failure.

    ``n_sersic`` keeps the 0.5 floor for the b_n reason in the module docstring.
    """
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(0.4), 0.6),
            n_sersic=tfd.Uniform(0.5, 6.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.5),
            center_y=tfd.Normal(0.0, 0.5),
            Ie=tfd.LogNormal(jnp.log(SOURCE_IE), 1.2),
        )
    )


def _lens_light_component_v1(lens_light_n_max=10.0):
    """The ORIGINAL single-component deflector prior (v1), as fitted in every run to date.

    Kept verbatim for the same reason ``_source_component_v1`` is: checkpoints hold
    unconstrained samples, and ``n_sersic``'s Uniform support determines what physical
    value a stored z maps to. Reading an old run through the v2 bulge prior below --
    ``Uniform(1, 8)`` rather than ``Uniform(2, lens_light_n_max)`` -- would report a
    different n_sersic with no error raised.

    Light second moments of the deflector give e1 = -0.017, e2 = +0.036 -- nearly round,
    |e| = 0.04. The truncated prior below is centred on round and comfortably covers it.
    """
    return tfd.JointDistributionNamed(
        dict(
            R_sersic=tfd.LogNormal(jnp.log(0.8), 0.3),
            n_sersic=tfd.Uniform(2.0, lens_light_n_max),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
            center_x=tfd.Normal(0.0, 0.05),
            center_y=tfd.Normal(0.0, 0.05),
            Ie=tfd.LogNormal(jnp.log(LENS_IE), 0.5),
        )
    )


def _lens_light_component(which):
    """One deflector light component's prior (v2): ``which`` is "bulge" or "disk".

    Deliberately NOT identical across components, which is the opposite of the choice made
    for the source. The source components are exchangeable on purpose -- the data decides
    the morphology there. Here the two components mean distinct physical things, and giving
    them identical priors would only add a label-switching mode to the posterior.

    The centres are the 1D indicator fit's recovered geometry rescaled to this model's Ie
    convention (``LENS_IE`` = 80 for the total light; the indicator fit's Ie = 1.965 and
    0.240 are in data units and carry the 1.965 : 0.240 ratio, which is what is preserved).
    The sigmas are wide -- 0.5 dex on R_sersic, 0.8 on Ie -- so the split is a starting
    point, not a constraint. The two R_sersic priors overlap at about 1 sigma.

    ``n_sersic`` bounds differ by component and both stay inside the range where gigalens'
    ``bn = 1.9992 n - 0.3271`` approximation is accurate: the error is -0.37% at n = 1 and
    below 0.02% for n >= 2 (see the module docstring's table).
    """
    if which == "bulge":
        return tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.5),
                n_sersic=tfd.Uniform(1.0, 8.0),
                e1=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
                center_x=tfd.Normal(0.0, 0.05),
                center_y=tfd.Normal(0.0, 0.05),
                Ie=tfd.LogNormal(jnp.log(LENS_IE * 2.0), 0.8),
            )
        )
    if which == "disk":
        return tfd.JointDistributionNamed(
            dict(
                R_sersic=tfd.LogNormal(jnp.log(1.2), 0.5),
                n_sersic=tfd.Uniform(0.5, 2.5),
                e1=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0.0, 0.1, -0.3, 0.3),
                center_x=tfd.Normal(0.0, 0.05),
                center_y=tfd.Normal(0.0, 0.05),
                Ie=tfd.LogNormal(jnp.log(LENS_IE * 0.25), 0.8),
            )
        )
    raise ValueError(f"unknown lens-light component {which!r}; expected bulge or disk")


def build_prior(
    theta_E_center=1.5,
    theta_E_width=0.35,
    shear_width=0.1,
    n_source=1,
    n_lens_light=1,
    lens_light_n_max=10.0,
    source_prior_version=2,
    lens_light_prior_version=None,
):
    """Build the nested prior over ``[lens, lens_light, source_light]``.

    Returns a ``tfd.JointDistributionSequential`` of ``JointDistributionNamed`` dicts,
    matching the structure ``ProbabilisticModel.pack_bij`` expects. 22 free parameters at
    ``n_source=1, n_lens_light=1``; 29 with a second source; 36 with both.

    Args:
        theta_E_center: median of the Einstein-radius LogNormal, in arcsec. The default
            1.5 sits between the inner clumps and the E blob, with ``theta_E_width=0.35``
            putting real prior mass across 1.0-2.2" as the four-image reading requires.
            For the falsification run that fits the inner three clumps only, use
            ``theta_E_center=1.3, theta_E_width=0.2``.
        theta_E_width: sigma of the Einstein-radius LogNormal, in log space.
        shear_width: sigma of the external-shear Normals. Widened from the demo's 0.05:
            producing a four-image configuration spanning 1.06-2.11" in radius needs
            substantial angular structure, and a tight shear prior would fight the data.
        n_source: number of Sersic source components. Must match ``build_phys_model``'s
            ``n_source``, or ``LensSimulator``'s ``zip`` over profiles and parameters
            silently drops the extra one. Components get identical priors.
        n_lens_light: number of Sersic deflector-light components, 1 or 2. Must match
            ``build_phys_model``'s ``n_lens_light`` for the same zip reason. At 2 the
            components get *different* (bulge / disk) priors -- see the module docstring.
        lens_light_n_max: upper bound of the lens-light ``n_sersic`` Uniform. Applies to
            the v1 single-component prior only; the v2 bulge/disk priors carry their own
            bounds. The first production run pinned at the old ceiling of 6; the default 10
            frees it while staying in the range where gigalens' b_n approximation is exact
            to 0.01%.
        source_prior_version: 2 (default) for the wide priors described above; 1 for the
            narrow originals used by seed0 / seed1 / inner3. **Reading an old run's
            checkpoints requires version 1** -- the stored samples are unconstrained, and
            a different Uniform support maps the same value to a different number without
            raising. diagnose.py selects this from metadata.json.
        lens_light_prior_version: 1 for the single-component prior every run to date used,
            2 for the bulge/disk split. Defaults to whatever ``n_lens_light`` implies,
            which is the only consistent pairing -- the versions differ in component count,
            so they are not independently selectable. Same checkpoint hazard as
            ``source_prior_version``: the bulge ``n_sersic`` support differs from v1's.
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
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(theta_E_center), theta_E_width),
                    gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0),
                    # Widened from the demo's 0.1: a 2:1 radial spread between images
                    # generally needs either high ellipticity or strong shear.
                    e1=tfd.Normal(0.0, 0.15),
                    e2=tfd.Normal(0.0, 0.15),
                    # Arcsec relative to the crop centre, which is the deflector to
                    # within 0.1 px. 0.05" is ~0.8 px.
                    center_x=tfd.Normal(0.0, 0.05),
                    center_y=tfd.Normal(0.0, 0.05),
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
        lens_light_parts = [_lens_light_component_v1(lens_light_n_max)]
    else:
        lens_light_parts = [_lens_light_component(w) for w in ("bulge", "disk")]
    lens_light_prior = tfd.JointDistributionSequential(lens_light_parts)

    # One component per entry, each a fresh object. Identical priors at v2: the data
    # chooses the morphology, not the prior.
    make = _source_component_v1 if source_prior_version == 1 else _source_component
    source_light_prior = tfd.JointDistributionSequential(
        [make() for _ in range(n_source)]
    )

    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )


def build_phys_model(n_source=1, n_lens_light=1):
    """EPL + shear deflector, ``n_lens_light`` Sersic lens lights, ``n_source`` sources.

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
    the shapelets escalation path (MODELING_PLAN.md step 3b), which mixes lstsq and
    non-lstsq profiles in one model. The fix belongs upstream in gigalens/src; this file
    only records the hazard.
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
    """Total number of free parameters in a prior.

    Use this rather than the ``N_PARAMS`` constant anywhere ``n_source`` may vary: the
    constant describes only the default single-source model.
    """
    return sum(len(keys) for block in _structure(prior) for keys in block)


def param_names(prior):
    """Flat parameter names in the order ``bij.inverse`` produces them.

    Derived from the prior rather than hardcoded, because the ordering is *not* the
    declaration order of the dicts above. ``tfb.pack_sequence_as`` flattens each
    ``JointDistributionNamed`` by sorted key, so the lens block comes out as
    ``center_x, center_y, e1, e2, gamma, theta_E`` -- alphabetical, not as written.
    Hardcoding the declaration order would mislabel every diagnostic plot without
    raising an error. ``verify.py`` cross-checks this against a perturbation probe.

    Naming rule for repeated components: a sub-block index is appended to the group name
    **only when two sub-blocks in that group share a key**. So the lens block stays
    ``lens.theta_E`` / ``lens.gamma1`` (EPL and Shear have disjoint keys), a single source
    stays ``source.Ie`` -- preserving every name used by the seed0/seed1/inner3 runs --
    while two source components become ``source0.Ie`` and ``source1.Ie``. Without this,
    the two components' names would collide and every per-parameter diagnostic would be
    ambiguous.
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
    is what lets diagnose.py swap whole blocks when checking label switching.
    """
    structure = _structure(prior)
    offset = sum(len(keys) for block in structure[:-1] for keys in block)
    spans = []
    for keys in structure[-1]:
        spans.append((offset, offset + len(keys)))
        offset += len(keys)
    return spans
