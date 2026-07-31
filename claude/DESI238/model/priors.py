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


def build_prior(
    theta_E_center=1.5,
    theta_E_width=0.35,
    shear_width=0.1,
    n_source=1,
    lens_light_n_max=10.0,
    source_prior_version=2,
):
    """Build the nested prior over ``[lens, lens_light, source_light]``.

    Returns a ``tfd.JointDistributionSequential`` of ``JointDistributionNamed`` dicts,
    matching the structure ``ProbabilisticModel.pack_bij`` expects. 22 free parameters at
    ``n_source=1``, 29 at ``n_source=2``.

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
        lens_light_n_max: upper bound of the lens-light ``n_sersic`` Uniform. The first
            production run pinned at the old ceiling of 6; the default 10 frees it while
            staying in the range where gigalens' b_n approximation is exact to 0.01%.
        source_prior_version: 2 (default) for the wide priors described above; 1 for the
            narrow originals used by seed0 / seed1 / inner3. **Reading an old run's
            checkpoints requires version 1** -- the stored samples are unconstrained, and
            a different Uniform support maps the same value to a different number without
            raising. diagnose.py selects this from metadata.json.
    """
    if n_source < 1:
        raise ValueError(f"n_source must be >= 1, got {n_source}")
    if source_prior_version not in (1, 2):
        raise ValueError(
            f"source_prior_version must be 1 or 2, got {source_prior_version}"
        )
    if source_prior_version == 1 and n_source != 1:
        raise ValueError("source_prior_version=1 only ever described a single component")
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

    # Light second moments of the deflector give e1 = -0.017, e2 = +0.036 -- nearly round,
    # |e| = 0.04. The truncated prior below is centred on round and comfortably covers it.
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
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
        ]
    )

    # One component per entry, each a fresh object. Identical priors at v2: the data
    # chooses the morphology, not the prior.
    make = _source_component_v1 if source_prior_version == 1 else _source_component
    source_light_prior = tfd.JointDistributionSequential(
        [make() for _ in range(n_source)]
    )

    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )


def build_phys_model(n_source=1):
    """EPL + external shear deflector, one Sersic lens light, ``n_source`` Sersic sources.

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
    return PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
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
