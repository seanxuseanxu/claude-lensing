"""Diagnostics for a completed fit: convergence, residuals, corner plot.

Everything ``run_fit.py`` writes to ``.npz`` is in **unconstrained** space -- MAP, SVI and
HMC all return the internal parameterisation. Nothing in this file is interpreted
physically until ``prob_model.bij.forward`` has been applied. That transform is the single
most common source of silent nonsense in this codebase, so it happens in exactly one place
here: ``to_constrained``.

Two conventions worth stating, both verified against the running code rather than assumed:

- ``bij.forward`` takes a *list of n_params arrays* (one per parameter, each of length
  ``n``) and returns the nested ``[lens, lens_light, source]`` structure, where the lens
  block holds **two** dicts (EPL and Shear) and the source block holds one dict per
  component. It batches natively -- no per-sample Python loop needed.
- The flat parameter order is alphabetical within each block, not declaration order. It
  comes from ``priors.param_names``; never hardcode it.

HMC samples are shaped ``(num_results, n_hmc, n_params)`` -- draws on axis 0, chains on
axis 1. This differs from upstream gigalens, which returns a 4-D pmap-shaped array. See
``compat.hmc_single_device`` for why.

The model variant (``n_source``, ``lens_light_n_max``, the prior versions, and which lens
this is) is read back from the run's ``metadata.json``, because reconstructing the wrong
prior would transform the samples through the wrong bijector and produce plausible, wrong
numbers. Runs predating any of those keys fall back to the values they were in fact fitted
with, not to today's defaults. That reconstruction lives in ``rebuild_model`` and every
downstream tool calls it rather than reimplementing the fallbacks.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compat  # noqa: F401  -- applies JAX 0.11 shims on import

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from gigalens.jax.model import ForwardProbModel
from gigalens.jax.simulator import LensSimulator

import data
import lenses
import priors

RHAT_THRESHOLD = 1.01


def load_checkpoint(run_dir, name):
    """Return the npz for a stage, or None if that stage never ran."""
    path = Path(run_dir) / f"{name}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def load_run_config(run_dir):
    """Return the run's ``metadata.json`` as a dict, or ``{}`` if absent.

    The model variant lives here. Rebuilding the prior with the wrong ``n_source`` would
    push the samples through a bijector of the wrong dimension -- either a loud shape
    error, or worse, silently valid numbers for a different model.
    """
    path = Path(run_dir) / "metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


class RunModel:
    """The model a run was fitted with, reconstructed from its ``metadata.json``.

    Attributes:
        cfg, image, psf, sim_config, prior, phys_model, prob_model, pnames, spans,
        run_cfg (the raw metadata dict), and the resolved variant fields ``n_source``,
        ``n_lens_light``, ``lens_light_n_max``, ``theta_e_center``, ``theta_e_width``.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __iter__(self):
        """Unpack as ``image, psf, sim_config, prior, phys_model, prob_model, pnames``."""
        return iter(
            (
                self.image,
                self.psf,
                self.sim_config,
                self.prior,
                self.phys_model,
                self.prob_model,
                self.pnames,
            )
        )


def rebuild_model(run_dir, lens=None, theta_e_center=1.5, theta_e_width=0.35, quiet=False):
    """Rebuild the exact model a run was fitted with.

    Every tool that reads a checkpoint must go through here. Reconstructing a prior that
    differs from the one a run was fitted with does not raise -- it pushes the stored
    unconstrained samples through a different bijector and reports different physical
    numbers. So each fallback below is the value the runs *predating that key* actually
    used, never today's default:

    - ``lens``: from ``metadata.json``, else the run directory's grandparent name, which is
      correct for the ``claude/<LENS>/runs/<label>/`` layout. An explicit ``lens`` argument
      overrides both, for a run whose directory was moved.
    - ``n_source`` / ``n_lens_light``: 1, the single-component model.
    - ``lens_light_n_max``: 6.0, the original ceiling -- not the current default of 10.
    - ``source_prior_version``: absence of ``n_source`` identifies a pre-revision run,
      which was fitted against the narrow v1 source prior. Reading such a run through v2
      reports ``source.n_sersic = 0.6146`` where the fit said ``0.5729``. A run that
      records ``n_source`` but no version predates the key and used v2 -- note the literal
      2 below, deliberately *not* ``priors.SOURCE_PRIOR_VERSION``. Every version bump
      would otherwise silently retag those runs with the newest prior.
    - ``lens_light_prior_version``: whatever the component count implies, since runs
      predating ``--n-lens-light`` all used the single component.

    ``theta_e_center`` / ``theta_e_width`` are defaults used only when the run recorded
    none; a recorded value always wins.
    """
    run_dir = Path(run_dir)
    run_cfg = load_run_config(run_dir)

    lens = lens or run_cfg.get("lens") or lenses.from_run_dir(run_dir)
    cfg = lenses.load(lens)

    n_source = int(run_cfg.get("n_source", 1))
    n_lens_light = int(run_cfg.get("n_lens_light", 1))
    lens_light_n_max = float(run_cfg.get("lens_light_n_max", 6.0))
    theta_e_center = float(run_cfg.get("theta_e_center", theta_e_center))
    theta_e_width = float(run_cfg.get("theta_e_width", theta_e_width))
    source_prior_version = int(
        run_cfg.get("source_prior_version", 2 if "n_source" in run_cfg else 1)
    )
    lens_light_prior_version = int(
        run_cfg.get(
            "lens_light_prior_version",
            priors.lens_light_prior_version_for(n_lens_light),
        )
    )

    if not quiet:
        print(f"Lens: {cfg.name}")
        print(f"Model variant: n_source={n_source}, n_lens_light={n_lens_light}, "
              f"lens_light_n_max={lens_light_n_max}, "
              f"source_prior_v{source_prior_version}, "
              f"lens_light_prior_v{lens_light_prior_version}, "
              f"theta_E prior LogNormal(log {theta_e_center}, {theta_e_width})")

    image, psf, sim_config = data.load(cfg)
    prior = priors.build_prior(
        cfg,
        theta_E_center=theta_e_center,
        theta_E_width=theta_e_width,
        n_source=n_source,
        n_lens_light=n_lens_light,
        lens_light_n_max=lens_light_n_max,
        source_prior_version=source_prior_version,
        lens_light_prior_version=lens_light_prior_version,
    )
    phys_model = priors.build_phys_model(
        n_source=n_source, n_lens_light=n_lens_light
    )
    prob_model = ForwardProbModel(
        prior,
        jnp.asarray(image),
        background_rms=cfg.background_rms,
        exp_time=cfg.exp_time,
    )
    return RunModel(
        cfg=cfg,
        run_cfg=run_cfg,
        image=image,
        psf=psf,
        sim_config=sim_config,
        prior=prior,
        phys_model=phys_model,
        prob_model=prob_model,
        pnames=priors.param_names(prior),
        spans=priors.source_component_indices(prior),
        n_source=n_source,
        n_lens_light=n_lens_light,
        lens_light_n_max=lens_light_n_max,
        theta_e_center=theta_e_center,
        theta_e_width=theta_e_width,
        source_prior_version=source_prior_version,
        lens_light_prior_version=lens_light_prior_version,
    )


def to_constrained(prob_model, z_unc):
    """Map unconstrained samples ``(n, n_params)`` to constrained ``(n, n_params)``.

    Column order matches ``priors.param_names(prior)``: blocks in order
    ``lens, shear, lens_light, source``, alphabetical by key within each block. This is
    the same traversal ``tfb.pack_sequence_as`` uses, which is what makes the round trip
    consistent.
    """
    z_unc = np.atleast_2d(np.asarray(z_unc))
    nested = prob_model.bij.forward(list(jnp.asarray(z_unc).T))
    cols = []
    for block in nested:
        for sub in block:
            for key in sorted(sub.keys()):
                cols.append(np.asarray(sub[key]))
    return np.stack(cols, axis=-1)


def simulate_from_unconstrained(prob_model, phys_model, sim_config, z_unc):
    """Render the model image for a single unconstrained parameter vector.

    Returns ``(num_pix, num_pix)``. Note that ``LensSimulator.simulate`` squeezes the
    batch axis at ``bs=1`` -- it returns ``(80, 80)``, not ``(1, 80, 80)`` -- so reshape
    rather than indexing ``[0]``.
    """
    z = jnp.asarray(np.asarray(z_unc).reshape(1, -1))
    simulator = LensSimulator(phys_model, sim_config, bs=1)
    x = prob_model.bij.forward(list(z.T))
    im = np.asarray(simulator.simulate(x))
    return im.reshape(sim_config.num_pix, sim_config.num_pix)


def point_estimate(run_dir):
    """Return ``(z_unconstrained, label)`` for the best available point estimate.

    Preference: HMC posterior median > SVI mean > best MAP sample. The median is taken in
    unconstrained space, which is the right place for it -- the transform is monotonic
    per-coordinate, so the median commutes with it, and taking it here avoids the
    boundary pile-up that a constrained-space mean would suffer.
    """
    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is not None:
        samples = hmc["samples"]  # (num_results, n_hmc, n_params)
        return np.median(samples.reshape(-1, samples.shape[-1]), axis=0), "HMC median"

    svi = load_checkpoint(run_dir, "svi")
    if svi is not None:
        return np.asarray(svi["mean"]), "SVI mean"

    mp = load_checkpoint(run_dir, "map")
    if mp is not None:
        return np.asarray(mp["map_estimate"][int(mp["best_idx"])]), "MAP best"

    raise FileNotFoundError(f"No checkpoints in {run_dir}; expected at least map.npz.")


def residual_figure(out_dir, cfg, image, im_sim, label):
    """Four-panel figure: observed, model, residual, normalised residual.

    The normalised-residual panel is the decisive one. Coherent arc-shaped structure above
    ~3 sigma means the model is wrong regardless of what R-hat says. The error map is the
    same parameter-dependent form ForwardProbModel uses internally:
    ``sqrt(background_rms^2 + im_sim / exp_time)``.
    """
    err_map = np.sqrt(
        cfg.background_rms**2 + np.clip(im_sim, 0, np.inf) / cfg.exp_time
    )
    residual = image - im_sim
    norm_res = residual / err_map

    # Reduced chi-squared over the image, matching ForwardProbModel's second return value.
    chi2 = float(np.mean(norm_res**2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    vmax = float(np.percentile(image, 99.5))

    p0 = axes[0, 0].imshow(image, origin="lower", vmin=0, vmax=vmax)
    axes[0, 0].set_title("Observed (e/s)")
    plt.colorbar(p0, ax=axes[0, 0], fraction=0.046)

    p1 = axes[0, 1].imshow(im_sim, origin="lower", vmin=0, vmax=vmax)
    axes[0, 1].set_title(f"Model -- {label}")
    plt.colorbar(p1, ax=axes[0, 1], fraction=0.046)

    rlim = float(np.percentile(np.abs(residual), 99.5))
    p2 = axes[1, 0].imshow(residual, origin="lower", cmap="RdBu_r", vmin=-rlim, vmax=rlim)
    axes[1, 0].set_title("Residual (obs - model, e/s)")
    plt.colorbar(p2, ax=axes[1, 0], fraction=0.046)

    p3 = axes[1, 1].imshow(norm_res, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
    axes[1, 1].set_title(rf"Normalised residual  ($\chi^2_\nu$ = {chi2:.2f})")
    plt.colorbar(p3, ax=axes[1, 1], fraction=0.046, label=r"$\sigma$")

    fig.tight_layout()
    path = Path(out_dir) / "residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chi2, path


def label_switching_check(con, spans, pnames):
    """Detect label switching between exchangeable source components.

    Two source components with identical priors are formally exchangeable: swapping them
    leaves the posterior density unchanged. If chains wander between the two labellings,
    every per-component R-hat is meaningless -- not because the fit is bad, but because
    the parameterisation is degenerate.

    In practice this should NOT happen here. All chains initialise from ``q_z.sample()``,
    a single unimodal Gaussian fitted around one MAP mode, so they start in one labelling
    and would have to cross a density barrier to swap. That is a prediction about the fit,
    so it is measured rather than assumed -- and the samples are left untouched unless the
    measurement says otherwise.

    Correction is deliberately gated on detection for a second reason. If the data does
    not want a second component, that component's ``Ie`` collapses and its remaining
    parameters go unconstrained. Their R-hat will be poor *correctly* -- the model
    reporting an unidentified parameter. Sorting by ``R_sersic`` in that regime would
    scramble a meaningful diagnostic into noise.

    Args:
        con: constrained samples, ``(n_draws, n_chains, n_params)``.
        spans: ``[(start, stop), ...]`` flat index span per source component, from
            ``priors.source_component_indices``.
        pnames: flat parameter names.

    Returns:
        ``(switched, con_relabelled)``. ``con_relabelled`` is None when no switching was
        detected, in which case the caller must use ``con`` unmodified.
    """
    if len(spans) < 2:
        return False, None

    # Order on R_sersic. It is the parameter the two components are most likely to differ
    # in (compact knot vs extended host), so it is the most reliable discriminator.
    r_idx = []
    for si, (start, stop) in enumerate(spans):
        names = pnames[start:stop]
        match = [start + k for k, n in enumerate(names) if n.endswith(".R_sersic")]
        if not match:
            print("\nNo R_sersic in a source component; skipping label-switching check.")
            return False, None
        r_idx.append(match[0])

    print("\n=== Label switching between source components ===")

    # Only the first two components are compared. With n_source > 2 a full permutation
    # check is needed; say so rather than reporting a partial result as if it were whole.
    if len(spans) > 2:
        print(f"NOTE: {len(spans)} source components; only components 0 and 1 compared.")

    r0 = con[:, :, r_idx[0]]
    r1 = con[:, :, r_idx[1]]
    diff = r0 - r1
    sign = np.sign(diff)
    # A crossing is a sign change along the draw axis, within a single chain.
    crossings = np.sum(np.diff(sign, axis=0) != 0, axis=0)
    n_chains_switched = int(np.sum(crossings > 0))
    total = int(np.sum(crossings))

    # Separation of the two R_sersic posteriors. Heavy overlap means the ordering itself
    # is ill-determined, whatever the crossing count says.
    m0, m1 = float(np.median(r0)), float(np.median(r1))
    s0, s1 = float(np.std(r0)), float(np.std(r1))
    sep = abs(m0 - m1) / np.sqrt(s0**2 + s1**2) if (s0 or s1) else np.inf

    print(f"R_sersic  component 0: {m0:.4f} +/- {s0:.4f}")
    print(f"R_sersic  component 1: {m1:.4f} +/- {s1:.4f}")
    print(f"separation: {sep:.2f} sigma")
    print(f"chains with a label crossing: {n_chains_switched}/{con.shape[1]} "
          f"({total} crossings total)")

    if n_chains_switched == 0:
        print("No label switching detected; R-hat below is reported on the raw chains.")
        if sep < 1.0:
            print("CAUTION: the two components' R_sersic posteriors overlap heavily "
                  "(< 1 sigma). No chain crossed during this run, but the labelling is "
                  "weakly determined and component identities should not be over-read.")
        return False, None

    print("Label switching DETECTED. R-hat is reported twice below: raw, and after "
          "sorting components by R_sersic per draw. Component-wise numbers describe "
          "'the more compact one' and 'the more extended one', not two labelled objects.")

    # Relabel: per draw and chain, order the component blocks by R_sersic ascending.
    con_re = con.copy()
    swap = r0 > r1  # (n_draws, n_chains)
    (a0, a1), (b0, b1) = spans[0], spans[1]
    block_a = con[:, :, a0:a1].copy()
    block_b = con[:, :, b0:b1].copy()
    con_re[:, :, a0:a1] = np.where(swap[:, :, None], block_b, block_a)
    con_re[:, :, b0:b1] = np.where(swap[:, :, None], block_a, block_b)
    return True, con_re


def boundary_pinning_check(prior, con_flat, pnames):
    """Warn when a posterior sits against a Uniform prior's bound.

    HMC mixes badly along a boundary, and a pinned parameter means the prior -- not the
    data -- is setting the value. The first production run had two such pinnings
    (``lens_light.n_sersic`` at its ceiling of 6, ``source.n_sersic`` at its floor of 0.5)
    and only the first was noticed by eye. This makes the check automatic.

    Only ``Uniform`` priors are checked. A parameter in the tail of an unbounded prior is
    a different phenomenon and not necessarily pathological.
    """
    import jax
    from tensorflow_probability.substrates.jax import distributions as tfd

    sample = prior.sample(1, seed=jax.random.PRNGKey(0))
    # sample_distributions returns the nested distribution objects, but the inner blocks
    # are JointDistribution* instances that are not themselves iterable -- their children
    # hang off `.model` (a list for Sequential, a dict for Named).
    dists = prior.sample_distributions(1, seed=jax.random.PRNGKey(0))[0]

    # Walk the nested structure in the same traversal order as priors.param_names.
    bounds, i = {}, 0
    for block, dblock in zip(sample, dists):
        for sub, dsub in zip(block, dblock.model):
            dmodel = dsub.model
            for key in sorted(sub.keys()):
                d = dmodel.get(key)
                if isinstance(d, tfd.Uniform):
                    bounds[pnames[i]] = (float(d.low), float(d.high))
                i += 1

    if not bounds:
        return

    print("\n=== Boundary pinning (Uniform priors) ===")
    med = np.median(con_flat, axis=0)
    any_pinned = False
    for name, (lo, hi) in bounds.items():
        idx = pnames.index(name)
        v = med[idx]
        frac = (v - lo) / (hi - lo)
        tag = ""
        if frac < 0.02:
            tag, any_pinned = "  <-- PINNED at lower bound", True
        elif frac > 0.98:
            tag, any_pinned = "  <-- PINNED at upper bound", True
        print(f"{name:<24} {v:>9.4f}  in [{lo:g}, {hi:g}]  ({100*frac:5.1f}% of range){tag}")

    if any_pinned:
        print("\nA pinned parameter is set by its prior, not by the data, and HMC cannot "
              "mix along the boundary -- expect poor R-hat there and treat the value as a "
              "bound rather than a measurement.")


def convergence_table(run_dir, prob_model, pnames, prior=None, spans=None):
    """Print posterior summary with R-hat and ESS. Returns the max R-hat, or None.

    R-hat and ESS are computed in constrained space -- that is the space the numbers get
    quoted in, so it is the space the diagnostic has to apply to.

    With more than one source component this also runs the label-switching detector and
    the boundary-pinning check. The R-hat reported is the raw one unless switching was
    actually detected, in which case both raw and relabelled are shown.
    """
    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is None:
        print("\nNo HMC samples; convergence diagnostics unavailable.")
        return None

    samples = hmc["samples"]  # (num_results, n_hmc, n_params)
    n_draws, n_chains, n_params = samples.shape

    con = to_constrained(prob_model, samples.reshape(-1, n_params))
    con = con.reshape(n_draws, n_chains, n_params)

    switched, con_re = (False, None)
    if spans is not None:
        switched, con_re = label_switching_check(con, spans, pnames)

    rhat = np.asarray(tfp.mcmc.potential_scale_reduction(con, independent_chain_ndims=1))
    ess = np.asarray(tfp.mcmc.effective_sample_size(con, cross_chain_dims=1))

    if switched and con_re is not None:
        rhat_re = np.asarray(
            tfp.mcmc.potential_scale_reduction(con_re, independent_chain_ndims=1)
        )
        print(f"\nR-hat worst  raw: {np.nanmax(rhat):.3f}   "
              f"after relabelling: {np.nanmax(rhat_re):.3f}")
        print("If relabelling fixes it, the chains were switching labels and the fit is "
              "healthy. If both are bad, something other than exchangeability is wrong.")

    flat = con.reshape(-1, n_params)
    if prior is not None:
        boundary_pinning_check(prior, flat, pnames)

    lo, med, hi = np.percentile(flat, [16, 50, 84], axis=0)

    print(f"\n=== Posterior ({n_chains} chains x {n_draws} draws) ===")
    print(f"{'parameter':<22}{'median':>10}{'-1sig':>10}{'+1sig':>10}{'Rhat':>8}{'ESS':>9}")
    for i, name in enumerate(pnames):
        flag = "  <-- Rhat" if rhat[i] > RHAT_THRESHOLD else ""
        print(
            f"{name:<22}{med[i]:>10.4f}{med[i]-lo[i]:>10.4f}{hi[i]-med[i]:>10.4f}"
            f"{rhat[i]:>8.3f}{ess[i]:>9.0f}{flag}"
        )

    worst = float(np.nanmax(rhat))
    n_bad = int(np.sum(rhat > RHAT_THRESHOLD))
    if n_bad:
        print(f"\n{n_bad}/{n_params} parameters have Rhat > {RHAT_THRESHOLD} "
              f"(worst {worst:.3f}). The chains have not converged; do not quote these "
              f"values as measurements.")
    else:
        print(f"\nAll {n_params} parameters have Rhat <= {RHAT_THRESHOLD} "
              f"(worst {worst:.3f}); min ESS {np.nanmin(ess):.0f}.")
    return worst


def theta_E_check(run_dir, cfg, prob_model, pnames, theta_E_center, theta_E_width):
    """Compare the theta_E posterior against the config's measured clump radii.

    If every clump is a counter-image of one source, the Einstein radius should land inside
    their radial span. A posterior pinned against either prior edge means the radial scales
    are pulling the model apart, and that is the thing to report -- not the number.
    """
    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is None:
        return
    samples = hmc["samples"]
    con = to_constrained(prob_model, samples.reshape(-1, samples.shape[-1]))
    idx = pnames.index("lens.theta_E")
    lo, med, hi = np.percentile(con[:, idx], [16, 50, 84])

    mu, sigma = np.log(theta_E_center), theta_E_width
    p01, p99 = np.exp(mu - 2.326 * sigma), np.exp(mu + 2.326 * sigma)

    print("\n=== theta_E vs. measured clump radii ===")
    print(f"posterior   {med:.3f} -{med-lo:.3f} +{hi-med:.3f} arcsec")
    print(f"prior 1-99% [{p01:.3f}, {p99:.3f}]")
    for name, r in sorted(cfg.clump_radii.items(), key=lambda kv: kv[1]):
        print(f"  {name:<10} {r:.2f}")

    if med - p01 < 0.1 * (p99 - p01):
        print("WARNING: posterior pinned near the LOWER prior edge. The model is likely "
              "fitting only the inner clumps and ignoring the E blob.")
    elif p99 - med < 0.1 * (p99 - p01):
        print("WARNING: posterior pinned near the UPPER prior edge. The prior may be "
              "truncating a solution that wants a larger Einstein radius.")


def svi_health(run_dir):
    """Report whether the SVI stage converged, before any Rhat is worth reading.

    This is the check that would have caught the ``twosource`` run immediately. Its ELBO
    swung by 27,000 over the last few hundred steps and left a covariance conditioned at
    8.8e+11 -- a thousand times worse than the worst healthy run. ``hmc_single_device``
    inverted that in float32 without complaint, producing a matrix that is not an inverse
    (``max|inv(C)C - I| = 55``) and whose spectrum carries five negative eigenvalues. Every
    one of that run's 29 parameters then failed Rhat, which was read as slow mixing and was
    not: the mass matrix was effectively indefinite.

    Note that the covariance itself is positive definite, so a PD test does not catch this.
    Conditioning is the discriminant. See ``compat.check_svi_covariance``.

    Read this line before the convergence table. A rejected covariance invalidates the HMC
    output entirely -- there is nothing to diagnose downstream of it. Unlike the guard in
    the fitting path this only reports, since by the time a run is being diagnosed the
    samples already exist and the useful thing is to say why they cannot be trusted.
    """
    svi = load_checkpoint(run_dir, "svi")
    if svi is None:
        return
    try:
        min_eig, cond = compat.check_svi_covariance(scale_tril=svi["scale_tril"])
        failed = None
    except RuntimeError as exc:
        failed = str(exc)
        # Recompute unguarded so the numbers can still be printed alongside the failure.
        s = np.linalg.svd(np.asarray(svi["scale_tril"], dtype=np.float64), compute_uv=False)
        min_eig = float(s.min() ** 2)
        cond = float(s.max() ** 2 / min_eig) if min_eig > 0 else np.inf
    loss = np.asarray(svi["loss_hist"], dtype=np.float64)
    tail = loss[-300:]

    print("\n=== SVI health ===")
    print(f"covariance min eigenvalue {min_eig:.3e}, condition {cond:.3e} "
          f"(healthy runs 1e6-6e8, limit {compat.COND_MAX:.0e})")
    print(f"ELBO loss final {loss[-1]:.1f}, sd over last {len(tail)} steps "
          f"{np.std(tail):.1f}")
    if failed is not None:
        print(f"FAILED: {failed}")
        print("HMC was preconditioned on this covariance, so every Rhat below is "
              "meaningless. Do not interpret this run; re-run SVI.")
    elif np.std(tail) > 100:
        print("CAUTION: the ELBO is still moving at the end of the run (healthy 22-param "
              "runs settle to sd ~10). The Gaussian ansatz may not have converged.")
    else:
        print("Converged: well-conditioned covariance, settled ELBO.")


def svi_loss_plot(run_dir, out_dir):
    svi = load_checkpoint(run_dir, "svi")
    if svi is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.asarray(svi["loss_hist"]))
    ax.set_xlabel("step")
    ax.set_ylabel("ELBO loss")
    ax.set_title("SVI convergence")
    fig.tight_layout()
    path = Path(out_dir) / "svi_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def corner_plot(run_dir, out_dir, prob_model, pnames, subset=None):
    """Corner plot of the constrained posterior.

    Defaults to the mass parameters, which exist under every model variant. All 22 (or 29)
    in one figure is unreadable at print size.
    """
    try:
        import corner
    except ImportError:
        print("corner not installed; skipping corner plot.")
        return None

    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is None:
        return None

    samples = hmc["samples"]
    con = to_constrained(prob_model, samples.reshape(-1, samples.shape[-1]))

    if subset is None:
        subset = [
            "lens.theta_E", "lens.gamma", "lens.e1", "lens.e2",
            "lens.gamma1", "lens.gamma2",
        ]
    idx = [pnames.index(n) for n in subset if n in pnames]

    fig = corner.corner(
        con[:, idx],
        labels=[pnames[i] for i in idx],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
    )
    path = Path(out_dir) / "corner.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="Diagnostics for a completed fit")
    ap.add_argument("run_dir", help="Directory containing map/svi/hmc .npz checkpoints")
    ap.add_argument("--out", default=None, help="Output directory (default: run_dir)")
    lenses.add_argument(ap, required=False)
    ap.add_argument("--theta-e-center", type=float, default=1.5)
    ap.add_argument("--theta-e-width", type=float, default=0.35)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct the *same* model the run was fitted with -- see rebuild_model for why
    # each fallback is the value old runs used rather than today's default.
    rm = rebuild_model(
        run_dir,
        lens=args.lens,
        theta_e_center=args.theta_e_center,
        theta_e_width=args.theta_e_width,
    )
    cfg, image, sim_config = rm.cfg, rm.image, rm.sim_config
    prior, phys_model, prob_model, pnames = (
        rm.prior, rm.phys_model, rm.prob_model, rm.pnames
    )

    z_unc, label = point_estimate(run_dir)
    print(f"Point estimate: {label}")

    im_sim = simulate_from_unconstrained(prob_model, phys_model, sim_config, z_unc)
    chi2, res_path = residual_figure(out_dir, cfg, image, im_sim, label)
    print(f"Reduced chi-squared at point estimate: {chi2:.3f}")
    print(f"Residual figure: {res_path}")

    # Before Rhat: a non-PD SVI covariance invalidates the HMC output outright, so this
    # line has to be read first.
    svi_health(run_dir)

    worst_rhat = convergence_table(
        run_dir, prob_model, pnames, prior=prior, spans=rm.spans
    )
    theta_E_check(
        run_dir, cfg, prob_model, pnames, rm.theta_e_center, rm.theta_e_width
    )

    p = svi_loss_plot(run_dir, out_dir)
    if p:
        print(f"SVI loss: {p}")
    p = corner_plot(run_dir, out_dir, prob_model, pnames)
    if p:
        print(f"Corner plot: {p}")

    if worst_rhat is not None and worst_rhat > RHAT_THRESHOLD:
        print("\nReminder: quote R-hat alongside any parameter value. A theta_E without a "
              "convergence diagnostic is not a measurement.")


if __name__ == "__main__":
    main()
