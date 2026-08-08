"""Diagnostics for a completed fit: convergence, residuals, corner plot.

Everything ``run_fit.py`` writes to ``.npz`` is in **unconstrained** space -- MAP, SVI and
HMC all work in the flat ``z`` the samplers move in. Nothing in this file is interpreted
physically until the model's bijector has been applied. That transform is the single most
common source of silent nonsense in this codebase, so it happens in exactly one place
here: ``to_constrained``.

Three conventions worth stating, all verified against the running code rather than assumed:

- ``model.bijector.forward`` takes a **flat array** of shape ``(..., num_free_params)``.
  The retired ``list(z.T)`` list-of-columns convention now raises ``TypeError`` outright,
  which is the good outcome -- it used to be accepted.
- The flat column order is ``model.z_param_names``: path-keyed and sorted, so lens *light*
  precedes lens *mass* within the lens plane (``light`` < ``mass``). ``model.z_labels`` is
  the parallel display list. Never reconstruct either from declaration order.
- HMC samples are shaped ``(n_chains, n_steps, n_params)`` -- **chains first**, the ArviZ
  layout that ``diagnose_convergence`` requires, and the opposite of TFP's own.

The model variant (``n_source``, ``lens_light_n_max``, the prior versions, and which lens
this is) is read back from the run's ``metadata.json``, because reconstructing the wrong
prior would transform the samples through the wrong bijector and produce plausible, wrong
numbers. That reconstruction lives in ``rebuild_model`` and every downstream tool calls it
rather than reimplementing the fallbacks.

Runs predating the scene-API port are refused, not guessed at: the port changed both the
z-column order and the noise convention, so their stored samples mean something else here.
Read them at the pre-port commit, or re-fit.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

from gigalens.jax.analysis import diagnose_convergence
from gigalens.jax.scene_prob_model import ImageData, ProbModel

import data
import lenses
import priors
import svi_health

RHAT_THRESHOLD = 1.01


def load_checkpoint(run_dir, name):
    """Return the npz for a stage, or None if that stage never ran."""
    path = Path(run_dir) / f"{name}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def load_run_config(run_dir):
    """Return the run's ``metadata.json`` as a dict, or ``{}`` if absent.

    The model variant lives here. Rebuilding the model with the wrong ``n_source`` would
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
        cfg, image, psf, sim_config, model (``LensModel``), prob (``ProbModel``),
        pnames (``model.z_param_names``), labels (``model.z_labels``), spans, run_cfg (the
        raw metadata dict), and the resolved variant fields ``n_source``, ``n_lens_light``,
        ``lens_light_n_max``, ``theta_e_center``, ``theta_e_width``.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __iter__(self):
        """Unpack as ``image, psf, sim_config, model, prob, pnames``."""
        return iter(
            (self.image, self.psf, self.sim_config, self.model, self.prob, self.pnames)
        )

    @property
    def simulator(self):
        """The single imaging dataset's ``SceneSimulator``.

        Not bound to a batch size: the batch axis comes from the params and broadcasts
        onto a singleton grid axis, so one instance renders one sample or a thousand.
        """
        return self.prob.simulators[0]


def rebuild_model(run_dir, lens=None, theta_e_center=1.5, theta_e_width=0.35, quiet=False):
    """Rebuild the exact model a run was fitted with.

    Every tool that reads a checkpoint must go through here. Reconstructing a model that
    differs from the one a run was fitted with does not raise -- it pushes the stored
    unconstrained samples through a different bijector and reports different physical
    numbers. So:

    - ``"api": "scene"`` is **required**. A run without it predates the port; its
      z-columns are in the old ``[lens, lens_light, source]`` block order and its
      likelihood used a parameter-dependent noise map. Decoding it here would be silent
      and wrong, so it is refused.
    - ``lens``: from ``metadata.json``, else the run directory's grandparent name, which is
      correct for the ``claude/<LENS>/runs/<label>/`` layout. An explicit ``lens`` argument
      overrides both, for a run whose directory was moved.
    - ``n_source`` / ``n_lens_light`` / the prior versions: read from the metadata every
      scene-API run writes. The defaults below are the single-component model, which is
      also what ``run_fit.py`` defaults to.

    ``theta_e_center`` / ``theta_e_width`` are defaults used only when the run recorded
    none; a recorded value always wins.
    """
    run_dir = Path(run_dir)
    run_cfg = load_run_config(run_dir)

    if run_cfg.get("api") != "scene":
        raise ValueError(
            f"{run_dir} was not fitted with the scene API (metadata.json has "
            f"api={run_cfg.get('api')!r}). The port changed the flat-z column order and "
            f"the noise convention, so this run's stored unconstrained samples decode to "
            f"different physical values here -- with no error raised. Read it at the "
            f"pre-port commit, or re-fit it."
        )

    lens = lens or run_cfg.get("lens") or lenses.from_run_dir(run_dir)
    cfg = lenses.load(lens)

    n_source = int(run_cfg.get("n_source", 1))
    n_lens_light = int(run_cfg.get("n_lens_light", 1))
    lens_light_n_max = float(run_cfg.get("lens_light_n_max", 10.0))
    theta_e_center = float(run_cfg.get("theta_e_center", theta_e_center))
    theta_e_width = float(run_cfg.get("theta_e_width", theta_e_width))
    source_prior_version = int(
        run_cfg.get("source_prior_version", priors.SOURCE_PRIOR_VERSION)
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
    model = priors.build_model(
        cfg,
        theta_E_center=theta_e_center,
        theta_E_width=theta_e_width,
        n_source=n_source,
        n_lens_light=n_lens_light,
        lens_light_n_max=lens_light_n_max,
        source_prior_version=source_prior_version,
        lens_light_prior_version=lens_light_prior_version,
    )
    ds = ImageData(
        image,
        sim_config,
        background_rms=cfg.background_rms,
        exp_time=cfg.exp_time,
        sees="all",
    )
    prob = ProbModel(model, ds, mode="forward")

    n_recorded = run_cfg.get("n_free_params")
    if n_recorded is not None and int(n_recorded) != model.num_free_params:
        raise ValueError(
            f"{run_dir} recorded {n_recorded} free parameters but the model rebuilt from "
            f"its metadata has {model.num_free_params}. The reconstruction is wrong; "
            f"every physical value read from this run would be too."
        )

    return RunModel(
        cfg=cfg,
        run_cfg=run_cfg,
        image=image,
        psf=psf,
        sim_config=sim_config,
        model=model,
        prob=prob,
        pnames=list(model.z_param_names),
        labels=list(model.z_labels or model.z_param_names),
        spans=priors.source_component_spans(model),
        n_source=n_source,
        n_lens_light=n_lens_light,
        lens_light_n_max=lens_light_n_max,
        theta_e_center=theta_e_center,
        theta_e_width=theta_e_width,
        source_prior_version=source_prior_version,
        lens_light_prior_version=lens_light_prior_version,
    )


def column(model, suffix, kind=None):
    """Index of the one z-column whose path ends in ``/<suffix>``.

    Structural lookup, so nothing here hardcodes a column number. ``kind`` optionally
    restricts to ``"mass"`` or ``"light"`` sites. Raises if the match is not unique --
    an ambiguous name silently picking the first hit is exactly how a plot ends up
    labelled with one component's value and drawn from another's.
    """
    names = list(model.z_param_names)
    hits = [
        i
        for i, n in enumerate(names)
        if n.endswith("/" + suffix) and (kind is None or f"/{kind}/" in n)
    ]
    if len(hits) != 1:
        raise KeyError(
            f"expected exactly one z-column ending in /{suffix}"
            f"{f' among {kind} sites' if kind else ''}, found {[names[i] for i in hits]}"
        )
    return hits[0]


def to_constrained(model, z_unc):
    """Map unconstrained samples ``(..., n_params)`` to constrained ``(..., n_params)``.

    Column order is ``model.z_param_names`` on both sides: ``labeled_samples`` returns a
    name-keyed dict, and this restacks it in z-column order so index ``i`` means the same
    parameter before and after. Batch dims are preserved.
    """
    z = jnp.asarray(np.asarray(z_unc))
    if z.ndim == 1:
        z = z[None, :]
    labeled = model.labeled_samples(z)
    return np.stack([np.asarray(labeled[n]) for n in model.z_param_names], axis=-1)


def simulate_from_unconstrained(rm, z_unc):
    """Render the model image for a single unconstrained parameter vector.

    Returns ``(num_pix, num_pix)``. ``SceneSimulator.simulate`` squeezes the batch axis, so
    a single (unbatched) parameter vector renders straight to a 2-D image.
    """
    z = jnp.asarray(np.asarray(z_unc).reshape(-1))
    im = np.asarray(rm.simulator.simulate(rm.model.constrained(z)))
    return im.reshape(rm.sim_config.num_pix, rm.sim_config.num_pix)


def point_estimate(run_dir):
    """Return ``(z_unconstrained, label)`` for the best available point estimate.

    Preference: HMC posterior median > SVI mean > MAP best. The median is taken in
    unconstrained space, which is the right place for it -- the transform is monotonic
    per-coordinate, so the median commutes with it, and taking it here avoids the
    boundary pile-up that a constrained-space mean would suffer.
    """
    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is not None:
        samples = hmc["samples"]  # (n_chains, n_steps, n_params)
        return np.median(samples.reshape(-1, samples.shape[-1]), axis=0), "HMC median"

    svi = load_checkpoint(run_dir, "svi")
    if svi is not None:
        return np.asarray(svi["mean"]), "SVI mean"

    mp = load_checkpoint(run_dir, "map")
    if mp is not None:
        return np.asarray(mp["best_z"]).reshape(-1), "MAP best"

    raise FileNotFoundError(f"No checkpoints in {run_dir}; expected at least map.npz.")


def residual_figure(out_dir, image, err_map, im_sim, label):
    """Four-panel figure: observed, model, residual, normalised residual.

    The normalised-residual panel is the decisive one. Coherent arc-shaped structure above
    ~3 sigma means the model is wrong regardless of what R-hat says.

    ``err_map`` is the dataset's own ``ImageData.error_map`` --
    ``sqrt(background_rms^2 + clip(observed, 0, inf) / exp_time)``, fixed from the
    *observed* image. Pre-port runs used a map recomputed from the *simulated* image each
    step, so a chi-squared quoted here is not comparable with one from those runs.
    """
    err_map = np.asarray(err_map)
    residual = image - im_sim
    norm_res = residual / err_map

    # Reduced chi-squared over the image, matching the ProbModel's second return value.
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
        con: constrained samples, ``(n_chains, n_steps, n_params)``.
        spans: ``[(start, stop), ...]`` flat index span per source component, from
            ``priors.source_component_spans``. Each span must be contiguous -- that helper
            returns ``[]`` rather than a misaligned span if it is not.
        pnames: flat parameter names (``model.z_param_names``).

    Returns:
        ``(switched, con_relabelled)``. ``con_relabelled`` is None when no switching was
        detected, in which case the caller must use ``con`` unmodified.
    """
    if len(spans) < 2:
        return False, None

    # Order on R_sersic. It is the parameter the two components are most likely to differ
    # in (compact knot vs extended host), so it is the most reliable discriminator.
    r_idx = []
    for start, stop in spans:
        match = [
            start + k
            for k, n in enumerate(pnames[start:stop])
            if n.endswith("/R_sersic")
        ]
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
    sign = np.sign(r0 - r1)
    # A crossing is a sign change along the DRAW axis (axis 1), within a single chain.
    crossings = np.sum(np.diff(sign, axis=1) != 0, axis=1)
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
    print(f"chains with a label crossing: {n_chains_switched}/{con.shape[0]} "
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

    # Relabel: per chain and draw, order the component blocks by R_sersic ascending.
    con_re = con.copy()
    swap = r0 > r1  # (n_chains, n_steps)
    (a0, a1), (b0, b1) = spans[0], spans[1]
    block_a = con[:, :, a0:a1].copy()
    block_b = con[:, :, b0:b1].copy()
    con_re[:, :, a0:a1] = np.where(swap[:, :, None], block_b, block_a)
    con_re[:, :, b0:b1] = np.where(swap[:, :, None], block_a, block_b)
    return True, con_re


def boundary_pinning_check(model, con_flat, pnames):
    """Warn when a posterior sits against a Uniform prior's bound.

    HMC mixes badly along a boundary, and a pinned parameter means the prior -- not the
    data -- is setting the value. The first production run had two such pinnings
    (``lens_light.n_sersic`` at its ceiling of 6, ``source.n_sersic`` at its floor of 0.5)
    and only the first was noticed by eye. This makes the check automatic.

    ``model.prior`` is a ``JointDistributionNamed`` over the *unique* free parameters,
    keyed by exactly the ``z_param_names`` strings, so the walk is a flat dict lookup --
    no nested structure to traverse in step with anything.

    Only ``Uniform`` priors are checked. A parameter in the tail of an unbounded prior is
    a different phenomenon and not necessarily pathological.
    """
    from tensorflow_probability.substrates.jax import distributions as tfd

    bounds = {}
    for key, dist in (model.prior.model or {}).items():
        if isinstance(dist, tfd.Uniform) and key in pnames:
            bounds[key] = (float(dist.low), float(dist.high))

    if not bounds:
        return

    print("\n=== Boundary pinning (Uniform priors) ===")
    med = np.median(con_flat, axis=0)
    labels = list(model.z_labels or pnames)
    width = max(len(labels[pnames.index(n)]) for n in bounds)
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
        print(f"{labels[idx]:<{width}} {v:>9.4f}  in [{lo:g}, {hi:g}]  "
              f"({100*frac:5.1f}% of range){tag}")

    if any_pinned:
        print("\nA pinned parameter is set by its prior, not by the data, and HMC cannot "
              "mix along the boundary -- expect poor R-hat there and treat the value as a "
              "bound rather than a measurement.")


def convergence_table(run_dir, rm):
    """Print the posterior summary with R-hat and ESS. Returns the max R-hat, or None.

    R-hat and ESS come from ``gigalens.jax.analysis.diagnose_convergence``: ArviZ's
    rank-normalized split R-hat and bulk/tail ESS (Vehtari et al. 2021), computed in
    **unconstrained** space and keyed by ``z_param_names``. That replaces the hand-rolled
    ``tfp.mcmc.potential_scale_reduction`` call this file used to make. Two differences
    are worth knowing about when comparing against older reports:

    - The classic PSRF over-reports badly on the heavy-tailed z-marginals a prior bound
      induces; the rank-normalized statistic does not. Numbers will not match old runs'.
    - Both bulk- and tail-ESS are reported. Tail-ESS is the one that catches heavy-tail
      mixing failures, and it is usually the smaller.

    Percentiles are still quoted in constrained space -- that is the space the numbers get
    published in.

    With more than one source component this also runs the label-switching detector and
    the boundary-pinning check. The R-hat reported is the raw one unless switching was
    actually detected, in which case both raw and relabelled are shown.
    """
    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is None:
        print("\nNo HMC samples; convergence diagnostics unavailable.")
        return None

    samples = np.asarray(hmc["samples"])  # (n_chains, n_steps, n_params)
    if samples.ndim != 3 or samples.shape[-1] != rm.model.num_free_params:
        raise ValueError(
            f"hmc.npz holds {samples.shape}; expected (n_chains, n_steps, "
            f"{rm.model.num_free_params})."
        )
    n_chains, n_draws, n_params = samples.shape

    con = to_constrained(rm.model, samples.reshape(-1, n_params))
    con = con.reshape(n_chains, n_draws, n_params)

    switched, con_re = label_switching_check(con, rm.spans, rm.pnames)

    report = diagnose_convergence(samples, model=rm.model)
    rhat, ess_bulk, ess_tail = report.rhat, report.ess_bulk, report.ess_tail

    if switched and con_re is not None:
        # Relabelling is defined in constrained space (it sorts on R_sersic), so map the
        # relabelled samples back to z before re-diagnosing -- diagnose_convergence works
        # in z space, and mixing the two would compare statistics computed on different
        # parameterisations.
        z_re = np.asarray(
            rm.model.unconstrained(rm.model.to_params(
                {n: jnp.asarray(con_re[..., i])
                 for i, n in enumerate(rm.pnames)}))
        )
        report_re = diagnose_convergence(z_re, model=rm.model)
        print(f"\nR-hat worst  raw: {report.max_rhat:.3f}   "
              f"after relabelling: {report_re.max_rhat:.3f}")
        print("If relabelling fixes it, the chains were switching labels and the fit is "
              "healthy. If both are bad, something other than exchangeability is wrong.")

    flat = con.reshape(-1, n_params)
    boundary_pinning_check(rm.model, flat, rm.pnames)

    lo, med, hi = np.percentile(flat, [16, 50, 84], axis=0)

    labels = rm.labels
    width = max(len(l) for l in labels)
    print(f"\n=== Posterior ({n_chains} chains x {n_draws} draws) ===")
    print(f"{'parameter':<{width}}{'median':>10}{'-1sig':>10}{'+1sig':>10}"
          f"{'Rhat':>8}{'ESSbulk':>9}{'ESStail':>9}")
    for i, name in enumerate(labels):
        flag = "  <-- Rhat" if rhat[i] > RHAT_THRESHOLD else ""
        print(
            f"{name:<{width}}{med[i]:>10.4f}{med[i]-lo[i]:>10.4f}{hi[i]-med[i]:>10.4f}"
            f"{rhat[i]:>8.3f}{ess_bulk[i]:>9.0f}{ess_tail[i]:>9.0f}{flag}"
        )

    worst = float(report.max_rhat)
    n_bad = int(np.sum(rhat > RHAT_THRESHOLD))
    if n_bad:
        print(f"\n{n_bad}/{n_params} parameters have Rhat > {RHAT_THRESHOLD} "
              f"(worst {worst:.3f}). The chains have not converged; do not quote these "
              f"values as measurements.")
    else:
        print(f"\nAll {n_params} parameters have Rhat <= {RHAT_THRESHOLD} "
              f"(worst {worst:.3f}); min bulk-ESS {report.min_ess_bulk:.0f}, "
              f"min tail-ESS {report.min_ess_tail:.0f}.")
    return worst


def theta_E_check(run_dir, rm):
    """Compare the theta_E posterior against the config's measured clump radii.

    If every clump is a counter-image of one source, the Einstein radius should land inside
    their radial span. A posterior pinned against either prior edge means the radial scales
    are pulling the model apart, and that is the thing to report -- not the number.
    """
    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is None:
        return
    samples = np.asarray(hmc["samples"])
    con = to_constrained(rm.model, samples.reshape(-1, samples.shape[-1]))
    idx = column(rm.model, "theta_E", kind="mass")
    lo, med, hi = np.percentile(con[:, idx], [16, 50, 84])

    mu, sigma = np.log(rm.theta_e_center), rm.theta_e_width
    p01, p99 = np.exp(mu - 2.326 * sigma), np.exp(mu + 2.326 * sigma)

    print("\n=== theta_E vs. measured clump radii ===")
    print(f"posterior   {med:.3f} -{med-lo:.3f} +{hi-med:.3f} arcsec")
    print(f"prior 1-99% [{p01:.3f}, {p99:.3f}]")
    for name, r in sorted(rm.cfg.clump_radii.items(), key=lambda kv: kv[1]):
        print(f"  {name:<10} {r:.2f}")

    if med - p01 < 0.1 * (p99 - p01):
        print("WARNING: posterior pinned near the LOWER prior edge. The model is likely "
              "fitting only the inner clumps and ignoring the widest image.")
    elif p99 - med < 0.1 * (p99 - p01):
        print("WARNING: posterior pinned near the UPPER prior edge. The prior may be "
              "truncating a solution that wants a larger Einstein radius.")


def svi_report(run_dir):
    """Report whether the SVI stage converged, before any Rhat is worth reading.

    This is the check that would have caught the ``twosource`` run immediately. Its ELBO
    swung by 27,000 over the last few hundred steps and left a covariance conditioned at
    8.8e+11. ``HMC`` inverts that in float32 without complaint, producing a matrix that is
    not an inverse (``max|inv(C)C - I| = 55``) and whose spectrum carries five negative
    eigenvalues. Every one of that run's 29 parameters then failed Rhat, which was read as
    slow mixing and was not: the mass matrix was effectively indefinite.

    Note that the covariance itself is positive definite, so a PD test does not catch this.
    Conditioning is the discriminant. See ``svi_health.check_svi_covariance``.

    Read this line before the convergence table. A rejected covariance invalidates the HMC
    output entirely -- there is nothing to diagnose downstream of it. Unlike the guard in
    the fitting path this only reports, since by the time a run is being diagnosed the
    samples already exist and the useful thing is to say why they cannot be trusted.
    """
    svi = load_checkpoint(run_dir, "svi")
    if svi is None:
        return
    try:
        min_eig, cond = svi_health.check_svi_covariance(scale_tril=svi["scale_tril"])
        failed = None
    except RuntimeError as exc:
        failed = str(exc)
        # Recompute unguarded so the numbers can still be printed alongside the failure.
        s = np.linalg.svd(np.asarray(svi["scale_tril"], dtype=np.float64), compute_uv=False)
        min_eig = float(s.min() ** 2)
        cond = float(s.max() ** 2 / min_eig) if min_eig > 0 else np.inf
    loss = np.asarray(svi["loss_hist"], dtype=np.float64).reshape(-1)
    tail = loss[-300:]

    print("\n=== SVI health ===")
    print(f"covariance min eigenvalue {min_eig:.3e}, condition {cond:.3e} "
          f"(healthy runs 1e6-6e8, limit {svi_health.COND_MAX:.0e})")
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
    ax.plot(np.asarray(svi["loss_hist"]).reshape(-1))
    ax.set_xlabel("step")
    ax.set_ylabel("ELBO loss")
    ax.set_title("SVI convergence")
    fig.tight_layout()
    path = Path(out_dir) / "svi_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def corner_plot(run_dir, out_dir, rm, subset=None):
    """Corner plot of the constrained posterior.

    Defaults to the mass parameters, which exist under every model variant. All 22 (or 36)
    in one figure is unreadable at print size. Columns are located structurally by
    ``column``, so nothing here depends on a parameter's index.
    """
    try:
        import corner
    except ImportError:
        print("corner not installed; skipping corner plot.")
        return None

    hmc = load_checkpoint(run_dir, "hmc")
    if hmc is None:
        return None

    samples = np.asarray(hmc["samples"])
    con = to_constrained(rm.model, samples.reshape(-1, samples.shape[-1]))

    if subset is None:
        subset = [
            column(rm.model, s, kind="mass")
            for s in ("theta_E", "gamma", "e1", "e2", "gamma1", "gamma2")
        ]
    idx = list(subset)

    fig = corner.corner(
        con[:, idx],
        labels=[rm.labels[i] for i in idx],
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

    # Reconstruct the *same* model the run was fitted with. A run that predates the
    # scene-API port is refused here rather than silently mis-decoded.
    rm = rebuild_model(
        run_dir,
        lens=args.lens,
        theta_e_center=args.theta_e_center,
        theta_e_width=args.theta_e_width,
    )

    z_unc, label = point_estimate(run_dir)
    print(f"Point estimate: {label}")

    im_sim = simulate_from_unconstrained(rm, z_unc)
    chi2, res_path = residual_figure(
        out_dir, rm.image, rm.prob.error_map, im_sim, label
    )
    print(f"Reduced chi-squared at point estimate: {chi2:.3f}")
    print(f"Residual figure: {res_path}")

    # Before Rhat: an ill-conditioned SVI covariance invalidates the HMC output outright,
    # so this line has to be read first.
    svi_report(run_dir)

    worst_rhat = convergence_table(run_dir, rm)
    theta_E_check(run_dir, rm)

    p = svi_loss_plot(run_dir, out_dir)
    if p:
        print(f"SVI loss: {p}")
    p = corner_plot(run_dir, out_dir, rm)
    if p:
        print(f"Corner plot: {p}")

    if worst_rhat is not None and worst_rhat > RHAT_THRESHOLD:
        print("\nReminder: quote R-hat alongside any parameter value. A theta_E without a "
              "convergence diagnostic is not a measurement.")


if __name__ == "__main__":
    main()
