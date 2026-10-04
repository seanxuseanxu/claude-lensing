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

``main()`` prints a ``[timing]`` line per stage. Measured 2026-08-11 on the largest run to
date (``claude/DESI234/runs/r2_s2l2``, 50 chains x 40000 steps x 36 params = 2M draws),
CPU, 172s total: 79s in the ArviZ rank-R-hat/ESS pass, 25s in ``ess_vs_draws_plot``, 6s in
``convergence_table`` once the report is handed to it, 6s in the rank plots, 4s in the
corner plot after thinning, 2s in ``to_constrained``. Everything else -- simulate, critical
curves/caustics, the source render, label switching -- is under 2s combined.

The ArviZ pass is exact and lives in the read-only ``nanolens`` package, so its cost is not
reduced here, only made attributable: it is computed **once** in ``main()`` and threaded
into the three stages that need it. ``ess_vs_draws_plot`` costs a further ~0.3x of it by
carrying only the six worst-R-hat columns through its prefixes rather than all 36. If a
run's total grows well past what its draw count would predict from these numbers, suspect
the redundant work this file used to do (loading ``hmc.npz``, computing ``to_constrained``,
or re-running ``diagnose_convergence`` more than once), not ArviZ.
"""

import argparse
import dataclasses
import json
import time
from collections import namedtuple
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm

import jax.numpy as jnp

from gigalens.jax.analysis import (
    LensSolver,
    diagnose_convergence,
    image_border_in_source_plane,
)

import data
import lenses
import priors
import svi_health

RHAT_THRESHOLD = 1.01

# ESS floor, from Vehtari et al. 2021 (papers/r-hat_convergence/NOTES.md sec. 4). These are
# not a quality target -- they are the condition under which R-hat is *estimable at all*.
# R-hat is itself computed from the chains, and below these counts its sampling error swamps
# the 0.01 margin the gate is testing. The consequence that is easy to get backwards: below
# the floor a **passing** R-hat is uninformative too, so 1.002 on 200 effective draws is not
# evidence of convergence, it is an absence of evidence in either direction.
#
# The per-split-half number is the binding one. Split-R-hat halves every chain, so n_chains
# chains give 2*n_chains halves, and each needs its own >= 50 for the between/within variance
# estimate to mean anything -- an aggregate 400 concentrated in two good chains does not pass.
ESS_FLOOR_TOTAL = 400.0
ESS_FLOOR_PER_SPLIT_HALF = 50.0
MIN_CHAINS = 4

# Radius (arcsec) of the deflector-core excess-chi-squared region, for a *reporting* mask
# only -- it never touches the likelihood or any fit. KNOWN_ISSUES.md item 5's convention is
# to mask the core when computing chi-squared but not when modelling, so the core pixels
# stay in every ImageData/ProbModel exactly as they were fit; only residual_figure's and
# diagnose.main's SECOND, centre-masked chi-squared excludes them.
#
# Measured by claude/DESI257/analysis/core_checkerboard_extent.py: the radius enclosing 95%
# of the excess chi-squared above the local background is 7 px on DESI257 r3_unmasked and
# 8 px on DESI234 r3_unmasked (90% -> 7/6 px, 99% -> 8/8 px; insensitive to the choice). The
# two systems agree to one pixel, and all three modelled systems are HST WFC3/IR F140W at
# 0.065"/px, so this is an instrument-level constant rather than a per-lens config field.
# Set to the larger of the two (8 px), so neither system is under-masked.
#
# In arcsec, not pixels, because the artifact is believed to be PSF-scale (see that script's
# docstring) -- a property of the instrument, not of any one lens's pixel grid.
CORE_MASK_RADIUS_ARCSEC = 0.520


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
        raw metadata dict), ``mask`` (the keep-mask the fit used, or None), the resolved
        ``mask_circles`` that mask was built from, and the resolved variant fields
        ``n_source``, ``n_lens_light``, ``lens_light_n_max``, ``theta_e_center``,
        ``theta_e_width``, ``foreground`` (bool), ``foreground_prior_version``.
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
    # foreground: whether the run's model carries the SIS + Sersic interloper pair, and
    # which versioned support it used. Absent in metadata written before round 3b, which
    # is correctly read as False/no-op -- those runs have no such component.
    foreground_flag = bool(run_cfg.get("foreground", False))
    # run_fit.py writes "foreground_prior_version": null when --foreground was not given,
    # so a bare .get(..., default) would return that None rather than the default -- guard
    # explicitly rather than let int(None) raise somewhere less legible.
    recorded_fpv = run_cfg.get("foreground_prior_version")
    foreground_prior_version = (
        priors.FOREGROUND_PRIOR_VERSION if recorded_fpv is None else int(recorded_fpv)
    )
    foreground = cfg.foreground_prior if foreground_flag else None
    if foreground_flag and foreground is None:
        raise ValueError(
            f"{run_dir} was fitted with foreground=True but {cfg.name}'s config defines "
            f"no foreground_prior; the model cannot be rebuilt without the location/scale "
            f"parameters that produced this run."
        )

    if not quiet:
        print(f"Lens: {cfg.name}")
        print(f"Model variant: n_source={n_source}, n_lens_light={n_lens_light}, "
              f"lens_light_n_max={lens_light_n_max}, "
              f"source_prior_v{source_prior_version}, "
              f"lens_light_prior_v{lens_light_prior_version}, "
              f"foreground={foreground_flag}"
              + (f" (prior_v{foreground_prior_version})" if foreground_flag else "") +
              f", theta_E prior LogNormal(log {theta_e_center}, {theta_e_width})")

    # supersample is a PER-RUN choice, not a fact about the field like the crop -- DESI234
    # round 3 raised it 2 -> 4 (RESULTS_v3.md sec 1/3), and the same config cannot describe
    # both rounds. ``run_fit.py`` records the value it actually fit with under
    # "supersample" in metadata.json, so that recorded value wins here, exactly like
    # mask_circles above (KNOWN_ISSUES.md item 8); cfg.supersample is used only as a
    # fallback for runs written before that key existed. Using today's config unconditionally
    # would rebuild the ProbModel's rendering grid at a resolution the run was never fitted at.
    if "supersample" in run_cfg:
        supersample = int(run_cfg["supersample"])
    else:
        supersample = cfg.supersample
    if supersample != cfg.supersample and not quiet:
        print(f"NOTE: run's recorded supersample={supersample} differs from "
              f"{cfg.name}'s current config ({cfg.supersample}); using the recorded value.")
    elif "supersample" not in run_cfg and not quiet:
        print(f"NOTE: {run_dir} predates the recorded 'supersample' key; falling back to "
              f"{cfg.name}'s current config (supersample={cfg.supersample}), which may not "
              f"be what this run was actually fitted at.")

    image, psf, sim_config = data.load(cfg, supersample=supersample)
    model = priors.build_model(
        cfg,
        theta_E_center=theta_e_center,
        theta_E_width=theta_e_width,
        n_source=n_source,
        n_lens_light=n_lens_light,
        lens_light_n_max=lens_light_n_max,
        source_prior_version=source_prior_version,
        lens_light_prior_version=lens_light_prior_version,
        foreground=foreground,
        foreground_prior_version=foreground_prior_version,
    )
    # mask_circles is a PER-RUN choice, not a fact about the field like the crop --
    # DESI257's config masked its SW object for r1_s1l1/r2_s2l2 and later stopped doing so.
    # ``run_fit.py`` records the value it actually fit with under "mask_circles" in
    # metadata.json, so that recorded value wins here; cfg.mask_circles is used only as a
    # fallback for runs written before that key existed. Using today's config unconditionally
    # would compare a masked fit's samples against an unmasked likelihood, or vice versa.
    if "mask_circles" in run_cfg:
        raw = run_cfg["mask_circles"]
        # JSON round-trips a tuple of triples as a list of lists; restore the
        # tuple-of-triples shape data.build_mask expects.
        mask_circles = None if raw is None else tuple(tuple(c) for c in raw)
    else:
        mask_circles = cfg.mask_circles
    if mask_circles != cfg.mask_circles and not quiet:
        print(f"NOTE: run's recorded mask_circles {mask_circles} differs from "
              f"{cfg.name}'s current config ({cfg.mask_circles}); using the recorded value.")
    mask = data.build_mask(dataclasses.replace(cfg, mask_circles=mask_circles), image.shape)

    # n_masked_pixels is the kept-pixel count run_fit.py recorded (see its own comment) --
    # cross-check it against the rebuilt mask so this reconstruction is verified rather than
    # merely hoped correct.
    n_kept = int(mask.sum()) if mask is not None else int(image.size)
    n_kept_recorded = run_cfg.get("n_masked_pixels")
    if n_kept_recorded is not None and int(n_kept_recorded) != n_kept:
        raise ValueError(
            f"{run_dir} recorded n_masked_pixels={n_kept_recorded} but the mask rebuilt "
            f"from mask_circles={mask_circles} keeps {n_kept} pixels. The reconstructed "
            f"mask does not match what this run was fitted with."
        )
    prob = data.build_prob_model(cfg, model, image, sim_config, mask)

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
        mask=mask,
        mask_circles=mask_circles,
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
        foreground=foreground_flag,
        foreground_prior_version=foreground_prior_version,
    )


def column(model, suffix, kind=None):
    """Index of the one z-column whose path ends in ``/<suffix>``.

    Structural lookup, so nothing here hardcodes a column number. ``kind`` optionally
    restricts to ``"mass"`` or ``"light"`` sites. Raises if the match is not unique --
    an ambiguous name silently picking the first hit is exactly how a plot ends up
    labelled with one component's value and drawn from another's.

    With a ``foreground`` component the model carries a SECOND ``theta_E`` (the SIS's) and
    a second ``center_x``/``center_y`` (both the SIS's and its light's), so a bare
    ``suffix="theta_E"`` is no longer unique among ``kind="mass"`` sites. ``priors.py``
    always builds the host EPL first, so ``suffix="mass/0/theta_E"`` picks it
    unambiguously regardless of how many other mass components follow -- callers that need
    the host's Einstein radius should use that, not the bare name.
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


def point_estimate(run_dir, hmc_samples=None):
    """Return ``(z_unconstrained, label)`` for the best available point estimate.

    Preference: HMC posterior median > SVI mean > MAP best. The median is taken in
    unconstrained space, which is the right place for it -- the transform is monotonic
    per-coordinate, so the median commutes with it, and taking it here avoids the
    boundary pile-up that a constrained-space mean would suffer.

    ``hmc_samples``, if given, is the already-loaded ``(n_chains, n_steps, n_params)``
    array -- pass it when the caller has already paid for ``hmc.npz``'s decompression
    (288 MB / ~1.5s on the largest run), so a second read is not needed here. ``None``
    (the default, and every external caller's usage) loads it fresh, unchanged from
    before.
    """
    if hmc_samples is None:
        hmc = load_checkpoint(run_dir, "hmc")
        hmc_samples = None if hmc is None else hmc["samples"]
    if hmc_samples is not None:
        samples = hmc_samples  # (n_chains, n_steps, n_params)
        return np.median(samples.reshape(-1, samples.shape[-1]), axis=0), "HMC median"

    svi = load_checkpoint(run_dir, "svi")
    if svi is not None:
        return np.asarray(svi["mean"]), "SVI mean"

    mp = load_checkpoint(run_dir, "map")
    if mp is not None:
        return np.asarray(mp["best_z"]).reshape(-1), "MAP best"

    raise FileNotFoundError(f"No checkpoints in {run_dir}; expected at least map.npz.")


def masked_chisq(norm_res, keep):
    """Reduced chi-squared: mean of ``norm_res**2`` over the ``keep`` (unmasked) pixels.

    This is the one formula both ``residual_figure`` and ``plot_overview.py`` use -- it used
    to be duplicated by hand in each, and the copy in ``plot_overview.py`` silently kept
    averaging over every pixel after the mask was fixed here. Matches ``ProbModel``'s second
    return value, ``Σ(r/σ)² / event_size``, when ``keep`` is the same mask its ``ImageData``
    was given. Returns ``(chi2, n_kept)``.
    """
    keep = np.asarray(keep, dtype=bool)
    return float(np.mean(norm_res[keep] ** 2)), int(keep.sum())


def core_excluded_keep(shape, centroid_pix, radius_px, keep=None):
    """``keep`` (default: all-True) with a disc of radius ``radius_px`` about
    ``centroid_pix`` additionally excluded -- the *reporting*-only core mask, never applied
    to the likelihood. ``centroid_pix`` is ``(row, col)``, matching ``LensConfig.centroid_pix``.

    Uses the same convention as ``core_checkerboard_extent.py``'s ``radius_map``/annulus
    selection (``rmap >= r``), so the two scripts agree on which pixels count as "inside":
    a pixel is excluded when its distance to the centroid is strictly less than
    ``radius_px``, i.e. the surviving annulus starts at ``rmap >= radius_px``.
    """
    if keep is None:
        keep = np.ones(shape, dtype=bool)
    else:
        keep = np.asarray(keep, dtype=bool)
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    dist = np.hypot(rows - centroid_pix[0], cols - centroid_pix[1])
    return keep & ~(dist < radius_px)


ChiSquares = namedtuple(
    "ChiSquares",
    ["full", "n_full", "core_masked", "n_core_masked", "radius_px", "radius_arcsec"],
)


def map_hmc_verdict(map_lp, hmc_max_lp):
    """Return ``(passed, deficit)``: MAP's log posterior must be >= the max HMC draw's.

    Pure and side-effect-free so ``map_convergence_check``'s pass/fail logic is testable on
    synthetic numbers without a checkpoint or a model. ``deficit`` is ``hmc_max_lp -
    map_lp`` in nats; positive means MAP scored lower than some HMC draw, which cannot
    happen at the true mode (MAP is by definition the highest-density point) and is the
    signature of an optimiser that stopped short.
    """
    deficit = float(hmc_max_lp) - float(map_lp)
    return deficit <= 0.0, deficit


def curve_radius(curve, center):
    """Mean distance of a curve's vertices from ``center`` (arcsec).

    ``curve`` is an ``(xs, ys)`` pair, the form ``LensSolver.critical_and_caustics``
    returns -- not an ``(N, 2)`` array.
    """
    xs, ys = curve
    return float(np.mean(np.hypot(xs - center[0], ys - center[1])))


def check_theta_E(curves, center, theta_E, gamma=None, extra_mass=None):
    """Print each critical curve's mean radius against theta_E; identify the tangential
    one (the largest) and flag loudly if it disagrees by more than 10%.

    ``extra_mass`` is the list of ADDITIONAL lens-plane mass components beyond the main
    EPL -- i.e. ``params["planes"][0]["mass"][1:]`` -- and it is not optional bookkeeping.
    The critical curve is a property of the TOTAL convergence on the lens plane, while
    ``theta_E`` here is the main deflector's alone. With a second mass present the two are
    different quantities and the 10% bound does not apply: a foreground SIS adds
    convergence, which pushes the tangential curve OUTWARD while the main EPL's own
    theta_E goes DOWN to compensate. Comparing them anyway reports a large disagreement
    and blames the fit for it. DESI234's ``r3_fg`` tripped exactly this: tangential 1.7610
    against a main theta_E of 1.4726, 19.6%, on a run whose chi2_nu was the best of the
    round. When ``extra_mass`` is non-empty this function reports the excess and says what
    it means, rather than asserting the fit is bad.

    HOW MANY CURVES TO EXPECT is a check in itself. The radial eigenvalue of a circular
    EPL is ``1 - (2-gamma)(theta/theta_E)^(1-gamma)``, which vanishes only where
    ``(theta/theta_E)^(1-gamma) = 1/(2-gamma)``. For **gamma >= 2 the right-hand side is
    non-positive, so there is no radial critical curve at all** -- a gamma > 2 fit must
    produce exactly ONE curve, the tangential one at theta_E. Two curves are expected
    only for gamma < 2. An extra inner curve at a radius of order the grid spacing is
    the signature of a contour finder tripping over the ``alpha ~ theta^(2-gamma)``
    divergence at the origin, not of real lens structure -- which is why the curves come
    from ``LensSolver`` rather than a hand-rolled finite-difference Jacobian.

    Only the tangential curve is expected to track theta_E closely.
    """
    if not curves:
        print("VALIDATION FAILED: no critical curve found at all.")
        return
    radii = sorted((curve_radius(c, center) for c in curves), reverse=True)
    tangential = radii[0]
    print(f"Critical curve radii (arcsec, sorted): {['%.4f' % r for r in radii]}")
    if gamma is not None and gamma >= 2.0 and len(radii) > 1:
        print(
            f"VALIDATION FAILED: gamma = {gamma:.3f} >= 2 admits no radial critical "
            f"curve, so exactly one curve is expected, but {len(radii)} were found. The "
            f"extra one(s) are a numerical artifact of the deflection's divergence at the "
            f"origin -- see check_theta_E's docstring. Distrust the caustics."
        )
    print(f"Tangential (largest) radius = {tangential:.4f}, theta_E = {theta_E:.4f}")
    frac = abs(tangential - theta_E) / theta_E

    if extra_mass:
        # The tangential curve traces the TOTAL convergence; theta_E is the main EPL's
        # alone. These are not the same quantity here, so the 10% bound is not applicable
        # and its failure message would be false. Report the excess as the physical
        # quantity it is.
        descr = ", ".join(
            f"{m.get('_profile', 'mass[%d]' % (i + 1))}"
            f"(theta_E={float(m['theta_E']):.4f})" if "theta_E" in m
            else f"mass[{i + 1}]"
            for i, m in enumerate(extra_mass)
        )
        print(
            f"Validation N/A (multi-mass lens plane): {len(extra_mass)} additional mass "
            f"component(s) beyond the main EPL -- {descr}. The tangential curve traces the "
            f"TOTAL lens-plane convergence, so it is not expected to equal the main "
            f"deflector's theta_E and the 10% bound does not apply. Excess over the main "
            f"theta_E is {tangential - theta_E:+.4f}\" ({frac * 100:.1f}%), which is the "
            f"additional mass doing lensing work -- if it is large, the main theta_E is "
            f"model-dependent at that level and belongs in the systematic budget, not the "
            f"statistical error. The critical curve and caustics themselves remain valid."
        )
    elif frac > 0.10:
        print(
            f"VALIDATION FAILED LOUDLY: tangential critical-curve radius disagrees with "
            f"theta_E by {frac * 100:.1f}% (> 10%). DO NOT trust this figure's critical "
            f"curve / caustics. With the curves coming from LensSolver this is more "
            f"likely to mean the fit itself is bad (an unconverged run, or a lens "
            f"centroid far from the light) than a geometry bug."
        )
    else:
        print(f"Validation OK: tangential radius agrees with theta_E to {frac * 100:.2f}%.")


def source_component_params(params, plane):
    """Light-component parameter dicts for ``plane``, in component-index order.

    ``model.constrained`` returns ``planes``, ``light`` and ``mass`` as **dicts keyed by
    integer index**, not lists. Iterating one yields the keys (ints), and ``params["planes"]``
    does not even iterate in index order -- so a ``for p in ...["light"]`` loop that looks
    exactly like a list iteration silently hands you integers. Index by ``range`` instead.
    """
    light = params["planes"][plane]["light"]
    return [light[j] for j in range(len(light))]


def render_source_plane(model, params, extent, n=400):
    """Render summed source-light surface brightness on a fresh (x, y) grid in the source
    plane -- i.e. calling ``light()`` directly on plain coordinates, not through the
    lens. ``extent = (xmin, xmax, ymin, ymax)`` in arcsec.

    The source plane is the last plane of the scene, and its light components are
    ``model.planes[-1].light`` with parameters at the matching path in the structured
    ``params`` dict. Indexing the plane list from the end rather than hardcoding ``1``
    keeps this correct if a deflector plane is ever inserted.
    """
    xmin, xmax, ymin, ymax = extent
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    Xg, Yg = np.meshgrid(xs, ys)  # indexing='xy', consistent with imshow(extent=...) below
    img = np.zeros_like(Xg)
    src = len(model.planes) - 1
    for j, comp in enumerate(model.planes[src].light):
        p = params["planes"][src]["light"][j]
        img = img + np.asarray(comp.profile.light(jnp.asarray(Xg), jnp.asarray(Yg), **p))
    return Xg, Yg, img


def residual_figure(out_dir, cfg, model, prob, image, err_map, im_sim, label, z_unc,
                     mask=None, filename="residuals.png"):
    """Four-panel figure: observed + critical curve, model + critical curve, unlensed
    source + caustics, normalised residual.

    The normalised-residual panel is the decisive one. Coherent arc-shaped structure above
    ~3 sigma means the model is wrong regardless of what R-hat says. The critical curve and
    caustic overlays are a second, geometric check: the curve should sit right on the arcs
    it is supposed to be forming, and the source panel should show one compact blob, not a
    smear -- ``check_theta_E`` below also checks the curve's radius against theta_E
    numerically, not just by eye.

    ``err_map`` is the dataset's own ``ImageData.error_map`` --
    ``sqrt(background_rms^2 + clip(observed, 0, inf) / exp_time)``, fixed from the
    *observed* image. Pre-port runs used a map recomputed from the *simulated* image each
    step, so a chi-squared quoted here is not comparable with one from those runs.

    ``mask`` is the run's keep-mask (True = kept), i.e. ``data.build_mask``'s output and
    the same array the fit's ``ImageData`` was given. It must be passed whenever the lens
    config sets ``mask_circles``: masked pixels are excluded from the likelihood, so a
    chi-squared averaged over them is not the quantity the fit optimised. Masked pixels
    are typically the *largest* residuals in the frame -- an unmodelled interloper is
    exactly what gets masked -- so including them inflates the number badly rather than
    negligibly. Measured on the round-1 fits (2026-08-08), masking ~6.5% of the pixels:
    DESI234 4.225 -> 1.560, DESI257 7.112 -> 1.533. Both models were being judged mostly
    on pixels their likelihood never saw.

    With the mask applied this matches ``ProbModel``'s second return value, which is
    ``Σ(r/σ)²`` over unmasked pixels divided by ``event_size`` (the unmasked count).
    Masked pixels are rendered in grey (NaN + ``cmap.set_bad``) and excluded from the
    normalised-residual colour scale, so the figure cannot imply the fit was judged on
    pixels it never saw.

    Returns a ``ChiSquares`` namedtuple (not a bare float), because this function reports
    TWO reduced chi-squareds and neither should be readable as the other by accident
    (KNOWN_ISSUES.md item 5). ``full`` is the number above -- everything the likelihood was
    fit on, mask included. ``core_masked`` is the same formula, ``masked_chisq`` again, over
    ``full``'s keep-mask with an additional disc of radius ``CORE_MASK_RADIUS_ARCSEC``
    excluded around ``cfg.centroid_pix`` -- a *reporting*-only exclusion of the deflector
    core's known excess-chi-squared artifact (measured by
    ``claude/DESI257/analysis/core_checkerboard_extent.py``; see that constant's comment).
    This never touches the likelihood: the core pixels were in the fit and stay there:
    ``core_masked`` differs from ``full`` only in which pixels this function's own average
    is taken over, after the fit is already done.

    Critical curves, caustics and the ray-traced cutout border all come from
    ``gigalens.jax.analysis``: ``LensSolver.critical_and_caustics`` and
    ``image_border_in_source_plane``. Nothing here computes det(A) by hand -- a
    finite-difference Jacobian got the tangential radius right but also reported a second,
    non-existent critical curve near the origin where ``alpha ~ theta^(2-gamma)`` diverges
    (see ``check_theta_E``). ``LensSolver`` uses forward-mode AD on
    ``LensModel.total_deflection`` and is pinned bit-identical to the simulator's own trace
    by gigalens' test suite, so it is both exact and incapable of drifting from the model
    that was fitted. ``prob`` must be the ``ProbModel`` built on the *same* ``model``
    passed here -- ``LensSolver`` reads deflection off it.

    ``z_unc`` is the flat unconstrained parameter vector (or a length-1 batch of one) the
    point estimate/synthetic draw was taken at; it is turned into the constrained
    ``params`` dict used for the ray tracing and the source render.
    """
    image = np.asarray(image)
    err_map = np.asarray(err_map)
    residual = image - im_sim
    norm_res = residual / err_map

    keep = (np.ones(image.shape, dtype=bool) if mask is None
            else np.asarray(mask, dtype=bool))
    n_cut = int(keep.size - keep.sum())

    # Reduced chi-squared over the KEPT pixels, matching the ProbModel's second return
    # value (Σ(r/σ)² / event_size). Averaging over masked pixels would report a
    # chi-squared for a likelihood that was never evaluated.
    chi2_full, n_full = masked_chisq(norm_res, keep)

    # Second, reporting-only number: the same formula, over the same keep-mask, with the
    # deflector-core excess-chi-squared region additionally excluded. Item 5 -- the core
    # stays in the likelihood; this is computed after the fact, for display only.
    radius_px = CORE_MASK_RADIUS_ARCSEC / cfg.delta_pix
    keep_core = core_excluded_keep(image.shape, cfg.centroid_pix, radius_px, keep=keep)
    chi2_core, n_core = masked_chisq(norm_res, keep_core)
    n_core_excluded = n_full - n_core

    # Grey out the excluded discs rather than drawing them: NaN + set_bad. The colour
    # scale is fixed (-5, 5 sigma) rather than data-derived, so this greying is what keeps
    # a masked interloper's huge residual from being implied by the panel at all.
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.75")
    norm_res_m = np.where(keep, norm_res, np.nan)

    # Raw lens parameters -- needed for the critical curve / caustic / source panels,
    # which is the reason this file and plot_overview.py used to be two implementations.
    params = model.constrained(jnp.asarray(np.asarray(z_unc).reshape(-1)))
    epl_params = {k: float(np.asarray(v)) for k, v in params["planes"][0]["mass"][0].items()}
    theta_E = epl_params["theta_E"]
    lens_center = (epl_params["center_x"], epl_params["center_y"])

    # LensSolver defaults its tracing window to the cutout's own angular extent and its
    # scale to delta_pix/supersample, so it needs neither here.
    solver = LensSolver(prob)
    curves, caustics = solver.critical_and_caustics(params)
    # Everything on the lens plane beyond the main EPL -- the foreground SIS, when the run
    # carries one. check_theta_E needs it to know that the tangential curve and theta_E are
    # different quantities for this model; see its docstring.
    # Filtered on `theta_E`, not on component index: SHEAR is also a lens-plane mass
    # component but carries no convergence (gamma1/gamma2 only), so it does not push the
    # tangential radius off theta_E the way a second deflector does. Only components with
    # an Einstein radius of their own qualify.
    lens_plane_mass = params["planes"][0]["mass"]
    extra_mass = [
        d for d in (
            {k: float(np.asarray(v)) for k, v in lens_plane_mass[i].items()}
            for i in sorted(lens_plane_mass)[1:]
        )
        if "theta_E" in d
    ]
    check_theta_E(
        curves, lens_center, theta_E, gamma=epl_params.get("gamma"), extra_mass=extra_mass
    )

    npix, delta = cfg.num_pix, cfg.delta_pix
    half = npix / 2 * delta
    extent_img = [-half, half, -half, half]

    # Source-plane window: sized from the fitted source centres/radii, then widened to the
    # ray-traced image of the cutout border. A source is observable only if one of its
    # images lands in the cutout, so beta(dR) is the edge of the observable region -- the
    # caustic is a property of the mass model, the border a property of the observation,
    # and the border is the one that actually bounds where a source could be. It is also
    # more robust: it is traced at the cutout boundary, where the deflection is small and
    # smooth, whereas a spurious critical curve near the origin (gamma > 2, see
    # check_theta_E) maps to an enormous caustic that would blow a caustic-derived window
    # out.
    src_plane = len(model.planes) - 1
    src_centers, src_radii = [], []
    for p in source_component_params(params, src_plane):
        src_centers.append((float(np.asarray(p["center_x"])),
                             float(np.asarray(p["center_y"]))))
        src_radii.append(float(np.asarray(p["R_sersic"])))
    cxs = [c[0] for c in src_centers]
    cys = [c[1] for c in src_centers]
    pad = max(src_radii) * 4 if src_radii else 0.3
    xmin, xmax = min(cxs) - pad, max(cxs) + pad
    ymin, ymax = min(cys) - pad, max(cys) + pad
    border_x, border_y = image_border_in_source_plane(solver, params)
    xmin = min(xmin, float(border_x.min()))
    xmax = max(xmax, float(border_x.max()))
    ymin = min(ymin, float(border_y.min()))
    ymax = max(ymax, float(border_y.max()))

    Xs, Ys, src_img = render_source_plane(model, params, (xmin, xmax, ymin, ymax), n=400)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # asinh stretch for the two image-plane brightness panels. The deflector core can be
    # decades brighter than the arcs, which are the only part of the image that constrains
    # the lens; asinh is linear below `linear_width` and logarithmic above, so both stay
    # visible at once. `linear_width` is tied to the noise (BACKGROUND_RMS), not a
    # percentile, so the linear region always means "consistent with background" rather
    # than drifting with how much bright galaxy happens to be in the crop. `vmin` is
    # slightly negative because real sky has negative pixels; clipping at 0 hides the
    # noise symmetry that makes an over-subtracted background obvious.
    vmax = float(np.percentile(image, 99.9))
    vmin = -5.0 * cfg.background_rms
    linear_width = 5.0 * cfg.background_rms
    bright_norm = AsinhNorm(linear_width=linear_width, vmin=vmin, vmax=vmax)

    # Panel 1: observed + critical curve.
    ax = axes[0, 0]
    im0 = ax.imshow(image, origin="lower", extent=extent_img, norm=bright_norm)
    for cx, cy in curves:
        ax.plot(cx, cy, color="lime", lw=1.2)
    ax.plot(*lens_center, "+", color="white", ms=12, mew=1.5)
    ax.set_title("Observed (asinh) + critical curve")
    plt.colorbar(im0, ax=ax, fraction=0.046, label="e/s")

    # Panel 2: model + critical curve. Shares `bright_norm` with panel 1 on purpose -- a
    # different stretch between data and model would make them incomparable by eye, which
    # is the whole point of putting them side by side.
    ax = axes[0, 1]
    im1 = ax.imshow(im_sim, origin="lower", extent=extent_img, norm=bright_norm)
    for cx, cy in curves:
        ax.plot(cx, cy, color="lime", lw=1.2)
    ax.plot(*lens_center, "+", color="white", ms=12, mew=1.5)
    ax.set_title(f"Model -- {label} (asinh) + critical curve")
    plt.colorbar(im1, ax=ax, fraction=0.046, label="e/s")

    # Panel 3: unlensed source + caustics. Also AsinhNorm, but with its own linear_width --
    # this is intrinsic surface brightness with no noise in it, so BACKGROUND_RMS is
    # meaningless here; a small fraction of the peak is the natural choice instead.
    ax = axes[1, 0]
    src_vmax = float(np.percentile(src_img, 99.9))
    im2 = ax.imshow(src_img, origin="lower", extent=[xmin, xmax, ymin, ymax],
                     cmap="viridis", aspect="equal",
                     norm=AsinhNorm(linear_width=max(src_vmax * 1e-3, 1e-8),
                                    vmin=0.0, vmax=src_vmax))
    for cx, cy in caustics:
        ax.plot(cx, cy, color="orange", lw=1.2)
    ax.plot(border_x, border_y, color="white", lw=1.2, ls="--")
    for i, (cx, cy) in enumerate(src_centers):
        ax.plot(cx, cy, "x", color="red", ms=10, mew=2)
        ax.annotate(f"src{i}", (cx, cy), color="red", fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Unlensed source (asinh) + caustics, dashed = cutout border")
    # Pin the axes to the computed window: `ax.plot` of a caustic otherwise autoscales the
    # panel out to its most distant vertex, so choosing a window would have no effect.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    plt.colorbar(im2, ax=ax, fraction=0.046, label="surface brightness (e/s/pix)")

    # Panel 4: normalised residual, mask-aware.
    ax = axes[1, 1]
    cut_note = "" if n_cut == 0 else f", {n_cut} px masked"
    im3 = ax.imshow(norm_res_m, origin="lower", extent=extent_img, cmap=cmap,
                     vmin=-5, vmax=5)
    ax.set_title(
        rf"Normalised residual  ($\chi^2_\nu$ = {chi2_full:.2f}{cut_note}, "
        rf"core-masked $\chi^2_\nu$ = {chi2_core:.2f} [r={radius_px:.1f}px, "
        rf"{n_core_excluded}px excl.])"
    )
    plt.colorbar(im3, ax=ax, fraction=0.046, label=r"$\sigma$")

    for ax in axes.flat:
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")

    fig.tight_layout()
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    chi_squares = ChiSquares(
        full=chi2_full, n_full=n_full, core_masked=chi2_core, n_core_masked=n_core,
        radius_px=radius_px, radius_arcsec=CORE_MASK_RADIUS_ARCSEC,
    )
    return chi_squares, path


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
    """Warn when a posterior sits against a hard prior bound -- Uniform's edges or a
    TruncatedNormal's truncation.

    HMC mixes badly along a boundary, and a pinned parameter means the prior -- not the
    data -- is setting the value. The first production run had two such Uniform pinnings
    (``lens_light.n_sersic`` at its ceiling of 6, ``source.n_sersic`` at its floor of 0.5)
    and only the first was noticed by eye. DESI257's ``gamma`` is a
    ``TruncatedNormal(2.0, 0.25, 1.0, 3.0)`` (priors.py) whose round-2 mode B posterior
    peaked at ~2.86 against the 3.0 ceiling, with samples reaching 2.885 -- a pinning this
    check did not catch until TruncatedNormal bounds were added here too. This makes both
    checks automatic.

    ``model.prior`` is a ``JointDistributionNamed`` over the *unique* free parameters,
    keyed by exactly the ``z_param_names`` strings, so the walk is a flat dict lookup --
    no nested structure to traverse in step with anything.

    Only ``Uniform`` and ``TruncatedNormal`` are checked -- both have a hard bound the
    posterior can pile up against. Unbounded priors (``Normal``, ``LogNormal``) are
    deliberately excluded: a parameter in their tail is a different phenomenon, not
    necessarily pathological, since there is no bound to pin against.
    """
    from tensorflow_probability.substrates.jax import distributions as tfd

    bounds = {}
    for key, dist in (model.prior.model or {}).items():
        if key not in pnames:
            continue
        if isinstance(dist, tfd.Uniform):
            bounds[key] = (float(dist.low), float(dist.high), "Uniform")
        elif isinstance(dist, tfd.TruncatedNormal):
            bounds[key] = (float(dist.low), float(dist.high), "TruncatedNormal")

    if not bounds:
        return

    print("\n=== Boundary pinning (Uniform bounds, TruncatedNormal truncation) ===")
    med = np.median(con_flat, axis=0)
    labels = list(model.z_labels or pnames)
    width = max(len(labels[pnames.index(n)]) for n in bounds)
    any_pinned = False
    for name, (lo, hi, family) in bounds.items():
        idx = pnames.index(name)
        v = med[idx]
        frac = (v - lo) / (hi - lo)
        tag = ""
        if frac < 0.02:
            tag, any_pinned = "  <-- PINNED at lower bound", True
        elif frac > 0.98:
            tag, any_pinned = "  <-- PINNED at upper bound", True
        print(f"{labels[idx]:<{width}} {v:>9.4f}  in [{lo:g}, {hi:g}]  "
              f"({100*frac:5.1f}% of range)  {family}{tag}")

    if any_pinned:
        print("\nA pinned parameter is set by its prior, not by the data, and HMC cannot "
              "mix along the boundary -- expect poor R-hat there and treat the value as a "
              "bound rather than a measurement.")


# Report a parameter when its posterior median sits beyond this many equivalent Gaussian
# sigma of its own prior. 3.0 is loose on purpose: with ~40 free parameters, a couple of
# 3-sigma displacements are expected by chance, and this is a prompt to look, not a verdict.
TENSION_SIGMA = 3.0


def prior_posterior_tension(model, con_flat, pnames, threshold=TENSION_SIGMA):
    """Report parameters whose posterior has moved far from the prior that generated it.

    This exists because ``boundary_pinning_check`` above answers a *different* question and
    the difference cost a wrong diagnosis. DESI234's ``r3_fg`` fitted
    ``gamma = 2.951 +/- 0.005`` under a ``TruncatedNormal(2.0, 0.25, 1.0, 3.0)`` prior. That
    is **not** pinned -- it sits 9.9 posterior sd below the 3.0 truncation with zero draws
    above 2.99, and the pinning check correctly stayed silent at 97.6% of range. It is
    nonetheless 3.8 prior sigma from the prior centre, with a posterior 50x narrower than
    the prior scale: the *likelihood* is confidently choosing a near-singular mass slope
    where galaxy-scale lenses sit near 2.0-2.2. Nothing in this file reported that.

    Pinning means the prior set the value. Tension means the data set it, against the
    prior's advice. The second is the more common signature of model misspecification --
    something else in the model is wrong and this parameter is absorbing it -- and it is
    invisible to every aggregate statistic, including ``chi2_nu``, which was 1.08 here.

    The measure is the prior quantile of the posterior median, converted to the equivalent
    Gaussian sigma. Working in quantiles rather than ``(x - mu) / sigma`` keeps it
    meaningful for the skewed and bounded priors this pipeline uses -- ``LogNormal`` for
    every Einstein radius, ``HalfNormal`` for the foreground mass, ``Uniform`` for Sersic
    indices -- where a moment-based z-score would be misleading.

    Reported, never raised. A prior is a statement about what is typical, and an atypical
    system is a thing that exists; see ``priors.py``'s plausibility warnings, which are
    flexible by the same reasoning. What this buys is that the atypical case has to be
    noticed and argued for rather than passing silently into a results table.
    """
    from tensorflow_probability.substrates.jax import distributions as tfd
    from scipy.special import ndtri

    med = np.median(con_flat, axis=0)
    sd = np.std(con_flat, axis=0)
    labels = list(model.z_labels or pnames)
    rows = []
    for key, dist in (model.prior.model or {}).items():
        if key not in pnames:
            continue
        idx = pnames.index(key)
        try:
            sf = float(dist.survival_function(med[idx]))
            cdf = float(dist.cdf(med[idx]))
        except Exception:
            continue
        if not np.isfinite(sf) or not np.isfinite(cdf):
            continue
        # Use whichever tail is small: cdf saturates to 1.0 in float long before the
        # survival function underflows, so reading sigma off the far tail of cdf alone
        # would silently cap the reported tension at ~8 sigma.
        tail = min(max(sf, 0.0), max(cdf, 0.0))
        tail = max(tail, 1e-300)
        sigma = abs(float(ndtri(tail)))
        if not np.isfinite(sigma):
            continue
        try:
            prior_sd = float(dist.stddev())
        except Exception:
            prior_sd = float("nan")
        ratio = sd[idx] / prior_sd if prior_sd and np.isfinite(prior_sd) else float("nan")
        if sigma >= threshold:
            rows.append((labels[idx], med[idx], sigma, ratio, cdf < sf))

    if not rows:
        return []

    rows.sort(key=lambda r: -r[2])
    width = max(len(r[0]) for r in rows)
    print(f"\n=== Prior-posterior tension (> {threshold:g} sigma of the prior) ===")
    print(f"{'parameter':<{width}}{'median':>10}{'prior sig':>11}{'post/prior sd':>15}  side")
    for name, v, sigma, ratio, low_side in rows:
        rs = f"{ratio:.3f}" if np.isfinite(ratio) else "n/a"
        print(f"{name:<{width}}{v:>10.4f}{sigma:>11.2f}{rs:>15}  "
              f"{'below' if low_side else 'above'} prior centre")
    print("\nThe data, not the prior, put these here. A narrow posterior far from its prior "
          "is the usual signature of a parameter absorbing someone else's error -- check it "
          "against what is physically typical before quoting anything that depends on it.")
    return rows


def hmc_and_constrained(run_dir, rm, hmc_samples=None, con=None, need_con=True):
    """Return ``(samples_z, con)``, both ``(n_chains, n_draws, n_params)``, or ``(None, None)``.

    The HMC-consuming stages below each need the same arrays and each used to load and
    transform them itself. ``main()`` computes both once and threads them in, so in the
    normal path this function only reshapes; the fallback exists so every stage stays
    callable on its own from a notebook or a test.

    ``need_con=False`` for the stages that work purely in z (the rank and ESS-vs-draws
    plots): they get ``con=None`` back rather than paying for a transform they discard.

    The shape check lives here rather than in ``convergence_table`` alone, so a transposed
    ``(n_steps, n_chains, n_params)`` array raises for *every* caller. That layout is the
    one silent hazard in this file's inputs -- it yields plausible, wrong split-R-hat rather
    than an error (see the module docstring, and ``diagnose_convergence``'s own warning).
    """
    if hmc_samples is None:
        hmc = load_checkpoint(run_dir, "hmc")
        if hmc is None:
            return None, None
        hmc_samples = hmc["samples"]
    samples = np.asarray(hmc_samples)
    if samples.ndim != 3 or samples.shape[-1] != rm.model.num_free_params:
        raise ValueError(
            f"hmc.npz holds {samples.shape}; expected (n_chains, n_steps, "
            f"{rm.model.num_free_params})."
        )
    n_chains, n_draws, n_params = samples.shape
    if not need_con:
        return samples, None
    if con is None:
        con = to_constrained(rm.model, samples.reshape(-1, n_params))
    return samples, np.asarray(con).reshape(n_chains, n_draws, n_params)


def worst_rhat_columns(report, n):
    """Indices of the ``n`` columns with the highest R-hat, worst first.

    The plots below diagnose *why* a fit is not mixing, so they show the parameters that are
    not mixing. A parameter at R-hat 1.000 contributes a flat line and a uniform histogram
    and crowds out the one that would have shown the problem.
    """
    order = np.argsort(np.asarray(report.rhat, dtype=float))[::-1]
    return [int(i) for i in order[:min(n, len(order))]]


def ess_floor_report(report, n_chains, labels=None):
    """Check bulk/tail ESS against the Vehtari floor. Returns True if the floor is met.

    Printed *before* the R-hat verdict it qualifies, because it decides whether that verdict
    carries information. See the ``ESS_FLOOR_*`` constants for why a failing floor
    invalidates a passing R-hat as much as a failing one.

    Both bulk- and tail-ESS are checked. Tail-ESS is usually the smaller and is the one that
    catches heavy-tail mixing failures, which is exactly the regime a prior bound induces in
    z space -- checking bulk alone would let the common failure through.

    ``labels`` names the offending parameter in the same vocabulary as the table above
    (``rm.labels``); without it the report falls back to the z-path names, which are correct
    but read as a different parameter to anyone skimming.
    """
    ess = np.minimum(np.asarray(report.ess_bulk, dtype=float),
                     np.asarray(report.ess_tail, dtype=float))
    worst = int(np.argmin(ess))
    min_ess = float(ess[worst])
    per_half = min_ess / (2 * n_chains)

    ok_total = min_ess >= ESS_FLOOR_TOTAL
    ok_half = per_half >= ESS_FLOOR_PER_SPLIT_HALF
    ok_chains = n_chains >= MIN_CHAINS
    ok = ok_total and ok_half and ok_chains

    print(f"\n=== ESS floor (Vehtari et al. 2021) ===")
    names = report.names if labels is None else labels
    print(f"min ESS over all parameters (bulk and tail) {min_ess:.0f}  "
          f"[{names[worst]}]")
    print(f"  total       >= {ESS_FLOOR_TOTAL:.0f}          "
          f"{min_ess:>8.0f}   {'ok' if ok_total else 'FAIL'}")
    print(f"  per split half >= {ESS_FLOOR_PER_SPLIT_HALF:.0f}      "
          f"{per_half:>8.1f}   {'ok' if ok_half else 'FAIL'}   "
          f"({n_chains} chains -> {2 * n_chains} split halves)")
    print(f"  chains      >= {MIN_CHAINS}            "
          f"{n_chains:>8d}   {'ok' if ok_chains else 'FAIL'}")
    if not ok:
        print("R-hat is NOT INTERPRETABLE below this floor -- and that cuts both ways: a "
              "passing R-hat here is an absence of evidence, not evidence of convergence. "
              "Draw more before reading anything from the table below.")
    return ok


def convergence_table(run_dir, rm, hmc_samples=None, con=None, report=None):
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

    This is the single most expensive stage in this file -- on a 2M-draw run (50 chains x
    40000 steps), ArviZ's rank-R-hat is ~30s and bulk+tail ESS another ~25s, scaling with
    total draws. That cost is exact and lives in ``nanolens`` (a read-only dependency), so
    it is not reduced here; ``main()`` prints it as its own timed stage instead, so it is
    attributable rather than mysterious.

    Percentiles are still quoted in constrained space -- that is the space the numbers get
    published in.

    With more than one source component this also runs the label-switching detector and
    the boundary-pinning check. The R-hat reported is the raw one unless switching was
    actually detected, in which case both raw and relabelled are shown.

    ``hmc_samples`` / ``con`` / ``report`` let a caller that already loaded ``hmc.npz``,
    computed ``to_constrained`` and ran ``diagnose_convergence`` (``main()`` does all three
    once, for the several stages that used to do them each) pass them in rather than pay
    again; ``None`` for any of them reproduces the original behaviour exactly. The
    ``report`` is the expensive one -- see the timing note above.
    """
    samples, con = hmc_and_constrained(run_dir, rm, hmc_samples, con)
    if samples is None:
        print("\nNo HMC samples; convergence diagnostics unavailable.")
        return None
    n_chains, n_draws, n_params = samples.shape

    switched, con_re = label_switching_check(con, rm.spans, rm.pnames)

    if report is None:
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
    prior_posterior_tension(rm.model, flat, rm.pnames)

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

    # Read this before the R-hat verdict: it says whether that verdict means anything.
    floor_ok = ess_floor_report(report, n_chains, labels=labels)

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
        if not floor_ok:
            print("But the ESS floor above failed, so this is not a pass -- it is an "
                  "undetermined result. Do not report this run as converged.")
    return worst


def theta_E_check(run_dir, rm, hmc_samples=None, con=None):
    """Compare the theta_E posterior against the config's measured clump radii.

    If every clump is a counter-image of one source, the Einstein radius should land inside
    their radial span. A posterior pinned against either prior edge means the radial scales
    are pulling the model apart, and that is the thing to report -- not the number.

    ``hmc_samples`` / ``con``: see ``convergence_table``'s docstring -- same reuse, same
    fallback to loading/transforming fresh when not given.
    """
    samples, con = hmc_and_constrained(run_dir, rm, hmc_samples, con)
    if samples is None:
        return
    con = con.reshape(-1, con.shape[-1])
    # "mass/0/theta_E", not bare "theta_E": with a foreground component the SIS also has a
    # theta_E, and the host EPL (always mass index 0) is the one this check is about.
    idx = column(rm.model, "mass/0/theta_E", kind="mass")
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


def map_convergence_check(run_dir, rm, hmc_samples=None, max_hmc_eval=3000):
    """Check that MAP actually found the maximum a posteriori point.

    MAP is by definition the highest-density point of the posterior, so an optimiser that
    stopped below typical HMC draws did not converge -- and since SVI starts from MAP and
    the sampler is seeded from SVI, that is a seeding problem for the whole pipeline, not a
    cosmetic one. Measured on DESI257 r2_s2l2: ``map.npz``'s ``best_lp = 16603.62`` while
    ordinary HMC draws score 17197-17253, a ~600 nat deficit invisible from ``best_lp``
    alone without this comparison.

    ``best_lp`` is independently RE-COMPUTED through ``rm.prob.log_prob`` rather than
    trusted as stored, and both are printed -- a disagreement between them would mean
    ``map.npz``'s own bookkeeping is stale, not that the model is wrong.
    ``ProbModel.log_prob(z)`` returns ``(log_posterior, red_chi2)`` with batch dims
    ``z.shape[:-1]``, so a single flat ``z`` yields a scalar and a batch of ``z`` yields one
    log-posterior per row -- the same convention ``log_like`` documents.

    HMC draws are thinned to at most ``max_hmc_eval`` via ``plot_corner.thin_to_cap`` (the
    same helper ``corner_plot`` uses) before being scored in one batch, so this stays cheap
    on a 500,000-draw run.

    Skipped with no output when ``map.npz`` or ``hmc.npz`` is absent -- there is nothing to
    compare.
    """
    mp = load_checkpoint(run_dir, "map")
    if mp is None:
        return
    hmc_samples, _ = hmc_and_constrained(run_dir, rm, hmc_samples, need_con=False)
    if hmc_samples is None:
        return

    best_z = np.asarray(mp["best_z"]).reshape(-1)
    stored_lp = float(np.asarray(mp["best_lp"]))
    map_lp, _ = rm.prob.log_prob(jnp.asarray(best_z))
    map_lp = float(np.asarray(map_lp))

    print("\n=== MAP vs. HMC log posterior ===")
    print(f"map.npz best_lp: stored {stored_lp:.3f}, recomputed {map_lp:.3f}")
    if abs(map_lp - stored_lp) > 1.0:
        print(f"NOTE: stored and recomputed best_lp disagree by "
              f"{abs(map_lp - stored_lp):.3f} nats -- map.npz's own bookkeeping may be "
              f"stale, independent of whatever the MAP-vs-HMC check below finds.")

    from plot_corner import thin_to_cap
    hmc_samples = np.asarray(hmc_samples)
    n_total = hmc_samples.shape[0] * hmc_samples.shape[1]
    thinned, factor = thin_to_cap(hmc_samples, max_hmc_eval)
    flat = thinned.reshape(-1, thinned.shape[-1])
    if factor > 1:
        print(f"Scoring {flat.shape[0]} HMC draws (thinned by {factor} from {n_total})")
    hmc_lp, _ = rm.prob.log_prob(jnp.asarray(flat))
    hmc_lp = np.asarray(hmc_lp)
    hmc_max, hmc_med = float(np.max(hmc_lp)), float(np.median(hmc_lp))
    print(f"HMC log posterior over {flat.shape[0]} draws: max {hmc_max:.3f}, "
          f"median {hmc_med:.3f}")

    passed, deficit = map_hmc_verdict(map_lp, hmc_max)
    if passed:
        print(f"OK: MAP's log posterior ({map_lp:.3f}) is >= the maximum scored HMC draw "
              f"({hmc_max:.3f}, margin {-deficit:.3f} nats).")
    else:
        print(f"FAILED: MAP's log posterior ({map_lp:.3f}) is {deficit:.3f} nats BELOW "
              f"the maximum scored HMC draw ({hmc_max:.3f}). MAP did not find the "
              f"posterior mode -- SVI (seeded from this MAP) and HMC (seeded from SVI) "
              f"both inherit a bad starting point.")

    chisq_hist = np.asarray(mp["chisq_hist"], dtype=np.float64).reshape(-1)
    if len(chisq_hist) >= 10:
        tail_len = max(1, len(chisq_hist) // 10)
        tail = chisq_hist[-tail_len:]
        slope = float(tail[-1] - tail[0]) / max(1, tail_len - 1)
        state = "still descending" if slope < 0 else "flat or rising"
        print(f"MAP chisq_hist tail slope (last {tail_len}/{len(chisq_hist)} steps): "
              f"{slope:+.5f}/step ({state} when MAP stopped)")


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


def ess_vs_draws_plot(run_dir, out_dir, rm, hmc_samples=None, report=None,
                      n_worst=6, n_prefix=8):
    """ESS against the number of draws used, on the run already on disk.

    The free test before spending a longer run (Vehtari et al. 2021 sec. 5, sec. 7;
    ``claude/guide/04-running-the-fit.md`` sec. 4.3 step 1). ESS should grow roughly
    linearly in the number of draws: a chain that is merely slow gets there by running
    longer, so 4x the draws buys ~4x the ESS. Sublinear growth, or a fall, means the chains
    are not sampling the same distribution and more draws will not fix it -- which sends you
    to the model or the surrogate without first paying for the 4x run to find out.

    The read is stated in the printed output as well as drawn, because the figure alone
    invites eyeballing a log-log slope. The slope is fitted and reported.

    Cost control: computed on the ``n_worst`` worst-R-hat parameters only, at ``n_prefix``
    prefix lengths. ESS is per-column, so this is ``n_prefix/2 * n_worst/n_params`` times
    the cost of the full ESS pass -- about 0.75x on a 36-parameter run, rather than 4.5x if
    every column were carried. The worst-R-hat parameters are the ones the question is about;
    a parameter already at R-hat 1.00 has nothing to diagnose.

    Prefixes are taken from the START of each chain, keeping all chains. That is the honest
    simulation of "what would a shorter run have looked like" -- and it means the earliest
    prefixes include the warmup-adjacent draws that a shorter run would also have included.
    """
    samples, _ = hmc_and_constrained(run_dir, rm, hmc_samples, need_con=False)
    if samples is None:
        return None
    n_chains, n_draws, n_params = samples.shape
    if n_draws < 8 * MIN_CHAINS:
        print(f"\nESS-vs-draws: {n_draws} draws per chain is too short to subdivide; "
              f"skipping.")
        return None

    if report is None:
        report = diagnose_convergence(samples, model=rm.model)
    idx = worst_rhat_columns(report, n_worst)
    names = [rm.labels[i] for i in idx]

    # Geometric spacing: the interesting behaviour (a curve flattening) shows up as a
    # departure from the unit slope on log-log axes, and linear spacing wastes most of the
    # points on the top half of the run where nothing changes.
    lengths = np.unique(np.geomspace(max(64, n_draws // 64), n_draws, n_prefix).astype(int))

    ess = np.full((len(lengths), len(idx)), np.nan)
    for k, n in enumerate(lengths):
        sub = np.ascontiguousarray(samples[:, :n, idx])
        rep = diagnose_convergence(sub, z_param_names=[report.names[i] for i in idx])
        ess[k] = np.minimum(np.asarray(rep.ess_bulk, dtype=float),
                            np.asarray(rep.ess_tail, dtype=float))

    total = lengths * n_chains
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for j, name in enumerate(names):
        ax.plot(total, ess[:, j], marker="o", ms=3, label=name)
    # Reference: exactly linear growth anchored at the final point of the worst parameter.
    ref_j = int(np.argmin(ess[-1]))
    ax.plot(total, ess[-1, ref_j] * total / total[-1], "k--", lw=1,
            label="linear growth")
    ax.axhline(ESS_FLOOR_TOTAL, color="r", ls=":", lw=1,
               label=f"floor {ESS_FLOOR_TOTAL:.0f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total draws used (all chains)")
    ax.set_ylabel("min(bulk, tail) ESS")
    ax.set_title(f"ESS vs draws -- {n_worst} worst-Rhat parameters")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = Path(out_dir) / "ess_vs_draws.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # The slope over the last decade-ish is what the decision hangs on; an early transient
    # while the chains are still finding the typical set would bias a whole-range fit.
    half = len(lengths) // 2
    lg = np.log(total[half:])
    print("\n=== ESS vs draws (the free test before a longer run) ===")
    print(f"{'parameter':<{max(len(n) for n in names)}}{'ESS now':>10}{'slope':>8}")
    slopes = []
    for j, name in enumerate(names):
        y = ess[half:, j]
        good = np.isfinite(y) & (y > 0)
        slope = (np.polyfit(lg[good], np.log(y[good]), 1)[0]
                 if good.sum() >= 2 else np.nan)
        slopes.append(slope)
        print(f"{name:<{max(len(n) for n in names)}}{ess[-1, j]:>10.0f}{slope:>8.2f}")
    worst_slope = float(np.nanmin(slopes)) if np.any(np.isfinite(slopes)) else np.nan
    if not np.isfinite(worst_slope):
        print("Slope undefined (ESS non-positive or non-finite somewhere) -- read the "
              "figure directly.")
    elif worst_slope > 0.7:
        print(f"Slope ~{worst_slope:.2f}, near-linear: ESS is still growing with draws, so "
              f"a longer run buys what it costs. 4x the draws for ~4x the ESS.")
    else:
        print(f"Slope ~{worst_slope:.2f}, SUBLINEAR: more draws will not fix this. Go to "
              f"the model or the surrogate (guide 04 sec. 4.3 step 3), not to a longer run.")
    return path


def rank_plot(run_dir, out_dir, rm, hmc_samples=None, report=None, n_worst=6, n_bins=20,
              max_chains=10):
    """Pooled-rank histograms, one panel per chain, for the worst-R-hat parameters.

    Vehtari et al. recommend these over trace plots: ranks are computed over the pooled
    draws of all chains, so if every chain targets the same distribution each chain's ranks
    are uniform. A chain that is offset shows as a sloped or one-sided histogram, and it
    shows *which* chain -- which R-hat, a single number over all chains, cannot say. This
    catches a single misbehaving chain at R-hat values that clear a loose gate.

    Computed here rather than through ``arviz.plot_rank``: in arviz 1.2 that function draws
    a rank Delta-ECDF, not the per-chain histogram described above, and returns an
    arviz-plots ``PlotCollection`` rather than a matplotlib figure. The statistic itself is
    a pooled ``argsort``-of-``argsort``, which is exact and worth no dependency.

    Ranks are taken in unconstrained space, matching where R-hat is computed. Every prior's
    scalar z->x map is strictly monotone, so per-parameter ranks are identical in either
    space; taking them in z keeps the figure aligned with the table it explains.

    At most ``max_chains`` chains are drawn. Runs here go to 50 chains, and a 6x50 grid of
    panels is ~56 inches wide and unreadable. The chains shown are the ones that deviate
    most from uniform, by the chi-square statistic of their rank histogram, so the panel
    that would have revealed the problem is the one that survives the cap -- a first-N slice
    could drop the only bad chain. Ranks are always pooled over ALL chains before the cut;
    only the drawing is subset, so each panel still means "this chain against the whole
    run". The chain indices drawn are printed, and they index the original chain axis.
    """
    samples, _ = hmc_and_constrained(run_dir, rm, hmc_samples, need_con=False)
    if samples is None:
        return None
    n_chains, n_draws, n_params = samples.shape

    if report is None:
        report = diagnose_convergence(samples, model=rm.model)
    idx = worst_rhat_columns(report, n_worst)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    expected = n_draws / n_bins

    # Pooled ranks first, for every chain: the cut below is about what to draw, never about
    # what the ranks are relative to.
    counts = np.empty((len(idx), n_chains, n_bins))
    for r, i in enumerate(idx):
        x = samples[:, :, i].reshape(-1)
        # Fractional pooled rank in [0, 1): scattering arange into argsort order gives each
        # draw its rank among all n_chains*n_draws of them.
        ranks = np.empty(x.size, dtype=float)
        ranks[np.argsort(x, kind="stable")] = np.arange(x.size)
        ranks = (ranks / x.size).reshape(n_chains, n_draws)
        for c in range(n_chains):
            counts[r, c] = np.histogram(ranks[c], bins=edges)[0]

    # Chi-square of each chain's histogram against uniform, summed over the plotted
    # parameters. Largest first, so a cut keeps the chains worth looking at.
    chisq = ((counts - expected) ** 2 / expected).sum(axis=(0, 2))
    shown = sorted(int(c) for c in np.argsort(chisq)[::-1][:min(max_chains, n_chains)])

    n_rows, n_cols = len(idx), len(shown)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.1 * n_cols + 1.5, 1.5 * n_rows),
        squeeze=False, sharex=True,
    )
    centres = 0.5 * (edges[:-1] + edges[1:])
    for r, i in enumerate(idx):
        for k, c in enumerate(shown):
            ax = axes[r][k]
            ax.bar(centres, counts[r, c], width=1.0 / n_bins, color="0.4")
            ax.axhline(expected, color="r", ls=":", lw=0.8)
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"chain {c}", fontsize=7)
            if k == 0:
                ax.set_ylabel(f"{rm.labels[i]}\nRhat {report.rhat[i]:.3f}", fontsize=6)
    for k in range(n_cols):
        axes[-1][k].set_xlabel("rank", fontsize=6)
        axes[-1][k].tick_params(labelsize=5)
    fig.suptitle("Pooled-rank histograms -- uniform if the chains agree", fontsize=9)
    fig.tight_layout()
    path = Path(out_dir) / "rank_plots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"\nRank plots ({len(idx)} worst-Rhat parameters x {n_cols} chains): {path}")
    if n_cols < n_chains:
        print(f"Showing the {n_cols} least-uniform of {n_chains} chains "
              f"(indices {list(shown)}); ranks are pooled over all {n_chains}.")
    print("Uniform panels mean the chains agree. A sloped or one-sided panel names the "
          "chain that disagrees -- something Rhat, one number over all chains, cannot do.")
    return path


def corner_plot(run_dir, out_dir, rm, subset=None, hmc_samples=None, con=None):
    """Corner plot of the constrained posterior.

    Defaults to the mass parameters, which exist under every model variant. All 22 (or 36)
    in one figure is unreadable at print size. Columns are located structurally by
    ``column``, so nothing here depends on a parameter's index.

    The figure itself is drawn by ``plot_corner.make_corner`` rather than by a second
    ``corner.corner`` call site here. Those two call sites had drifted: ``make_corner``
    carries a guard that refuses a corner with fewer draws than it has dimensions (which
    ``corner`` renders as an unreadable mess of empty panels rather than raising), and this
    function did not. Sharing the renderer is what makes the guard apply to both.

    Thinned to ``plot_corner.MAX_SAMPLES_FULL`` draws via ``plot_corner.thin_to_cap``. On
    the largest run (2M draws) this figure used to cost ~2.4s in ``corner.corner`` and
    ~14.4s in ``savefig``; thinning is for the *figure* only -- it never touches the
    R-hat/ESS table or the percentiles, which are computed on every draw. The thinning
    factor is printed so the figure never silently implies more draws than it shows.

    ``plot_corner`` is imported locally, not at module scope, to avoid a
    diagnose<->plot_corner import cycle.

    ``hmc_samples`` / ``con``: see ``convergence_table``'s docstring -- same reuse.
    """
    samples, con_3d = hmc_and_constrained(run_dir, rm, hmc_samples, con)
    if samples is None:
        return None
    n_chains, n_draws, n_params = con_3d.shape

    from plot_corner import thin_to_cap, make_corner, MAX_SAMPLES_FULL
    thinned, factor = thin_to_cap(con_3d, MAX_SAMPLES_FULL)
    if factor > 1:
        print(f"Corner plot: thinning {n_chains * n_draws} -> "
              f"{thinned.shape[0] * thinned.shape[1]} draws (factor {factor})")
    con_flat = thinned.reshape(-1, n_params)

    if subset is None:
        # "mass/0/theta_E" not bare "theta_E": with a foreground component the SIS also
        # has a theta_E, and this default subset is about the host EPL specifically.
        # gamma/e1/e2 stay bare -- the SIS carries neither, so they are still unique.
        subset = [
            column(rm.model, s, kind="mass")
            for s in ("mass/0/theta_E", "gamma", "e1", "e2", "gamma1", "gamma2")
        ]
    idx = list(subset)

    try:
        return make_corner(
            con_flat[:, idx],
            labels=[rm.labels[i] for i in idx],
            truths=None,
            out_path=Path(out_dir) / "corner.png",
            show_titles=True,
            dpi=150,
            # Held at the size corner chose for this 6-parameter figure before the two call
            # sites were merged, so the output is unchanged by the refactor rather than
            # quietly reformatted. make_corner's default (0.9 in/param) is tuned for the
            # 36-parameter full corner and would shrink this one.
            figsize=(12, 12),
        )
    except ImportError:
        print("corner not installed; skipping corner plot.")
        return None


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

    t_start = time.perf_counter()

    # Reconstruct the *same* model the run was fitted with. A run that predates the
    # scene-API port is refused here rather than silently mis-decoded.
    t0 = time.perf_counter()
    rm = rebuild_model(
        run_dir,
        lens=args.lens,
        theta_e_center=args.theta_e_center,
        theta_e_width=args.theta_e_width,
    )
    print(f"[timing] rebuild_model: {time.perf_counter() - t0:.1f}s")

    # Load hmc.npz once here and thread the array through point_estimate,
    # convergence_table, theta_E_check and corner_plot below, all of which used to load it
    # separately -- four decompressions of what can be a >250MB file (~1.5s each on the
    # largest run to date).
    t0 = time.perf_counter()
    hmc_ckpt = load_checkpoint(run_dir, "hmc")
    hmc_samples = None if hmc_ckpt is None else np.asarray(hmc_ckpt["samples"])
    shape_note = "" if hmc_samples is None else f" {hmc_samples.shape}"
    print(f"[timing] load hmc.npz: {time.perf_counter() - t0:.1f}s{shape_note}")

    z_unc, label = point_estimate(run_dir, hmc_samples=hmc_samples)
    print(f"Point estimate: {label}")

    t0 = time.perf_counter()
    im_sim = simulate_from_unconstrained(rm, z_unc)
    chi_squares, res_path = residual_figure(
        out_dir, rm.cfg, rm.model, rm.prob, rm.image, rm.prob.error_map, im_sim, label,
        z_unc, mask=rm.mask,
    )
    print(f"[timing] residual figure: {time.perf_counter() - t0:.1f}s")
    n_kept = rm.image.size if rm.mask is None else int(np.asarray(rm.mask).sum())
    print(f"Full-field reduced chi-squared at point estimate: {chi_squares.full:.3f} "
          f"(over {n_kept}/{rm.image.size} unmasked pixels)")
    n_core_excluded = chi_squares.n_full - chi_squares.n_core_masked
    print(
        f"Core-masked reduced chi-squared (REPORTING ONLY -- core pixels remain in the "
        f"likelihood, this is a display-only exclusion): {chi_squares.core_masked:.3f} "
        f"(radius {chi_squares.radius_arcsec:.3f}\" = {chi_squares.radius_px:.1f}px, "
        f"{n_core_excluded} px excluded, {chi_squares.n_core_masked}/{rm.image.size} "
        f"pixels kept)"
    )
    print(f"Residual figure: {res_path}")

    # MAP is by definition the highest-density point; if HMC draws outscore it, MAP did not
    # converge and SVI/HMC inherited a bad seed. Cheap (one batched log_prob call), and
    # worth reading before svi_report -- a failed MAP casts doubt on everything seeded from
    # it, including the SVI covariance svi_report is about to judge.
    t0 = time.perf_counter()
    map_convergence_check(run_dir, rm, hmc_samples=hmc_samples)
    print(f"[timing] MAP vs HMC log posterior: {time.perf_counter() - t0:.1f}s")

    # Before Rhat: an ill-conditioned SVI covariance invalidates the HMC output outright,
    # so this line has to be read first.
    svi_report(run_dir)

    # Compute the unconstrained -> constrained transform once and share it with
    # convergence_table, theta_E_check and corner_plot below, rather than recomputing the
    # same (n_draws, n_params) transform three times (~1.4s each on the largest run).
    con = None
    if hmc_samples is not None:
        t0 = time.perf_counter()
        con = to_constrained(rm.model, hmc_samples.reshape(-1, hmc_samples.shape[-1]))
        print(f"[timing] to_constrained: {time.perf_counter() - t0:.1f}s")

    # This is the expensive stage. ArviZ's rank-normalized R-hat and bulk/tail ESS are
    # exact and live in the read-only nanolens package, so their cost is not reduced here
    # -- only reported, so it is attributable rather than mysterious. It scales with total
    # draws: ~30s rank-Rhat + ~25s bulk/tail-ESS on a 2M-draw run (50 chains x 40000 steps),
    # A100. See convergence_table's docstring.
    report = None
    if hmc_samples is not None:
        t0 = time.perf_counter()
        report = diagnose_convergence(hmc_samples, model=rm.model)
        print(f"[timing] ArviZ rank-Rhat/ESS: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    worst_rhat = convergence_table(
        run_dir, rm, hmc_samples=hmc_samples, con=con, report=report)
    print(f"[timing] convergence table: {time.perf_counter() - t0:.1f}s")

    theta_E_check(run_dir, rm, hmc_samples=hmc_samples, con=con)

    p = svi_loss_plot(run_dir, out_dir)
    if p:
        print(f"SVI loss: {p}")

    # The two diagnostics Vehtari et al. recommend for deciding what to do about an elevated
    # Rhat, both computed from the draws already on disk. ESS-vs-draws answers "would a
    # longer run help?" without paying for the longer run; the rank plots answer "which
    # chain?", which Rhat cannot. See guide 04 sec. 4.3.
    t0 = time.perf_counter()
    p = ess_vs_draws_plot(run_dir, out_dir, rm, hmc_samples=hmc_samples, report=report)
    print(f"[timing] ESS vs draws: {time.perf_counter() - t0:.1f}s")
    if p:
        print(f"ESS vs draws: {p}")

    t0 = time.perf_counter()
    rank_plot(run_dir, out_dir, rm, hmc_samples=hmc_samples, report=report)
    print(f"[timing] rank plots: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    p = corner_plot(run_dir, out_dir, rm, hmc_samples=hmc_samples, con=con)
    print(f"[timing] corner plot: {time.perf_counter() - t0:.1f}s")
    if p:
        print(f"Corner plot: {p}")

    if worst_rhat is not None and worst_rhat > RHAT_THRESHOLD:
        print("\nReminder: quote R-hat alongside any parameter value. A theta_E without a "
              "convergence diagnostic is not a measurement.")

    print(f"[timing] total: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
