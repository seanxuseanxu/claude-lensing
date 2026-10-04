"""Unit tests for three pure/near-pure checks added 2026-08-09:

    - ``diagnose.map_hmc_verdict``: MAP must score >= the best HMC draw.
    - ``diagnose.boundary_pinning_check``: TruncatedNormal bounds, not just Uniform.
    - ``run_fit.merge_prior_metadata``: a ``--stages hmc`` rerun must not overwrite prior
      stages' recorded metadata with today's argparse defaults (KNOWN_ISSUES.md item 1).

Plus, added 2026-08-11, one test that is NOT CPU-only/scene-free like the rest of this
file: ``test_rebuild_model_uses_recorded_supersample_over_todays_config`` needs DESI257's
real data files and builds a real model, the same way ``test_foreground_roundtrip.py``
does, because it exercises ``diagnose.rebuild_model`` end to end. See its own docstring.

Run directly (pytest is not installed in this env):

    PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_diagnose_and_metadata.py
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

import diagnose
import lenses
import run_fit

from tensorflow_probability.substrates.jax import distributions as tfd


# -- map_hmc_verdict ---------------------------------------------------------------------

def test_map_hmc_verdict_passes_when_map_is_highest():
    passed, deficit = diagnose.map_hmc_verdict(map_lp=17300.0, hmc_max_lp=17253.0)
    assert passed, "MAP scoring above the max HMC draw must pass"
    assert abs(deficit - (17253.0 - 17300.0)) < 1e-9


def test_map_hmc_verdict_fails_on_desi257_r2_s2l2_numbers():
    # The confirmed instance from the task: map.npz best_lp = 16603.62 against HMC draws
    # scoring 17197-17253. This is the case the check exists to catch.
    passed, deficit = diagnose.map_hmc_verdict(map_lp=16603.62, hmc_max_lp=17253.0)
    assert not passed, "MAP scoring ~600 nats below the best HMC draw must fail"
    assert abs(deficit - (17253.0 - 16603.62)) < 1e-9
    assert deficit > 600  # the measured deficit, so the printed number stays meaningful


def test_map_hmc_verdict_equal_scores_pass():
    # deficit == 0 (MAP land exactly on the best HMC draw) is the boundary case and must
    # pass -- ">=" per the docstring/spec, not strictly ">".
    passed, deficit = diagnose.map_hmc_verdict(map_lp=100.0, hmc_max_lp=100.0)
    assert passed and deficit == 0.0


# -- boundary_pinning_check: TruncatedNormal coverage ------------------------------------

@dataclass
class FakePrior:
    model: dict


@dataclass
class FakeModel:
    """Just enough of a ``LensModel`` for ``boundary_pinning_check``: ``prior.model``
    (the JointDistributionNamed's dict of per-parameter distributions) and ``z_labels``.
    """

    prior: FakePrior
    z_labels: Optional[list] = None


def test_pinning_flags_truncated_normal_at_upper_bound():
    # DESI257's actual gamma prior (priors.py:266): TruncatedNormal(2.0, 0.25, 1.0, 3.0).
    # Round-2 mode B's posterior peaked near 2.86-2.885, against the 3.0 ceiling.
    prior_dict = {
        "gamma": tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0),
        "theta_E": tfd.Uniform(0.5, 3.0),
    }
    pnames = ["gamma", "theta_E"]
    model = FakeModel(prior=FakePrior(prior_dict), z_labels=pnames)

    n = 500
    con_flat = np.column_stack([
        np.full(n, 2.98),  # 99% of [1, 3] -- pinned at the TruncatedNormal's upper bound
        np.full(n, 1.75),  # 50% of [0.5, 3] -- not pinned
    ])

    buf = io.StringIO()
    with redirect_stdout(buf):
        diagnose.boundary_pinning_check(model, con_flat, pnames)
    out = buf.getvalue()

    assert "TruncatedNormal" in out, "the TruncatedNormal row must be labelled as such"
    assert "Uniform" in out, "the Uniform row must still be labelled as such"
    gamma_line = next(l for l in out.splitlines() if l.startswith("gamma"))
    assert "TruncatedNormal" in gamma_line and "PINNED at upper bound" in gamma_line
    theta_line = next(l for l in out.splitlines() if l.startswith("theta_E"))
    assert "PINNED" not in theta_line
    assert "pinned parameter is set by its prior" in out


def test_pinning_silent_when_truncated_normal_not_pinned():
    prior_dict = {"gamma": tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0)}
    pnames = ["gamma"]
    model = FakeModel(prior=FakePrior(prior_dict), z_labels=pnames)

    con_flat = np.full((500, 1), 2.0)  # dead centre of [1, 3]

    buf = io.StringIO()
    with redirect_stdout(buf):
        diagnose.boundary_pinning_check(model, con_flat, pnames)
    out = buf.getvalue()

    assert "TruncatedNormal" in out
    assert "PINNED" not in out
    assert "pinned parameter is set by its prior" not in out


# -- prior_posterior_tension ---------------------------------------------------------------

def test_tension_calibrates_against_a_known_gaussian():
    """A standard Normal prior with the posterior median at 3.0 must report 3.00 sigma.

    This is the calibration case: the answer is known in closed form, so it tests the
    instrument rather than agreeing with it.
    """
    prior_dict = {"x": tfd.Normal(0.0, 1.0)}
    pnames = ["x"]
    model = FakeModel(prior=FakePrior(prior_dict), z_labels=pnames)
    con_flat = np.full((500, 1), 3.0)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rows = diagnose.prior_posterior_tension(model, con_flat, pnames, threshold=2.0)

    assert len(rows) == 1
    assert abs(rows[0][2] - 3.0) < 0.01, f"expected 3.00 sigma, got {rows[0][2]}"
    assert "above prior centre" in buf.getvalue()


def test_tension_reports_the_low_side_and_scales_with_prior_width():
    """Same displacement, a prior 10x wider: 3 sigma must become 0.3 sigma, and the side
    must be reported as below rather than above."""
    model = FakeModel(prior=FakePrior({"x": tfd.Normal(0.0, 10.0)}), z_labels=["x"])
    rows = diagnose.prior_posterior_tension(model, np.full((500, 1), -3.0), ["x"], threshold=0.1)
    assert len(rows) == 1
    assert abs(rows[0][2] - 0.3) < 0.01
    assert rows[0][4] is True, "median below the prior centre must report the low side"


def test_tension_finds_desi234_r3_fg_gamma_where_pinning_cannot():
    """The case the check was built for: gamma = 2.9514 under TruncatedNormal(2, 0.25, 1, 3).

    ``boundary_pinning_check`` must stay silent -- 97.6% of range is under its 98% threshold,
    and correctly so, since the posterior sits ~10 sd from the bound. The tension check must
    speak. Both assertions matter: the point is that they answer different questions.
    """
    prior_dict = {"gamma": tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0)}
    pnames = ["gamma"]
    model = FakeModel(prior=FakePrior(prior_dict), z_labels=pnames)
    rng = np.random.default_rng(0)
    con_flat = rng.normal(2.9514, 0.0049, size=(2000, 1))

    buf = io.StringIO()
    with redirect_stdout(buf):
        diagnose.boundary_pinning_check(model, con_flat, pnames)
    assert "PINNED" not in buf.getvalue(), "gamma is ~10 sd from the bound; not pinned"

    rows = diagnose.prior_posterior_tension(model, con_flat, pnames)
    assert len(rows) == 1 and rows[0][0] == "gamma"
    assert rows[0][2] > 3.0, f"expected >3 sigma of prior tension, got {rows[0][2]}"
    assert rows[0][3] < 0.05, "posterior should be >20x narrower than the prior"


def test_tension_silent_on_a_healthy_parameter():
    """DESI234 r2_s2l2's gamma = 2.199 on the same prior: 0.8 prior sigma, no report."""
    model = FakeModel(prior=FakePrior({"gamma": tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0)}),
                      z_labels=["gamma"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rows = diagnose.prior_posterior_tension(model, np.full((500, 1), 2.199), ["gamma"])
    assert rows == []
    assert buf.getvalue() == "", "a healthy parameter must produce no output at all"


def test_tension_handles_a_skewed_prior_in_quantile_space():
    """LogNormal is why the measure is a quantile and not (x - mu) / sigma: the median of a
    LogNormal(log 1.43, 0.25) is exp(log 1.43) = 1.43 by construction, so a posterior sitting
    there is 0 sigma of tension however asymmetric the distribution's moments are."""
    model = FakeModel(prior=FakePrior({"theta_E": tfd.LogNormal(np.log(1.43), 0.25)}),
                      z_labels=["theta_E"])
    rows = diagnose.prior_posterior_tension(model, np.full((500, 1), 1.43), ["theta_E"],
                                            threshold=0.05)
    assert rows == [], "the prior's own median must register as zero tension"


# -- rebuild_model: supersample is per-run, not read from today's config -----------------

def test_rebuild_model_uses_recorded_supersample_over_todays_config():
    """KNOWN_ISSUES.md item 8's fix, extended from ``mask_circles`` to ``supersample``: a
    run recorded at supersample=2 must rebuild at 2 even when the lens's config.py says 4
    *today*. This is the exact hazard the task that added this test exists to close --
    DESI257's config was about to move 2 -> 4, which would otherwise have silently
    corrupted re-diagnosis of the three archived runs (r1_s1l1, r2_s2l2, r3_unmasked),
    all fitted at ss=2.

    Needs DESI257's real data files and builds a real model -- unlike the rest of this
    file, which is pure/scene-free -- because the thing under test is
    ``diagnose.rebuild_model`` end to end, through ``data.load``'s ``supersample=``
    override. ``lenses.load`` is monkeypatched to hand back a config claiming
    supersample=4 (simulating "today's config changed") without touching the file on
    disk, while the metadata.json this test writes records supersample=2 -- the value the
    run was actually fitted at.
    """
    real_cfg = lenses.load("DESI257")
    assert real_cfg.supersample == 2, (
        "this test assumes DESI257's config is at its historical supersample=2; if it "
        "has since been deliberately raised, update this assertion and the scenario it "
        "guards against"
    )
    bumped_cfg = replace(real_cfg, supersample=4)

    orig_load = lenses.load
    lenses.load = lambda name: bumped_cfg if name == "DESI257" else orig_load(name)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            metadata = {
                "api": "scene",
                "lens": "DESI257",
                "n_source": 2,
                "n_lens_light": 2,
                "mask_circles": None,
                "n_masked_pixels": real_cfg.num_pix * real_cfg.num_pix,
                "supersample": 2,
            }
            with open(Path(tmp) / "metadata.json", "w") as f:
                json.dump(metadata, f)
            rm = diagnose.rebuild_model(tmp, quiet=True)
    finally:
        lenses.load = orig_load

    assert rm.sim_config.supersample == 2, (
        f"rebuild_model used supersample={rm.sim_config.supersample}, but the run's "
        f"metadata.json recorded 2 and today's (monkeypatched) config says 4 -- the "
        f"recorded value must win"
    )


def test_rebuild_model_falls_back_to_config_when_supersample_key_absent():
    """A run written before ``run_fit.py`` recorded ``supersample`` (no such key in its
    metadata.json) must fall back to the lens's current config, same as ``mask_circles``
    did before KNOWN_ISSUES.md item 8's fix. Uses DESI257 unmodified, so the fallback
    value and the config value agree and there is nothing to silently get wrong.
    """
    real_cfg = lenses.load("DESI257")
    with tempfile.TemporaryDirectory() as tmp:
        metadata = {
            "api": "scene",
            "lens": "DESI257",
            "n_source": 2,
            "n_lens_light": 2,
            "mask_circles": None,
            "n_masked_pixels": real_cfg.num_pix * real_cfg.num_pix,
            # no "supersample" key
        }
        with open(Path(tmp) / "metadata.json", "w") as f:
            json.dump(metadata, f)
        rm = diagnose.rebuild_model(tmp, quiet=True)

    assert rm.sim_config.supersample == real_cfg.supersample


# -- run_fit.merge_prior_metadata ---------------------------------------------------------

def _base_config(**overrides):
    """A stand-in for the fully-populated ``config`` dict ``run_fit.main`` builds from
    argparse before writing metadata.json -- only the fields ``merge_prior_metadata``
    reads/writes matter here."""
    cfg = dict(
        api="scene", lens="DESI257",
        n_samples=1024, map_steps=350,           # map defaults
        n_vi=1000, svi_steps=1500, svi_peak_lr=2e-3, svi_warmup_frac=0.1,  # svi defaults
        sampler="hmc", n_hmc=50, burnin=250, num_results=750,
        target_acceptance=0.9, desired_energy_variance=5e-4, max_leapfrog=30,  # hmc
        n_devices=1, stages=["hmc"],
        timings={"hmc": 123.4},
    )
    cfg.update(overrides)
    return cfg


def test_merge_carries_forward_unrun_stage_fields():
    # The confirmed DESI257 r2_s2l2 instance: an hmc-only rerun's argparse defaults must
    # not clobber the svi_peak_lr the checkpoint was actually made with.
    prev_metadata = {
        "n_samples": 900, "map_steps": 300,
        "svi_peak_lr": 1e-3, "n_vi": 800, "svi_steps": 1200, "svi_warmup_frac": 0.1,
        "timings": {"map": 10.0, "svi": 20.0},
    }
    config = _base_config(svi_peak_lr=2e-3, n_samples=1024)  # today's argparse defaults
    merged = run_fit.merge_prior_metadata(config, prev_metadata, stages=["hmc"])

    assert merged["svi_peak_lr"] == 1e-3, "svi_peak_lr must come from prev_metadata, not today's default"
    assert merged["n_vi"] == 800
    assert merged["svi_steps"] == 1200
    assert merged["n_samples"] == 900
    assert merged["map_steps"] == 300


def test_merge_keeps_this_invocations_ran_stage_fields():
    prev_metadata = {"sampler": "mclmc", "n_hmc": 30}
    config = _base_config(sampler="hmc", n_hmc=50)
    merged = run_fit.merge_prior_metadata(config, prev_metadata, stages=["hmc"])
    # hmc WAS re-run this invocation, so its fields are this call's values, not prev's.
    assert merged["sampler"] == "hmc"
    assert merged["n_hmc"] == 50


def test_merge_does_not_fabricate_missing_prior_fields():
    # prev_metadata predates svi_warmup_frac -- merge must not invent a value for it.
    prev_metadata = {"svi_peak_lr": 1e-3}
    config = _base_config(svi_warmup_frac=0.25)
    merged = run_fit.merge_prior_metadata(config, prev_metadata, stages=["hmc"])
    assert merged["svi_peak_lr"] == 1e-3
    assert merged["svi_warmup_frac"] == 0.25, "a field absent from prev_metadata keeps this invocation's value"


def test_merge_stages_field_is_this_invocations_stages():
    prev_metadata = {"stages": ["map", "svi", "hmc"]}
    config = _base_config(stages=["hmc"])
    merged = run_fit.merge_prior_metadata(config, prev_metadata, stages=["hmc"])
    assert merged["stages"] == ["hmc"]


def test_merge_timings_preserves_prior_stages_and_overlays_new():
    prev_metadata = {"timings": {"map": 10.0, "svi": 20.0}}
    config = _base_config(stages=["hmc"])
    config["timings"] = {"hmc": 99.0}
    merged = run_fit.merge_prior_metadata(config, prev_metadata, stages=["hmc"])
    assert merged["timings"] == {"map": 10.0, "svi": 20.0, "hmc": 99.0}


def test_merge_no_prior_metadata_is_a_no_op():
    # First-ever run of a fresh out_dir: no metadata.json yet, prev_metadata == {}.
    config = _base_config(stages=["map", "svi", "hmc"])
    merged = run_fit.merge_prior_metadata(config, {}, stages=["map", "svi", "hmc"])
    assert merged == config


# -- core_excluded_keep / ChiSquares (item 5's centre-masked chi-squared) ----------------
#
# The core mask is a REPORTING-only exclusion (KNOWN_ISSUES.md item 5): it never touches
# the likelihood. These tests are all against ``diagnose.core_excluded_keep``, the pure
# helper, plus one test of ``residual_figure``'s own source (see test_check_theta_E.py for
# the precedent of pinning an expression rather than exercising a function that needs a
# full model/prob to invoke) to confirm the radius is DERIVED, not a second hardcoded 8.

def _brute_force_disc_excluded(shape, centroid_pix, radius_px, keep=None):
    """Independent (non-vectorised-clever) computation of the same disc, so the test does
    not just re-run ``core_excluded_keep``'s own arithmetic against itself."""
    keep = np.ones(shape, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    out = keep.copy()
    for r in range(shape[0]):
        for c in range(shape[1]):
            d = ((r - centroid_pix[0]) ** 2 + (c - centroid_pix[1]) ** 2) ** 0.5
            if d < radius_px:
                out[r, c] = False
    return out


def test_core_excluded_keep_matches_independent_disc_count():
    shape = (41, 41)
    centroid = (20.0, 20.0)  # dead centre of an odd-sized grid
    radius_px = 5.0
    got = diagnose.core_excluded_keep(shape, centroid, radius_px)
    expected = _brute_force_disc_excluded(shape, centroid, radius_px)
    assert got.dtype == bool
    assert got.shape == shape
    assert np.array_equal(got, expected), "excluded pixel set disagrees with an independent computation"
    n_excluded = int((~got).sum())
    assert n_excluded == int((~expected).sum())
    assert n_excluded > 0


def test_core_excluded_keep_is_symmetric_about_a_centred_centroid():
    shape = (41, 41)
    centroid = (20.0, 20.0)
    radius_px = 6.0
    keep = diagnose.core_excluded_keep(shape, centroid, radius_px)
    # 180-degree rotation about the centre pixel must map the excluded disc onto itself.
    assert np.array_equal(keep, keep[::-1, ::-1]), "disc about a centred centroid must be point-symmetric"


def test_core_excluded_keep_handles_a_noninteger_centroid():
    # DESI257's actual centroid_pix, per config.py: (59.397 - 20, 59.236 - 20).
    shape = (79, 79)
    centroid = (39.397, 39.236)
    radius_px = 8.0
    got = diagnose.core_excluded_keep(shape, centroid, radius_px)
    expected = _brute_force_disc_excluded(shape, centroid, radius_px)
    assert np.array_equal(got, expected)
    n_excluded = int((~got).sum())
    # Sanity: a disc of radius 8px covers pi*64 ~ 201 px; a non-integer centroid near a
    # pixel centre should land close to that, well inside a generous band.
    assert 150 < n_excluded < 260, f"excluded count {n_excluded} is not a sensible disc for r=8px"


def test_core_excluded_keep_composes_with_an_existing_mask():
    shape = (41, 41)
    centroid = (20.0, 20.0)
    radius_px = 5.0
    keep = np.ones(shape, dtype=bool)
    # Exclude a stripe far from the centroid, unrelated to the core disc.
    keep[0, :] = False
    got = diagnose.core_excluded_keep(shape, centroid, radius_px, keep=keep)
    # Pixels already excluded by `keep` must stay excluded.
    assert not got[0, :].any(), "pixels already excluded by keep must not be revived"
    # The core disc must additionally be excluded on top of that.
    assert not got[20, 20], "the centroid pixel itself must be excluded"
    # Pixels outside both the stripe and the disc must remain kept.
    assert got[20, 35], "a pixel far from both the stripe and the disc must remain kept"


def test_core_masking_materially_lowers_chi2_on_an_injected_core_excess():
    # The discriminating case: without this, a bug that returned the same number for
    # `full` and `core_masked` would pass every test above.
    shape = (60, 60)
    centroid = (30.0, 30.0)
    radius_px = 8.0
    rng = np.random.default_rng(0)
    norm_res = rng.normal(scale=1.0, size=shape)  # ordinary background residual, ~chi2_nu 1

    keep_all = np.ones(shape, dtype=bool)
    disc = _brute_force_disc_excluded(shape, centroid, radius_px, keep=keep_all)
    core_pixels = ~disc  # True where the disc excludes, i.e. the injected-excess region
    norm_res[core_pixels] = 40.0  # large injected core excess, as observed in DESI234/257

    chi2_full, n_full = diagnose.masked_chisq(norm_res, keep_all)

    keep_core = diagnose.core_excluded_keep(shape, centroid, radius_px, keep=keep_all)
    chi2_core, n_core = diagnose.masked_chisq(norm_res, keep_core)

    assert n_core < n_full, "core masking must remove pixels"
    assert chi2_core < chi2_full - 1.0, (
        f"core-masked chi2 ({chi2_core:.3f}) must be materially lower than full-field "
        f"({chi2_full:.3f}) once the injected core excess is excluded"
    )
    # Independently computed: the mean over exactly the surviving (non-disc) pixels.
    direct = float(np.mean(norm_res[keep_core] ** 2))
    assert abs(chi2_core - direct) < 1e-9


def test_residual_figure_derives_radius_px_from_the_constant_not_a_hardcoded_8():
    import inspect
    src = inspect.getsource(diagnose.residual_figure)
    assert "CORE_MASK_RADIUS_ARCSEC / cfg.delta_pix" in src, (
        "residual_figure's radius_px must be derived from CORE_MASK_RADIUS_ARCSEC / "
        "cfg.delta_pix, not a literal pixel count; update this test if the expression "
        "legitimately changed shape"
    )
    # And the constant itself must not silently become the DESI257 measurement (7px) or
    # any other value than the documented 8px-at-0.065"/px choice.
    assert diagnose.CORE_MASK_RADIUS_ARCSEC == 0.520
    assert abs(diagnose.CORE_MASK_RADIUS_ARCSEC / 0.065 - 8.0) < 1e-9


def main():
    tests = [
        test_map_hmc_verdict_passes_when_map_is_highest,
        test_map_hmc_verdict_fails_on_desi257_r2_s2l2_numbers,
        test_map_hmc_verdict_equal_scores_pass,
        test_pinning_flags_truncated_normal_at_upper_bound,
        test_pinning_silent_when_truncated_normal_not_pinned,
        test_tension_calibrates_against_a_known_gaussian,
        test_tension_reports_the_low_side_and_scales_with_prior_width,
        test_tension_finds_desi234_r3_fg_gamma_where_pinning_cannot,
        test_tension_silent_on_a_healthy_parameter,
        test_tension_handles_a_skewed_prior_in_quantile_space,
        test_rebuild_model_uses_recorded_supersample_over_todays_config,
        test_rebuild_model_falls_back_to_config_when_supersample_key_absent,
        test_merge_carries_forward_unrun_stage_fields,
        test_merge_keeps_this_invocations_ran_stage_fields,
        test_merge_does_not_fabricate_missing_prior_fields,
        test_merge_stages_field_is_this_invocations_stages,
        test_merge_timings_preserves_prior_stages_and_overlays_new,
        test_merge_no_prior_metadata_is_a_no_op,
        test_core_excluded_keep_matches_independent_disc_count,
        test_core_excluded_keep_is_symmetric_about_a_centred_centroid,
        test_core_excluded_keep_handles_a_noninteger_centroid,
        test_core_excluded_keep_composes_with_an_existing_mask,
        test_core_masking_materially_lowers_chi2_on_an_injected_core_excess,
        test_residual_figure_derives_radius_px_from_the_constant_not_a_hardcoded_8,
    ]
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"FAIL  {t.__name__}  {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - n_fail}/{len(tests)} tests passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
