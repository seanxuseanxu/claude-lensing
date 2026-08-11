"""Unit tests for three pure/near-pure checks added 2026-08-09:

    - ``diagnose.map_hmc_verdict``: MAP must score >= the best HMC draw.
    - ``diagnose.boundary_pinning_check``: TruncatedNormal bounds, not just Uniform.
    - ``run_fit.merge_prior_metadata``: a ``--stages hmc`` rerun must not overwrite prior
      stages' recorded metadata with today's argparse defaults (KNOWN_ISSUES.md item 1).

Run directly (pytest is not installed in this env), CPU-only like ``test_mask.py`` --
nothing here builds a real scene or runs inference, so no ``--lens`` and no GPU:

    PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_diagnose_and_metadata.py
"""

import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import diagnose
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
        test_merge_carries_forward_unrun_stage_fields,
        test_merge_keeps_this_invocations_ran_stage_fields,
        test_merge_does_not_fabricate_missing_prior_fields,
        test_merge_stages_field_is_this_invocations_stages,
        test_merge_timings_preserves_prior_stages_and_overlays_new,
        test_merge_no_prior_metadata_is_a_no_op,
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
