"""Unit tests for the ESS floor and the two Vehtari diagnostic plots (added 2026-08-11).

Run directly (pytest is not installed in this env), CPU-only -- nothing here builds a real
scene or runs inference, so no ``--lens`` and no GPU:

    PYTHONPATH=claude/pipeline python claude/pipeline/tests/test_ess_floor_and_plots.py

The three things under test are the ones that can be wrong *silently*, which is why they
are tested against synthetic chains whose answer is known by construction rather than
against a real run whose answer is what we are trying to find out:

  (a) ``ess_floor_report`` must fail a run that clears R-hat on too few effective draws.
      That is the whole point of the floor: a passing R-hat below it is uninformative, so
      a floor that only fires when R-hat also fails would add nothing.
  (b) ``ess_vs_draws_plot``'s slope must come out near 1 for chains that really are iid --
      i.e. the case where a longer run genuinely buys ESS -- and clearly below 1 for chains
      that are stuck. Getting this backwards would recommend the expensive 4x run in
      exactly the case where it cannot help, which is the decision the plot exists to make.
  (c) ``rank_plot``'s chain cut must KEEP the misbehaving chain. The cut exists because a
      50-chain run cannot be drawn in full; a cut that took the first N chains would drop
      the one bad chain whenever it is not among the first N, and the figure would then
      look clean precisely when it should not.

Everything uses a stub model. ``diagnose_convergence`` needs only ``z_param_names`` from a
model, and the two plot functions never leave unconstrained space, so no bijector, no
priors and no FITS data are involved.
"""

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import diagnose


class StubModel:
    """The two attributes ``diagnose_convergence`` and ``hmc_and_constrained`` read."""

    def __init__(self, n_params):
        self.num_free_params = n_params
        self.z_param_names = tuple(f"p{i}" for i in range(n_params))


class StubRM:
    """Stands in for ``RunModel`` for the z-space stages."""

    def __init__(self, n_params):
        self.model = StubModel(n_params)
        self.labels = list(self.model.z_param_names)


def report_for(samples, rm):
    return diagnose.diagnose_convergence(samples, model=rm.model)


# --- (a) the ESS floor -------------------------------------------------------------------

def test_floor_passes_a_long_well_mixed_run():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(8, 4000, 3))
    rm = StubRM(3)
    with redirect_stdout(io.StringIO()) as buf:
        ok = diagnose.ess_floor_report(report_for(samples, rm), n_chains=8)
    assert ok, buf.getvalue()
    assert "FAIL" not in buf.getvalue()
    print("  ok: 8 chains x 4000 iid draws clears the floor")


def test_floor_fails_a_short_run_that_would_pass_rhat():
    """The load-bearing case: R-hat is fine, ESS is not, and the floor must say so.

    32 iid draws x 4 chains is 128 total, well under 400, and iid chains from one
    distribution give a clean R-hat -- so R-hat alone would call this converged.
    """
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(4, 32, 3))
    rm = StubRM(3)
    report = report_for(samples, rm)
    assert report.max_rhat < 1.05, f"setup broken: R-hat {report.max_rhat} is not clean"

    with redirect_stdout(io.StringIO()) as buf:
        ok = diagnose.ess_floor_report(report, n_chains=4)
    out = buf.getvalue()
    assert not ok, out
    assert "FAIL" in out
    assert "NOT INTERPRETABLE" in out
    assert "absence of evidence" in out
    print(f"  ok: R-hat {report.max_rhat:.3f} passes but the floor fails, and says why")


def test_floor_fails_on_too_few_chains():
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(2, 20000, 3))  # plenty of draws, only 2 chains
    rm = StubRM(3)
    with redirect_stdout(io.StringIO()) as buf:
        ok = diagnose.ess_floor_report(report_for(samples, rm), n_chains=2)
    out = buf.getvalue()
    assert not ok, out
    assert f"chains      >= {diagnose.MIN_CHAINS}" in out
    print("  ok: 2 chains fails the chain count even with 40,000 draws")


def test_floor_uses_the_smaller_of_bulk_and_tail():
    """Tail-ESS is usually the smaller and is the one that catches heavy tails.

    Built by hand rather than sampled, so the assertion is about which number the function
    reads, not about a distribution.
    """
    report = diagnose.diagnose_convergence(
        np.random.default_rng(3).normal(size=(4, 2000, 2)),
        z_param_names=("a", "b"),
    )
    doctored = type(report)(
        names=report.names,
        rhat=report.rhat,
        ess_bulk=np.array([5000.0, 5000.0]),
        ess_tail=np.array([5000.0, 100.0]),   # only the tail of "b" is short
        n_chains=report.n_chains,
        n_steps=report.n_steps,
    )
    with redirect_stdout(io.StringIO()) as buf:
        ok = diagnose.ess_floor_report(doctored, n_chains=4)
    out = buf.getvalue()
    assert not ok, out
    assert "[b]" in out, out
    print("  ok: a short tail-ESS fails the floor and names the parameter")


# --- (b) ESS vs draws --------------------------------------------------------------------

def sticky_chains(rng, n_chains, n_draws, n_params, rho=0.999):
    """AR(1) chains with an autocorrelation long enough to starve ESS growth."""
    x = np.empty((n_chains, n_draws, n_params))
    x[:, 0] = rng.normal(size=(n_chains, n_params))
    scale = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n_draws):
        x[:, t] = rho * x[:, t - 1] + scale * rng.normal(size=(n_chains, n_params))
    return x


def test_ess_grows_linearly_for_iid_chains():
    rng = np.random.default_rng(10)
    samples = rng.normal(size=(4, 4000, 3))
    rm = StubRM(3)
    with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()) as buf:
        path = diagnose.ess_vs_draws_plot(
            td, td, rm, hmc_samples=samples, report=report_for(samples, rm), n_worst=3)
        # Inside the with: the temporary directory is gone by the time it closes.
        wrote = path is not None and Path(path).exists()
    out = buf.getvalue()
    assert wrote, f"no figure written; output was:\n{out}"
    assert "near-linear" in out, out
    assert "SUBLINEAR" not in out, out
    print("  ok: iid chains give a near-linear slope and the 'longer run helps' verdict")


def test_ess_growth_is_sublinear_for_stuck_chains():
    """Chains that are barely moving: ESS must NOT be reported as growing linearly.

    A false 'near-linear' here is the expensive error -- it recommends 4x the draws on a
    run that will not improve.
    """
    rng = np.random.default_rng(11)
    samples = sticky_chains(rng, 4, 4000, 3, rho=0.9995)
    # Push the chains apart so they are sampling visibly different regions, which is the
    # situation "more draws will not fix it" is about.
    samples += np.array([-6.0, -2.0, 2.0, 6.0])[:, None, None]
    rm = StubRM(3)
    with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()) as buf:
        path = diagnose.ess_vs_draws_plot(
            td, td, rm, hmc_samples=samples, report=report_for(samples, rm), n_worst=3)
    out = buf.getvalue()
    assert path is not None
    assert "SUBLINEAR" in out, out
    assert "near-linear" not in out, out
    print("  ok: stuck, separated chains give the 'more draws will not fix this' verdict")


def test_ess_vs_draws_skips_a_run_too_short_to_subdivide():
    rng = np.random.default_rng(12)
    samples = rng.normal(size=(4, 12, 2))
    rm = StubRM(2)
    with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()) as buf:
        path = diagnose.ess_vs_draws_plot(td, td, rm, hmc_samples=samples, n_worst=2)
    assert path is None
    assert "too short to subdivide" in buf.getvalue()
    print("  ok: a 12-draw run is skipped with a sentence, not a traceback")


# --- (c) the rank plot's chain cut -------------------------------------------------------

def test_rank_plot_keeps_the_offset_chain_under_the_cap():
    """One chain of twelve is offset; only four may be drawn. It must be one of them."""
    rng = np.random.default_rng(20)
    n_chains, bad = 12, 9
    samples = rng.normal(size=(n_chains, 1500, 3))
    samples[bad] += 8.0
    rm = StubRM(3)
    with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()) as buf:
        path = diagnose.rank_plot(
            td, td, rm, hmc_samples=samples, report=report_for(samples, rm),
            n_worst=2, max_chains=4)
        wrote = path is not None and Path(path).exists()
    out = buf.getvalue()
    assert wrote, f"no figure written; output was:\n{out}"
    assert "Showing the 4 least-uniform of 12 chains" in out, out
    shown = out.split("indices [")[1].split("]")[0]
    assert str(bad) in [s.strip() for s in shown.split(",")], (
        f"the offset chain {bad} was cut from the figure; shown = [{shown}]")
    print(f"  ok: offset chain {bad} survives a cut from 12 chains to 4")


def test_rank_plot_draws_every_chain_when_it_fits():
    rng = np.random.default_rng(21)
    samples = rng.normal(size=(4, 800, 2))
    rm = StubRM(2)
    with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()) as buf:
        path = diagnose.rank_plot(
            td, td, rm, hmc_samples=samples, report=report_for(samples, rm),
            n_worst=2, max_chains=10)
    out = buf.getvalue()
    assert path is not None
    assert "x 4 chains" in out, out
    assert "Showing the" not in out, "no cut should be announced when none was made"
    print("  ok: 4 chains under a cap of 10 are all drawn, with no cut notice")


# --- the shared loader -------------------------------------------------------------------

def test_hmc_and_constrained_rejects_a_transposed_array():
    """The (n_steps, n_chains, n_params) layout must raise for every caller, not just one.

    This is the file's one silent input hazard: transposed, split-R-hat is computed across
    the wrong axis and comes out plausible and wrong.
    """
    rm = StubRM(3)
    bad = np.zeros((500, 4, 7))  # 7 columns for a 3-parameter model
    try:
        diagnose.hmc_and_constrained("unused", rm, hmc_samples=bad, need_con=False)
    except ValueError as e:
        assert "expected (n_chains, n_steps, 3)" in str(e), str(e)
        print("  ok: a wrong-width array raises naming the expected shape")
    else:
        raise AssertionError("hmc_and_constrained accepted a wrong-shaped array")


def test_hmc_and_constrained_skips_the_transform_when_not_needed():
    """``need_con=False`` must not call ``to_constrained`` -- that is the point of it."""
    rm = StubRM(3)
    samples = np.zeros((4, 100, 3))
    called = []
    real = diagnose.to_constrained
    diagnose.to_constrained = lambda *a, **k: called.append(1)
    try:
        s, con = diagnose.hmc_and_constrained("unused", rm, hmc_samples=samples,
                                              need_con=False)
    finally:
        diagnose.to_constrained = real
    assert con is None and s.shape == (4, 100, 3)
    assert not called, "to_constrained was called despite need_con=False"
    print("  ok: need_con=False returns con=None without transforming")


def test_missing_hmc_returns_none_pair():
    rm = StubRM(3)
    with tempfile.TemporaryDirectory() as td:
        s, con = diagnose.hmc_and_constrained(td, rm)
    assert s is None and con is None
    print("  ok: a run directory with no hmc.npz gives (None, None)")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for fn in TESTS:
        print(f"{fn.__name__}:")
        try:
            fn()
        except AssertionError as e:
            failures += 1
            print(f"  FAILED: {e}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)
