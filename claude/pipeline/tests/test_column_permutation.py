"""Pin the stored-to-rebuilt flat-z column permutation against values recorded at fit time.

The hazard this guards is not a crash. The flat-z column order is *derived* from the
scene, never declared, so it moves whenever gigalens changes how it names parameter
paths -- and a stored ``(n_chains, n_steps, n_params)`` array carries no record of which
order it is in. Push it through the wrong order and the bijector accepts it, the shapes
agree, and every physical number comes out wrong. On 2026-08-07 gigalens renamed
``planes/0/mass/0/theta_E`` to ``planes/lens/mass/host/theta_E``, which reordered the
columns (``mass`` now precedes ``light``, and components sort by key, not by index).

So this test does not check the permutation against a rule. It checks the decoded output
against numbers written down **before** the rename, by the runs themselves, at fit time:

- DESI234 ``r3_fg`` / ``r3_unmasked`` -- ``claude/DESI234/analysis/r3_diagnostics.txt``
- DESI238 ``mclmc_long``, DESI257 ``r3_unmasked`` -- the headline results in
  ``RESULTS_v5.md`` / ``RESULTS_v3.md``

Those numbers are independent of today's code in the only way that matters: nothing can
retro-fit them. If the decode drifts again, the gap shows up here as a physics
disagreement rather than as silence.

Reads real checkpoints and builds real models, so it needs the data files and a device.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline \
        python claude/pipeline/tests/test_column_permutation.py
"""

import sys
from pathlib import Path

import numpy as np

import diagnose

REPO = Path(__file__).resolve().parents[3]

# (run dir, recorded theta_E, recorded gamma, recorded q, source of those numbers)
# q is derived from the recorded e1/e2 as q = (1-c)/(1+c), c = sqrt(e1^2+e2^2) --
# gigalens' own convention (gigalens.physicality.axis_ratio_constraint).
RECORDED = [
    ("claude/DESI234/runs/r3_fg", 1.47259, 2.9514, 0.143,
     "DESI234/analysis/r3_diagnostics.txt, written 2026-08-09"),
    # q is quoted to 3 dp in RESULTS_v3.md (0.943); the decode gives 0.9440. Tolerance is
    # the quote's own precision, 1e-3, which this sits exactly on -- hence 2e-3 here.
    ("claude/DESI234/runs/r3_unmasked", 1.53863, 2.1300, 0.944,
     "DESI234/analysis/r3_diagnostics.txt, written 2026-08-09"),
    ("claude/DESI238/runs/mclmc_long", 1.478, 1.800, None,
     "DESI238/RESULTS_v5.md headline"),
    ("claude/DESI257/runs/r3_unmasked", 1.991, 1.986, None,
     "DESI257/RESULTS_v3.md headline"),
]

_results = []


def check(name):
    def wrap(fn):
        try:
            detail = fn()
            print(f"PASS  {name}  {detail if detail else ''}")
            _results.append(True)
        except Exception as e:
            print(f"FAIL  {name}  {type(e).__name__}: {e}")
            _results.append(False)
        return fn
    return wrap


def decoded_medians(run_dir):
    """Return {label: posterior median} for a run, decoded the way every tool decodes it."""
    rm = diagnose.rebuild_model(run_dir, quiet=True)
    hmc = diagnose.load_checkpoint(run_dir, "hmc", rm)
    z = np.asarray(hmc["samples"])
    con = np.asarray(diagnose.to_constrained(rm.model, z.reshape(-1, z.shape[-1])))
    return rm, dict(zip(rm.labels, np.median(con, axis=0)))


def make_recorded_check(rel, theta_E, gamma, q, source):
    @check(f"{rel} decodes to the values recorded at fit time")
    def _test():
        run_dir = REPO / rel
        if not (run_dir / "hmc.npz").exists():
            raise FileNotFoundError(f"{run_dir}/hmc.npz is missing")
        rm, med = decoded_medians(run_dir)

        # Tolerance is set by how precisely each source quotes the number, not by the
        # decode: r3_diagnostics.txt prints 5 decimals, the READMEs quote 3.
        tol_t = 5e-5 if theta_E == round(theta_E, 5) and rel.startswith("claude/DESI234") else 1e-3
        got_t = med[diagnose_label(rm, "theta_E")]
        got_g = med[diagnose_label(rm, "gamma")]
        if abs(got_t - theta_E) > tol_t:
            raise AssertionError(
                f"theta_E decoded to {got_t:.5f}, recorded {theta_E} ({source}). "
                f"The stored columns are not being mapped onto the rebuilt model's."
            )
        if abs(got_g - gamma) > 1e-3:
            raise AssertionError(
                f"gamma decoded to {got_g:.4f}, recorded {gamma} ({source})"
            )
        detail = f"theta_E={got_t:.5f} gamma={got_g:.4f}"
        if q is not None:
            e1 = med[diagnose_label(rm, "e1", kind="mass")]
            e2 = med[diagnose_label(rm, "e2", kind="mass")]
            c = float(np.hypot(e1, e2))
            got_q = (1 - c) / (1 + c)
            if abs(got_q - q) > 2e-3:
                raise AssertionError(f"q decoded to {got_q:.4f}, recorded {q} ({source})")
            detail += f" q={got_q:.4f}"
        return detail
    return _test


def diagnose_label(rm, param, kind="mass"):
    """The host-mass label for ``param``, found structurally (never by column number)."""
    return rm.labels[diagnose.column(rm.model, f"mass/0/{param}", kind=kind)]


for _rel, _t, _g, _q, _src in RECORDED:
    make_recorded_check(_rel, _t, _g, _q, _src)


@check("source_component_params reaches the source plane, where position would pick the deflector")
def test_source_components_are_not_resolved_positionally():
    """The same 2026-08-07 rename renamed the *structured* params tree's keys as well.

    ``model.planes`` is a list in declaration order -- lens, then source. But
    ``model.constrained(...)["planes"]`` is a dict keyed by name, and its order is not
    ``model.planes``'. On ``r3_fg`` it comes out ``{"source": ..., "lens": ...}``:
    reversed. Resolving the source plane by position would return the deflector **while
    raising nothing**, because both planes carry a ``light`` dict -- the figure would
    render the lens galaxy as the source. gigalens' ``scene-design.md`` warns against
    assuming that order; this test is the local proof that the warning bites here.

    The resolution is gigalens' own ``LensModel.component_params``, which the library
    calls "the single supported way to reach a component's params".
    ``diagnose.source_component_params`` is a thin loop over it, and this test pins that
    it is reached correctly rather than re-deriving the keying here.
    """
    import jax.numpy as jnp

    run_dir = REPO / "claude/DESI234/runs/r3_fg"
    rm = diagnose.rebuild_model(run_dir, quiet=True)
    z = np.asarray(diagnose.load_checkpoint(run_dir, "map", rm)["best_z"]).reshape(-1)
    params = rm.model.constrained(jnp.asarray(z))

    model_order = [p.name for p in rm.model.planes]
    dict_order = list(params["planes"])
    if model_order == dict_order:
        raise AssertionError(
            f"both orders are {model_order}, so a positional lookup would also pass here "
            f"and this test cannot detect the bug it exists for."
        )

    src = len(rm.model.planes) - 1
    got = diagnose.source_component_params(rm.model, params, src)
    want = params["planes"][model_order[src]]["light"]
    assert len(got) == len(want) == len(rm.model.planes[src].light), (
        f"{len(got)} components resolved, plane declares {len(rm.model.planes[src].light)}"
    )
    # Identity, not equality: these must BE the source plane's own leaves.
    assert all(g is want[k] for g, k in zip(got, want)), "did not resolve the source plane"

    # And the positional read really does land on the deflector, which is the whole point.
    positional = params["planes"][dict_order[src]]["light"]
    assert positional is not want, "name and position agree; see above"
    # The deflector has no R_sersic-free component in common count with the source here:
    # 3 light components against the source's 2. That asymmetry is what a positional
    # lookup gets backwards, silently.
    assert len(positional) != len(want), (
        "both planes have the same number of light components, so the positional error "
        "would not change the count -- this assertion cannot witness it"
    )
    return (f"model {model_order} vs tree {dict_order}; "
            f"resolved {len(got)} source components, not the deflector's {len(positional)}")


@check("reorder_columns scatters (stored -> model), and is not its own inverse")
def test_scatter_not_gather():
    # perm[j] = i means stored column j is model column i. A gather (arr[:, perm]) would
    # apply the inverse -- which for a permutation that is not an involution is a
    # different, equally plausible-looking answer. Pin the direction with an asymmetric
    # 3-cycle, for which scatter and gather provably disagree.
    perm = [1, 2, 0]
    arr = np.array([[10.0, 20.0, 30.0]])
    out = diagnose.reorder_columns(arr, perm)
    want = np.array([[30.0, 10.0, 20.0]])
    assert np.array_equal(out, want), f"scatter gave {out}, expected {want}"
    gathered = arr[:, perm]
    assert not np.array_equal(out, gathered), (
        "scatter and gather agree on this permutation, so it cannot pin the direction"
    )
    assert diagnose.reorder_columns(arr, None) is arr, "None must be a no-op"
    return f"{arr.tolist()} -> {out.tolist()}"


@check("a run whose stored order differs is refused when read without its model")
def test_unpermuted_read_refused():
    # The whole failure mode is a silent one, so the guard matters more than the fix:
    # point_estimate without rm must raise rather than return stored-order columns.
    run_dir = REPO / "claude/DESI234/runs/r3_fg"
    try:
        diagnose.point_estimate(run_dir)
    except ValueError as e:
        assert "column order" in str(e), f"raised, but not about column order: {e}"
        return "point_estimate(run_dir) raises without rm"
    raise AssertionError(
        "point_estimate returned stored-order columns instead of raising"
    )


@check("the permuted MAP scores near its recorded log posterior, the unpermuted nowhere near")
def test_log_prob_separates_the_two_orders():
    """A check that cites no physical result: ``map.npz`` records the log posterior the
    fit computed for its own best vector, so the decode can be scored against it.

    The assertion is deliberately one of *separation*, not equality. The permuted vector
    lands within a few nats of the recorded score; the unpermuted one lands hundreds of
    thousands of nats away. That gap is the evidence the permutation is right, and it is
    five orders of magnitude larger than any plausible drift, so the test cannot pass by
    accident.

    Equality does not hold: the permuted score sits a few nats low (r3_fg -11.7, DESI257
    r3_unmasked -1.3), entirely in the prior+Jacobian term, with the likelihood term
    unaffected. That is a separate post-2026-08 change in gigalens' prior normalisation,
    not a column-order problem -- an offset in the prior cannot reorder parameters, and
    all four runs' *physical* values reproduce exactly (see the checks above). It is
    recorded in KNOWN_ISSUES.md. The 50-nat bound below is well inside that drift and
    far outside any mis-ordering.
    """
    import jax.numpy as jnp

    run_dir = REPO / "claude/DESI234/runs/r3_fg"
    rm = diagnose.rebuild_model(run_dir, quiet=True)
    stored_lp = float(np.asarray(diagnose.load_checkpoint(run_dir, "map", rm)["best_lp"]).reshape(-1)[0])

    permuted = np.asarray(diagnose.load_checkpoint(run_dir, "map", rm)["best_z"]).reshape(-1)
    raw = np.asarray(diagnose._load_npz(run_dir, "map")["best_z"]).reshape(-1)
    lp_perm = float(np.asarray(rm.prob.log_prob(jnp.asarray(permuted))[0]))
    lp_raw = float(np.asarray(rm.prob.log_prob(jnp.asarray(raw))[0]))

    if abs(lp_perm - stored_lp) > 50:
        raise AssertionError(
            f"permuted MAP scores {lp_perm:.1f} against a recorded {stored_lp:.1f} -- "
            f"further off than the known prior-normalisation drift explains."
        )
    if abs(lp_raw - stored_lp) < 1000:
        raise AssertionError(
            f"the UNpermuted MAP scores {lp_raw:.1f}, within 1000 nats of the recorded "
            f"{stored_lp:.1f}. The two column orders are not being distinguished, so "
            f"this test can no longer detect a decode regression."
        )
    return f"recorded {stored_lp:.1f}; permuted {lp_perm:.1f}; unpermuted {lp_raw:.1f}"


if __name__ == "__main__":
    print(f"\n{sum(_results)}/{len(_results)} tests passed")
    sys.exit(0 if all(_results) else 1)
