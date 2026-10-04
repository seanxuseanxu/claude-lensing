"""Regression test: ``verify.py``'s summary counts a SKIPPED check as skipped, not passed.

Run directly (pytest is not installed in this env; needs no lens data, CPU-only):

    PYTHONPATH=claude/pipeline CUDA_VISIBLE_DEVICES="" \\
        python claude/pipeline/tests/test_verify_summary.py

Why this exists (2026-08-11). ``verify.py``'s check decorator records a triple
``(label, ok, detail)``, and a check that cannot run on a given lens -- check 5, the SVI
covariance guard, needs a known-good/known-diverged pair the lens config may not name --
signals that by *returning* a ``"SKIPPED: ..."`` detail rather than raising. It therefore
lands with ``ok=True``. The old summary line printed ``len(RESULTS) - n_fail`` as the pass
count, so that skip was counted as a pass: ``verify.py --lens DESI257`` reported **5/5
checks passed** when only four checks had actually run, and the one that did not run was the
one guarding against a silently-non-positive-definite SVI covariance.

That is the failure mode this repository cares about most: absent evidence reported as
evidence. A skipped check is honest and must stay visible, so the fix is to report it as a
third category -- never to let it inflate a pass count, and never to weaken the check so it
can "pass" instead.

The assertions below pin both directions, because only one of them would have caught the
bug and only the other proves the fix did not simply reclassify everything as a skip:

  - a suite containing a skip must NOT report it as a pass (the bug), and
  - an ordinary pass whose detail merely *mentions* skipping elsewhere is still a pass
    (the fix must key on the documented ``SKIP_PREFIX`` at the start of the detail, not on
    the substring appearing anywhere).

``summarize`` and ``status_of`` were split out of ``verify.main`` for this test: the
accounting used to be inline in a function that runs the whole suite, and so was reachable
only by running every check against a real lens.
"""

import sys

import verify

RESULTS = []


def check(name):
    def wrap(fn):
        def run(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True, detail or ""))
            except Exception as exc:  # noqa: BLE001
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                import traceback
                traceback.print_exc()

        return run

    return wrap


@check("A skipped check is counted as skipped, not passed")
def test_skip_not_counted_as_pass():
    # DESI257's actual round-3 shape: four real passes plus check 5 skipped.
    results = [
        ("1. Data conditioning", True, "crop (80, 80) max 8.47"),
        ("2. Parameter count and ordering", True, "22 parameters"),
        ("3. Simulator flux scale", True, "ratio 1.19"),
        ("4. Short end-to-end", True, "MAP -> SVI -> HMC"),
        ("5. SVI covariance guard", True, "SKIPPED: no svi_guard_reference runs"),
    ]
    n_pass, n_skip, n_fail, line = verify.summarize(results)

    assert (n_pass, n_skip, n_fail) == (4, 1, 0), (
        f"expected 4 pass / 1 skip / 0 fail, got {n_pass}/{n_skip}/{n_fail}"
    )
    # The literal string the old code produced, which must never reappear.
    assert "5/5" not in line, (
        f"summary line still reports the skip as a pass: {line!r}"
    )
    assert line.startswith("4/5 checks passed"), f"unexpected summary line: {line!r}"
    assert "skipped" in line, (
        f"summary line does not make the skip visible: {line!r}"
    )
    assert verify.status_of(True, "SKIPPED: no reference") == "SKIP"

    return f"{line!r}; status_of -> SKIP"


@check("A real pass and a real failure are unaffected")
def test_pass_and_fail_still_classified():
    # No skips at all: the summary must read exactly as it always did.
    n_pass, n_skip, n_fail, line = verify.summarize(
        [("1. ok", True, "fine"), ("2. bad", False, "AssertionError: boom")]
    )
    assert (n_pass, n_skip, n_fail) == (1, 0, 1), (
        f"expected 1 pass / 0 skip / 1 fail, got {n_pass}/{n_skip}/{n_fail}"
    )
    assert line.startswith("1/2 checks passed"), f"unexpected summary line: {line!r}"
    assert "skipped" not in line, (
        f"summary mentions skipping when nothing was skipped: {line!r}"
    )
    assert verify.status_of(True, "fine") == "PASS"
    assert verify.status_of(False, "AssertionError: boom") == "FAIL"

    # A pass whose detail merely mentions the word must NOT be reclassified: the prefix is
    # the contract, so keying on a bare substring would silently downgrade real passes.
    n_pass2, n_skip2, _, _ = verify.summarize(
        [("1. ok", True, "guard fired; nothing was SKIPPED this run")]
    )
    assert (n_pass2, n_skip2) == (1, 0), (
        f"a pass mentioning 'SKIPPED' mid-detail was miscounted: "
        f"{n_pass2} pass / {n_skip2} skip"
    )

    return f"{line!r}; mid-detail 'SKIPPED' stays a pass"


def main():
    test_skip_not_counted_as_pass()
    test_pass_and_fail_still_classified()

    print()
    width = max(len(n) for n, _, _ in RESULTS)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
