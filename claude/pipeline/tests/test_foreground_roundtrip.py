"""Regression test: round 3b's foreground-interloper model metadata round-trips through
``diagnose.rebuild_model``.

Run directly (pytest is not installed in this env; needs a real lens config and gigalens,
so unlike this directory's CPU-only tests it is not GPU-free):

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=claude/pipeline \\
        python claude/pipeline/tests/test_foreground_roundtrip.py

Uses DESI257, the only lens with a ``foreground_prior`` set. ``priors.build_model(...,
foreground=cfg.foreground_prior)`` adds a SIS + Sersic pair to the lens plane (10 extra
free params, 36 -> 46) that must survive being written to and read back from
``metadata.json`` exactly the way ``n_source``/``lens_light_prior_version`` already do --
see ``run_fit.py``'s "foreground"/"foreground_prior_version" fields and
``diagnose.rebuild_model``'s reading of them. Getting this wrong does not raise: it
silently rebuilds a 36-param model (or the wrong bijector) from a run that was actually
fitted at 46, and every physical value read back would be wrong. This test writes a
metadata.json by hand -- not by running run_fit.py -- so it stays fast and needs no
checkpoint files, only ``diagnose.rebuild_model``'s metadata contract (it requires
map.npz/svi.npz/hmc.npz to exist only for the stages that get READ, not to rebuild the
model itself).
"""

import json
import sys
import tempfile
from pathlib import Path

import lenses
import priors
import diagnose

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


@check("foreground=True rebuilds the same 46-param model from metadata.json")
def test_foreground_roundtrip():
    cfg = lenses.load("DESI257")
    assert cfg.foreground_prior is not None, "DESI257's config must define foreground_prior"

    built = priors.build_model(
        cfg, n_source=2, n_lens_light=2, foreground=cfg.foreground_prior
    )
    assert built.num_free_params == 46, (
        f"expected 46 free params (36 + 10 for the SIS + Sersic pair), got "
        f"{built.num_free_params}"
    )
    built_names = list(built.z_param_names)

    with tempfile.TemporaryDirectory() as tmp:
        metadata = {
            "api": "scene",
            "lens": "DESI257",
            "n_source": 2,
            "n_lens_light": 2,
            "lens_light_n_max": 10.0,
            "theta_e_center": cfg.mass_prior.theta_E_median,
            "theta_e_width": cfg.mass_prior.theta_E_logsigma,
            "source_prior_version": priors.SOURCE_PRIOR_VERSION,
            "lens_light_prior_version": priors.lens_light_prior_version_for(2),
            "foreground": True,
            "foreground_prior_version": priors.FOREGROUND_PRIOR_VERSION,
            "mask_circles": None,
            "n_masked_pixels": cfg.num_pix * cfg.num_pix,
            "n_free_params": 46,
        }
        (Path(tmp) / "metadata.json").write_text(json.dumps(metadata))

        rm = diagnose.rebuild_model(tmp, quiet=True)

    assert rm.model.num_free_params == 46, (
        f"rebuilt model has {rm.model.num_free_params} free params, expected 46"
    )
    rebuilt_names = list(rm.model.z_param_names)
    assert rebuilt_names == built_names, (
        "rebuilt param names differ from the freshly-built model's:\n"
        f"  built:    {built_names}\n"
        f"  rebuilt:  {rebuilt_names}"
    )
    assert rm.foreground is True
    assert rm.foreground_prior_version == priors.FOREGROUND_PRIOR_VERSION

    # The disambiguated column lookup diagnose.py's own call sites use: must resolve to
    # the HOST's theta_E -- the FIRST mass component priors.py declares -- and not raise
    # on the SIS's own theta_E. Asserted positionally rather than against a spelling:
    # gigalens renamed these paths on 2026-08-07 (planes/0/mass/0/... ->
    # planes/lens/mass/host/...), and pinning the literal string is what let that rename
    # break every caller of this idiom silently. What must hold is that the column is the
    # first mass theta_E, whatever it is called.
    idx = diagnose.column(rm.model, "mass/0/theta_E", kind="mass")
    first_mass_theta_E = next(
        i for i, n in enumerate(rebuilt_names)
        if "/mass/" in n and n.endswith("/theta_E")
    )
    assert idx == first_mass_theta_E, (
        f"column() resolved {rebuilt_names[idx]} but the first mass theta_E is "
        f"{rebuilt_names[first_mass_theta_E]}"
    )

    return f"{rm.model.num_free_params} params, names identical, host theta_E at column {idx}"


@check("foreground absent from metadata.json rebuilds the unchanged 36-param model")
def test_no_foreground_is_unaffected():
    # A metadata.json predating round 3b (no "foreground" key at all) must still rebuild
    # the plain 36-param model -- the default this whole feature must not disturb.
    cfg = lenses.load("DESI257")
    built = priors.build_model(cfg, n_source=2, n_lens_light=2)
    assert built.num_free_params == 36

    with tempfile.TemporaryDirectory() as tmp:
        metadata = {
            "api": "scene",
            "lens": "DESI257",
            "n_source": 2,
            "n_lens_light": 2,
            "mask_circles": None,
            "n_masked_pixels": cfg.num_pix * cfg.num_pix,
            "n_free_params": 36,
        }
        (Path(tmp) / "metadata.json").write_text(json.dumps(metadata))
        rm = diagnose.rebuild_model(tmp, quiet=True)

    assert rm.model.num_free_params == 36
    assert rm.foreground is False
    assert list(rm.model.z_param_names) == list(built.z_param_names)
    return "36 params, foreground=False, names identical"


def main():
    tests = [test_foreground_roundtrip, test_no_foreground_is_unaffected]
    for t in tests:
        t()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
