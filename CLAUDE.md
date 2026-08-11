# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two unrelated top-level directories, tracked in one git repo:

- `papers/` — LaTeX sources for five papers. **`papers/INDEX.md` is the map**: what each one is, which question it answers, and a list of the places they bear on decisions made in this repo. Every paper has a `NOTES.md` beside it — a reading note with section and equation numbers cited, so a claim can be checked against the source rather than trusted. Read the note before the `.tex`. The two AASTeX papers are `gigalens_paper/` (the GIGA-Lens method paper, arXiv:2202.07663; entry point `ms_SL-model.tex`) and `strong_lens_foundry_I/` (entry point `main.tex`), which is the survey our three lenses come from; the other three are Narayan & Bartelmann's lensing lectures, the MCLMC paper, and Vehtari et al.'s improved R̂.
- `claude/` — the lens-modelling work. `claude/pipeline/` is a lens-independent MAP→SVI→HMC fitting pipeline; `claude/<LENS>/` holds one system's data, its `config.py`, its `runs/`, and its own README. `claude/guide/` and `WORKING_WITH_AGENTS.md` are lessons learned, using DESI238 as the worked case — **`claude/guide/INDEX.md` is the map**, ten chapters numbered to match the `§N` citations scattered through the results files; read `09-the-short-version.md` if you read only one.

There used to be a third, `gigalens/` — a vendored copy of the package, which was the code you modified. It is gone from the working copy (still reachable in git history). gigalens is now a pip-installed dependency, developed elsewhere.

## Environment

The active conda env is `gigajax7.0` (Python 3.14).

- **gigalens is installed, editable, from `/global/u1/s/seanjx/nanolens`** (`pip show gigalens` → version 2.2, `Editable project location`). `import gigalens.jax...` works from anywhere; there is no `PYTHONPATH=gigalens/src` any more. **That directory is a different project — do not edit it from here.** If something in it is broken or wrong, report it; do not patch around it silently either.
- **`src/gigalens/old_api/`** holds the superseded `PhysicalModel` / `ForwardProbModel` / `ModellingSequence` interface. Nothing here uses it. If you find yourself importing from it, you are working from a stale example.

gigalens' own profile tests assert numerical parity against the corresponding `lenstronomy` profile, typically `rtol=1e-5, atol=1e-4`. **A lenstronomy reference is the accepted ground truth for profile shape**.

## The gigalens scene API

One object describes the system. `LensModel` holds `Plane`s of `Component`s, a `Component`
is a profile bundled with a dict of per-parameter priors, and the joint prior, the
unconstraining bijector, the flat-`z` column order and the constants dict are all
**derived** from it rather than hand-synchronised. `gigalens/jax/scene.py` is the file to
read; `nanolens/docs/` and `nanolens/demos/simple_demo.ipynb` are the worked examples.

| Concern | Where |
|---|---|
| Scene spec, prior, bijector, ray trace | `gigalens/jax/scene.py` — `Component`, `Plane`, `LensModel`, `shared`, `coupled` |
| Rendering | `gigalens/jax/scene_simulator.py` — `SceneSimulator` |
| Data + likelihood | `gigalens/jax/scene_prob_model.py` — `ImageData`, `ProbModel` |
| Inference | `gigalens/jax/inference/` — free functions `MAP`, `SVI`, `HMC`, plus `MCLMC`, `MAMS`, `NUTS` |
| Convergence | `gigalens/jax/analysis/diagnostics.py` — `diagnose_convergence` |
| Lens geometry | `gigalens/jax/analysis/lens_solver.py` — `LensSolver` (images, critical curves, caustics); `source_plane.py` — `image_border_in_source_plane` |

### Finding things in gigalens

The table above is the load-bearing surface, not the whole package — there are ~90
modules. **`claude/GIGALENS_INDEX.md` is a generated symbol index**: every public class
and function with its signature, one-line summary and `file:line`, plus a module-level
table of contents and a list of nanolens' own prose docs. Grep it for a name, or read its
head for "which module is this in".

```bash
/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python claude/gigalens_index.py
```

Regenerate after pulling nanolens; the index stamps its own generation date and the source
path it resolved, so staleness is visible rather than silent. Check it **before writing
anything numerical** — a hand-rolled version of something the library already has is not
just wasted effort, it is unvalidated. 

## Modelled systems

Each has its own README with the dataset, the conditioning decisions, and the results.

- `claude/DESI238/` — HST WFC3/IR F140W. Five rounds, converged at R̂ ≤ 1.003 (`RESULTS_v5.md`), the first fit under the scene API. It reproduces the four pre-port rounds' `1.480 ± 0.003"` to within 0.002" across the changed likelihood normalisation, parameter ordering, sampler and codebase — so the port is validated, but pre-port checkpoints are still unreadable and `diagnose.rebuild_model` refuses them. See `claude/DESI238/README.md`.
- `claude/DESI257/`, `claude/DESI234/` — in progress, see their respective `claude/<LENS>/README.md`.

## Working notes

- `nanolens/demos/simple_demo.ipynb` (rendered as `docs/tutorials/first-fit.ipynb`) is the best worked example of a full model → MAP → SVI → HMC run. Note that optax schedules use *negative* learning rates composed with `scale_by_adam` (i.e. descent by scaling), which is easy to misread.
- `nanolens/docs/known-issues-core-api.md` and `known-issues-profiles.md` are worth reading before blaming your own code.
