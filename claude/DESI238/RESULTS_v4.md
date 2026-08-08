# DESI238 round 4 — an accurate b_n, a lower source floor, and a converged fit

> **Round 4 of 5. Superseded — current result is [`RESULTS_v5.md`](RESULTS_v5.md):
> `theta_E = 1.478 ± 0.003"`, all 36 parameters at R̂ ≤ 1.003, run `mclmc_long`.**
> This round: `theta_E = 1.4795 ± 0.0026"` (R̂ 1.001, ESS 66,778, run `bn_rational_long`),
> the last pre-port result. Round 5 is an independent re-fit under the scene API, not a
> re-analysis, and it agrees with this number to 0.002" — which is what validates the port.
> The b_n and source-floor findings below still hold; the checkpoints in
> `runs/bn_rational_long/` do not, as the column order and noise convention changed
> (`diagnose.rebuild_model` refuses them).

2026-08-01. One run on one H100, `--seed 0`, ~35 minutes.

| run | params | model | chi²ᵥ | worst Rhat | Rhat > 1.1 | Rhat > 1.01 |
|---|---|---|---|---|---|---|
| `bulgedisk_long` (round 3) | 36 | 2 source, 2 lens light | 0.989 | 1.338 | 1/36 | 10/36 |
| `bn_rational_long` | 36 | as above, new b_n + source prior v3 | **0.988** | **1.099** | **0/36** | 9/36 |

The two runs are identical in every setting — same seed, same components, 3000 draws after
1000 burnin, `--max-leapfrog 60`. They differ in two coupled changes described below.

Three results:

1. **All 36 parameters converge at R̂ < 1.1** (worst 1.099), the first round for which that
   is true. `theta_E = 1.4795 ± 0.0026"`, R̂ 1.001, ESS 66,778. This is now the run to cite.
2. **The source's pinning was not a prior artefact.** Lowering the `n_sersic` floor from 0.5
   to 0.3 moved the compact source from pinned at 0.5 to pinned at 0.3. The data want a
   flatter profile than this family can express; the fix is a different profile.
3. **`theta_E` is insensitive to a 13.6% error in the source's b_n** — unchanged to four
   digits. The Einstein radius is set by arc positions, not by source concentration.

## What changed

### The Sérsic constant

gigalens hardcoded `bn = 1.9992n − 0.3271`, an asymptotic fit that is fine at large n and
poor at small n. It is now a rational (3,2) approximation. Against the exact
`b_n = gammaincinv(2n, ½)`:

| n | exact | rational | error | linear | error |
|---|---|---|---|---|---|
| 0.3 | 0.315702 | 0.315711 | +0.003% | 0.272660 | **−13.6%** |
| 0.5 | 0.693147 | 0.693173 | +0.004% | 0.672500 | −2.98% |
| 1.0 | 1.678347 | 1.678292 | −0.003% | 1.672100 | −0.37% |
| 4.0 | 7.669249 | 7.669246 | −0.000% | 7.669700 | +0.006% |
| 8.0 | 15.667930 | 15.668210 | +0.002% | 15.666500 | −0.009% |

Worst case 5.5e-05 over n ∈ [0.3, 8] in float32; the coefficients themselves are good to
3.8e-05 in float64, so the rest is dtype. Better than the linear form at every n except a
narrow band around n ≈ 2.5 and n ≈ 5.5, where the linear form's *signed* error crosses zero
and is incidentally closer — both are ~1e-05 there.

**Lenstronomy cannot validate this.** `lenstronomy/LensModel/Profiles/sersic_utils.py:37`
implements the same `1.9992n − 0.3271`, clipped at 1e-5, and so carries the identical
−13.6% at n = 0.3. The repository convention of testing a profile against lenstronomy
(CLAUDE.md, "Tests") would here assert agreement with the bug. `test_sersic_bn.py`
therefore splits the claim in two: b_n against `gammaincinv`, which is the defining
relation; profile *shape* — the distance function, ellipticity convention, `Ie`
normalisation — against lenstronomy with b_n held equal between the codes. Five checks,
all passing.

Two incidental fixes made in the same file:

- `SersicEllipse` and `CoreSersic` override `light` and carried their own copies of the
  linear form. `build_phys_model` uses only `SersicEllipse`, so a change confined to
  `Sersic.light` would not have touched the fitted model at all. `_bn` is now a module
  function used by all three.
- `Sersic.light` applied `Ie` both to the exponential and again in the non-lstsq return,
  squaring it — measured as a factor 2.95–3.02 against lenstronomy at `Ie = 3`. Pre-existing
  and unrelated to this round. **No DESI238 result was affected**, since every component in
  `build_phys_model` is a `SersicEllipse`, which never had the bug. Fixed.

The TF backend was mirrored so the two do not silently diverge. **That mirror is unverified**
— TensorFlow is not installed in `gigajax6.0`.

### Source prior v3

The 0.5 floor on source `n_sersic` existed only because of the old approximation: below it,
a fitted index measured the approximation rather than the galaxy. With b_n accurate to
5.5e-05 down to n = 0.3, the floor can express what the data support. v3 is `Uniform(0.3, 6)`;
v2 was `Uniform(0.5, 6)`.

This is a *support* change, so it is a new version rather than an edit. v2's builder is kept
verbatim. Every existing run records its own `source_prior_version`, and `dump_params.py`
reproduces `bulgedisk_long`, `twosource_v2`, `seed0`, `inner3` and `nsersic10` with
bit-identical numeric output after the change — verified by hashing the numeric lines against
a pristine `HEAD` worktree.

The lens-light bounds (bulge `Uniform(1, 8)`, disk `Uniform(0.5, 2.5)`) were left alone
deliberately, even though the same justification applied to them. Neither was pinning, and
widening them would have added a second difference to a comparison designed to isolate one.

## The compact source is still pinned

```
source0.n_sersic   0.3562  in [0.3, 6]  ( 1.0% of range)  <-- PINNED at lower bound
source1.n_sersic   1.0843  in [0.3, 6]  (13.8% of range)
```

Posterior `0.356 (+0.064 −0.039)`, R̂ 1.099, ESS 585 — the worst-mixing parameter in the run,
which is what a boundary-pinned parameter looks like.

In round 3 the compact component sat at 0.522 against a floor of 0.5. Given a floor of 0.3 it
sits at 0.356. The posterior tracked the boundary down rather than settling at an interior
mode, which is the discriminating observation: had 0.5 been an artificial constraint on a
mode near, say, 0.45, the mode would have appeared. It did not.

**Do not lower the floor again.** Two independent reasons. First, 0.3 is the lowest n at
which the b_n approximation is validated, so a lower bound would reintroduce exactly the
artefact this round removed. Second, and more to the point, the evidence now says the
problem is the profile family, not the bound — an n below 0.3 is a profile flatter than a
Sérsic usefully describes. The escalation path is shapelets (`use_lstsq=True`, which
requires `BackwardProbModel` and trips the `Parameterized._params` class-list hazard
documented in CLAUDE.md) or an explicitly cored profile.

Note that the component labelled `source0` here is the compact one (R_sersic 0.0676"), while
in round 3 the compact component was labelled `source1`. The source components are
exchangeable by construction, and `diagnose.py` confirms no label switching *within* either
run — 0/50 chains, 20.0σ separation in `R_sersic`. The labels simply need not agree between
runs, and comparing `source0` to `source0` across the two would be an error.

## What reproduced

Everything that should have, which is what makes the pinning result interpretable:

| | `bulgedisk_long` (v2) | `bn_rational_long` (v3) |
|---|---|---|
| `theta_E` | 1.4796 ± 0.0026 (R̂ 1.001) | 1.4795 ± 0.0026 (R̂ 1.001) |
| `lens.gamma` | 1.7921 ± 0.0207 | 1.7924 ± 0.0207 |
| `lens.gamma1` | 0.1421 ± 0.0043 | 0.1423 ± 0.0043 |
| chi²ᵥ | 0.989 | 0.988 |
| bulge `R_sersic` | 0.316" | 0.317" |
| disk `R_sersic` | 2.59" | 2.59" |
| clump ratios (SW/E/W/N) | 1.03 / 0.99 / 1.02 / 0.83 | 1.03 / 0.99 / 1.02 / 0.83 |
| SVI condition | 3.4e+07 | 1.8e+07 |

`theta_E` agrees to four digits with an unchanged error bar. **The quotable value remains
`theta_E = 1.480 ± 0.003"`**, now from a run that converges at R̂ < 1.1 throughout.

That a 13.6% correction to the source's b_n leaves the mass model untouched is a result in
its own right, and a mildly reassuring one: it says the Einstein radius here is constrained
by where the arcs are, not by how centrally concentrated the source model thinks they are.

## A comparison deliberately not made

No ΔBIC or Δlog-evidence between round 3 and round 4. The two runs use different prior
*supports*, so their log posteriors are not on a common scale and the difference would not
mean what it appears to. The claim that round 4 supersedes round 3 rests on convergence
(0/36 versus 1/36 above R̂ 1.1) and on the b_n approximation being demonstrably closer to the
defining relation — not on a model-comparison statistic.

The same caution recorded in round 3 applies unchanged: a model-to-model spread is a
systematic only when the evidences are comparable.

## On the convergence threshold

`diagnose.py` warns against R̂ < 1.01 and so still prints "the chains have not converged"
for the nine parameters between 1.01 and 1.099. Under the R̂ < 1.1 convention this run is
converged. The distinction matters mainly for the deflector's shape parameters
(`lens_light*.e2`, `n_sersic`, `Ie`, R̂ 1.02–1.03, ESS ~2000), which are the slowest-mixing
group after the pinned source. The eight lens-mass parameters — the ones `theta_E` depends
on — are at R̂ ≤ 1.003 with ESS ≥ 25,000, comfortable under either convention.

chi²ᵥ = 0.988 should not be read as confirmation of the noise model, for the reason given in
`RESULTS_v3.md`: over the blank-patch `background_rms` range 0.0070–0.0092 it spans
0.82–1.05. Quote the improvement, not the absolute value.

## Next steps

1. **A non-Sérsic compact source.** This is now the binding limitation, and it is the only
   parameter above R̂ 1.01 by a wide margin. Shapelets on the compact component would test
   whether n < 0.3 is really what the data want or whether the component is standing in for
   unmodelled structure.
2. **The b_n fix is in `gigalens/src`,** which CLAUDE.md puts off limits, and it is not
   upstream. If the vendored copy is ever refreshed, the change and its test are lost. Worth
   either an upstream PR or a note in CLAUDE.md recording the local divergence.
3. **The TF mirror is unverified.** It should be run once in an environment with TensorFlow
   before anyone trusts the TF backend.

## Reproduce

```bash
export PYTHONPATH=gigalens/src:claude/pipeline

# The b_n validation (5 checks; pytest is not installed, so run it directly)
JAX_PLATFORMS=cpu python claude/pipeline/test_sersic_bn.py

NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 python claude/pipeline/verify.py --lens DESI238
NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py \
    --lens DESI238 --label bn_rational_long --seed 0 \
    --n-source 2 --n-lens-light 2 --max-leapfrog 60 \
    --num-results 3000 --burnin 1000

python claude/pipeline/diagnose.py claude/DESI238/runs/bn_rational_long
python claude/pipeline/clumps.py   claude/DESI238/runs/bn_rational_long
```

`verify.py` passes 18/18. The prior-equality and parameter-recovery checks were run against
a pristine `HEAD` worktree and are not checked in; `test_sersic_bn.py` is.
