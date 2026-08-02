# DESI238

One lens system, HST WFC3/IR in F140W, `EXPTIME = 1197.699` s, `NCOMBINE = 3`. Fitted with
the shared pipeline in `claude/pipeline/` (see its README for the machinery and for how the
config seam works); this file covers everything specific to *this* system.

Every number below was measured from the files, not read off documentation.

## The data

| File | Contents |
|---|---|
| `final_94_drz.fits` | DrizzlePac product. HDU1 `SCI` (2258×1997, float32, `BUNIT = ELECTRONS/S`, full WCS), HDU2 `WHT`, HDU3 `CTX`, HDU4 `HDRTAB`. HDU0 has no data. |
| `cutout238.npy` | 120×120, dtype `>f4`. **Bit-for-bit identical to `SCI[1010:1130, 1165:1285]`** — verified, not inferred. |
| `psf238.fits` | Single HDU, 27×27, dtype `>f8`. |
| `config.py` | Every fitted number, with the evidence for each in comments. |

- `delta_pix = 0.065` — from the SCI CD matrix, `sqrt(|det CD|) × 3600`. The cutout carries
  no WCS of its own, so this comes from its provenance as a crop.
- `exp_time = 1197.699` (EXPTIME).
- `num_pix = 80` after cropping; the cutout itself is 120×120.

### Three traps in the raw files

1. **Both arrays are big-endian, and JAX refuses them.** `jnp.asarray(np.load(...))` raises
   `TypeError: Error interpreting argument ... as a JAX value`. `data.load_image` casts.
2. **The PSF is not normalized** — it sums to 1.1445, so an unnormalised kernel scales every
   simulated image up by 14%. It is odd-sized and centered (flux centroid within 0.12 px of
   pixel 13,13), so no recentering is needed. `data.load_psf` divides by the sum.
3. **The PSF carries no WCS or scale keywords**, so its scale cannot be read off the file.
   It is at the same 0.065"/pix as the science image — confirmed by the repository owner,
   not derivable from the data. It is therefore not oversampled relative to the drizzled
   grid, and `subgrid_kernel` handles any `supersample > 1` on its own.

### Data conditioning decisions

- **`background_rms = 0.0075`, not 0.0315.** `sigma_clipped_stats` over the whole cutout
  gives 0.0315, and that number is wrong for likelihood use — the cutout is mostly galaxy,
  so the clip never finds sky. Measured instead from six widely separated 60×60 blank
  patches in the parent drizzle frame: mean −0.0020 to +0.0014, std 0.0070–0.0092. Using
  0.0315 down-weights the arcs into irrelevance. Kept here only as a documented dead end.
- **No sky subtraction.** The crop's outer ring reads ~0.035 e/s, but that is the
  deflector's own Sersic wings, not sky: the radial profile falls smoothly 0.057 (2.3") →
  0.0054 (5.9") → 0.0007 (24"), and true blank sky is consistent with zero. Subtracting
  0.035 would remove real galaxy light and bias the lens-light `R_sersic` and `n_sersic`
  low. The Sersic profile is there to model those wings; let it.
- **Crop is `cutout[18:98, 18:98]`.** Puts the deflector centroid (57.48, 57.64) at
  (39.48, 39.64) against an array centre of (39.5, 39.5), and excludes the companion galaxy
  at ~(11, 11). The exclusion matters: `grep -rn "mask" gigalens/src` returns nothing, so
  the likelihood runs over every pixel unconditionally and an in-frame companion would be
  fit as part of the lens system.
- **`Ie` priors are order 100, not order 1.** `LensSimulator` multiplies by
  `conversion_factor = delta_pix² = 0.004225`, so a unit-`Ie` Sersic pair peaks at 0.083
  e/s against a data peak of 7.22. `verify.py` check 3 guards this.

### On units

The data is a rate (electrons/s), not counts, which is what `ForwardProbModel`'s
`sqrt(background_rms² + max(model,0)/exp_time)` expects — the Poisson variance of a rate
image scales as `rate / exp_time`. Passing `exp_time = EXPTIME` is correct to first order
only: `NCOMBINE = 3` means coverage varies across the frame, and the WHT map shows coverage
falling to ~half depth (WHT 150 vs a median 308) within 2.5 px of the deflector. The noise
is therefore understated exactly where it matters most. `ForwardProbModel` accepts only a
scalar; the fix, if it proves to matter, is a per-pixel exposure map from the WHT extension,
which needs a `ForwardProbModel` subclass. **State this in any writeup.**

## Running

```bash
export PYTHONPATH=gigalens/src:claude/pipeline
export NCCL_NVLS_ENABLE=0

# Pre-flight. ~5 minutes. Do this before spending a GPU-hour.
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/verify.py --lens DESI238

# Production fit — the bulge+disk, two-source model that RESULTS_v3.md reports.
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py --lens DESI238 \
    --label bulgedisk_long --n-source 2 --n-lens-light 2 \
    --burnin 1000 --num-results 3000 --max-leapfrog 60

# Diagnostics. The lens is inferred from the run's metadata.json.
python claude/pipeline/diagnose.py claude/DESI238/runs/bulgedisk_long
```

`--stages` restricts which stages run (`--stages map`, `--stages svi,hmc`, …). Later stages
load the earlier stages' checkpoints from the output directory, so a crashed HMC can be
retried without redoing MAP and SVI.

**One GPU per process.** The multi-device path is broken (see below). Use several GPUs for
*independent* runs, not for splitting one batch — the model is too small to saturate one
H200:

```bash
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py --lens DESI238 --label seed0 --seed 0 &
CUDA_VISIBLE_DEVICES=1 python claude/pipeline/run_fit.py --lens DESI238 --label seed1 --seed 1 &
wait
```

Two seeds is the multimodality check: a `theta_E` prior spanning 1.0–2.2" can support
distinct solutions fitting different subsets of the clumps, and one batch on one device
cannot detect that. If two runs disagree on `theta_E` beyond their quoted uncertainties,
report the disagreement rather than picking the better-looking one.

The falsification run — fit the inner three clumps only, and ask whether the E blob is a
counter-image or an interloper:

```bash
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py --lens DESI238 \
    --label inner3 --theta-e-center 1.3 --theta-e-width 0.2
```

## Results

Four rounds have run. `RESULTS.md` covers round 1 (single Sersic source, 22 params);
`RESULTS_v2.md` covers round 2 (29 params); `RESULTS_v3.md` covers round 3 (`twosource_v2`
at 29 params and `bulgedisk` at 36). Both of the first two carry inline corrections dated
2026-07-31 — read those before the sections they correct. `RESULTS_v4.md` covers round 4
(`bn_rational_long`), summarised below. `MODELING_PLAN.md` records how the priors were
derived from this image's flux scale.

**`theta_E = 1.480 ± 0.003"` is the quotable value.** Round 4 is the run to cite: it is the
only one converged at R̂ < 1.1 across all 36 parameters, and gives
`theta_E = 1.4795 ± 0.0026"` (R̂ 1.001, ESS 66,778). Round 3's `bulgedisk_long` gives
1.4796 ± 0.0026" — the same number, but with one parameter at R̂ 1.338. Both supersede
round 1's 1.516 ± 0.002".

- **The 0.036" gap between them is not a systematic.** Round 1 is rejected at ΔBIC ≈ −5294,
  so the gap is a bias that was removed, not an uncertainty that remains. Quote ± 0.003".
  A model-to-model spread is only a systematic when the evidences are comparable.
- **Bulge+disk beats a single-Sersic deflector: Δlog L = +912 for 7 params, ΔBIC ≈ −1762**,
  robust across the whole `background_rms` range. chi²ᵥ went 1.302 → 0.991, core chi²
  (r ≤ 8 px) 7.09 → 2.97, and the coherent radial S-curve in the core residual flattened
  into sign-alternating scatter within ±0.75σ. Recovered geometry is a textbook bulge+disk
  (R_sersic 0.33" and 2.73", an 8.4× separation from priors overlapping at ~1σ), with **no
  arc cannibalisation** — clump ratios improved to 1.03 / 0.99 / 1.02 / 0.83 and `theta_E`
  moved only 1.484 → 1.480.
- **chi²ᵥ = 0.99 is not evidence the noise model is right.** Over the blank-patch range
  0.0070–0.0092 it spans 0.82–1.05, so ±10% on the noise scale is ±20% on chi². Quote the
  improvement, not the absolute value. This makes the WHT per-pixel exposure map more worth
  building, not less.
- **Convergence was chain length, and is fixed.** `bulgedisk_long` re-ran `bulgedisk`
  changing only the HMC length (3000 draws after 1000 burnin, 4×): worst Rhat 3.242 →
  1.338, failing count 33/36 → 10/36, chi²ᵥ unchanged at 0.989. Rhat fell instead of
  plateauing, so the residue was under-mixing, not the posterior ridge it was read as — no
  reparametrization needed. What still fails is the two components' shape parameters plus
  `source1.n_sersic`, which is pinned at the 0.5 floor and therefore cannot mix by
  construction — the sole parameter above R̂ 1.1 in that round. All eight lens-mass
  parameters sit at Rhat ≤ 1.002. Round 4 clears R̂ < 1.1 outright.
- **Round 2's 1.484 ± 0.002" was not usable** — 29/29 parameters failed Rhat (worst 7.17)
  because its SVI stage diverged and HMC ran on the resulting ill-conditioned
  preconditioner. Label switching, a dead component, and multimodality were each ruled out;
  the "slow mixing" conclusion drawn from that was wrong. The SVI fix took conditioning
  8.8e+11 → 3.4e+07, ELBO tail sd 7,384 → 6.6, worst Rhat 7.17 → 1.544, and reproduced
  `theta_E` at 1.4837 ± 0.0024 — confirming the point estimate was never affected, only the
  error bar.

### Clumps

Four clumps, all read as counter-images of one source (the repository owner's decision):

| clump | r (") | PA (deg) | residual flux (e/s) |
|---|---|---|---|
| SW arc | 1.06 | 224 | 18.8 |
| N arc | 1.25 | 339 | 8.7 |
| W clump | 1.64 | 278 | 12.5 |
| E blob | 2.11 | 88 | 24.0 |

**These PA values are array-space angles**, measured from the +row axis toward +col in the
crop — *not* sky PA. Converting them through the image's WCS the standard way (North
through East) places every marker on empty sky.

Requiring one source to produce images at both 1.06" and 2.11" is a strong demand on a
single elliptical deflector, and it is what widens the `theta_E`, ellipticity and shear
priors relative to the gigalens demo.

### A recurring pinning problem

A source `n_sersic` has pinned at its lower floor in every round so far.
`lens_light.n_sersic` also pinned at its round-1 ceiling of 6 (fixed by raising it to 10; it
lands at 7.43), and `source.n_sersic` pinned at 0.5 went unnoticed for a full round.
`diagnose.py` now flags any Uniform-prior parameter within 2% of a bound; check it routinely.

Round 4 tested whether the floor itself was responsible — see below. It was not.

### Round 4: the b_n floor was not what pinned the source

Full write-up in `RESULTS_v4.md`. Run `bn_rational_long` is identical to `bulgedisk_long` in
every setting except two coupled changes: gigalens' Sérsic constant is now a rational (3,2)
approximation accurate to 5.5e-05 over n ∈ [0.3, 8] (against −13.6% for the old linear form
at n = 0.3), and source prior v3 lowers the `n_sersic` floor 0.5 → 0.3, which the first
change made defensible.

**The pinning is not a prior artefact.** The compact source moved from pinned at 0.5 to
pinned at 0.3 — it tracked the boundary down rather than settling at an interior mode. The
data want a profile flatter than this family can express, so the fix is a different source
profile, not another version bump. **Do not lower the floor again**: 0.3 is the lowest n
where the approximation is validated, and the evidence now points at the profile family.

**All 36 parameters converge at R̂ < 1.1** (worst 1.099), the first round for which that
holds; `bulgedisk_long` had `source1.n_sersic` at 1.338. `diagnose.py` warns at the stricter
1.01 and so still prints a non-convergence line for the nine parameters in between. Under
the R̂ < 1.1 convention this is a publishable result.

`theta_E` is unchanged to four digits (1.4795 ± 0.0026 vs 1.4796 ± 0.0026), as are chi²ᵥ,
`gamma`, both `R_sersic` and all four clump ratios. That a 13.6% correction to the source's
b_n leaves the mass model untouched says the Einstein radius is set by arc positions, not by
source concentration.

## Four JAX 0.11 breakages

`gigalens.jax.inference` was written against JAX 0.2.24. All four below were reproduced by
traceback in this environment, not inferred. The fixes live in `claude/pipeline/compat.py`;
`gigalens/src/**` is not modified.

1. `jax.tree_map` removed → `AttributeError` in `SVI`. Patched.
2. `float(jnp.nanmin(x, keepdims=True))` → `TypeError` in `MAP`, on step 1. Patched.
3. `jnp.array(jnp.split(...))` under `pmap` → sharding `ValueError` in `MAP`. **Not
   patched** — the multi-device path is unused. `require_single_device()` refuses to start
   rather than dying mid-run.
4. `HMC`'s `run_chain` is `@pmap`-wrapped while `q_z` is built outside it, so Auto-sharded
   arrays meet a Manual mesh → context-mesh `ValueError`. This fires **even with one
   device**, so the guard in (3) is no defence. Worked around by `compat.hmc_single_device`,
   which reproduces the identical kernel stack without `pmap`.

Because of (4), HMC output is `(num_results, n_hmc, n_params)` — draws on axis 0, chains on
axis 1 — not upstream's 4-D pmap shape.

## Reporting

Quote R̂ alongside any parameter value — a `theta_E` without a convergence diagnostic is
not a measurement. Under the four-image interpretation, expect `theta_E` between the inner
clumps and the E blob, roughly 1.3–1.9". A posterior pinned at either prior edge means the
two radial scales are pulling the model apart; `diagnose.py` warns on this. The decisive
plot is the normalised residual: coherent arc-shaped structure above ~3σ means the model is
wrong regardless of R̂.

`report/desi238.tex` is the write-up of `runs/bulgedisk_long`; `make -C report` rebuilds it
(needs `module load texlive/2024`). It pulls figures from that run directory via
`\graphicspath` rather than copying them, so it always reflects the run it describes.
