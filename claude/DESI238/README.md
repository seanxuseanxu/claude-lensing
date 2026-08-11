# DESI238

One lens system, HST WFC3/IR in F140W, `EXPTIME = 1197.699` s, `NCOMBINE = 3`. Fitted with
the shared pipeline in `claude/pipeline/` (see its README for the machinery and for how the
config seam works); this file covers everything specific to *this* system.

Every number below was measured from the files, not read off documentation.

> **Round 5 (`RESULTS_v5.md`, 2026-08-08) is the current result: `theta_E = 1.478 ± 0.003"`,
> all 36 parameters at R̂ ≤ 1.003.** It is the first fit under the GIGA-Lens scene API, and
> it reproduces round 4's pre-port 1.480 ± 0.003" to within 0.002" — across a changed
> likelihood normalisation, a changed parameter ordering, a different sampler and different
> code. That agreement was the outcome genuinely at risk in the port, so it is evidence, not
> bookkeeping.
>
> **Rounds 1–4 remain unreadable by the current pipeline.** They were fitted with the old
> `PhysicalModel` / `ForwardProbModel` interface, whose likelihood recomputed the noise map
> from the *simulated* image each step; the scene API derives a fixed noise map once from the
> *observed* image. The port also reordered the flat-`z` columns, so those checkpoints decode
> to different physical values under today's code and `diagnose.rebuild_model` refuses them
> outright rather than reporting the wrong numbers. To read them, check out the commit before
> the port. Their `svi.npz` files stay useful at any commit — `verify.py`'s conditioning check
> replays them, and that statistic does not depend on the parameterisation.

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
  at ~(11, 11). The exclusion matters because the pipeline sets no mask, so the likelihood
  runs over every pixel unconditionally and an in-frame companion would be fit as part of the
  lens system. (The old justification — that gigalens had no masking at all — is obsolete:
  `ImageData` takes a `mask=`. The pipeline still declines to set one, because which pixels
  to drop is a per-lens judgement. Cropping remains the choice here.)
- **`Ie` priors are order 100, not order 1.** `LensSimulator` multiplies by
  `conversion_factor = delta_pix² = 0.004225`, so a unit-`Ie` Sersic pair peaks at 0.083
  e/s against a data peak of 7.22. `verify.py` check 3 guards this.

### On units

The data is a rate (electrons/s), not counts. `ImageData` derives its noise map once, from
the **observed** image, as `sqrt(background_rms² + clip(image, 0, ∞)/exp_time)` — the Poisson
variance of a rate image scales as `rate / exp_time`. (The pre-port `ForwardProbModel`
recomputed the same expression from the *simulated* image every step, which made the variance
parameter-dependent. That is the likelihood change the port introduced.)

Passing `exp_time = EXPTIME` is correct to first order only: `NCOMBINE = 3` means coverage
varies across the frame, and the WHT map shows coverage falling to ~half depth (WHT 150 vs a
median 308) within 2.5 px of the deflector. The noise is therefore understated exactly where
it matters most. **State this in any writeup.**

**The scene API removes the obstacle that used to make this hard to fix.** `ImageData` accepts
`background_rms` and `exp_time` as either scalars *or* per-pixel arrays matching the image
shape, and also takes a fully precomputed `error_map=`. A WHT-derived per-pixel exposure map
is now a config-and-`data.py` change, not a `ForwardProbModel` subclass. It has not been
built; it remains the most worthwhile outstanding improvement to the noise model, and
`RESULTS_v5.md` §9 quotes χ²ᵥ with that caveat attached.

## Running

gigalens is pip-installed (editable). There is no `PYTHONPATH=gigalens/src` and no
`NCCL_NVLS_ENABLE=0` any more.

```bash
export PYTHONPATH=claude/pipeline

# Pre-flight. ~5 minutes. Do this before spending a GPU-hour. 18/18 expected.
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/verify.py --lens DESI238

# The surrogate, on all four GPUs, ~21 min. This stage is what buys the convergence.
python claude/pipeline/run_fit.py --lens DESI238 --label prod_svi \
    --n-source 2 --n-lens-light 2 --stages map,svi \
    --n-samples 4096 --map-steps 1500 --n-vi 2048 --svi-steps 12000 --svi-peak-lr 1e-3

# The headline sample, one GPU, ~8 min. Copying the checkpoints in holds the surrogate fixed.
mkdir -p claude/DESI238/runs/mclmc_long
cp claude/DESI238/runs/prod_svi/{map,svi}.npz claude/DESI238/runs/mclmc_long/
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py --lens DESI238 --label mclmc_long \
    --n-source 2 --n-lens-light 2 --stages hmc --sampler mclmc \
    --n-hmc 64 --burnin 4000 --num-results 12000

# Diagnostics. The lens is inferred from the run's metadata.json.
python claude/pipeline/diagnose.py claude/DESI238/runs/mclmc_long
```

`--stages` restricts which stages run (`--stages map`, `--stages svi,hmc`, …). Later stages
load the earlier stages' checkpoints from the output directory, so a crashed sampler can be
retried without redoing MAP and SVI.

**All four GPUs are usable for one fit.** MAP, SVI and the sampling stage shard across
`jax.device_count()`, and each floors its batch to a device multiple. The old "one GPU per
process, the multi-device path is broken" rule died with `compat.py`.

**Use MCLMC, not HMC** (`--sampler {hmc,mams,mclmc}`, default `hmc` for backwards
compatibility). On this problem MCLMC is ~9× faster per draw and reaches the same worst R̂
with 1.7× the minimum ESS at two thirds the wall time. See `RESULTS_v5.md` §5.

**3,000 draws is too small a budget here.** Five HMC runs differing only in seed gave worst R̂
of 1.003 / 1.019 / 1.009 / 1.093 / 1.019 — only 2 of 5 clear R̂ < 1.01, which is what a
marginal budget looks like. Use the `mclmc_long` budget (64 chains × 12,000 draws) or larger,
where the margin is wide; if you want R̂ lower still, buy draws. `RESULTS_v5.md` §6.

Running independent seeds concurrently on separate GPUs remains the multimodality check: a
`theta_E` prior spanning 1.0–2.2" can support distinct solutions fitting different subsets of
the clumps. If two runs disagree on `theta_E` beyond their quoted uncertainties, report the
disagreement rather than picking the better-looking one.

```bash
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s python claude/pipeline/run_fit.py --lens DESI238 \
      --label seed$s --seed $s --n-source 2 --n-lens-light 2 &
done
wait
```

Long runs must outlive the shell that launched them. Prefer whatever background-execution
facility your tooling offers, since it will also notify you when the job exits — polling with
`sleep` wastes the tail. Failing that, detach by hand:
`setsid nohup env ... python -u ... > log 2>&1 < /dev/null & disown`. A job backgrounded with
a bare `&` from a subagent's shell dies with that agent's process group. Always `python -u`; a
killed job with buffered stdout leaves a log containing nothing diagnostic.

The falsification run — fit the inner three clumps only, and ask whether the E blob is a
counter-image or an interloper:

```bash
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/run_fit.py --lens DESI238 \
    --label inner3 --theta-e-center 1.3 --theta-e-width 0.2
```

## Results

Five rounds have run. `RESULTS.md` covers round 1 (single Sersic source, 22 params);
`RESULTS_v2.md` covers round 2 (29 params); `RESULTS_v3.md` covers round 3 (`twosource_v2`
at 29 params and `bulgedisk` at 36). Both of the first two carry inline corrections dated
2026-07-31 — read those before the sections they correct. `RESULTS_v4.md` covers round 4
(`bn_rational_long`), summarised below. **`RESULTS_v5.md` covers round 5 (`mclmc_long`) and
is the current result.** `MODELING_PLAN.md` records how the priors were derived from this
image's flux scale.

[`runs/INDEX.md`](runs/INDEX.md) lists all 21 run directories — which round each belongs
to, what it was for, and whether its numbers or its checkpoints are still usable. Start there
rather than opening `metadata.json` files.

**`theta_E = 1.478 ± 0.003"` is the quotable value**, from round 5's `mclmc_long`:
`1.4784 ± 0.0025"`, **all 36 parameters at R̂ ≤ 1.003**, min bulk-ESS 18,692, χ²ᵥ 0.968.
Round 4's pre-port `bn_rational_long` gave 1.4795 ± 0.0026" at R̂ < 1.1, and round 3's
`bulgedisk_long` 1.4796 ± 0.0026" with one parameter at R̂ 1.338. All three supersede round
1's 1.516 ± 0.002".

- **The round-4 → round-5 agreement is the port's validation.** 1.480 → 1.478, a shift of
  0.002" against a ±0.003" error bar, produced by a different likelihood normalisation, a
  different parameter ordering, a different sampler and different code. The noise convention
  moved from a parameter-dependent variance to one fixed from the observed image — a real
  change to the likelihood, not a refactor — so this was the outcome at risk.
- **R̂ values are not comparable across the port.** Convergence now comes from ArviZ
  rank-normalized split R̂ via `diagnose_convergence`; the old pipeline used a classic PSRF
  that over-reports on these marginals. Compare pass/fail against a stated threshold, not the
  statistic.
- **What produced the convergence was the preconditioner, not the chain length.** Round 4
  stalled at worst R̂ 1.338 and more sampling was not the fix. 21 minutes of MAP+SVI on four
  GPUs (4,096 particles, 1,500 MAP steps; n_vi 2,048, 12,000 SVI steps, peak LR 1e-3) took
  the SVI covariance condition to 2.3e+06 and the ELBO tail sd to **0.1** — two orders of
  magnitude better than a healthy 22-parameter run. Every sampler then cleared or nearly
  cleared the target from that one surrogate. At 36 parameters the surrogate is the binding
  constraint.
- **`src1` is unresolved; quote its index as a bound.** `R_sersic = 0.069"` is below the
  0.065" pixel and `n_sersic` pins at the 0.3 floor in every sampler independently. It carries
  11% of source flux, sits 0.32" from `src0` inside `src0`'s own `R_sersic` — a knot on the
  host — and is uncorrelated with `theta_E` (r = 0.018), so the headline is unaffected.
  Report `n_sersic ≤ 0.37 (prior-limited)`; its position, flux and size are measured and
  converged.
- **Use `diagnose.py`'s percentiles, not `dump_params.py`'s mean/sd, for `mclmc_long`.** A
  suspected in-place mutation in `diagnose_convergence` corrupts one column of the latter.
  Reported but not independently verified, and it lives in read-only `nanolens` —
  `RESULTS_v5.md` §9.

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
  preconditioner. Label switching, a dead component, and multimodality were each ruled out
  (`RESULTS_v2.md`, "Convergence got worse", items 1-3 — measured, not assumed: 0 label
  crossings in 50 chains, components separated by 24.4 sigma);
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

**JUDGEMENT (unmeasured):** requiring one source to produce images at both 1.06" and 2.11" is
a strong demand on a single elliptical deflector, and that is what widens the `theta_E`,
ellipticity and shear priors relative to the gigalens demo. *Falsified by:* fitting the
four-clump reading against a reading that drops the E blob, and comparing chi-squared — the
one-source assignment of all four clumps is the repository owner's decision (above) and has
never been tested against its alternative. The radii are measured; "strong demand" is not
quantified anywhere, and the prior widths it justifies are therefore justified by argument.

### A recurring pinning problem

A source `n_sersic` has pinned at its lower floor in every round so far.
`lens_light.n_sersic` also pinned at its round-1 ceiling of 6 (fixed by raising it to 10; it
lands at 7.43), and `source.n_sersic` pinned at 0.5 went unnoticed for a full round.
`diagnose.py` now flags any Uniform-prior parameter within 2% of a bound; check it routinely.

Round 4 tested whether the floor itself was responsible — see below. It was not. Round 5
reproduces the pinning under a different likelihood, a different parameter ordering and four
independent samplers, and adds the physical reading: the pinned component is *unresolved*
(`R_sersic` 0.069" against a 0.065" pixel). The data want a flatter core than a Sérsic in the
validated range can give. **Do not lower the floor a third time** — 0.3 is the lowest index
at which `test_sersic_bn.py` validates the rational-(3,2) `b_n` approximation, so a fitted
index below it reports a profile-accuracy artefact, not a morphology. The remaining move is a
different source profile.

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

## The JAX 0.11 shim layer is gone

Four JAX 0.11 breakages in `gigalens.jax.inference` once required `claude/pipeline/compat.py`
— a removed `jax.tree_map`, a `float(jnp.nanmin(..., keepdims=True))` `TypeError`, a `pmap`
sharding failure, and a context-mesh `ValueError` that forced a `pmap`-free HMC. **Every one
was already fixed upstream.** The shim was avoidable work, it is deleted, and the multi-device
path it disabled now works. The one piece that survived was never a shim: the SVI-covariance
conditioning guard, now `claude/pipeline/svi_health.py`. The library still inverts that
covariance in float32 without checking it.

The lesson, recorded in `claude/guide/07-tools-and-provenance.md` §7: check for a newer upstream release
*before* writing a compatibility shim.

**Sampler output layouts differ, and getting one wrong is silent.** `HMC` returns
`(n_steps, n_devices, n_chains_per_device, n_params)`; `MAMS` and `MCLMC` return
`(n_chains, num_results, n_params)`, chains first. Determined empirically with tiny runs, not
from the docs. `run_fit.normalize_hmc_samples` validates each layout against the shape it must
have and raises otherwise — a silent chain/step swap yields a plausible but wrong split-R̂,
which is the failure mode that convinces you an unconverged run converged. `hmc.npz` stores
`(n_chains, n_steps, n_params)` regardless of sampler, because the analysis tools key on it.

## Reporting

Quote R̂ alongside any parameter value — a `theta_E` without a convergence diagnostic is
not a measurement. Under the four-image interpretation, expect `theta_E` between the inner
clumps and the E blob, roughly 1.3–1.9". A posterior pinned at either prior edge means the
two radial scales are pulling the model apart; `diagnose.py` warns on this. The decisive
plot is the normalised residual: coherent arc-shaped structure above ~3σ means the model is
wrong regardless of R̂.

`report/desi238.tex` is the write-up of `runs/bulgedisk_long` — **round 3, and therefore
superseded**. `make -C report` rebuilds it (needs `module load texlive/2024`). It pulls figures
from that run directory via `\graphicspath` rather than copying them, so it always reflects the
run it describes; pointing it at round 5 means changing the run directory and re-checking every
number, not just the path. Until that is done, quote `RESULTS_v5.md`, not the PDF.
