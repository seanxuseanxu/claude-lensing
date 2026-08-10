# Pipeline known issues and next steps

Cross-cutting concerns that span lenses, or belong to the pipeline code rather than any one
system's data. Per-lens results, convergence status and modelling judgements stay in
`claude/<LENS>/README.md` / `RESULTS_*.md`, which link here for anything that follows the
tooling or the dataset rather than one fit.

---

## 1. `run_fit.py --stages hmc` overwrote `metadata.json` with argparse defaults — FIXED

A run that re-ran only the sampling stage (`--stages hmc`) wrote `metadata.json` fresh from
its own argument list, which never mentioned the MAP/SVI stages it did not re-run — so those
stages' fields were overwritten with argparse **defaults**, not the values actually used.

**Confirmed instance, in full.** `claude/DESI257/runs/r2_s2l2/metadata.json` was written by a
`--stages hmc` re-run, and misreports **five** fields, not the one originally recorded here.
Ground truth is `runs/r2_s2l2.log`'s own printed lines and `RESULTS_v2.md` §9:

| field | metadata.json | actually used |
|---|---|---|
| `n_samples` | 1024 | 4096 |
| `map_steps` | 350 | 1500 |
| `n_vi` | 1000 | 2048 |
| `svi_steps` | 1500 | 12000 |
| `svi_peak_lr` | 0.002 | 1e-3 |

`n_devices: 2` is **correct** and is not part of this bug: `metadata.json` is byte-identical
to `metadata_10k.json.bak` (md5 `7e720b80…`), i.e. it describes the 10,000-draw run, which
`r2_long.log` confirms ran on two devices. The single-device run was the 80,000-draw one
(item 2), whose metadata was replaced when the good checkpoint was restored.

**What it does *not* break, stated so the blast radius is known rather than feared:**
`diagnose.rebuild_model` reads only `api`, `lens`, `n_source`, `n_lens_light`,
`lens_light_n_max`, the two prior versions, `theta_e_center`/`theta_e_width`,
`mask_circles`/`n_masked_pixels`, and guards on `n_free_params` — all of which are correct in
that file. So the physical decoding of `r2_s2l2` was never in question. What was corrupted is
the record of *how the surrogate was optimised*, which is exactly what one needs to reproduce
or to diagnose a bad seed (item 6).

**Fixed 2026-08-09.** `run_fit.py` now merges through `merge_prior_metadata(config,
prev_metadata, stages)`: fields are mapped to stages in one auditable dict literal
(`STAGE_METADATA_FIELDS`), and any stage not in this invocation's `--stages` keeps the
previous `metadata.json`'s recorded value rather than today's argparse default. A field
absent from the previous metadata is not fabricated. `stages` always records this
invocation's own stages, and `timings` merges per stage. Tested in
`claude/pipeline/tests/test_diagnose_and_metadata.py`.

**The existing `r2_s2l2` metadata is left as it stands**, wrong fields and all. Rewriting a
finished run's provenance record by hand is a worse habit than carrying a documented error,
and this table is the correction.

---

## 2. MCLMC frozen-chain failure mode

An 80,000-draw / 10,000-burnin MCLMC re-sample on DESI257 (`claude/DESI257/runs/r2_s2l2`,
attempted from the same MAP+SVI checkpoint as the working 10,000-draw run) came back with
every chain frozen:

| | good 10,000-draw run | frozen 80,000-draw run |
|---|---|---|
| mean within-chain sd | 0.08155 | **0.0006021** (135x smaller) |
| between-chain sd of chain means | — | 0.09816 (unchanged in scale) |
| all values finite | yes | yes |
| R-hat | — | **7.86e15** |
| bulk-ESS | — | 50 (= chain count; one effective sample per chain) |

All values finite, and the between-chain spread stayed at the real posterior's scale — the
signature of 50 independent point-samples with no internal exploration, not 50 chains
converging to a point. The frozen checkpoint is preserved, not deleted:
`claude/DESI257/runs/r2_s2l2/hmc_80k_FROZEN_do_not_use.npz`.

**Cause unattributed.** Two variables changed at once relative to DESI234's successful
40,000-draw run (`claude/DESI234/RESULTS_v2.md`, burnin 5,000, two GPUs): this attempt used
burnin **10,000** and a **single** GPU. Either the longer burn-in degenerated MCLMC's
step-size adaptation, or the single-device chain layout did (recall from DESI238 round 5 that
HMC and MCLMC/MAMS use different chain-axis layouts, and a silent mismatch there produces
plausible-looking wrong output rather than an error).

**Next step, and it is a one-variable test:** 80,000 draws, burnin 5,000, on two GPUs — i.e.
DESI257's failed configuration with only the burn-in and GPU count changed to match DESI234's
working configuration. This isolates which of the two changed variables is the cause. This is
a pipeline robustness question, since MCLMC is used for every lens; it happened to surface on
DESI257 and could recur on any of them.

### 2a. Update (DESI234 round 3): it recurred in the *working* configuration, so neither
### candidate variable is the cause

`claude/DESI234/runs/r3_unmasked` (36 params, 50 chains, MCLMC) froze at **burnin 5,000 on two
GPUs** — the exact configuration nominated above as the control. That largely disposes of both
hypotheses: the freeze is not caused by burn-in 10,000, and it is not caused by the
single-device chain layout, because it happens without either.

The signature is the same one, measured the same way
(`claude/DESI234/analysis/mode_structure.py`, `RESULTS_v3.md` §5.4):

| | DESI257 80k (item 2) | DESI234 `r3_unmasked` |
|---|---|---|
| burn-in / devices | 10,000 / 1 | **5,000 / 2** |
| between-chain sd ÷ within-chain sd | 163 | 15.4 (on `theta_E`); 20.9 on the worst parameter |
| bulk ESS | 50 (= chain count) | 50.7 (= chain count) |
| max R-hat | 7.9e15 | 13.9 |

**`bulk ESS ≈ n_chains` is the diagnostic to key on**, not R-hat's magnitude — DESI234's freeze
is far milder in R-hat than DESI257's yet identical in kind. The new discriminating test, added
this round, is **chain clustering with R-hat recomputed within each cluster**: partition the
chains by their per-chain medians and re-run `diagnose_convergence` on each subset. Frozen
chains keep `bulk ESS ≈ n_chains_in_cluster` under *every* partition (DESI234: R-hat stayed
above 5 at k = 2, 3, 4 and 5); genuinely multimodal chains produce at least one cluster with
R-hat near 1. This separates the two failures, which have opposite fixes, and it costs one
script that reads the finished checkpoint.

Worth noting that the companion run `r3_fg` — same lens, same launcher, same burn-in, same
device count, 46 parameters instead of 36 — did **not** freeze (R = 1.59, bulk ESS 71–380). So
the trigger is neither the configuration nor the machine but something about the posterior
geometry the sampler was handed. `r3_unmasked` is the model with a 110-sigma object in the field
and no component to put on it, which the data rejects by Δchi² ≈ 3700; a plausible reading is
that MCLMC's step-size adaptation degenerates on a badly misspecified target. That is a
hypothesis, not a measurement.

**Revised next step:** the one-variable test on DESI257 is no longer the highest-value probe,
since its two variables are both now excluded by a counterexample. Instead, capture MCLMC's
adapted step size and trajectory length from a frozen run and compare them against a healthy
one on the same lens — the freeze is almost certainly visible there rather than in the samples.

---

## 3. The mask-aware chi-squared bug, and its fix

`diagnose.residual_figure` and `plot_overview.py`'s panel used to average normalised squared
residuals over **every** pixel, including masked ones. Since a masked interloper carries the
field's largest residuals, this inflated reported chi2_nu badly: DESI234's round-1 value read
4.225 instead of a true 1.560; DESI257's read 7.112 instead of 1.533.

Both call sites now share one helper, `diagnose.masked_chisq`, which sums `(residual/sigma)^2`
over unmasked pixels only, divided by the unmasked count — matching `ProbModel.log_prob`'s own
second return value exactly, per `CLAUDE.md`'s rule that gigalens is the ground truth.
`claude/pipeline/tests/test_masked_chisq.py` (moved out of `verify.py`, was check 7, on
2026-08-09 — it asserts pipeline correctness, not something per-lens) exercises the fixed
function directly (not a reimplementation of the formula) and would fail if either call
site regressed.

Per-lens files may keep a one-line note that their own chi-squared values use the corrected
definition — that is a fact about their numbers — but this is the general lesson: **a
chi-squared computed outside `diagnose.masked_chisq` (or `ProbModel`'s own return value) is
suspect until shown otherwise.**

---

## 4. The four-panel `residuals.png` format and the `plot_overview.py` consolidation

`residuals.png` is now a four-panel figure: observed image + critical curve, model + critical
curve, unlensed source + caustics (with the ray-traced cutout border overlaid, dashed), and
the mask-aware normalised residual. `diagnose.residual_figure` and `plot_overview.py` used to
be two separate implementations of the same panel; they now share `diagnose.masked_chisq` (see
item 3) and the same critical-curve/caustic machinery (`LensSolver.critical_and_caustics`,
`image_border_in_source_plane`, both from `gigalens.jax.analysis` — nothing here reimplements
a Jacobian or a caustic by hand). See `diagnose.residual_figure`'s docstring for the full
rationale of each panel.

---

## 5. Cross-lens: a pixel-scale checkerboard at the deflector core

**The observation, per lens:** DESI234's and DESI257's `residuals.png` both show a pixel-scale
alternating +/-sigma checkerboard at the deflector core, reaching roughly +/-5 sigma. See each
system's own `RESULTS_v2.md` for its residual panel and chi-squared.

**The cross-cutting inference, stated here because it spans both systems:** a checkerboard
that alternates sign pixel-to-pixel is the signature of a **PSF mismatch**, not an inadequate
light profile — a light-profile error smooths out over several pixels, it does not alternate
sign at the pixel scale. The same pattern at the same location relative to the deflector core,
appearing independently in two different lenses fitted with two different delivered PSF
kernels, points at a problem shared by the dataset rather than a per-lens light-profile
failure.

**Precedent and previously recommended remedy:** `claude/DESI238/PSF_REPLICATION.md` documents
rebuilding that system's PSF directly from its own parent frame with
`claude/pipeline/make_psf.py`, rather than trusting an undocumented delivered kernel — and
found three defects in the delivered PSF that way, including a real 0.008" astrometric bias
with no prior record. This section used to call rebuilding DESI234's and DESI257's PSFs the
highest-value next step for both systems.

### 5a. Correction (DESI234 round 3): the remedy above was already applied, and the two
### leading explanations have now both been tested

**The remedy was stale for DESI234.** `psf234.json` records `psf234.fits` as already built by
the *fixed* `make_psf.py` from DESI234's own parent frame on 2026-08-08 — 16 stars, 21 extended
objects rejected by the size cut, full provenance — i.e. after the defects `PSF_REPLICATION.md`
found were fixed. The checkerboard persists anyway. **Rebuilding DESI234's PSF is not the next
step; it has been done.**

**The competing explanation — that this is gigalens' `supersample >= 2` subgrid-kernel
quadrature error — is also refuted, and was refuted by direct measurement rather than by
argument.** `claude/DESI234/analysis/undersampling_check.py` runs
`gigalens.jax.analysis.diagnose_undersampling` on a 9-sample batch and compares the predicted
`delta/sigma` map against the observed residual in a 10x10 px box at the deflector core, at the
same parameter point `residuals.png` was rendered at:

| | |
|---|---|
| observed normalised-residual peak | 6.582 sigma |
| ss=2 predicted `delta/sigma` peak | 2.446 sigma |
| **Pearson r over the box** | **−0.109** |

The predicted quadrature error is a smooth, single-signed blob; the observation is sharp
pixel-scale sign alternation. They share a location and nothing else.

**What is nonetheless true, and is a separate finding that applies to both lenses:** DESI234
**is** undersampled at `supersample = 2`. Against the diagnostic's 0.2 sigma/pixel tolerance,
max `|delta|/sigma` runs 3.132 at ss=2 and 0.997 at ss=4 (certified against a ss=32 reference;
`analysis/undersampling_certified.txt`). DESI234 round 3 therefore moved to `supersample = 4`.
Two things make that generalisable to DESI257, whose bulge is comparably cuspy:

- ss=2 and ss=4 are both the `subgrid` PSF convention, so 2 -> 4 refines quadrature and does
  **not** move the PSF model. It is 2 -> 1 that crosses into `native` and changes the PSF by
  `convention_gap_max` = 5.02 sigma, a gap between conventions that supersampling cannot close.
- On an 80x80 image the wall cost of 2 -> 4 is 1.05x at batch 50 and 0.90x at batch 1 —
  kernel-launch overhead dominates the O(ss^2) compute, so the naive 4x is not what it costs.
  **The memory cost is real, though — see item 9.**

**So the honest status of the checkerboard is: cause unknown.** Both leading explanations have
been tested and neither survives. It is not a supersampling artifact, and the PSF-rebuild
remedy has already been applied to the system that still shows it. Recommended next step for
both lenses is now a *diagnostic* one — a per-pixel comparison of the delivered kernel against
a star in each system's own frame at the sub-pixel offset the fit's centroid implies — not a
rebuild. Until then it is an acknowledged floor on `chi2_nu` and belongs in each system's
systematic budget rather than in its statistical error.

---

## 6. The MAP stage can stop hundreds of nats short of the mode, silently

**The measurement.** `claude/DESI257/runs/r2_s2l2/map.npz` records `best_lp = 16603.62`.
Recomputing that same point through `ProbModel.log_prob` gives **16603.617**, so the stored
scalar is right. But the run's own MCLMC draws, scored through the identical `log_prob`,
reach **max 17285.41** and sit at **median 17198.24**. The MAP is by definition the
highest-density point in the posterior, so an optimiser finishing **681.8 nats below the best
draw of the chain it seeded** — and below the *typical* draw — has not converged. That is a
fact about the optimiser, not about where the posterior mass is.

`chisq_hist`'s tail slope over the last 150 of 1500 steps is **-0.00018 per step**: still
descending when it stopped, not plateaued.

**Why it matters more than a bad point estimate.** SVI starts from the MAP point and the
sampler is seeded from SVI, so a MAP that stopped in the foothills propagates into where
every chain begins. On DESI257 that is measurable: the SVI surrogate mean sits at
`gamma = 2.4188`, and **48 of 50 chains have `gamma < 2.66` at post-burn-in draw 0** — i.e.
the 73.4% "dominant mode" fraction that round 2 quoted is substantially a measurement of the
seed, not of the posterior's mode weights. See `claude/DESI257/RESULTS_v2.md` §3.

**Now checked, not assumed.** `diagnose.map_convergence_check` (called from `diagnose.main`)
recomputes the MAP point's log posterior, scores a thinned set of sampler draws through the
same `log_prob`, prints the `chisq_hist` tail slope, and FAILS loudly with the deficit in
nats when `map best_lp < max(sampler log posterior)`. `map_hmc_verdict` is the pure predicate
behind it, tested in `claude/pipeline/tests/test_diagnose_and_metadata.py` on both the
passing case and this measured failing one.

**The remedy is `--map-steps`, and it has to be measured per model.** 350 steps (the argparse
default) and 1500 both fell short at 36 parameters. Raise the step count until `best_lp`
plateaus *and* clears the best draw of the sampler it seeds, and record which value did it.

---

## 7. `boundary_pinning_check` was blind to `TruncatedNormal` priors — FIXED, with a residual limit

The check inspected only `tfd.Uniform` priors. `gamma`'s prior is
`tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0)` (`priors.py`), so a posterior pressing on the 3.0
ceiling was never reported. Round 1's `n_sersic` pinning was caught only because that
parameter's prior happened to be Uniform — the check was one prior family away from missing
that too.

Fixed 2026-08-09: `TruncatedNormal` bounds are read off `.low`/`.high` and reported alongside
the Uniform ones, each row labelled with its family so a hard bound is distinguishable from a
truncation. Unbounded priors (`Normal`, `LogNormal`) stay excluded on purpose — a parameter in
the tail of an unbounded prior is a different phenomenon.

**The residual limitation, stated because it bit on the very run that motivated the fix.**
The check tests the **median** of the pooled draws. On DESI257 `r2_s2l2` it now prints
`lens/mass/host/gamma 2.5365 in [1, 3] (76.8% of range) TruncatedNormal` — not flagged, and
correctly so for a median. But that posterior is bimodal: its minority mode peaks near
`gamma = 2.84` with draws reaching 2.885, and *that* is what is pressing on the ceiling. A
median-based pinning test cannot see a mode it is not centred on. Read the pinning table
together with the R-hat column: `gamma`'s R-hat there is 1.453, which is the warning that a
single summary statistic is not describing this parameter.

---

## 8. `mask_circles` is a per-run choice, and is now read back from the run

`diagnose.rebuild_model` used to build the likelihood mask from the lens config as it stands
*today*, on the reasoning that a mask is a fact about the field like the crop. DESI257 broke
that: its SW object was masked in rounds 1-2 and is unmasked from round 3, so the same config
cannot describe both, and diagnosing an old run against the new config would silently compare
a different likelihood than the one that was fitted.

`run_fit.py` already recorded `mask_circles` and `n_masked_pixels` per run. As of 2026-08-09
`rebuild_model` uses the **recorded** value when present, falls back to the config only for
runs predating the key, prints a `NOTE:` when the two differ, and raises if the rebuilt mask's
kept-pixel count disagrees with the recorded `n_masked_pixels`. That last check is what makes
the reconstruction verifiable rather than hopeful.

General lesson: anything that changes the *likelihood* — the mask, the noise model, the crop —
must be recorded per run and read back from the run, not from today's config. A prior's
support is already handled this way (`priors.py`'s versioning); the mask was the gap.

---

## 9. Raising `supersample` costs memory as ss², and the MAP/SVI batch buffer is per-device

Found the expensive way on DESI234 round 3: with `supersample` raised 2 -> 4, the previously
working `--n-samples 4096` died with
`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 178.02GiB [executable_name='jit_run_map']`
on **two** 80 GB A100s. Halving to 1536 asked for 67.07 GiB and still failed.

**The second failure is the informative one.** 67 GiB is comfortably under 2 x 80 GB, so the
allocation is not being sharded across devices — the MAP/SVI batch buffer is allocated **per
device**, and the ceiling is one card after JAX's 75% preallocation, about **61 GiB**. Adding
GPUs does not raise it.

The scaling is clean enough to size a run by arithmetic instead of by trial:

| | cost/sample | 4096 | 2048 | 1024 |
|---|---|---|---|---|
| ss=2 | 10.9 MiB | 44.6 GiB | 22.3 GiB | 11.2 GiB |
| ss=4 | **43.6 MiB** | 178 GiB ✗ | 89 GiB ✗ | **43.6 GiB** ✓ |

(43.6 MiB/sample is read off both failures — 178 GiB at 4096 and 67 GiB at 1536 — which agree.)

**The practical rule: raising `supersample` by a factor k divides the affordable `--n-samples`
and `--n-vi` by k².** DESI234 round 3 went 4096 -> 1024 and 2048 -> 1024, landing on the same
43.6 GiB footprint round 2 ran at, and paid for the smaller batch with `--map-steps` 1500 ->
5000. Smoke-test `--stages map,svi` with `--map-steps 3 --svi-steps 3` on the **largest**
model variant before launching a long fit; it takes under a minute and it is the difference
between finding this at launch and finding it an hour in. This applies to DESI257 whenever it
follows DESI234 to ss=4.

---

## Next steps, ordered by value

1. ~~**Rebuild the PSF for DESI234 and DESI257** (item 5)~~ — **withdrawn, see item 5a.**
   DESI234's PSF was already rebuilt from its own frame on 2026-08-08 and the checkerboard
   persists; the supersampling explanation was also tested and refuted. The replacement step is
   diagnostic: compare each delivered kernel per-pixel against a star in that system's own
   frame, at the sub-pixel offset the fit's centroid implies.
2. ~~**Isolate the MCLMC frozen-chain cause** (item 2) with the one-variable test: 80,000 draws,
   burnin 5,000, two GPUs on DESI257.~~ — **superseded, see item 2a.** DESI234's `r3_unmasked`
   froze in exactly that configuration, so neither burn-in nor device count is the cause. The
   replacement step is to capture and compare MCLMC's *adapted step size and trajectory length*
   between a frozen run and a healthy one; and to run `mode_structure.py`-style within-cluster
   R-hat on any future R-hat failure before calling it multimodal.
