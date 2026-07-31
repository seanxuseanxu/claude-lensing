# DESI238 fit results, round 2: freed light profile and two-component source

Follows `RESULTS.md`. Two changes, run as an ablation so each is attributable:

| Run | Change | Params | GPU |
|---|---|---|---|
| `nsersic10` (A) | `lens_light.n_sersic` ceiling 6 → 10, single source | 22 | 0 |
| `twosource` (B) | A, plus a second Sersic source component with wide identical priors | 29 | 1 |

Both at seed 0, sampler settings identical to `seed0` (1024×350 MAP, 1000×1500 SVI,
50 chains × 750 draws), so the comparison against it is clean.

## Headline

| | seed0 (baseline) | A: n_sersic freed | B: + 2nd source |
|---|---|---|---|
| reduced chi-squared | 1.837 | 1.760 | **1.302** |
| worst Rhat | 1.877 | 1.306 | 7.169 |
| params with Rhat > 1.01 | 18/22 | 10/22 | 29/29 |
| theta_E | 1.516 ± 0.002" | 1.515 ± 0.002" | 1.484 ± 0.002" |
| N arc flux ratio | 0.55 | — | **0.91** |

**Both changes helped the fit; the second badly hurt convergence.** Neither number should
be read alone.

## The N arc is fixed

This was the point of the exercise. Aperture photometry (r ≤ 3 px on the azimuthal-median
subtracted image, via the new `clumps.py`):

| Clump | data | model B | ratio | (seed0 ratio) |
|---|---|---|---|---|
| SW arc | 5.79 | 5.90 | 1.02 | 0.93 |
| E blob | 5.37 | 5.52 | 1.03 | 1.07 |
| W clump | 3.34 | 3.38 | 1.01 | 0.90 |
| N arc | 3.55 | 3.23 | **0.91** | **0.55** |

All four clumps now within 9%, where the single-source model under-produced the N arc by
roughly half. **The morphology changed too, which matters more than the fluxes.** In
`runs/twosource/az_median_compare.png` the model produces a detached E blob, a distinct SW
arc, and a separate N-arc/W-clump segment — the clumpy structure of the data. The seed0
model produced a smooth continuous ring whose brightness merely modulated. That was the
real deficiency behind the N-arc number, and it is gone.

The two source components did not split into "two separated clumps"; they became a
**compact bright knot plus a larger fainter companion**, separated by 24σ in `R_sersic`:

    source0:  R = 0.243"  Ie = 15.6   n = 1.28   at (0.158, 0.005)
    source1:  R = 0.070"  Ie = 77.3   n = 0.80   at (0.442, 0.144)

Both are compact — neither is an extended host. Given identical, deliberately wide priors
(R spanning 0.10–1.62" at 1–99%), this is the data's choice, not the prior's.

## Convergence got worse, and it is not label switching

29/29 parameters fail Rhat < 1.01, worst 7.17, ESS down to 58. Three candidate
explanations, checked rather than assumed:

1. **Label switching?** No. The detector added to `diagnose.py` counts sign changes of
   `R_sersic[comp0] − R_sersic[comp1]` per chain: **0 crossings in 50 chains**, components
   separated by 24.4σ. Every chain stayed in one labelling, as predicted from all chains
   initialising off one SVI mode. The samples were left untransformed.
2. **A component switching off?** No. `source1.Ie = 77.3` is the *brighter* of the two.
   Both components are doing real work.
3. **Multimodality?** No. Per-chain `theta_E` means form a single unimodal cluster
   spanning 1.479–1.488. But between-chain sd (0.0018) exceeds within-chain sd (0.0013) —
   each chain explores a narrow slice of a posterior wider than one chain covers in 750
   draws. **This is slow mixing, not separated modes.** The 29-parameter posterior is more
   strongly correlated and the trajectory length tuned at 22 parameters is too short.

So run B's chi-squared and clump fluxes are trustworthy (they are point-estimate
quantities), and its **error bars are not**. `theta_E = 1.484 ± 0.002"` in particular
should not be quoted: the ±0.002 is a within-chain width, and the between-chain spread is
larger than it.

## Run A, in isolation

Freeing the lens-light ceiling did what it was supposed to. `n_sersic` moved to 7.43 —
off the boundary, comfortably inside the new `Uniform(2, 10)` — and Rhat improved from
1.877 to 1.306, with the bad-parameter count halving. theta_E was unchanged at 1.515",
confirming the light-profile pinning was never driving the lensing result.

`lens_light.R_sersic` moved with it, 1.88" → 2.65", as expected from the
n–R_sersic degeneracy.

**But `source.n_sersic` pinned harder** (0.534, within 0.6% of its floor) and became the
worst Rhat in the run. The single-source model wants a sharper-edged source than a Sersic
with n ≥ 0.5 can provide. That is the pressure the second component relieves: in run B,
both components sit off the floor (1.28 and 0.80).

This is the case for the two changes being related rather than independent — a point the
ablation was needed to see.

## Two pipeline bugs found and fixed

Both were latent in the original pipeline and did not fire in seed0 by luck.

1. **`jnp.argmax` selects a diverged MAP start.** MAP runs 1024 starts and picked the best
   with `jnp.argmax(lp)`, which propagates NaN — one diverged start (1/1024 in run A,
   4/1024 in run B) and argmax returns *its* index. Reported best chi-squared was 3907.6
   when the true best was 2.006. SVI then produced NaN for 1500 steps. The MAP progress bar
   read chi2 = 2.003 throughout, so the failure was invisible until SVI died. Fixed with
   `np.nanargmax` plus a warning; the `--stages svi,hmc` resume path now recomputes the
   index rather than trusting a checkpoint written before the fix.
2. **Changing a prior's support silently rewrites old runs.** Checkpoints store
   *unconstrained* samples, so the physical value depends on the bijector. Reading `seed0`
   through the new wider source prior reported `source.n_sersic = 0.6146` where the fit had
   said `0.5729`, and chi-squared 1.841 instead of 1.837 — no error, just wrong numbers.
   Fixed with `priors.SOURCE_PRIOR_VERSION` and a preserved `_source_component_v1()`;
   `diagnose.py` reads the version from `metadata.json` and infers v1 for pre-revision runs.

An upstream gigalens hazard was also documented but not fixed (it does not affect these
runs): `Parameterized.__init__` binds `self.params` to the *class* list, so constructing
any `use_lstsq=True` profile appends `"amp"` to the class attribute and retroactively
corrupts every other instance. This will bite the shapelets escalation path.

## What to do next

1. **Re-run B with a longer HMC** before quoting any parameter from it — more draws, more
   burn-in, and a longer `max_leapfrog_steps` than the 30 tuned for 22 parameters. The
   posterior is unimodal, so this is a sampler-budget problem, not a model problem.
2. **Re-run B at a second seed** once it converges, as the multimodality check.
3. **Only then compare theta_E** between the 1- and 2-source models. The shift 1.515 →
   1.484 is 15× the quoted error bar, but the error bar is untrustworthy, so the shift is
   currently uninterpretable. This is the one number a reader will most want, and it is not
   yet ready.
4. The central chi-squared excess (mean 8.3 in the inner 8 px, per `RESULTS.md`) is
   untouched by either change and still looks like a sub-pixel centroid/PSF mismatch.

## Artifacts

- `runs/nsersic10/`, `runs/twosource/` — checkpoints, metadata.json, residuals.png,
  corner.png, svi_loss.png
- `runs/twosource/az_median_compare.png` — the plot that carries the verdict
- `clumps.py` — the azimuthal-median clump photometry, previously ad-hoc, now reproducible.
  Reproduces `RESULTS.md`'s seed0 ratios to within a few percent; absolute data fluxes run
  ~5–11% low against that table for three clumps, an unresolved difference in the
  az-median binning convention from the original throwaway script.
