# DESI238 fit results

> **Round 1. Partly superseded by `RESULTS_v2.md` (2026-07-30).** The N-arc underfit
> reported here is fixed there by a two-component source, and both boundary pinnings are
> resolved. The theta_E value below (1.516 ± 0.002") is still the quotable one — the
> two-source run that supersedes the *fit* has not converged well enough to supersede the
> *error bar*.

Three fits, all through the MAP → SVI → HMC pipeline in `run_fit.py`, HMC at 50 chains ×
750 draws:

| Run | GPU | theta_E prior | Purpose |
|---|---|---|---|
| `seed0` | 0 | LogNormal(log 1.5, 0.35), seed 0 | Four-image baseline |
| `seed1` | 1 | LogNormal(log 1.5, 0.35), seed 1 | Multimodality check |
| `inner3` | 2 | LogNormal(log 1.3, 0.20) | Falsification: inner three clumps only |

Pre-flight (`verify.py`) passed 5/5 before any of these ran.

## theta_E and convergence

| Run | theta_E (posterior median) | Rhat (worst) | Params with Rhat > 1.01 |
|---|---|---|---|
| seed0 | 1.516 ± 0.002" | 1.877 | 18/22 |
| seed1 | 1.516 ± 0.002" | 1.489 | 12/22 |
| inner3 | 1.516 ± 0.002" | 1.432 | 3/22 |

**No multimodality.** seed0 and seed1 (independent seeds, identical prior) agree on
theta_E to four significant figures, and the MAP multi-start distributions (1024 starts
each) show no separate low-chi-squared cluster elsewhere in the prior — the best 50 MAP
starts in each run cluster within theta_E = 1.48–1.57". A wide 1.0–2.2" prior does not, in
practice, support more than one basin here.

**Convergence is marginal, not clean.** All three runs fail the Rhat < 1.01 criterion set
in the plan — badly for seed0 (18/22 parameters), much better for inner3 (3/22, worst
1.43). `lens_light.n_sersic` is the worst offender in every run (Rhat up to 1.88, ESS as
low as 108) because it is pinned at the prior's upper bound of 6 (median 5.99 in both
seed0 and seed1) — the light profile wants a steeper core than the prior allows, and HMC
struggles to mix along a boundary. `theta_E` itself has good Rhat (1.007–1.017) and is not
the problem parameter; the light-profile block is. **Report theta_E = 1.516 ± 0.002" with
the caveat that the joint chain has not fully converged**, driven by the light profile, not
the lensing parameters.

**A second boundary pinning went unnoticed here and is recorded in `RESULTS_v2.md`:**
`source.n_sersic` sits at its *lower* bound (0.573 against a floor of 0.5) in seed0. There
are two pinnings, not one. Freeing the lens-light ceiling alone makes this one worse
(0.534); it takes a second source component to relieve it.

## The falsification test did not falsify

The inner3 run, prior-centered at 1.3" with 99% mass below 2.07", converged to the *same*
theta_E = 1.516" as the wide-prior runs and the *same* reduced chi-squared (1.837 vs
1.837, to three decimals) — the narrower prior did not pull the fit toward a
distinctly-worse, inner-only solution. This means the two priors were not actually testing
different hypotheses: 1.516" already sits inside the inner3 prior's 1–99% range, so
"fit the inner three only" and "fit all four" converged to the identical basin. **The
falsification test as designed does not discriminate the two scenarios** and should be
re-run with a prior that actually excludes the E blob (e.g. truncated at 1.5" or lower) if
a genuine A/B comparison is wanted.

## Residual quality: correcting an early misread

The plan's clump table gives position angle as "PA (deg)" without specifying a convention.
I initially converted PA using the standard astronomical convention (from North through
East, via WCS) and got clump positions that landed on empty sky in the residual — nowhere
near actual flux excesses. Re-deriving PA directly from the azimuthal-median residual image
(finding the actual local maxima and computing their angle in **array/pixel space**, from
the +row axis toward +col) matched the four documented clumps almost exactly. **The PA
values in `MODELING_PLAN.md` are array-space angles, not sky position angles.** This
matters for anyone using that table with WCS-aware tools later.

With correct positions, aperture photometry (r ≤ 3 px, on the azimuthal-median-subtracted
image) comparing data to the seed0 model:

| Clump | data flux | model flux | ratio |
|---|---|---|---|
| SW arc | 5.93 | 5.49 | 0.93 |
| W clump | 3.76 | 3.37 | 0.90 |
| E blob | 5.71 | 6.14 | 1.07 |
| N arc | 3.58 | 1.96 | 0.55 |

**Three of the four clumps, including the E blob, are reproduced to within ~10% flux** —
this is materially different from what the raw normalized-residual map suggests, because
that map is dominated by small (~1 px) positional offsets that create large point-wise
sigma even when integrated flux matches (this is why the SW arc and W clump both show
6σ+ peak pixels despite matching to 7-10% in aperture flux — sub-pixel astrometric offset,
not a flux deficit). **N arc is the outlier**, under-produced by roughly half regardless of
aperture size (3–7 px tested) and offset by ~6 px in position from the model's counterpart
feature. This is the one clump the model genuinely struggles with, not the E blob.

Chi-squared is not uniformly distributed: the central 8-pixel-radius region (203 px, mostly
deflector light) carries a mean chi² of 8.3 and accounts for 14% of the total chi² despite
being <3% of the image area. The outer region (arcs + background, 6197 px) has mean chi²
1.63 — closer to unity, and consistent with the known first-order approximation that a
scalar `exp_time` understates noise near the deflector (per `CLAUDE.md` / `README.md`).

> **Correction, 2026-07-31 — the core excess is not a centroid offset.** The sub-pixel
> astrometric explanation given two paragraphs above is right for the *arcs*, where a ~1 px
> offset does produce large point-wise sigma at matched aperture flux. It is wrong for the
> central region, and that was tested directly: scanning the model over a ±0.6 px shift
> grid, the fitted position is optimal to within 0.15 px and the chi² surface rises steeply
> in every direction — core chi² 7.10 at centre, 9.48 one 0.15 px step away, 45+ at the
> edges. There is no better centroid to move to.
>
> It is an amplitude/shape mismatch. The residual about the lens-light centre, in sigma:
>
>     r [px]   0.5    1.5    2.5    3.5    4.5    5.5    6.5    7.5    8.5    9.5   10.5
>     res/sig -2.68  +0.66  +0.58  +0.83  +0.44  +0.71  -0.23  -0.38  -0.58  -0.77  -0.72
>
> Over-bright at the centre, uniformly too faint at r = 1.5–5.5 px, too bright again
> further out. The amplitude is only 1–3%, but it is coherent over hundreds of pixels,
> which is what makes it dominate the core chi². A single Sersic is a one-shape family and
> cannot produce that S-curve; the fix is a second deflector light component (round 3).
>
> One further correction to an earlier reading of the residual figure: the apparent
> "striping" in the core is a plotting artefact, not a rendering one. Nyquist (2-pixel)
> power is 0.0000 in the model against 0.0025 in the data — the model has no 2-pixel
> structure at all. The barcode appearance is the diverging colormap saturating on adjacent
> pixels that straddle zero in a smooth dipole. Only the coherent radial pattern above is
> real.

## Verdict on the four-image interpretation

**Supported, with one exception.** The wide-prior fit reproduces three of the four
clumps (SW arc, W clump, E blob) to 7–10% flux accuracy in the same single-source,
single-EPL-deflector model, including the 2.11" E blob that motivated widening the prior
in the first place. It does not do this by sacrificing the inner clumps — SW arc and W
clump fit about as well as the E blob does. This is evidence for, not against, the E blob
being a genuine counter-image rather than an interloper.

N arc is under-reproduced by ~2x in flux and offset in position. Given three
per-clump flux calls matched at the 90-107% level, this reads as a single Sersic source
being insufficiently flexible to capture the sharper counter-image morphology at that one
location — consistent with the plan's stated escalation path (a second source component or
shapelets), not evidence that the whole four-image interpretation is wrong.

## What would need to happen next

Items 1 and 3 were done on 2026-07-30 — see `RESULTS_v2.md`. Items 2 and 4 stand.

1. ~~**Escalate the source model**~~ **Done.** A second Sersic component brings all four
   clumps within 9% (N arc 0.55 → 0.91) and chi² to 1.302. See `RESULTS_v2.md`.
3. ~~**Loosen `lens_light.n_sersic`'s upper bound**~~ **Done.** `Uniform(2, 10)` moves it
   to 7.43, off the boundary, and improves worst Rhat 1.877 → 1.306 with theta_E
   unchanged — confirming the pinning never drove the lensing result.

Original list, for the record:

1. **Escalate the source model** (second Sersic component or shapelets, per
   `MODELING_PLAN.md`'s escalation path) specifically to address the N arc underfit before
   trusting `theta_E` as a final number — the current chi² of 1.84 is not "converged
   residuals," it is "the two known systematics (light-profile boundary pinning, N-arc
   underfit) account for essentially all of the excess."
2. **Re-run the falsification test with a genuinely exclusionary prior** (e.g. theta_E
   truncated below ~1.7") to get an actual A/B comparison; the current inner3 run shares a
   basin with the four-image runs and answers a different question than intended.
3. **Loosen or reparametrize `lens_light.n_sersic`'s upper bound** (currently
   Uniform(2,6), pinned at 6 in every run) — this is the single largest driver of the
   non-convergence reported above.
4. State plainly in any external write-up: `background_rms = 0.0075` (not the 0.0315 in
   `CLAUDE.md`, which is contaminated by galaxy light — see `README.md`), no sky
   subtraction applied, and the scalar `exp_time` approximation understates noise near the
   deflector (per `MODELING_PLAN.md`'s "Known approximation").

## Artifacts

- `runs/seed0/`, `runs/seed1/`, `runs/inner3/` — MAP/SVI/HMC checkpoints, metadata.json,
  residuals.png, corner.png, svi_loss.png.
- `runs/seed0/residual_with_clumps_v2.png`, `runs/inner3/residual_with_clumps_v2.png` —
  normalised residual with the four clumps marked at corrected (array-space PA) positions.
- `runs/seed0/az_median_compare.png` — azimuthal-median-subtracted data vs. model,
  side by side; the plot that actually supports the four-image verdict above.
