# DESI234

HST WFC3/IR F140W imaging of DESI-234.4783+14.7232. This file records the data and the
conditioning decisions, so that when a number does appear it is clear what it was
conditioned on. **For the quotable result, see `RESULTS_v2.md`** (round 2, converged);
**for the current state of the modelling, see `RESULTS_v4.md`** (round 4, did not converge;
the ellipticity bound did not rescue the foreground model). `RESULTS_v2.md` and
`RESULTS_v3.md` carry correction banners dated 2026-08-10: round 3 does not supersede round
2's `theta_E`, and neither does round 4.

> **Current status — round 4, 2026-10-05: the foreground model still does not give a
> physical deflector, and round 2's `theta_E` stands.**
>
> | round | run | params | converged | `chi2_nu` | mass model `q` | `gamma` |
> |---|---|---|---|---|---|---|
> | 1 | `r1_s1l1` | — | — | — | 0.736 | 2.677 |
> | 2 | `r2_s2l2` | 36 | yes, R-hat <= 1.008 | 1.300 | 0.900 | 2.199 |
> | 3 | `r3_unmasked` | 36 | **no** — chains frozen, max R-hat **13.94** | 1.662 | 0.943 | 2.130 |
> | 3 | `r3_fg` | 46 | **no** — under-mixed, max R-hat **1.92** | **1.082** | **0.143** | **2.951** |
> | 4 | `r4_fg_q` | 46 | **no** — max R-hat **2.125**, ESS floor **failed** (min 63) | 1.199 | **0.295** | **2.898** |
> | 4b | `r4_fg_q2` | 46 | **no** — max R-hat **2.330**, ESS floor **failed** (min 56) | 1.276 | **0.302** | **2.917** |
>
> **Round 4 bounded the deflector's ellipticity and the deflector went to the new wall.**
> With `mass_prior_version=2` enforcing gigalens' soft `q >= 0.2` floor, the host fits
> `e1 = -0.4692` against a bound at `-0.4714` — `diagnose.py` reports it PINNED, with 96.3%
> of draws within 0.01 of the bound. **This is not an artifact of the fit's poor starting
> point:** among the top 10% of draws by log posterior the pinned fraction is 96.0%, against
> 96.3% over all draws. The likelihood prefers the excluded region. The bound relocated the
> pathology from `q = 0.143` to `q = 0.295`; it did not remove it.
>
> **Round 4's own question is therefore still unanswered.** Whether a foreground mass
> component survives when the deflector is required to be round cannot be read off a fit in
> which the deflector is still not round. The fitted foreground SIS `theta_E = 0.4752"`
> (R-hat 1.609) **must not be quoted** — it is a statistic on a non-converged fit with an
> unphysical host, the same error retracted on 2026-08-10.
>
> **The sampling explanations are now spent.** A second run (`r4_fg_q2`) tripled MAP and SVI
> and closed two thirds of the MAP-vs-HMC deficit; convergence did not improve, and the
> deflector pinned at the same place (`q = 0.302`). Chain length had already measured
> sublinear twice. Of guide 04 §4.3's ordered list, only "improve the model" is left.
>
> See `RESULTS_v4.md` §4 for the tagged judgement that the host EPL and the foreground SIS
> are competing for the same deflection, and the fit that would falsify it.
>
> **`r3_fg`'s deflector is a 7:1 needle with a near-singular slope.** Axis ratio
> `q = 0.143` (16/84: 0.134–0.153) and `gamma = 2.951 ± 0.005`, against a prior
> `TruncatedNormal(2.0, 0.25, 1, 3)` — 3.95 prior sigma out, with a posterior 50x narrower
> than the prior. Galaxy-scale deflectors are round and near-isothermal. Every other run on
> this system is physically sane; `r3_fg` alone is not, and adding the foreground component
> is what broke it. Its `chi2_nu = 1.082`, the best on the system, bought that fit with an
> unphysical mass distribution. Found by `diagnose.prior_posterior_tension`, added
> 2026-08-10 after none of the existing checks caught it — the `e1`/`e2` priors are
> unbounded `Normal(0, 0.15)`, so `boundary_pinning_check` cannot see them by construction.
>
> **RETRACTED — the previous version of this banner.** It claimed round 3 had settled, "on
> grounds non-convergence cannot overturn", that the interloper carries mass (foreground SIS
> `theta_E = 0.370"`, 12.2 sd from zero) and that modelling it moves the main `theta_E` by
> 0.066". Both numbers come from `r3_fg` and neither survives. The 0.066" shift is the
> difference between a sane mass model and a broken one, not a measurement of the
> interloper's effect; the SIS detection is what a needle-shaped deflector needs nearby to
> stay consistent with the arc. The statistic itself was sound — the prior is a `HalfNormal`
> with mode at zero (`pipeline/priors.py:190-212`), so zero was available and was excluded —
> but a sound statistic computed on a rejected model measures nothing.
>
> **Round 2's `theta_E = 1.539 +0.003/-0.002"` is the system's current number**, with the
> interloper masked and its mass therefore unmodelled — a known and still-unquantified
> systematic. Supporting it: `r3_unmasked` has a sane mass model (`q = 0.943`,
> `gamma = 2.130`) and lands at `theta_E = 1.5386`, reproducing round 2 to **0.0007"**
> despite its frozen chains. Two independent conditionings agreeing to under a milliarcsecond
> is worth more than one converged fit alone.
>
> **Round 4 ran that re-run, and it was necessary but not sufficient.** The reading that
> round 3's `r3_fg` failure was modelling rather than sampling is upheld — bounding the
> ellipticity was the right move and it was not enough. What round 4 adds is that the
> *unbounded prior* was not the whole fault: the likelihood presses against the bound
> wherever the bound is put, so there is something wrong with the 46-parameter model itself,
> most likely a degeneracy between the host EPL and the foreground SIS (`RESULTS_v4.md` §4).
> `r3_unmasked`'s frozen chains remain a genuine and separate sampler problem; float64 is
> still the first thing to vary there (`pipeline/data.py:118` pins float32) and is still
> untried.

Everything below is measured from this system's own files. Nothing is inherited from
DESI238 except the *structure* of `config.py` and two prior widths that are explicitly
flagged as unmeasured (`SOURCE_PRIOR`'s scales, and the bulge/disk split).

## The data

| | |
|---|---|
| Parent frame | `final_92_drz.fits` (1997 x 2258, HDU 1 = SCI, HDU 2 = WHT) |
| Filter | F140W |
| EXPTIME | 1197.699 s, NCOMBINE 3 |
| Pixel scale | 0.06500 "/px, from `sqrt(|det CD|) * 3600` on the SCI CD matrix |
| Cutout | `cutout234.npy`, 120 x 120, parent slice `[1008:1128, 1170:1290]`-equivalent (see `cutout234.json`) |
| Crop used for fitting | `[20:100, 20:100]` -> 80 x 80 |
| PSF | `psf234.fits`, 27 x 27, stacked from 16 stars (`psf234.json`) |
| Peak | 6.63 e/s (the deflector core) |

`cutout234.json` and `psf234.json` record the parent frame's sha256, the exact slice, and
every threshold used, so both products are reproducible from `claude/pipeline/make_cutout.py`
and `make_psf.py`.

### Why the cutout is centred on light, not on the catalogue position

The deflector centroid is 3.4 px from where `all_world2pix` puts the catalogue RA/Dec.
`make_cutout.py` centres on the measured centroid instead. The crop then places the
deflector at (39.13, 39.47) against an array centre of (39.5, 39.5) -- 0.37 px in row,
0.03 px in column. The row residual is 0.024", half of `MASS_PRIOR`'s `center_sigma`, so the
deflector-centre prior still covers the truth comfortably. No integer crop does better, and
shifting the data to do so would correlate neighbouring pixels and invalidate the
independent-noise likelihood.

## Conditioning decisions

### Background RMS: 0.0075 e/s

200 random 40x40 patches in the *parent* frame, rejecting any patch containing a pixel above
0.15 e/s or whose WHT median is below half the frame's. A single seed's median is not a
measurement, so the estimator was run over five:

| seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| median | 0.00756 | 0.00743 | 0.00746 | 0.00752 | 0.00767 |

The 16-84th percentile *within* a seed is 0.0069-0.0100. So the patch-to-patch spread
(+/-15%) dwarfs the seed-to-seed spread (+/-1.5%), and 0.0075 is the frame's robust median.

The rule is deliberately **self-contained per frame** -- no cross-frame rescaling -- so that
DESI234 and DESI257 are conditioned identically and neither inherits a judgement made on
DESI238. What that costs: the same estimator returns 0.0078 on DESI238's frame against the
0.0075 adopted there by hand, so it reads ~4% high, and rescaling would give 0.0072 here.
4% in sigma is 8% in chi-squared, smaller than the +/-15% patch-to-patch spread the
measurement carries anyway. **A reduced chi-squared near 1 is not by itself evidence that
the noise model is right.**

`sigma_clipped_stats` over the cutout is the documented trap: the cutout is mostly galaxy,
the clip never lands on sky, and the answer comes out several times too large -- which would
down-weight the arc into irrelevance.

### No sky subtraction

The cutout's clean corners read 0.004-0.015 e/s. That is the deflector's own Sersic wings
plus its neighbours, not a sky pedestal -- true blank sky in this frame is consistent with
zero. Subtracting it would remove real galaxy light and bias the lens-light `R_sersic` and
`n_sersic` low. The Sersic profile is there to model those wings.

### One masked interloper, at 2.30", PA 314

A compact source at crop (64, 14) peaks at 0.82 e/s -- 110 sigma above the subtracted
deflector light -- and is the brightest thing in the field after the deflector.

**No square crop removes it without clipping the arc.** For a box of half-width h centred on
the deflector, the interloper is inside whenever h > 25.5 px, while the arc runs out to
h = 26.5 px. The two requirements do not overlap, so per `claude/guide/01-the-data.md` the options
are mask or model, and cropping is not among them.

Rounds 1-2 mask it, radius 12 px: its residual profile falls through 1 sigma between r = 11
and 12 px, and the nearest arc pixel is 48 px away. That costs 441 of 6400 pixels (6.9%) and,
more importantly, declines to use the interloper's own light. **Round 3 replaces the mask
with a modelled Sersic and compares.** Until then the fit carries an unmodelled interloper as
an accepted systematic and must be reported as carrying one.

**JUDGEMENT (unmeasured):** leaving it unmasked is not an option, because the source
components' priors put no constraint on where a component may sit, so a free Sersic will go and
sit on a 110-sigma blob — the cheapest way for the model to buy chi-squared. *Falsified by:*
an unmasked fit with no component at the interloper's position, checking whether a source
component's fitted centroid lands on it. Round 3's `r3_unmasked` is that configuration, but its
chains froze (`RESULTS_v3.md` §5.4), so it does not answer the question. DESI257 ran the same
check successfully and found the hazard did *not* materialise there
(`claude/DESI257/README.md`), which is reason to doubt this as stated.

## Lens configuration

Read as a **cusp quad**. After subtracting an elliptical-isophote median about the deflector:

| clump | r (") | PA (deg, array-space) | peak (sigma) | \|e\| | major axis vs tangential |
|---|---|---|---|---|---|
| S arc, E end | 1.81 | 145 | 85 | 0.31 | 2.5 deg |
| S arc, mid | 1.71 | 184 | 62 | 0.39 | 2.9 deg |
| S arc, W end | 1.74 | 215 | 57 | 0.40 | 3.7 deg |
| N counter | 1.11 | 358 | 37 | 0.05 | -- (round) |

**These PAs are array-space angles**, measured from the +row axis toward +col in the crop --
not sky PA. Converting them through the WCS the standard way (North through East) puts the
markers on empty sky. This is the single most misreadable field in `config.py`.

The southern structure is continuous from PA 125 to PA 235 -- 110 degrees -- at a radius that
stays within 1.66-1.82" the whole way, with three local maxima and no gap between them, each
elongated to within 4 degrees of tangential.

**JUDGEMENT (unmeasured):** a chance alignment of three galaxies does not produce a
constant-radius, tangentially-elongated, unbroken ridge over 110 degrees, so the southern
structure is lensed. *Falsified by:* a measured rate — the frequency of three-object alignments
at fixed radius and tangential orientation in blank-field imaging of this depth. No such rate
has been computed here; the radius range and the orientations above are measured, the
inference from them to "not a chance alignment" is not.

Three merging images along one arc with a fourth opposite is a cusp configuration, and the
fourth is where a cusp quad puts it: the N clump at PA 358, opposite the arc's midpoint to
within 5 degrees, at the smaller radius 1.11", and the faintest of the four. Taking theta_E
as the mean of the two radii gives **1.43"**, with `theta_E_logsigma = 0.25` (1-99% range
0.80-2.55").

The competing reading -- that the southern structure is a single fold pair and the three
maxima are source structure -- is not excluded by the image and does not move theta_E much,
since both put the tangential critical curve between 1.11" and 1.75". It changes the *source*
model, which is what round 2's ladder tests.

The interloper at 2.30" is deliberately **not** in the clump list, and that is a judgement.
Against it being a fifth image: it is round (|e| = 0.036) with its major axis 66 degrees from
tangential; at 3.5x the counter-image's brightness where a fifth image should be faintest;
and at PA 314 it sits 44 degrees from the counter-image on the same side, which no standard
quad geometry produces. If that is wrong, theta_E is biased low.

**Settled, 2026-08-10, by the repository owner's reading:** the object is almost certainly not a
fifth image. It is much brighter than the other images and completely out of position for one.
That is the same evidence the paragraph above lists, read by someone with the modelling
experience to weigh it, and it is accepted here as the answer.

**The fifth-image run is therefore retired, not deferred.** It was named as the falsifier in
round 1 and stayed unrun through three rounds; it is closed by judgement rather than left as a
forward reference that never resolves. Note what does *not* follow: round 3's foreground SIS
(`theta_E = 0.370"`, 12.2 sd from zero) has been withdrawn along with the rest of `r3_fg`'s
posterior — see the correction banner at the top of `RESULTS_v3.md`. So the object is foreground
rather than lensed, and whether it carries enough mass to matter for `theta_E` is **unestablished**
and is round 4's question.

## Flux scale

The simulator multiplies its output by `delta_pix**2 = 0.004225`, so `Ie` is of order 10-100,
not order 1. A value copied from another lens can be 50x wrong, so both are calibrated here
by rendering 200 draws from this system's own prior and comparing the median rendered peak
with the data (measured 2026-08-08, `n_source=1, n_lens_light=1`):

| | median rendered peak | 16-84th pct | data | scale | adopted |
|---|---|---|---|---|---|
| lens light | 7.375 e/s | 3.44-19.5 | 6.63 e/s (core) | 0.899 | `LENS_IE = 72` |
| lensed source | 0.6222 e/s | 0.112-2.95 | 0.64 e/s (arc) | 1.029 | `SOURCE_IE = 21` |

**The median is a weak anchor.** The LogNormal `Ie` prior's 16-84th percentile spans nearly
two orders of magnitude, so this fixes the prior's *centre* to within a factor of a few, not
better. That the provisional placeholders (80/20) came out within 10-20% of the calibrated
values is coincidence, not a validation of guessing.

## Status

- [x] Cutout and PSF built, both with reproducible sidecars
- [x] `config.py` written, every number measured on this system
- [x] `LENS_IE = 72` / `SOURCE_IE = 21` calibrated by prior-draw render (see below)
- [x] `verify.py --lens DESI234` passing — 5/5, check 5 honestly skipped (`RESULTS_v3.md` §3)
- [x] Round 1 fit (`r1_s1l1`, 22 params) — **did not converge**, worst R-hat 1.652;
      lens-light `n_sersic` pinned at 89.5% of its range. Not a measurement; see `RESULTS_v2.md` section 3.
- [x] Round 2 fit (`r2_s2l2`, 36 params, bulge+disk light + 2 source components) —
      **converged**, all 36 parameters R-hat <= 1.008. `theta_E = 1.539 +0.003/-0.002"`.
      See `RESULTS_v2.md` for the full result, the residual-structure caveats (PSF mismatch
      at the core, unmodelled arc knots), and section 6's finding that the second source
      component sits outside the caustic and is not multiply imaged.
      **Its `theta_E` stands as the system's current number**, qualified by the unmodelled
      interloper mass. It was briefly held superseded by round 3; that reading was retracted
      on 2026-08-10 when `r3_fg` was rejected — see the banner at the head of this file and
      the one on `RESULTS_v2.md`.
- [x] Round 3 fits (`r3_unmasked` 36 params, `r3_fg` 46 params; both unmasked, `supersample = 4`)
      — **neither converged**, max R-hat 13.94 and 1.92, and **`r3_fg` is additionally rejected
      as unphysical**: `q = 0.143`, `gamma = 2.951`. Its positive conclusions are withdrawn; see
      the correction banner on `RESULTS_v3.md`. What survives is `r3_unmasked`'s
      `theta_E = 1.5386` on a sane mass model, reproducing round 2 to 0.0007" despite frozen
      chains — support for round 2, not a replacement.
- [x] Round 4 fit (`r4_fg_q`, 46 params, `mass_prior_version=2`) — **did not converge**:
      46/46 parameters above the 1.01 gate, worst R-hat 2.125, and the Vehtari ESS floor
      **failed** (min bulk-ESS 63 against 400 total / 50 per split half), so R-hat is
      uninterpretable in both directions. **Its question is not answered.** Bounding the
      ellipticity moved the deflector from `q = 0.143` to `q = 0.295` and pinned it against
      the new bound (96.3% of draws within 0.01 of it; 96.0% among the top 10% by log
      posterior, so not a start-point artifact). The ESS-vs-draws slope is 0.08 — sublinear
      — after 5x round 3's burnin, so chain length is not the remaining lever. See
      `RESULTS_v4.md`.
- [x] Round 4b (`r4_fg_q2`, same model, MAP and SVI tripled) — **the surrogate lever, pulled
      and exhausted.** It closed the MAP-vs-HMC deficit from 193.75 to 69.60 nats and
      produced no convergence improvement at all: worst R-hat 2.125 -> 2.330, min ESS
      63 -> 56, slope 0.08 -> 0.04. The deflector pins identically (`q = 0.302` against
      0.295), so two surrogates 124 nats apart agree on the same unphysical mass model.
      `RESULTS_v4.md` §8.
- [ ] The modelling question round 4 exposed: is the host EPL pinning because the foreground
      SIS is taking its mass? `RESULTS_v4.md` §4 states the judgement and names the falsifier
      — fix the SIS `theta_E` and see whether the host relaxes to `q > 0.5`.
- [ ] `r3_unmasked`'s frozen chains: try float64 (`pipeline/data.py:118` pins float32).
      Untouched by round 4.
- [x] ~~The fifth-image falsification run~~ — **retired 2026-08-10, not run.** Closed by the
      repository owner's reading: the 2.30" object is much brighter than the other images and
      out of position for a fifth image. The object is foreground. Whether it carries mass that
      matters for `theta_E` is a separate question, now unestablished and owned by round 4.

`verify.py`'s check 6 (the SVI guard) will report itself **skipped** until round 1 gives this
lens its own known-good and known-diverged runs. A skipped check is honest; it must not be
weakened to pass.

## Results

| round | run | params | converged | theta_E | notes |
|---|---|---|---|---|---|
| 1 | `r1_s1l1` | 22 | no (worst R-hat 1.652) | 1.532" (unreliable) | lens-light `n_sersic` pinned at bound |
| 2 | `r2_s2l2` | 36 | yes (worst R-hat 1.008) | **1.539 +0.003/-0.002" — the system's current number** | `RESULTS_v2.md` — converged, with the interloper masked and its mass therefore unmodelled: a known, still-unquantified systematic |
| 3 | `r3_unmasked` | 36 | no (worst R-hat 13.94) | 1.5386 (unconverged; agrees with round 2 to 0.0007") | `RESULTS_v3.md` — chains **frozen** (§5.4); chi2_nu 1.662. Mass model is sane (`q = 0.943`, `gamma = 2.130`), so it supports round 2 rather than replacing it |
| 3 | `r3_fg` | 46 | no (worst R-hat 1.92) | — | `RESULTS_v3.md` — **rejected as unphysical**: `q = 0.143`, `gamma = 2.951`. chi2_nu 1.082 was the best of the round, bought with a 7:1 needle. Its foreground SIS `theta_E = 0.370"` is a statistic computed on a rejected model and measures nothing |
| 4 | `r4_fg_q` | 46 | no (worst R-hat 2.125; **ESS floor failed**, min 63) | — | `RESULTS_v4.md` — same model under `mass_prior_version=2` (`q >= 0.2`). Deflector **pinned at the new bound**, `q = 0.295`, `gamma = 2.898`. Pinning holds among the highest-posterior draws, so the bound did not fix the model, it relocated the pathology. Foreground SIS `theta_E = 0.4752"` **not quotable** |
| 4b | `r4_fg_q2` | 46 | no (worst R-hat 2.330; **ESS floor failed**, min 56) | — | `RESULTS_v4.md` §8 — same model, MAP and SVI tripled. A 124-nat better surrogate gave no convergence gain and the same pinned deflector (`q = 0.302`, `gamma = 2.917`). Rules the surrogate out as the cause |

**Round 2's `theta_E = 1.539 +0.003/-0.002"` is the quotable number**, qualified by the
unmodelled interloper mass. The claim that modelling that mass shifts `theta_E` by 0.066"
was **retracted 2026-08-10**: the shift is the difference between a sane mass model and a
broken one, not a measurement of the interloper's effect. `RESULTS_v3.md` §7 gives the
systematic budget.

**What round 4 was to establish, it did not.** Whether a foreground mass component survives
once the deflector is required to be round is still open, because the deflector under a
`q >= 0.2` bound is still not round — it sits on the bound. The interloper's mass therefore
remains an **unquantified systematic on round 2's number**, exactly as it was before round 4,
and no round has yet produced a converged fit that both models that mass and keeps a
physical host. `RESULTS_v4.md` §4 and §6 state what would.
