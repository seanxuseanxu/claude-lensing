[Guide index](INDEX.md) · previous: [00-what-kind-of-problem.md](00-what-kind-of-problem.md) · next: [02-building-the-model.md](02-building-the-model.md)

---

# 1. Before anything: measure the data, do not read about it

Every number that enters the likelihood should be derived from the files in front of you
and recorded with its derivation. Documentation, headers, and the human's memory are all
starting points, not sources.

**Pixel scale.** Take it from the WCS: `sqrt(|det CD|) * 3600` arcsec/pixel. In DESI238
this gave 0.065"/pix. A cutout usually carries no WCS of its own, so this comes from its
provenance — *verify the provenance*. I confirmed the DESI238 cutout was bit-for-bit
identical to `SCI[1010:1130, 1165:1285]` of the parent drizzle rather than assuming it.

**The PSF.** Check three things, in this order:
1. **Normalization.** The DESI238 PSF summed to 1.1445. Passing it unnormalized inflates
   every rendered image by 14% — which the fit absorbs into the light amplitudes, so
   nothing looks wrong.
2. **Centering.** Odd-sized and centroid within a fraction of a pixel of the middle, or
   you have introduced an astrometric offset between model and data.
3. **Scale.** Whether it is oversampled relative to the science grid. This is frequently
   *not derivable from the file* — the DESI238 PSF had no WCS or scale keywords. When a
   number cannot be derived, ask the human and record the answer as sourced from them.

**The noise.** This is the single most consequential number and the easiest to get wrong.

> `sigma_clipped_stats` on a cutout centered on a galaxy does not measure the sky. It
> measures the galaxy. In DESI238 it returned `background_rms ≈ 0.0315`; the true value,
> measured from six widely separated blank 60×60 patches in the parent frame, was
> **0.0075** — a factor of 4. At 0.0315 the arcs are down-weighted into irrelevance and
> the fit quietly becomes a fit to the deflector alone.

Measure the background from blank sky, in several separated patches, and **record the
spread across patches** (0.0070–0.0092 here). You will need that spread later, in §6.

**Units.** Know whether the image is counts or a rate. A rate image (electrons/s) has
Poisson variance `rate / exp_time`, which is what a `sqrt(bg_rms² + max(model,0)/exp_time)`
noise model expects. Passing a scalar exposure time to a drizzled mosaic is correct only to
first order: `NCOMBINE = 3` means coverage varies across the frame, and the weight (`WHT`)
extension is the honest way to build a per-pixel exposure map.

**Interlopers: mask, crop, or model — and modeling is often the best of the three.** You
have three ways to handle a companion or foreground galaxy in the frame, and they are not
equally good:

1. **Model it.** Add a light profile (and, if it is at the lens redshift and massive
   enough, a mass profile) for the interloper. This is the only option that uses the
   information rather than discarding it, and the only one available when the contaminant
   overlaps the arcs. A nearby satellite can perturb the deflection field measurably, so
   for precision work on θ_E or a time delay this is not optional.
2. **Mask it.** Clean, but throws away pixels. Check that your package supports per-pixel
   masking before planning around it; not all do.
3. **Crop it out.** The crudest option, and it works only when the contaminant is well
   separated from the arcs.

For DESI238 I used (3): the companion sat in a corner far from the lensed structure, and a
crop that centres the deflector to within 0.1 px excludes it. I added an automated check that
the corner maximum stays below a threshold, so the companion cannot creep back in when
someone adjusts the crop. That was pragmatic, not the best available physics. **If an
interloper is close enough to affect the deflection, cropping is not a solution to it — it is
a decision to accept an unmodelled systematic, and should be reported as one.**

**Settle the choice by measurement, and publish the ambiguity if it survives.** Foundry I faced
exactly this on an object embedded in one of its arcs, of unknown nature — Milky Way star or
low-mass perturber. They fitted the system with a mass model for it and without, found
"negligible difference for the main lens parameters," masked it, and stated in print that its
nature is unresolved (`papers/strong_lens_foundry_I/NOTES.md` §5). That is the whole procedure:
the decision is made by the size of the shift in the parameter you care about, not by an
argument about what the object probably is. Two further conventions from the same paper worth
knowing — a nearby group member was given a **light-only** model with no mass component at all,
and **no formal model selection criterion is used anywhere in the paper**: no Bayes factor, no
BIC. Component decisions are argued from measured parameter shifts. If you are reaching for an
information criterion to justify a component, check first whether a with/without comparison on
the headline parameter answers the question more directly (§6).

## Write these down as a preflight test, not as prose

Everything above should live in a check that runs in seconds: array shapes, finiteness, PSF
sum, PSF centring, corner contamination. Prose in a README describes what you believed; a
test enforces it. Mine took ~5 minutes against fits that took hours, and every check in it
exists because something once went wrong.

## Mark the readings you have not measured

Conditioning prose is written before any fit exists, and it mixes two things that read
identically: measurements, and readings of the image that seem obvious. Four headline claims
in this project have been retracted, and every one was the second kind asserted in the voice
of the first — the 0.036" systematic above; DESI257's brightness-ordering argument against the
fourth-image reading, which was a physics error; the demagnification argument that replaced it,
whose premise the next round falsified; and round 2's "dominant mode" fraction, which measured
where the chains were seeded.

So tag them, in the README and in `config.py` alike:

```
**JUDGEMENT (unmeasured):** <the reading>. *Falsified by:* <the run or measurement that
would overturn it>.
```

Two rules keep this from becoming decoration. **A claim with a number, a command, or a file
citation in it is not tagged** — the whole value of the tag is that it separates the two, and
tagging everything destroys that. And **every tag names its falsifier**: a judgement with no
stated falsifier is either measurable now, in which case measure it, or it is not a claim and
should be deleted. `grep -rn "JUDGEMENT (unmeasured)"` then returns the list of things a
system's results are resting on that nothing has yet tested.

The failure this catches is not writing the reading down — writing it down is right, and is
what makes it falsifiable later. The failure is writing it in the same voice as a measurement
and then never scheduling the falsifier. DESI234's fifth-image run was named in round 1 and is
still unrun three rounds later; it only became visible as a debt when it was moved out of a
forward reference and into the Status checklist.

---

[Guide index](INDEX.md) · previous: [00-what-kind-of-problem.md](00-what-kind-of-problem.md) · next: [02-building-the-model.md](02-building-the-model.md)
