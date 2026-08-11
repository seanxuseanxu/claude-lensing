[Guide index](INDEX.md) · previous: [01-the-data.md](01-the-data.md) · next: [03-parameter-conventions.md](03-parameter-conventions.md)

---

# 2. Building the model: start small, escalate on evidence

The temptation is to write down the most flexible model you can afford and let the data
sort it out. Resist it. A model you cannot diagnose is a model you cannot defend, and
extra components in a degenerate problem do not sit idle — they eat each other's flux.

**Escalate one axis at a time, and let the residual choose the axis.** The DESI238
sequence, which I would repeat:

| Round | Model | Params | Motivated by | Result |
|---|---|---|---|---|
| 1 | EPL + shear, 1 Sérsic deflector, 1 Sérsic source | 22 | baseline | χ²ᵥ 1.837; N arc underfit 2× |
| 2 | + 2nd source component, freed `n_sersic` ceiling | 29 | one clump reproduced at 0.55 of its flux | χ²ᵥ 1.302; all four clumps within 9% |
| 3 | + 2nd deflector component (bulge+disk) | 36 | coherent 1–3% radial S-curve in the core residual | χ²ᵥ **0.991**; core χ² 7.09 → 2.97 |

Each step was justified by a **specific, localized, structured** feature of the residual —
not by a global χ² that was merely "too high." A coherent radial S-curve about the lens
centre (over-bright at r < 1 px, too faint at 1.5–5.5 px, too bright at 6.5–11.5 px) is a
shape a single Sérsic *cannot* produce, because a Sérsic is a one-shape family. That is an
argument for a second component. "χ² is 1.3" is not.

**Rule out the boring explanation first.** My initial reading of that core residual blamed
a sub-pixel centroid offset. A ±0.6 px shift scan disproved it: the fitted position was
optimal to within 0.15 px with χ² rising steeply in every direction. Only then was
"amplitude/shape mismatch" earned. (The centroid explanation did still hold for the *arcs*
— two features, two causes, and conflating them cost a round.)

## Priors on exchangeable versus physically distinct components

This distinction matters more than it looks:

- **Two source components: give them identical priors.** They are exchangeable —
  you have no prior knowledge of which clump is which. Let the data assign the morphology.
  Make the amplitude prior wide enough that an unwanted component can *switch itself off*
  by going faint (`Ie ~ LogNormal(log 20, 1.2)`, spanning 1.2–326 at 1–99%). Accept that
  you now have a label-switching symmetry and check for it (§5).
- **Bulge and disk: give them different priors.** They are physically distinct populations
  — compact/high-n and extended/low-n. Identical priors would add a label-switching mode
  for nothing. In DESI238 I derived the split from a 1D indicator fit to the inner-arcsecond
  profile (compact Rₑ = 0.216", n = 0.72; extended Rₑ = 1.150", n = 1.13) and set
  `R_sersic ~ LogNormal(log 0.25, 0.5)` and `LogNormal(log 1.2, 0.5)` — priors that
  overlap at about 1σ, so the separation the fit found (0.33" vs 2.73", 8.4×) is a
  statement from the data, not from me.

## Derive amplitude priors by rendering, not by reasoning

Surface-brightness normalizations depend on your simulator's internal conventions in ways
that are very hard to predict. Mine multiplied every render by the pixel area — a factor of
0.004225 at 0.065"/pix — so a unit-amplitude Sérsic peaked at 0.083 against data peaking at
7.2. My first-guess amplitude priors, copied from a demo built on a synthetic image with a
different flux scale, were **50× too faint**. The fix is mechanical and takes two minutes:

> Draw ~64 samples from your prior, render them, and compare the median peak to the data
> peak. Require the ratio to land within a factor of ~3. Then keep that as a permanent
> preflight check.

Calibrated values were `LENS_IE = 80`, `SOURCE_IE = 20`, giving median peak 5.26 against
data 7.18.

## Know where your profile family stops being valid

Analytic profiles are usually implemented with an approximation somewhere — the Sérsic
constant *b(n)* is the common one, and closed forms for it degrade badly at low index. In my
case the error reached −13.6% at n = 0.3 while being negligible above n ≈ 1. **Measure that
error against the defining relation rather than trusting the range the profile is advertised
for, and keep the measurement as a test.** Then set the prior's floor where the profile is
still trustworthy, and treat that floor as a physical statement rather than a formality.

**A boundary placed for a known reason becomes a diagnostic.** When a parameter pins against
one, the question is whether the bound is too tight or the family is wrong, and there is a
cheap experiment that distinguishes them: **move the bound once, deliberately.**

- A parameter that *settles at an interior mode* once you widen it was genuinely prior-limited.
- A parameter that *follows the bound down* is telling you the data want something outside
  the family. The fix is a different profile, not another bound.

Mine followed. Lowering the source index floor from 0.5 to 0.3 moved the pinning to 0.3, and
it pinned again in a later round under a different likelihood, a different parameter ordering
and four independent samplers. That is a profile-family verdict, not a prior artefact.

This is standard practice, not a local invention, and the published form of it is worth
copying. Foundry I re-ran its whole fit under two alternative `gamma` priors specifically to
test whether the fiducial one was pinning `gamma` near a boundary, reported that χ² *improved*
by 42 and by 552 when the boundary was raised, and stated plainly that it did not know why
`gamma` wanted to go so low (`papers/strong_lens_foundry_I/NOTES.md` §5). Two things to take
from that: the experiment is worth a full re-run, and **reporting an unresolved boundary
preference is an acceptable result.** You do not have to explain it to publish it; you have to
not hide it. Note also that their fiducial slope prior is `TruncNormal(2, 0.25, [1, 2.7])`
while this project's allows `gamma` up to 3.0 — which is how `r3_fg` reached 2.951 at all. A
bound inherited without a reason is not a physical statement.

**Then ask whether the component is resolved at all.** The same component's half-light radius
came out at 0.069" against a 0.065" pixel — below the resolution of the data. A shape
parameter fitted to an unresolved component is not a morphology measurement in the first
place. Report it as a bound, describe the component as point-like, and check that it is not
correlated with your headline parameter before you stop worrying about it (here the
correlation with θ_E was r = 0.018, so the headline was safe).

---

[Guide index](INDEX.md) · previous: [01-the-data.md](01-the-data.md) · next: [03-parameter-conventions.md](03-parameter-conventions.md)
