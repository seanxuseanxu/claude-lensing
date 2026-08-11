[Guide index](INDEX.md) · previous: [05-diagnostics.md](05-diagnostics.md) · next: [07-tools-and-provenance.md](07-tools-and-provenance.md)

---

# 6. Reporting: the part most likely to be wrong even when the fit is right

**Quote R̂ and ESS alongside every parameter value.** A θ_E without a convergence
diagnostic is not a measurement. Apply this even when it produces an uncomfortable
sentence: round 1's θ_E was reported to three significant figures *with* the note that the
joint chain had not fully converged, rather than dropping the caveat because the number
looked precise. **Say which R̂ you computed** — the classic PSRF and the rank-normalized folded
statistic share a name and differ by roughly a square (§5 step 3). Quoting "R̂ < 1.01" without
that is quoting a number a reader cannot place.

**Name the alternative explanation in the same paragraph as the detection.** Not as a falsifier
deferred to a future round — those do not get run. DESI234's fifth-image test was named as the
falsifier in round 1 and was still unrun three rounds later, closed in the end by judgement
rather than measurement. The discipline is to state, at the moment of writing the detection,
what else produces this signal and why it is excluded. That single habit would have caught
round 3's foreground SIS: a 12.2 sd detection of a *rare* thing, in a fit whose deflector had
collapsed to `q = 0.143`, where "the model is absorbing an error elsewhere" was an obvious
competing account and was never stated. Note the statistic itself was computed correctly, on a
prior with mode at zero, so zero was genuinely available and genuinely excluded — **a sound
statistic on a rejected model measures nothing**, and no amount of care in the statistic
substitutes for checking the model that produced it.

**A spread between models is only a systematic if the models are competitive.** This is
where I got the DESI238 write-up wrong, and the error is an easy one to make because the
cautious-sounding move is the incorrect one.

The headline moved from 1.516 ± 0.002" (round 1) to **1.480 ± 0.003"** (round 3), and I
reported the 0.036" gap as "the honest systematic floor on θ_E from light-profile choice,"
reasoning that a model-to-model spread an order of magnitude larger than either error bar
must dominate. That reasoning is wrong. **Averaging over models, or inflating an error bar
to span them, is only meaningful when the models have comparable evidence.** Round 3 beats
round 2 by Δlog L = +911.9 for 7 parameters (ΔBIC ≈ −1762, robust across the whole
background_rms range), and round 1's single-Sérsic source is worse still. A model that the
data rejects this decisively contributes nothing to a systematic budget — its θ_E is not a
competing estimate, it is a wrong one. The correct statement is that round 1 is superseded,
full stop; the 0.036" is the size of a *bias that has been removed*, not an uncertainty that
remains.

> **The test.** Before quoting a model-to-model spread as a systematic, compute the
> evidence gap. If one model is decisively preferred, quote its error bar and say the
> others are excluded. Reserve the spread-as-systematic treatment for models that are
> genuinely hard to choose between — where the evidences are close and the difference is a
> real ambiguity in what the data supports. Inflating an error bar to cover a rejected
> model is not conservatism; it is reporting a number the data has already ruled out.

The caveat that survives: my ΔBIC is evaluated at point estimates, so it is a
maximum-likelihood comparison, not a Bayes factor. A proper marginal likelihood would
penalize the 36-parameter model's larger prior volume more heavily — though nowhere near
enough to close a 1762 gap. Say which one you computed.

**χ²ᵥ ≈ 1 is not evidence your noise model is right.** Propagate the uncertainty you
measured in §1. Over the blank-patch range 0.0070–0.0092, DESI238's χ²ᵥ = 0.989 spans
0.82–1.05 — ±10% on the noise scale is ±20% on χ², larger than the distance from unity.
**Quote the improvement (1.302 → 0.991), not the absolute value.** Conversely, this makes
the per-pixel exposure map more worth building, not less.

**Model comparison: report Δlog L, ΔBIC, and the parameter count together**, and check the
verdict survives your nuisance uncertainties. Bulge+disk over single-Sérsic deflector was
Δlog L = +912 for 7 parameters, ΔBIC ≈ −1762, robust across the whole background_rms range.
A ΔBIC that flips when you move the noise scale within its measured error is not a result.

**Supersede explicitly; never silently overwrite.** Each round of DESI238 got its own
results file (`RESULTS.md`, `RESULTS_v2.md`, `RESULTS_v3.md`), and corrections were added
*inline and dated* to the superseded documents — including struck-through text where a
conclusion was withdrawn — rather than by editing the old claim away. When a number should
no longer be used, say so in the place someone would go looking for it:
*"Do not quote θ_E = 1.484 ± 0.002" from that run."*

**Report partial validations as partial.** When I productionized an ad hoc photometry
script, it reproduced the original clump ratios to within a few percent but left a 5–11%
discrepancy in absolute flux for three clumps, from an unresolved difference in the binning
convention. That went in the writeup as an unresolved discrepancy. Rounding a partial check
up to "confirmed" is the single easiest way to lose a reader's trust permanently.

---

[Guide index](INDEX.md) · previous: [05-diagnostics.md](05-diagnostics.md) · next: [07-tools-and-provenance.md](07-tools-and-provenance.md)
