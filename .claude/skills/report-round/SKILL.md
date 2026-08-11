---
name: report-round
description: Write up a completed modelling round — the RESULTS_vN.md file, the README status update, and any correction to a superseded claim. Use when a fit is diagnosed and its numbers are about to be written down. Covers the four ways this project has published something it later had to retract.
---

# Write up a round

`claude/guide/06-reporting.md` and `01-the-data.md`. Four headline claims in this project have been retracted, and none of them
was a computational error: the 0.036" systematic, DESI257's brightness ordering, the
demagnification argument, round 2's mode fraction. Each was a reading asserted in the voice
of a measurement.

## Before writing

Run `/diagnose-run` on the run first. A number without its diagnostics is not ready to be
written down.

## The file

New round → new `RESULTS_v<N>.md`. **Never edit a superseded claim away.** Corrections go
inline and dated into the document that carried the claim, with the withdrawn conclusion
struck through, and say plainly in the place someone would go looking:
*"Do not quote θ_E = 1.484 ± 0.002" from that run."*

Then update the system's `README.md` banner so the top of the file agrees with the Status
section at the bottom. The header is written before a fit exists and goes stale silently;
`DESI257/README.md` and `DESI238/README.md` are the templates.

## Checks, each pinned to a specific past failure

- [ ] **Every quoted number carries its R̂ and ESS.** Including when that produces an
      uncomfortable sentence. Say *which* R̂ — rank-normalized folded, gated 1.01 here, versus
      the classic PSRF gated ~1.2 in the Foundry paper. A threshold without its definition is
      not a number.
- [ ] **Convergent-but-partial is stated as such**, with the failing parameters named and a
      sentence on whether it matters for the quantity quoted.
- [ ] **The alternative explanation is named in the same paragraph as the detection** — not
      deferred to a future round, because those do not get run. DESI234's fifth-image test was
      named as the falsifier in round 1 and was unrun three rounds later. A sound statistic on
      a rejected model measures nothing.
- [ ] **A model-to-model spread is quoted as a systematic only if the evidences are
      comparable.** Compute the gap first. A decisively rejected model is *superseded*, and
      its θ_E is a wrong number, not a competing estimate. Inflating an error bar to cover it
      is not conservatism. If you compute ΔBIC at point estimates, say that it is a maximum-
      likelihood comparison, not a Bayes factor.
- [ ] **A component decision is argued from how much the headline parameter moved** with and
      without it — that is what the sibling paper does, and it uses no formal model-selection
      criterion anywhere. If the ambiguity survives, publish it as unresolved. Foundry I did.
- [ ] **χ²ᵥ is quoted as an improvement, not an absolute**, and with the noise-scale
      uncertainty propagated. ±10% on the noise is ±20% on χ², larger than the distance from
      unity. Note that this repository's χ²ᵥ includes the core pixels the field masks by
      convention, so it is inflated relative to a published one.
- [ ] **Partial validations are reported as partial.** Rounding a partial check up to
      "confirmed" is the fastest way to lose a reader permanently.
- [ ] **Verified and inferred are labelled differently**, every time.
- [ ] **Unmeasured readings carry the tag**, with a falsifier named:

      **JUDGEMENT (unmeasured):** <the reading>. *Falsified by:* <the run or measurement>.

      A claim with a number, a command, or a file citation behind it is **not** tagged — the
      tag's whole value is that it separates the two. `python3 .claude/hooks/judgement_lint.py
      --all` checks every tag names a falsifier; `--strict` also flags tags that look like
      measurements.
- [ ] **Round numbering matches the results files.** It has drifted once already, inherited
      from another system's count.

## Where it goes

The repository, not private notes. If a human would ever want to read it or correct it, it
belongs in the versioned write-up where it is dated and cannot drift out of step with the
code (§7).
