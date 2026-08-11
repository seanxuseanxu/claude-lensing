# A Guide to Bayesian Strong-Lens Modeling for Language Models

Written by Claude, 2026-07-31, after fitting one HST lens system (DESI238) through three
rounds of a MAP → SVI → HMC pipeline; revised 2026-08-08 after two further rounds and a move
onto a new interface, reviewing the session records for what actually went wrong; revised again
2026-08-10 after reading the five papers now in `papers/` (see `papers/INDEX.md`); split from a
single 1,138-line file into the chapters below on 2026-08-11, with the prose unchanged.

Where a lesson has a published source, it is cited to the note beside that paper rather than
asserted. Several things this guide had worked out the expensive way turn out to be standard
practice with a literature behind them, and two things it took for granted turn out to be
conventions rather than facts.

It is about **modelling** — the data, the priors, the inference, the reporting, and the person
you are doing it for. It is deliberately not about software engineering, and nothing in it
should depend on which package you use.

The guide is organized around one claim: **in lens modeling, almost every error you will
make is silent.** It does not raise. It does not produce a NaN, an obviously wrong
magnitude, or a plot that looks broken. It produces a plausible number. Every hour you
spend building redundant checks buys back a round of fitting you would otherwise have to
throw away.

---

## The chapters

Cross-references inside the guide are written `§N` and `§N.M`. **`§N` is chapter N**, the row
below; `§N.M` is a section within it. The numbering is unchanged from the single-file version,
so a `§5` citation in an older results file still resolves here.

| | Chapter | What it settles |
|---|---|---|
| §0 | [What kind of problem this is](00-what-kind-of-problem.md) | Why every error here is silent; convergence as the target, and what is reportable before you reach it. |
| §1 | [Before anything: measure the data](01-the-data.md) | Pixel scale, PSF, background RMS, units, interlopers. The preflight test. Marking the readings you have not measured. |
| §2 | [Building the model](02-building-the-model.md) | Escalate one axis at a time on structured residuals. Exchangeable versus distinct priors. Amplitude priors by rendering. Where a profile family stops being valid. |
| §3 | [Parameter-space conventions](03-parameter-conventions.md) | Unconstrained space, why a prior's *support* becomes part of your file format, and probing the column order. |
| §4 | [Running the three stages](04-running-the-fit.md) | MAP's NaN `argmax`, SVI conditioning, the elevated-R̂ remedy list in priority order, and why a marginal R̂ is a budget problem. |
| §5 | [Diagnostics, in this order](05-diagnostics.md) | The ordered checklist: SVI health, pinning, prior–posterior tension, convergence, multimodality, residual structure, invariants. |
| §6 | [Reporting](06-reporting.md) | When a model spread is a systematic and when it is a removed bias. Superseding explicitly. Partial validations. |
| §7 | [Tools, records, and provenance](07-tools-and-provenance.md) | Use the library's numerics. The published method is not the installed implementation. What to record to reconstruct a run. |
| §8 | [Working with the human](08-working-with-the-human.md) | Ask what is normal, compute what is specific. Silence is not agreement. Retract loudly. |
| §9 | [The short version](09-the-short-version.md) | All thirty rules on one page. Read this first if you are reading only one file. |

## How to use it

- **Starting a system:** §1 → §2 → §3, then the `lens-preflight` skill.
- **A fit just finished:** §5 in order, then the `diagnose-run` skill.
- **R̂ missed the gate:** §4.3, then the `elevated-rhat` skill.
- **Writing a result down:** §6 and §1's tagging rule, then the `report-round` skill.
- **In a hurry:** §9 alone.

`.claude/README.md` records which of these lessons were made to fire on their own as skills
and hooks, and — the more important half — which were deliberately left as prose.
`claude/WORKING_WITH_AGENTS.md` is the companion written for the human rather than the agent.
