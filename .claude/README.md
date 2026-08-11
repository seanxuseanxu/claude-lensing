# What is automated, and what deliberately is not

`claude/guide/` is 1,138 lines of prose across ten chapters (`INDEX.md` is the map). Prose
describes what was believed; a check enforces it. This directory is the subset of the guide
that could be moved out of prose and into something that fires on its own.

Four skills, three subagents, two hooks. The rest of the guide stays prose, and the last
section here says why.

## Hooks — fire without being invoked

| Hook | Event | Enforces |
|---|---|---|
| `hooks/guard_external_repo.py` | PreToolUse on Edit/Write/NotebookEdit | Blocks edits under `/global/*/seanjx/nanolens`. CLAUDE.md says that tree is another project and must be reported on, not patched. That rule previously held only as long as the agent remembered it. **Does not cover `Bash`** — a `sed -i` still gets through, because every substring test that catches it also fires on `pip show gigalens`. |
| `hooks/judgement_lint.py` | PostToolUse on Edit/Write of `claude/**/*.md` and `config.py` | Guide §1: every `JUDGEMENT (unmeasured)` tag names a falsifier. Warns, never blocks. `--all` sweeps the repository; `--strict` adds the heuristic rule that a tagged claim should not contain its own evidence — three tags here trip it and all three are fine, which is why it is opt-in. A lint that cries wolf on every save is the same failure as a tag on every sentence. |

Wired in `settings.json` (checked in). `settings.local.json` is gitignored and holds
per-user overrides.

## Skills — invoked at a decision point

| Skill | When | Guide |
|---|---|---|
| `lens-preflight` | Before the first fit on a system, or after any change to its data, priors or model size | §1–§3 |
| `diagnose-run` | After every `run_fit.py`, before quoting any number from it | §5 |
| `elevated-rhat` | When R̂ comes back above the gate — before proposing a longer run | §4.3–§4.4 |
| `report-round` | When a diagnosed fit is about to be written down | §6, §1 |

Each carries the failure it exists to prevent, with the number attached. `diagnose-run` is
the load-bearing one: it makes *opening `residuals.png` and `lensing_overview.png`* a step
rather than an intention, which is the guide's most-violated instruction and has no automated
substitute.

## Subagents — delegate breadth, keep judgement

Per `WORKING_WITH_AGENTS.md` §7. All three are read-only and report claims, not conclusions.

| Agent | Model | Job |
|---|---|---|
| `paper-reader` | sonnet | Answer a question against `papers/`, citing sections. "The paper is silent on X" is a first-class answer — it is how we learned the GIGA-Lens masking convention is not in the GIGA-Lens paper. |
| `claim-auditor` | haiku | Sweep `claude/**/*.md` for numbers with no traceable provenance, stale forward references, superseded values still quoted as current. |
| `run-reader` | haiku | Report a run's configuration and state from its artifacts and their mtimes. Exists because a subagent here once reported a fit as running after its process had died. |

Each has an explicit instruction against the specific way a delegate has misled this project:
reporting a circular check as confirmation, reporting an intention as a state, filling a gap
in a file from a default.

## What is not automated, on purpose

These are the guide's real content and none of them reduces to a check:

- **Deciding a residual needs explaining.** That judgement is made *before* any computation,
  and it is the one an agent is worst equipped to make alone. Ask.
- **Reading the mass model against physical typicality.** A gate here would have rejected
  Foundry I's refereed `gamma = 1.372`. The value of the typicality table is that an atypical
  value gets *noticed and argued for*, which is a conversation, not a threshold.
- **Whether a component is justified.** Argued from how far the headline parameter moves, not
  from an information criterion — the sibling paper uses no formal model selection anywhere.
- **Whether two models are competitive enough for their spread to be a systematic.** This one
  was got wrong by reasoning that *sounded* cautious.
- **Anything outward-facing**: pushing to a remote, credentials, and the final decision about
  what number goes in a paper. Kept by the human by standing agreement, not by a hook.

The unifying reason the automated list is short: **these failures are silent.** None of the
real errors in this project raised an exception. Each was caught by comparing independent
runs, re-deriving a quantity a second way, running a single-variable experiment, asserting an
invariant, or by a human asking one sentence. A hook can hold a rule that is already known.
It cannot notice that a number is wrong.
