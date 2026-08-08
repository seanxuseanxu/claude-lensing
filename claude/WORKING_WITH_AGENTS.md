# Working with agents: a short guide for humans

Companion to `LENS_MODELING_GUIDE.md`, which is written for the agent. This one is written
for you. It is drawn from the DESI238 modeling work, where nearly every real error was
caught by a human doing one of the things below.

---

## 1. The failure mode is confident wrongness, not refusal

An agent that cannot do something will usually tell you. An agent that has done something
subtly wrong will report it in the same tone as work that is right. There is no tell in the
prose. This has one consequence that shapes everything else here: **your review time is
better spent on claims than on code.**

Reading a diff catches syntax and logic. It does not catch "I picked the best MAP start
with `argmax`, which propagated a NaN and selected a diverged sample" — that code is
correct-looking and was wrong for a full round. What catches it is asking *how do you know
that number is right?*

## 2. Ask for the derivation, not the answer

The single highest-yield question is some form of:

> How did you determine that? Show me the measurement.

Numbers an agent produces come from three places, and they look identical in the output:
measured from your data, read from documentation, or inferred from context. Only the first
is trustworthy without checking. On DESI238 the background RMS was quoted from a
sigma-clipped statistic that was contaminated by galaxy light — plausible, documented,
and wrong by a factor of four. It was caught by asking where the number came from.

A useful habit: when an agent states a quantity, ask whether it would still know that
number if the file it came from were deleted. If yes, it was inferred.

## 3. Your domain intuition is the most valuable thing you contribute

You will be tempted to defer on things you are unsure about. Don't — say you are unsure and
say what your experience is anyway.

Two examples from this project. The agent wrote a long passage on diagnosing "plateaued
R-hats" as evidence of posterior geometry requiring reparametrization. The correction was
one sentence: *I've never encountered plateaued R-hats; I usually got convergence after
tweaking the model, sampling longer, or switching sampler.* That was not a fact from a
paper. It was experience, and it replaced a whole section of theoretically-defensible
advice that would have sent future readers chasing a rare problem before the common ones.

Similarly: *systematic errors derived from competing models are only reportable if the
models have similar evidences.* One sentence, and it retracted a headline claim that had
propagated into four files, a LaTeX report, and two stored memories.

An agent has read more than you. You have seen more than it has. The second is scarcer.

## 4. Review in one pass, section by section, and say what is fine

The most productive exchange in this project was a single message walking §0 through §9,
with a verdict on each — several "looks good," several corrections, one apology for
unnecessary work. It was worth more than ten rounds of narrow questions, for three reasons:

- The agent could see which corrections were local and which were thematic.
- Marking sections as fine is information. It tells the agent what not to touch, which
  prevents the common failure of a revision quietly degrading something that was already
  right.
- Batching keeps the whole document in scope at once.

Close with an invitation to push back — *let me know if you disagree with any of this.*
Agents comply readily, including with instructions that are mistaken. Explicitly opening
the door to disagreement is how you get told when you are wrong, and it costs one sentence.

## 5. Give the agent a way to be checkable

The most useful standing instruction you can set is that output must be verifiable — by you
or by the agent itself. In practice this means asking for:

- The command that produced a number, not just the number.
- Diagnostics that fail loudly rather than reports that read well. A script that hard-fails
  when a covariance's condition number exceeds a threshold is worth more than any amount of
  careful prose about conditioning.
- Corrections written inline and dated, rather than files silently rewritten. You want to
  be able to see what changed its mind.

Set this once, in a project or global instruction file, rather than repeating it.

## 6. Write down what the environment actually is

Agents assume standard environments and lose hours to the gap. A short file stating the
real state — this package isn't pip-installed, this dependency is absent, these version
pins are fiction, this directory is off limits — pays for itself immediately, and the
agent can maintain it.

Related: check for a newer upstream release before authorizing compatibility work. In this
project the agent wrote and tested a shim layer for a version incompatibility that had
already been fixed upstream. Neither party checked. That is a ten-second query.

## 7. Delegate breadth, keep judgment

Subagents are good at sweeping a repository, extracting from long transcripts, and running
independent searches in parallel — work where you want the conclusion and not the material.
They are poor at anything requiring a judgment call you would want to make yourself, and
their reports carry the same confident tone as everything else. One subagent on this
project validated a change by comparing two code paths that were the same code path, and
reported the agreement as confirmation.

A second case is worth knowing because it looks like a factual report rather than a judgment.
A subagent said its fit was still running. It was not — the job had died with the agent's own
process group minutes earlier. The agent was not being careless; it simply had no more
visibility into whether its backgrounded job survived than anyone else, and reported its
intention as a state. **Reports about the state of the machine are claims too.** `nvidia-smi`,
`ps`, and the timestamps on the output files settle them in seconds.

So: delegate the reading, review the conclusions. Treat a subagent's report as a claim from
a competent stranger, not as a measurement.

One thing you control that helps: **state the agent's resources in the present tense every
time you delegate.** A line reading "the GPUs are busy" that was true an hour ago will be
acted on as though it were true now.

## 8. Say when the work is good, and be specific about why

Not politeness — calibration. "The bulge+disk ablation was the right call because it tested
the arc-cannibalisation risk directly" tells the agent which move to repeat. "Good job"
does not. The same applies to correcting: the reason matters more than the correction,
because the reason generalizes and the correction doesn't.

## 9. Decide deliberately what you keep for yourself

Some things should not be delegated even when they could be: pushing to shared remotes,
anything touching credentials, external communication, and the final decision about what
number goes in a paper. Not because the agent will do them badly, but because they are
outward-facing and hard to reverse, and because you are the one accountable for them.

Say which these are up front. An agent given a clear boundary will stop at it and ask.

---

## The short version

1. The failure mode is confident wrongness. Review claims, not just code.
2. Ask how a number was determined. Measured, documented, and inferred look identical.
3. State your experience even when unsure of it. It is scarcer than the agent's reading.
4. Review in one pass, section by section. Say what is fine, not only what is wrong.
5. Ask for verifiable output and loud diagnostics; set it once as a standing instruction.
6. Write down the real environment. Check upstream before authorizing workarounds.
7. Delegate breadth to subagents; treat their reports as claims, not measurements.
8. Explain why something was right, not just that it was.
9. Name up front what you are keeping for yourself.
