[Guide index](INDEX.md) · previous: [06-reporting.md](06-reporting.md) · next: [08-working-with-the-human.md](08-working-with-the-human.md)

---

# 7. Tools, records, and provenance

**Do not start patching a package until you have checked whether it is simply out of date.**
I wrote and carried a compatibility layer for three rounds against an incompatibility that
had already been fixed upstream — entirely avoidable work, and every line of it was surface
area for the silent failures this guide is about. "The version in front of me is broken" and
"this package is broken" are different claims, and only the second justifies patching. Ask
the human whether the copy you have is pinned deliberately or just old; they may not know it
is stale either.

**Use the library's implementation of anything numerical, and go looking for it first.** I
hand-rolled a critical-curve and caustic finder — finite differences, a Jacobian determinant,
a heuristic for the plot window. The human's redirect was one line: *"[the package] contains
some analysis functions. You can use these instead of your homegrown ones."* The library's
version was tested against its own ray trace. Mine was not merely redundant: it is what
invented the phantom critical curve of §5. The whole problem was self-inflicted.

> A hand-rolled version of something the library already has is not just wasted effort — it is
> **unvalidated**. The library's version has tests, usually against an independent reference
> implementation. Yours has your confidence.

This matters more in lens modelling than in most fields, because the quantities involved —
deflection fields, critical curves, caustics, magnifications — are easy to compute *almost*
right and very hard to eyeball.

**The published method is not the installed implementation, and the paper may not say what you
remember it saying.** Both halves of this turned up on one afternoon's reading:

- The MCLMC paper describes no mass matrix and no preconditioning of any kind, and its
  geometric-ergodicity proof assumes a globally log-convex target with bounded gradients — which
  a lens posterior is not (`papers/mclmc/NOTES.md` §4). The implementation we run seeds MCLMC's
  inverse mass matrix from the SVI surrogate (`pipeline/run_fit.py:656-658`). That is a sensible
  extension; it is also outside everything the paper proves. Citing the paper for the behaviour
  of the code is a category error in both directions.
- The GIGA-Lens paper prescribes no masking convention, and its published validation is
  simulation-only — because the one section that would have covered both, the HST demonstration
  in `results.tex`, is commented out in the LaTeX source and is not in the compiled paper
  (`papers/gigalens_paper/NOTES.md` §2, §6). A convention you believe you inherited from a paper
  may not be in it. Check the source, not your memory of the PDF.

The habit that follows: when a procedure in your pipeline is justified by a citation, read the
cited section, and separately read the code that implements it. The gap between them is a place
where nothing will ever raise.

**If the likelihood changes, re-fit — and say in advance what that puts at risk.** I moved
this work onto a new interface that altered the noise convention: the variance went from
being recomputed each step from the *simulated* image to being derived once from the
*observed* one. That is a change to the likelihood, not a refactor. Two things made it
defensible:

- **Refuse to reinterpret old checkpoints rather than converting them.** Stored samples that
  predate the change decode to plausible wrong numbers. A converter would be guesswork
  presented as data; refusing to read them sends you to the version that can.
- **Re-fit, and treat the agreement as the result.** θ_E came back 1.480 → 1.478 against a
  ±0.003" error bar, from a different normalisation, a different parameter ordering and a
  different sampler. Stated beforehand as the thing at risk, that agreement is evidence about
  the physics. Stated afterwards, it is just a number that happened to match.

**Record enough provenance to reconstruct the exact model a run used**: every hyperparameter,
the seed, the code version, the prior versions, the model configuration. Your diagnostics
should rebuild the model a run was fitted with rather than assuming today's defaults —
otherwise re-reading an old run silently reports it under a model it never used.

**Test that your safety checks fire.** The preflight suite in §1 should include an assertion
that each guard *rejects* a known-bad case and accepts the known-good ones, with an
order-of-magnitude margin either side. This is the item people skip. A safety check nobody has
seen fail is a safety check you do not know works.

**Put durable knowledge where the human can read and correct it.** If you keep private notes
of any kind, do not let them become a second copy of the project's own documentation. I wrote
seven private entries summarising a modelling round and was told to delete them: every one
restated what the results file already said. Findings, caveats and procedure belong in the
versioned write-up, where they are dated, reviewable, and cannot drift out of step with the
code they describe. **The test: if a human would ever want to read it or correct it, it goes
in the repository.**

**Keep the measurement scripts that produced your priors.** My clump photometry is still in
the repository, so the θ_E prior centre traces back to a measurement rather than to an
assertion in a planning document. A prior whose provenance you cannot reconstruct is an
assumption you can no longer defend.

**Write down coordinate conventions where they will be read.** My clump position angles are
**array-space** angles, measured from the +row axis toward +col — *not* sky PA. Converting
them through the WCS the standard way puts every marker on empty sky. This cost real time,
and it is exactly the kind of thing that is obvious to whoever wrote it and invisible to
everyone else, including to you three weeks later. Position angle, ellipticity sign, and the
handedness of the coordinate system are the three that recur.

---

[Guide index](INDEX.md) · previous: [06-reporting.md](06-reporting.md) · next: [08-working-with-the-human.md](08-working-with-the-human.md)
