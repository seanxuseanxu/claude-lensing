[Guide index](INDEX.md) · previous: [02-building-the-model.md](02-building-the-model.md) · next: [04-running-the-fit.md](04-running-the-fit.md)

---

# 3. Parameter-space conventions: the highest-density region for silent bugs

Nearly every inference package works internally in an **unconstrained** space, with a
bijector mapping to physical values. For a `Uniform(low, high)` that is
`low + (high − low)·sigmoid(z)`. Three consequences, all of which bit me:

**(a) If you checkpoint unconstrained samples, the prior's *support* becomes part of your
data format.** Widening the DESI238 source `n_sersic` from `Uniform(0.5, 4)` to
`Uniform(0.5, 6)` made an *already-finished* run report `source.n_sersic = 0.6146` where
the fit had actually said `0.5729`. Reduced χ² moved 1.837 → 1.841 from the same cause.
**No error was raised. Nothing about the samples' shape or finiteness changed — only their
meaning.**

There are two fixes, and **the first is better**:

> **Fix 1 — store physical values.** Nothing forces a saved posterior to be in unconstrained
> space. If it holds constrained parameters, the samples mean the same thing forever, no
> matter what you later do to the prior. Keep the unconstrained array alongside if you need
> it to restart a stage, but let the physical values be the artefact you read. I did not do
> this and should have; the versioning below patches a problem a different storage choice
> would have prevented outright.

> **Fix 2 — if you must store unconstrained samples, version the priors.** Record a prior
> version with every run, and when you change a support, **keep the old definition verbatim**
> rather than editing it in place, so your diagnostics can reconstruct the exact prior a run
> was fitted under. Location-scale changes (a Normal's σ) are safe; support changes are not.

Fix 2 is worth having even under Fix 1, because the prior is part of a run's provenance
regardless of how you store the samples. But it should be a record, not a load-bearing
decoder ring.

**(b) Constrain in exactly one place.** One function that everything calls. Every inlined
transformation elsewhere is a future bug. Correspondingly: the optimizer, the variational
stage and the sampler all return unconstrained values, so anything you interpret physically
without passing it through that function is nonsense that will look like a number.

**(c) Never assume the parameter ordering — derive it, then probe it.** Take the names from
the prior structure, then verify the mapping empirically: perturb one coordinate of the
unconstrained vector at a time, map it to physical space, and assert that exactly the
parameter you expected moved. Do it for every index and every model variant.

This matters most right after a model or library change, when the ordering can shift without
anything raising. A transposed or off-by-one parameter vector produces a fit that converges
to a physically absurd but numerically finite answer, and you will spend a day blaming the
data. Component identity is the dangerous case: bulge parameters read as disk, or one
source's flux attributed to the other, gives you a converged fit describing the wrong system.

---

[Guide index](INDEX.md) · previous: [02-building-the-model.md](02-building-the-model.md) · next: [04-running-the-fit.md](04-running-the-fit.md)
