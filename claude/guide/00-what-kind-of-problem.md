[Guide index](INDEX.md) · next: [01-the-data.md](01-the-data.md)

---

# 0. What kind of problem this is

A strong-lens fit is a high-dimensional (20–100 parameter), non-linear inverse problem in
which:

- The likelihood is a per-pixel Gaussian whose **variance you have to supply**, and getting
  it wrong rescales your entire posterior without changing its shape.
- The model is a **sum of degenerate components** — mass and light both live at the same
  place, source structure and lens structure can each explain the same residual, and two
  identical source components can swap labels.
- The parameters have **hard physical boundaries** (Sérsic index, ellipticity, radii > 0)
  that the sampler will happily press against.
- The thing you actually want to report (θ_E, the mass slope, a time delay) is often
  **better constrained than the model is identified**, so a broken fit can still give you
  the right answer, and a converged fit can still give you a biased one.

**The target is convergence in every parameter.** That is the goal, and in practice it is
reachable — by improving the model, getting a better variational surrogate, sampling
longer, or switching sampler. Do not treat partial convergence as an acceptable resting
state; treat it as unfinished work with a known remedy list (§4.3).

But the two things are separable, and that matters for how you report. A result can be
**preliminarily reportable** while some parameters remain unconverged, *provided* the
quantity you are quoting is one of the converged ones and you say which are not. For three
rounds DESI238 sat exactly there: all eight lens-mass parameters at R̂ ≤ 1.002 while several
light-profile shape parameters failed, so θ_E was quotable and the morphology was not.

**It was eventually reachable, and the route was not more sampling.** Round 5 got all 36
parameters to R̂ ≤ 1.003 by spending twenty minutes on a better variational surrogate and
switching sampler (§4.3). Chain length had already been increased fourfold and had stopped
helping. Treat partial convergence as unfinished work with a known remedy list, and work the
list in the right order — the surrogate before the chain length.

One caveat that §4.4 develops: an R̂ that only just clears the threshold is a sign the budget
is too small, not a finish line.

---

[Guide index](INDEX.md) · next: [01-the-data.md](01-the-data.md)
