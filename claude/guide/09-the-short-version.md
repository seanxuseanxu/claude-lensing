[Guide index](INDEX.md) · previous: [08-working-with-the-human.md](08-working-with-the-human.md)

---

# 9. The short version

**The data (§1)**

1. Measure every likelihood input from the files; record the derivation; test it in preflight.
2. Background RMS from blank sky in several separated patches — and keep the spread.
3. Normalize the PSF. Check its centering. Ask about its scale; it is often not in the file.
4. Interlopers: model them if you can, mask them if you must, crop only when they are far
   from the arcs — and call a crop what it is, an accepted unmodelled systematic.

**The model (§2–§3)**

5. Start simple; add a component only when a *structured* residual demands it; rule out the
   boring explanation first.
6. Calibrate amplitude priors by rendering prior draws, never by reasoning.
7. Identical priors for exchangeable components; distinct priors for physically distinct ones.
8. Know where your profile family stops being valid, and put the prior's floor there.
9. Store parameters in physical space. If you store unconstrained samples, the prior's
   *support* becomes part of your file format — version it and never edit one in place.
10. Never assume the parameter ordering. Derive it, then probe it one coordinate at a time.

**Running the fit (§4)**

11. NaN-aware selection, never bare `argmax`, over multi-start optimizers.
12. Check the variational covariance's *conditioning* — not its positive-definiteness —
    before reading any R̂. Passing the guard is a floor; read the ELBO tail sd as quality.
13. Elevated R̂: 4× the draws, change nothing else. If it does not fall, work the remedy list
    before blaming the geometry — and at high dimension reach for a **better surrogate and a
    different kernel before more draws**. Hold the surrogate fixed when comparing samplers.
    Cheaper first: plot ESS against draws used on the run you already have. Sublinear growth
    means more draws will not fix it.
14. An R̂ that only just clears the threshold means the budget is too small. Buy draws.
14b. "R̂" names two statistics differing by roughly a square — the classic PSRF (gated ~1.2 in
    the Foundry paper) and the rank-normalized folded one (gated 1.01, what we compute). Say
    which. And R̂ is only meaningful above an ESS floor: ≥50 per split half, ≥400 total, ≥4
    chains. Below it, a *passing* R̂ means nothing either.

**Diagnostics (§5)**

15. Automate prior-boundary pinning detection at the 2% level. A pinned parameter cannot mix.
    Move a suspect bound *once*: a parameter that follows it wants a different profile family;
    one that settles at an interior mode was genuinely prior-limited. And ask whether the
    component is resolved at all before reading its shape parameter as a morphology.
    - Run prior–posterior tension too, on *every* parameter including unbounded ones. Pinning
      means the prior set the value; tension means the data set it against the prior's advice,
      and that is the commoner sign of misspecification. Then read the mass model against
      physical typicality: round (`q ≳ 0.6`), near-isothermal (`gamma` 2.0–2.2), low shear,
      interlopers with mass rare. Prompts, not gates.
    - Open `residuals.png` and `lensing_overview.png`. A source component with `R_eff` several
      times the caustic scale, centred outside the caustic, is a pedestal, not a lensed galaxy.
16. Structure in the residual, not magnitude, tells you what to add. χ²ᵥ is the diagnostic
    least able to fail informatively — weight it last, and report it with the core masked.
17. Assert closed-form invariants on any geometry you derive — how many critical curves a
    given slope admits, signs, monotonicity. A phantom critical curve survived every
    quantitative check I had; only counting them exposed it.
18. Persistent multimodality is rare; suspect your sampler before your posterior. Detect
    label switching always, correct only if detected.
19. Check for flux cannibalisation whenever you add a component near the arcs.
19b. R̂ compares chains to each other, never to the truth. Chains can agree perfectly on a
    distribution whose mean is 3 SD wrong, with clean R̂ and ESS. No threshold catches it —
    only physical typicality and a cross-check against the picture do.

**Reporting (§6–§7)**

20. Report R̂ with every number. Convergence in all parameters is the goal and is reachable;
    a result is reportable before you get there if the quoted parameter is converged and you
    name the ones that are not.
21. A model-to-model spread is a systematic only if the evidences are comparable. A
    decisively rejected model is superseded, not a source of uncertainty.
22. χ²ᵥ ≈ 1 is not evidence the noise model is right — propagate the noise-scale uncertainty.
23. Supersede explicitly and date your corrections. Report partial validations as partial.
24. Test that your safety checks fire. Keep the scripts that produced your priors. Write down
    your coordinate conventions where they will be read.
24b. Decide a disputed component by measuring how much the headline parameter moves with and
    without it, then publish the ambiguity if it survives — that is what the sibling paper did,
    and it uses no formal model-selection criterion anywhere.
24c. The published method is not the installed implementation, and the paper may not say what
    you remember. Read the cited section and the code that implements it; the gap between them
    never raises.

**The human (§8)**

25. Ask what the data cannot tell you — PSF scale, prior philosophy, your compute allocation.
    Label verified versus inferred. Never invent a fact about the person.
26. A check whose description explains why it must pass has not been run. Read it.
27. Retract loudly, in the place the claim was made, with a date.
28. Before buying more compute, check whether the stated goal is already met.
29. If the deciding check is already running, say so rather than guessing its answer.
30. Durable knowledge goes where the human can read and correct it.

The unifying reason for all of it: **these failures are silent.** None of the real errors in
this project — the contaminated background estimate, the array-space PA convention, the
NaN-argmax selection, the prior-version drift, the indefinite mass matrix, the float32
eigenvalue artifact, the mislabeled progress bar, the non-discriminating falsification prior,
the phantom critical curve, the dict that impersonated a list — raised an exception or
produced an obviously wrong magnitude. Not one was caught by code review or by reasoning a
priori. Each was caught by comparing independent runs, re-deriving a quantity by a second
method, running a deliberate single-variable experiment, asserting an invariant, or chasing a
downstream anomaly backwards.

Worth noting which of them a *human* caught, and how cheaply. "The background is contaminated
by galaxy light." "Gamma greater than 2 should only have one critical curve." "Hasn't it
already converged?" Each was one sentence, none required reading my code, and each cost me a
round of work to have missed. §8 is not an afterthought to the technical sections; it is where
most of the real errors in this project were actually found.

Build the redundant check. The absence of an error message is not evidence of correctness.

---

[Guide index](INDEX.md) · previous: [08-working-with-the-human.md](08-working-with-the-human.md)
