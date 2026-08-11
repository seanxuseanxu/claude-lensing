---
name: lens-preflight
description: Condition a new lens system, or a new model variant on an existing one, before spending compute — measuring every likelihood input from the files, calibrating amplitude priors by rendering, and probing the parameter ordering. Use before the first run_fit.py on a system or after any change to its priors, data or model size.
---

# Preflight a system

`claude/guide/` chapters 1–3. Every number that enters the likelihood is derived from the files in front of
you and recorded with its derivation. Documentation, headers and memory are starting points,
not sources.

## Run the automated part first

```bash
export PYTHONPATH=claude/pipeline
CUDA_VISIBLE_DEVICES=0 python claude/pipeline/verify.py --lens <LENS> \
    --n-source <N> --n-lens-light <M>
```

~75 s. Add `--all-variants` (~210 s) after a change to the pipeline itself rather than to one
lens. It covers data, params, simulator, end-to-end and the SVI guard.

## What `verify.py` does not cover — do these by hand for a new system

**Pixel scale** from the WCS: `sqrt(|det CD|) * 3600`. A cutout carries no WCS of its own, so
this comes from its provenance — *verify the provenance*, do not assume it. DESI238's cutout
was confirmed bit-for-bit identical to a named slice of the parent drizzle.

**Background RMS from blank sky**, in several separated patches, **and record the spread**.
`sigma_clipped_stats` on a cutout centred on a galaxy measures the galaxy: it gave 0.0315
against a true 0.0075, a factor of 4, which down-weights the arcs into irrelevance and
quietly turns the fit into a fit to the deflector alone. You need the spread again at
reporting time.

**PSF**: normalization (an unnormalized PSF inflates every render and the fit absorbs it into
the amplitudes), centring, and scale. Scale is frequently not derivable from the file — ask,
and record the answer as sourced from the human.

**Units**: counts or a rate. A rate image has Poisson variance `rate / exp_time`. A scalar
exposure time on a drizzled mosaic is first-order only; the `WHT` extension is the honest
per-pixel exposure map.

**Amplitude priors, by rendering**: draw ~64 samples from the prior, render, compare the
median peak to the data peak, require within a factor of ~3. Surface-brightness conventions
are unpredictable — this project's first-guess priors were 50× too faint because the
simulator multiplies by pixel area.

**Parameter ordering**: derive the names from the prior structure, then probe it — perturb
one coordinate of the unconstrained vector at a time and assert exactly the expected
parameter moved. Do this after every model or library change. A transposed ordering produces
a converged fit describing the wrong system, and nothing raises.

## Interlopers

Model > mask > crop. Modelling is the only option that uses the information and the only one
available when the contaminant overlaps the arcs. **Settle it by measurement**: fit with and
without a mass model for the object and compare the shift in the headline parameter — that
is what the sibling paper did, then masked, then stated in print that the object's nature is
unresolved. If you crop, say what it is: a decision to accept an unmodelled systematic.

## Prior bounds

Know where the profile family stops being valid and put the floor there, for a stated reason
— then a parameter pinning against it is a diagnostic rather than a formality. A bound
inherited without a reason is not a physical statement: this project allows `gamma` up to
3.0, which is how one fit reached 2.951, against Foundry I's fiducial `[1, 2.7]`.

## Record it

Conditioning prose goes in `claude/<LENS>/README.md` and the `config.py` comment blocks. Tag
the readings you have not measured:

    **JUDGEMENT (unmeasured):** <the reading>. *Falsified by:* <the run or measurement>.

and **schedule the falsifier** rather than leaving it as a forward reference. One has now
gone unrun for three rounds. Keep the scripts that produced the priors — a prior whose
provenance you cannot reconstruct is an assumption you can no longer defend.
