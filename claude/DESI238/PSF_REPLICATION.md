# Replicating psf238.fits from the parent frame

**Date:** 2026-08-08. **Code:** `claude/pipeline/make_psf.py`. **Frame:** `final_94_drz.fits`.

The modelling plan called for one validation with a known answer before building PSFs for
DESI234 and DESI257: rebuild the DESI238 PSF from its parent frame and compare against the
delivered `psf238.fits`, which the published `theta_E = 1.478 +/- 0.003"` was fitted with.
The plan said both outcomes get written down. This is that record.

**Verdict: the builder is validated, and the delivered `psf238.fits` is worse than it.**
Three defects in my own selection were caught by this test, and one property of the
delivered kernel was caught that had not been recorded anywhere.

## 1. What the delivered psf238.fits is

Reading it before building anything:

| | value |
|---|---|
| shape | 27 x 27, `float64`, no header beyond the FITS minimum |
| sum | 1.1445 (`data.load_psf` normalises, so this never mattered) |
| FWHM | 3.009 px = 0.196" |
| EE(3 / 5 / 10 px) | 0.652 / 0.864 / 0.960 |
| centroid offset | (-0.121, -0.021) px from the central pixel |

It carries no provenance: no star count, no positions, no record of the frame or the
thresholds it came from. That is the reason this test was worth running at all — there was
no way to check it except by rebuilding it.

**Its row offset of -0.12 px is real, not a wing artifact.** Measuring the centroid in
apertures of increasing half-width gives -0.069, -0.107, -0.118, -0.130, -0.130, -0.122,
-0.121 for half = 2, 3, 4, 5, 6, 8, 13 px. A wing asymmetry would grow with aperture; this
is flat from half = 3 outward, so the core itself sits off centre. Convolving a symmetric
source with a kernel displaces the result by exactly the kernel's flux-weighted centroid,
so this is a 0.008" astrometric bias that the DESI238 fits absorbed into the deflector
centre. It does not invalidate `theta_E`, which is a radius rather than a position, but it
was not previously written down.

My independent rebuild lands at -0.140 px in row, the same sign and nearly the same size.
Two constructions sharing no code agreeing on that offset is good evidence it is a property
of the frame — the drizzle of three exposures — rather than of either construction.

## 2. Three defects this test found in my builder

None of these would have raised an exception. Each would have produced a plausible, wrong
PSF for DESI234 and DESI257, where there is no reference kernel to catch it.

### 2.1 The faint cut was an absolute flux

`PEAK_MIN = 0.15 e/s` is meaningless on a frame with a different exposure time, and both new
lenses are different frames. Worse, it was too low: the first stack reached down to 21 sigma
and put the kernel's flux-weighted centroid 0.24 px off in column while its *core* stayed
centred to 0.06 px. That is the signature of `EPSFBuilder` normalising each star by its flux
before combining, which amplifies a faint star's local background into the stacked wings.

Now `PEAK_NSIGMA = 50` times the frame's own sky sigma. At 50 sigma a wing pixel at 1% of
the peak is still a half-sigma measurement; at 15 sigma it is a sixth of a sigma, and what
stacks there is background rather than PSF.

### 2.2 `sharpness` does not reject galaxies

Reading the QA figure rather than only its numbers: of the 39 objects then being stacked,
about five were visibly extended. Measuring a flux-weighted rms radius inside r < 5 px split
the candidates into two groups with a gap between them, and stacking each group alone
settled it:

| group | n | FWHM | EE(3) | EE(5) | EE(10) |
|---|---|---|---|---|---|
| compact (size 2.25-2.48 px) | 18 | 2.829 | 0.696 | 0.920 | 1.001 |
| extended (size 2.6-2.9 px) | 21 | 3.824 | 0.403 | 0.633 | 0.868 |
| both together | 39 | 3.065 | 0.634 | 0.855 | 0.968 |

**21 of 39 were compact galaxies.** `DAOStarFinder`'s `sharpness` did not separate them at
all: the two groups span 0.41-0.87 and 0.50-0.73 respectively, overlapping almost entirely.

The fix (`SIZE_TOL = 1.10`) anchors on the *small* end of the size distribution, because
nothing in a field is more compact than a point source, and keeps sources within 10% of that
anchor. Anchoring on the median would have failed here, since the contaminants were the
majority.

### 2.3 The global sky median leaves the stack with no halo

The compact-only stack came out with EE(10) = 1.001 — negative net flux beyond 10 px, which
no real PSF has. One sigma-clipped median over a 2258 x 1997 mosaic does not describe the
background under each individual star, and `EPSFBuilder`'s per-star normalisation preserves
whatever pedestal is left.

| background treatment | FWHM | EE(10) | flux beyond r = 10 | rms vs psf238 |
|---|---|---|---|---|
| global median | 2.829 | 1.001 | -0.0010 | 0.0162 |
| none at all (control) | 2.791 | 0.998 | +0.0026 | 0.0155 |
| per-star, 15-25 px annulus | 2.858 | 0.983 | +0.0182 | 0.0134 |

Per-star local backgrounds are the only one of the three that produces a physically sensible
positive halo, and they are also closest to the delivered kernel. Now `BKG_ANNULUS =
(15.0, 25.0)`, median rather than mean so one neighbour in the ring cannot move it.

### 2.4 A fourth, found by the contract rather than by inspection

`to_science_grid` block-sums outward from the oversampled grid's central element, assuming
photutils centred the ePSF there. It does not: measured, the ePSF's own centroid sat
(-0.32, -1.09) oversampled px from its grid centre, which is (-0.107, -0.363) science px
passed straight into the kernel. `check_contract` refused the result, correctly.

`recentre_epsf` now shifts the oversampled grid by a whole number of oversampled pixels
before summing — a reindex, so exact and flux-conserving, no interpolation. On this frame it
applied (+0, +1) oversampled px and moved the column offset from -0.156 to +0.056.

**Known limitation, currently not binding.** Integer shifts correct in steps of
1/oversampling, so the residual is bounded by half an oversampled pixel = 0.167 px at
`OVERSAMPLING = 3`, against `check_contract`'s 0.15 px tolerance. A kernel can therefore fail
the contract on granularity alone, with no bug present. All three kernels built here pass
(worst is DESI238 at 0.140 px), so this has not bitten yet. The fix if it does is to raise
`OVERSAMPLING`, not to relax the tolerance — but note that oversampling 5 needs 137^2 grid
points against roughly 11,000 star pixel samples, so it would trade a centring bound for a
noisier ePSF and should not be done without checking that trade.

## 3. The replication result

The honest answer depends on which selection is used, and both are worth recording.

**With the flawed selection, the reference is reproduced almost exactly.** Sweeping the
faint cut with no size cut and a global background, `peak_min = 0.50 e/s` (37 stars) gives
FWHM 3.011 px against the reference's 3.009 — agreement to 0.001 px — with EE within 0.014,
0.005 and 0.008. That is strong evidence that my use of photutils is correct and that the
delivered `psf238.fits` was built from a similarly contaminated star list.

**With the corrected selection, it is not reproduced, and should not be.**

| | rebuilt | delivered | diff |
|---|---|---|---|
| FWHM | 2.805 px | 3.009 px | -0.205 |
| EE(3) | 0.660 | 0.652 | +0.008 |
| EE(5) | 0.880 | 0.864 | +0.016 |
| EE(10) | 0.970 | 0.960 | +0.009 |
| centroid offset | (-0.140, +0.056) | (-0.121, -0.021) | |

rms difference 2.35% of peak. The rebuilt kernel is narrower and more concentrated at every
radius, which is what removing 21 galaxies from a stack does. The delivered kernel's EE(3)
of 0.652 sits between the pure-star 0.696 and the fully contaminated 0.634, consistent with
it being partly contaminated.

So: the builder is validated — it reproduces the reference when run with the reference's
apparent assumptions, and departs from it in the specific, understood direction when run
with better ones.

## 4. An open question, flagged rather than resolved

The three kernels built with the final code do not agree on width, and DESI238 is the
outlier:

| kernel | n stars | FWHM | EE(3) | EE(5) | EE(10) | centroid offset |
|---|---|---|---|---|---|---|
| `psf234.fits` | 16 | 3.060 | 0.674 | 0.883 | 0.970 | (+0.139, -0.001) |
| `psf257.fits` | 21 | 3.057 | 0.669 | 0.878 | 0.964 | (+0.073, -0.094) |
| `psf238_rebuilt.fits` | 18 | 2.805 | 0.660 | 0.880 | 0.970 | (-0.140, +0.056) |

DESI234 and DESI257 agree with each other to **0.003 px in FWHM** and 1.10% rms — two
independent frames, independent star lists, no shared inputs. Against that, DESI238 is 0.25
px (8%) narrower than both.

The size-cut anchor is not the explanation: the three frames' anchors are 2.324, 2.344 and
2.341 px, agreeing to better than 1%. Nor is it the star count. Either the DESI238 visit was
genuinely sharper — plausible, since focus drifts between visits and these are three-exposure
drizzles — or something in the DESI238 field is selecting a narrower subset. I have not
settled which, and it should be settled before the DESI238 result is revised on the strength
of the new kernel.

**Nothing in the existing DESI238 result has been changed.** `psf238.fits` is untouched and
remains what `config.py` points at. The rebuild is written alongside as
`psf238_rebuilt.fits`, with `psf238_rebuilt.json` and `psf238_rebuilt_qa.png`.

## 5. Reproducing this

```bash
export PYTHONPATH=claude/pipeline
PY=/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python

# Rebuild. The keep-out zone stops the lens itself being taken for a star.
$PY claude/pipeline/make_psf.py --lens DESI238 --frame final_94_drz.fits \
    --out psf238_rebuilt.fits --qa psf238_rebuilt_qa.png --exclude "1067.5,1222.7,60"

# Compare.
$PY claude/pipeline/make_psf.py --compare claude/DESI238/psf238_rebuilt.fits \
    claude/DESI238/psf238.fits
```

Budget about five minutes per build, nearly all of it the photutils import (~240 s cold on
this filesystem). Background it rather than running under a short timeout.
