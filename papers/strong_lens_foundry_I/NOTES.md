# NOTES on `strong_lens_foundry_I` (main.tex)

Status: this looks like a complete draft (all sections present, `\bibliography` and
`main.bbl` both exist, full author list). Not marked as a draft in-text, but many
`\ed{}`-wrapped edits and inline TODO comments (`%%%% DO NOT DELETE`, red-text markers)
suggest it is still under revision. Title: "DESI Strong Lens Foundry I: HST Observations
and Modeling with GIGA-Lens." Journal: submitted to ApJ (`\submitjournal{ApJ}`).

## 1. What the paper is

- Programme: the **DESI Strong Lens Foundry**, built on ~3500 new strong-lens candidates
  found in the DESI Legacy Imaging Surveys (DECaLS+BASS+MzLS, ~19,000 deg², §"HST SNAP
  Program"/`ls-hst-snap.tex`→`discovery.tex`) via residual neural networks across three
  searches (H20, H21, S24 = Huang et al. 2020/2021, Storfer et al. 2024).
- This paper (Paper I of the series) presents the **HST SNAP follow-up**: SNAP program
  GO-15867 (PI Huang), "Confirming Strong Galaxy Gravitational Lenses in the DESI Legacy
  Imaging Surveys" (§2 / `ls-hst-snap.tex`, `hst-observations.tex`).
  - 112 candidates submitted; 53 targeted; 2 lost to guide-star failure; **51 systems
    successfully observed, all 51 confirmed as strong lenses** (100% success rate, stated
    again in the Conclusion as apparently the first HST SNAP program for lens confirmation
    to do so).
  - Instrument/filter: **WFC3/IR, F140W**, 3×399.23 s exposures (1197.7 s total), no CR
    split. **Native pixel scale 0.13″, drizzled to 0.065″**.
  - PSF: for the lens-model demo system, a 27×27 pixel **empirical PSF** built by
    stacking PSF-like objects identified by SExtractor (via the AstroObjectAnalyser repo).
    For the GALFIT photometry pass (all 51 systems), PSFs were built by stacking star
    cutouts and iteratively refining via `PSFr` in `Lenstronomy` (§ "Photometry").
- Targets were selected half galaxy-scale, half group/small-cluster scale, chosen for two
  science goals: (1) targeted lensed-SN search for H0, (2) low-mass (<10^9 Msun) halo
  detection (§ `ls-hst-snap.tex`).
- Spectroscopy: DESI Strong Lensing Secondary Target Program (redshift success >90% for
  lens galaxies) and Keck NIRES for high-z sources DESI can't reach; both deferred to
  Papers II and III (in prep) — this paper only gives redshifts for the two systems it
  discusses in detail (§3, `spect.tex`).

## 2. Naming/ID conventions and search for DESI-234 / DESI-238 / DESI-257

**Naming convention** (stated explicitly in the Fig. 1 caption, `hst-observations.tex`):
targets are named `DESI-RRR.RRRR±DD.DDDD`, i.e. full decimal **RA and Dec**, not a
sequence number. Figures are ordered by ascending RA and captioned "DESI-<RA>±<Dec>".

I grepped every `.tex` file for "234", "238", "257" (and DESI- generally). Findings:

- **No lens in this paper is called "DESI-234", "DESI-238", or "DESI-257" as a short
  numbered designation** — that naming (DESI234/238/257) does not appear anywhere in
  this paper's text or tables. The paper uses full RA-Dec names throughout.
- However, systems whose RA rounds to 234, 238, 257 **do appear**, in
  Table `tab:hst-phot` (`photometry-single-arc.tex`, the 51-system HST isophotal
  photometry table) and nowhere else in the running text except one HST-panel figure
  caption. By RA these are almost certainly the same systems the pipeline repo shortens
  to DESI234/DESI238/DESI257 (confirmed by cross-checking `claude/DESI234/README.md`,
  `claude/DESI238/README.md`, `claude/DESI257/README.md`, which cite these exact
  RA/Dec strings):
  - **DESI-234.4783+14.7232** — appears only in Table `tab:hst-phot`: RA 234.4780,
    Dec 14.7227, F140W isophotal mag `25.12 ± 0.04`, contour area `0.0254 arcsec²`.
    No redshift, morphology note, or theta_E is given anywhere in the paper.
  - **DESI-234.8707+16.8379** — same table only: RA 234.8703, Dec 16.8391, F140W mag
    `24.47 ± 0.03`, contour area `0.0803 arcsec²`. No other mention.
  - **DESI-238.5690+04.7276** — in the table (RA 238.5695, Dec 4.7277, F140W mag
    `24.24 ± 0.03`, contour area `0.1352 arcsec²`), **and** it gets a specific note:
    called a "galaxy-scale lens" in the Figure 5 (`fig:hstpanel-5`) caption, "likely
    with two lensed sources, which are quadruply (fainter) and doubly (brighter)
    lensed, respectively." A commented-out footnote (not in the compiled text) adds
    that preliminary lens modeling suggested the two apparent sources may actually be
    two components of a single source — this alternative is only in a `%`-commented
    footnote, not asserted in the main text. No theta_E, no redshift given.
  - **DESI-257.4348+31.9046** — table only: RA 257.4352, Dec 31.9046, F140W mag
    `24.13 ± 0.02`, contour area `0.0380 arcsec²`. No other mention, no redshift, no
    morphology note beyond being one of the 51 confirmed lenses.
  - (Not asked for, but adjacent in RA and worth flagging: **DESI-254.4235+34.8162**
    also gets a specific note — Fig. 5 caption says its HST image "possibly reveals a
    radial counter arc very close to the lensing galaxy.")

None of DESI-234, -238, -257 is the worked example of §"Lens Modeling" — that is
**DESI-165.4754-06.0423** (see §3 below), so this paper reports no theta_E, no
redshifts, and no detailed morphology discussion for the three systems the user is
modelling. Everything known about them from this paper is the one-line photometry
table row (and, for 238, the one-sentence morphology note above).

## 3. Modelling setup (§"Lens Modeling" / `lens-model.tex`)

The only system modelled in this paper is **DESI-165.4754-06.0423** (a demonstration
system, group-scale, quadruply-lensed, discovered in H21 with human grade 2.5/4 = "C").
This is the template the agent should expect for future papers in the series (Papers on
DESI-234/238/257 presumably follow the same conventions once modelled).

- **Pipeline**: `GIGA-Lens`, full forward-modeling Bayesian: MAP (multi-start gradient
  descent) → SVI (surrogate Gaussian covariance) → HMC. No staged/incremental fitting —
  everything is fit jointly from the start, explicitly contrasted with pipelines that
  first subtract lens light or fix Sérsic indices early on.
- **Mass model**: elliptical power law (EPL) for the main lens + external shear. EPL
  convergence given as
  κ(x,y) = (3−γ)/2 · (θ_E / sqrt(q x² + y²/q))^(γ−1),
  with θ_E in the **intermediate-axis convention**, γ the 3D deprojected slope. q and
  position angle φ are reparameterized as eccentricities
  (ε1, ε2) = ((1−q)/(1+q))·(cos 2φ, sin 2φ). External shear as (γ_ext,1, γ_ext,2).
- **Light model**: lens galaxy = **two elliptical Sérsic profiles**; nearby group
  member galaxy = **two elliptical Sérsic profiles** (light only, no mass model for it);
  source = **one elliptical Sérsic + shapelets, n_max = 6** (Birrer 2015 convention).
  Sérsic form given explicitly: I = I0 exp(−b_n((sqrt(q x²+y²/q)/R)^(1/n) − 1)),
  b_n = 1.9992 n − 0.3271.
- **Masking** (Fig. `fig:what-to-mask`): three faint background objects visible only in
  HST are masked out; a small object embedded in arc A (in projection) is masked (its
  mass is untested — found to not affect main lens parameters either way, so its
  nature — MW star vs. small perturber — is left for future work); the central 2.5-pixel-
  radius region of the nearby galaxy's light is masked (Sérsic doesn't fit ellipticals'
  cores well).
- **Priors** (Table `tab:parameters`, all in the paper's own notation):
  - θ_E ~ exp(N(ln 2.5, 0.25)); γ ~ TruncNormal(2, 0.25, [1, 2.7]);
    ε1, ε2 ~ N(0, 0.1); x, y ~ N(0, 0.05); γ_ext,1, γ_ext,2 ~ N(0, 0.05).
  - Lens light (×2 Sérsic, priors given with a slash between the two components):
    R_l ~ exp(N(ln 1, 0.15)) for both; n_l ~ U(0.5, 10) for both;
    ε_l,1/2 ~ N(0, 0.15); x_l ~ N(0,0.05)/N(0,0.10); y_l ~ N(0,0.05)/N(0,0.10).
  - Source light: R_s ~ exp(N(ln 0.25, 0.15)); n_s ~ U(0.5, 6); ε_s ~ N(0, 0.15);
    x_s, y_s ~ N(0, 0.1); shapelet scale β_shp ~ exp(N(ln 0.1, 0.1)); shapelet
    center x_shp, y_shp ~ N(0, 0.05).
  - Nearby galaxy light (×2 Sérsic): R_g ~ exp(N(ln 0.4, 0.2)) both; n_g ~ U(0.5,5)
    both; ε_g ~ TruncNormal(0, 0.15, [−0.3, 0.3]) both; x_g ~ TruncNormal(3.7, 0.05,
    [3.55, 3.85]) both; y_g ~ TruncNormal(0.25, 0.05, [0.1, 0.4]) both — i.e. the
    nearby galaxy's *position* is tightly, informatively priored (not centered at 0),
    reflecting its known offset in the cutout.
  - Total: **41 parameters** (`\nparams`).
- Coordinate origin: mass-model x,y priors centered at N(0,0.05) — i.e. origin is at
  (or very near) the main lens's own centroid; the nearby galaxy's light position is
  given in the same cutout-relative arcsec frame (offset ~3.7″, 0.25″).

## 4. Quoted results (Tables `tab:best-fit-params`, `tab:best-fit-light-params`)

For DESI-165.4754-06.0423 (the only system with a full result set):

- **θ_E = 2.6463 (+0.0017/−0.0016)″** (macro `\thetaEfit` = 2.646 ± 0.002″), γ = 1.372
  (+0.023/−0.022) (macro `\gammafit` = 1.37 ± 0.02).
- Full mass-parameter row: ε1 = 0.1091(20), ε2 = −0.1320(20), x = 0.0272(+23/−22),
  y = −0.0018(17), γ_ext,1 = 0.0657(24), γ_ext,2 = −0.0939(22) (all arcsec/dimensionless,
  1σ from HMC).
- Total mass within the critical curve: 1.7390 (+0.0046/−0.0043) × 10^12 M_sun.
- Lens light effective radius (larger of the two Sérsic components) = 1.89 ± 0.02″ =
  11.8 ± 0.1 kpc (H0=67.4±0.5, ΩΛ=0.6847±0.0073, ΩM=0.315±0.007 from Planck 2020).
- Magnifications (avg of two methods, Appendix "Magnification"): A=25.0±1.6,
  B=59.0±1.6, C=51.1±2.7, D=35.43±0.79. Three independent magnification methods (pixel-
  threshold ratio, convex-hull ray-traced area, Lenstronomy point-wise magnification)
  agree to ≲10%; that 10% spread — not the statistical/sampling uncertainty — is what's
  adopted as the quoted magnification error bar (paper explicitly notes this is larger
  than the naive statistical uncertainty and that published magnification errors
  elsewhere are likely underestimated for the same reason).
- Full light-parameter table (`tab:best-fit-light-params`) gives R_e, n, ε1, ε2, x, y for
  lens (2 Sérsic), source (1 Sérsic+shapelets: β_shp=0.0505, x_shp=−0.0249, y_shp=0.0354),
  and nearby galaxy (2 Sérsic) — see file for full numbers if needed.
- **Credible/convergence criteria explicitly endorsed**: R-hat (PSRF, Gelman & Rubin
  1992) and ESS. Paper adopts **R-hat < 1.2** as the threshold, citing Gelman 1992,
  and notes their R-hat is the TensorFlow-Probability definition, which is
  "approximately the square" of the more recent Gelman (2014) definition — so R-hat<1.2
  here ≈ R-hat<1.1 in the newer convention (see footnote `fn:rhat`). For this fit: all
  mass and light params of the lens and all light params of the source have R-hat<1.10;
  nearby-galaxy light params have R-hat<1.12. ESS: O(10^3) considered sufficient
  (citing Carpenter 2017, Mandel 2022); this fit's min ESS = 32,200, max = 40,000.
- Runtime: 3 hr 13 min total on 4×A100 GPUs (Perlmutter/NERSC) — MAP 1h9m, SVI 1h52m,
  HMC 12 min.

## 5. Model selection / second components / interlopers

- **No formal model-selection criterion (e.g. Bayes factor, BIC/AIC) is used anywhere
  in this paper.** Decisions about adding components are argued qualitatively.
- The **nearby small galaxy** (z_phot=0.579±0.166, r=23.2 mag, ~probable group member)
  is given a **light-only** model (2 Sérsic), explicitly **no mass component** — no
  justification beyond it being needed to explain flux near the source images; its
  mass is simply not modelled in this first-pass demonstration.
- The **small object embedded in arc A** (in projection) is masked out entirely rather
  than modelled. The paper explicitly tested including vs. excluding a mass model for
  it and found "negligible difference for the main lens parameters" — this empirical
  insensitivity is the stated justification for leaving it out, with its nature (MW
  star vs. small perturber/interloper) explicitly flagged as unresolved and deferred
  to future work (relevant to low-mass-halo detection science goal).
- **γ prior sensitivity test** (§ end of `lens-model.tex`): re-run with two alternative
  γ priors (TruncNormal(2,0.25,[1.5,2.5]) and Uniform(2,3)) to check whether the
  fiducial prior TruncNormal(2,0.25,[1,2.7]) was pinning γ near its boundary. Result:
  all other parameters stable; γ posterior pushes against whichever lower boundary is
  imposed (1.5 or 2.0 respectively); χ² *improves* by 42.13 and 551.87 respectively
  when the boundary is raised — i.e. the fit statistically prefers to push γ as low as
  allowed, and the authors report this as evidence in favor of their fiducial
  (lower-boundary 1) choice rather than the more restrictive alternatives, but do not
  resolve why γ wants to go so low.
- The paper **does not address** adding a second mass-model component (e.g. a second
  EPL/SIS for a satellite) for any system in this sample — no such case is described.

## 6. Validation, systematics, limitations

- Cross-check of SVI vs. HMC posteriors: "the marginal distributions of the SVI
  posterior match the HMC sampling results ... as well as they do for the simulated
  systems presented in Gu et al. 2022 ... although the match is not as close for all
  parameter pairs" — an explicit, honest caveat that SVI-HMC agreement is *not*
  uniformly as good as in the original GIGA-Lens validation paper, without giving a
  quantitative discrepancy metric.
- Magnification uncertainty is explicitly treated as **method-choice-dominated**, not
  statistical (§ above; Appendix "Magnification").
- Runtime and complexity are used as a validation/feasibility argument: 41 parameters,
  128×128 pixel cutout (large due to θ_E~2.65″), full forward Bayesian fit in 3h13m on
  4 GPUs — presented as proof the approach scales to HST-resolution, large-Einstein-
  radius, group-scale systems, "for the first time."
- **Discussion section (`discussion.tex`) explicitly states limitations/open items**:
  - No dark matter substructure or line-of-sight low-mass halo detected in this system
    (explicitly the negative result for one of the two science drivers); status for
    other systems in the program deferred to future publications.
  - Comparison to spatially-resolved stellar kinematics for an independent γ check is
    "left for a future investigation" — not done here.
  - The paper notes their γ=1.37±0.02 is low compared to canonical NFW predictions
    (γ→1 at small radii) but *within the range* reported for a handful of BELLS/SL2S/LSD
    systems (γ~1.4, citing LiR2018a, Treu2004a) and consistent to 2σ with several more
    SLACS/BELLS systems — but flags that none of those prior studies used a fully
    forward-modeling approach or applied R-hat/ESS convergence diagnostics, so a
    like-for-like comparison isn't really available yet.
  - States plainly this is a **single demonstration system**; broader conclusions about
    the sample (e.g. systematic mass-slope trends with redshift) are explicitly deferred
    to future papers in the series.

## 7. Redshifts, photometry, environment

- **Redshifts**: discussed only for the handful of systems singled out in the text
  (not tabulated systematically in this paper for the 51 systems as a set — that's
  deferred to Paper II for DESI spectroscopic redshifts and Paper III for Keck NIRES).
  For the modelled system, z_d = 0.4834 (DESI) and z_s = 1.6748 (Keck NIRES), both
  citing "Paper II & III" as the source, not derived here. Photometric redshifts,
  where quoted (e.g. for the nearby galaxy, z_phot=0.579±0.166; for other individual
  systems discussed in Appendix "All HST Systems"), are from Zhou et al. 2020 unless a
  spectroscopic z is available.
- **Photometry**: Table `tab:hst-phot` gives GALFIT/Sérsic-derived isophotal F140W AB
  magnitudes (75%-of-peak-pixel contour) and contour areas for the brightest source
  image in **all 51 systems**, including the three of interest (see §2 above). This is
  the only photometric result reported at full-sample scale in this paper; no g/r/z
  Legacy Survey photometry table is given (though depth/FWHM of the Legacy Surveys
  imaging is described qualitatively — see §1).
- **Environment**: discussed narratively, not tabulated. For DESI-165.4754-06.0423 the
  lens is identified as the brightest member (r=18.8 mag) of a small group, with one
  probable group member (the "nearby galaxy") explicitly modelled; other group members
  are noted as ≥12″ away and not modelled. For DESI-072.0873-19.4172 (an Einstein
  cross) the lens is identified as "at the center of a high redshift cluster" based on
  multiple similarly-colored ellipticals nearby. The paper does not give a systematic
  environment classification (e.g. richness, velocity dispersion) for the DESI-234,
  DESI-238, or DESI-257 systems specifically — the paper does not address their
  environment beyond their presence in the photometry table.
