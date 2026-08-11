# Notes on Narayan & Bartelmann, "Lectures on Gravitational Lensing" (SISSA.tex, astro-ph/9606001)

Audience: fitting parametric power-law elliptical + external shear + Sersic models to
galaxy-scale strong lenses (θ_E ~ 1-1.5", HST imaging). Section/eq numbers are the
paper's own (`\label{eq:X.Y}` / `\label{NB:...}`), so they are jump targets in the .tex.

## PART A — Section map

| # | Title | Contents | \label |
|---|---|---|---|
| 1 | Introduction | History: Soldner/Einstein deflection, Eddington multiple images, Zwicky galaxy lenses, Refsdal H0 via time delay, first lens (0957+561), microlensing, Einstein rings, cluster arcs/arclets. Roadmap. | `NB:sect1` |
| 2 | Lensing by Point Masses in the Universe | | `NB:sect2` |
| 2.1 | Basics of Gravitational Lensing | | `NB:subsect2.1` |
| 2.1.1 | Effective Refractive Index of a Gravitational Field | n=1-2Φ/c², Shapiro delay, point-mass deflection α̂=4GM/(c²b) | `NB:subsubsect2.1.1` |
| 2.1.2 | Thin Screen Approximation | Σ(ξ), deflection integral over mass sheet, circular-lens α(ξ)=4GM(ξ)/(c²ξ) | `NB:subsubsect2.1.2` |
| 2.1.3 | Lensing Geometry and Lens Equation | reduced deflection α, lens equation β=θ-α(θ), Σ_cr, supercritical lenses | `NB:subsubsect2.1.3` |
| 2.1.4 | Einstein Radius | θ_E definition and properties, mean Σ inside θ_E = Σ_cr | `NB:subsubsect2.1.4` |
| 2.1.5 | Imaging by a Point Mass Lens | θ_±, magnifications μ_±, parity | `NB:subsubsect2.1.5` |
| 2.2 | Microlensing in the Galaxy | basic relations, timescale t0, optical depth τ, MACHO searches, EROS/MACHO/OGLE results, parallax/finite-source effects | `NB:subsect2.2` (+ 2.2.1-2.2.4) |
| 2.3 | Extragalactic Microlenses | point masses as cosmological dark matter, Ω_M limits, QSO 2237+0305 microlensing | `NB:subsubsect2.3` (+2.3.1-2.3.3) |
| 3 | Lensing by Galaxies | **core section for galaxy-scale modelling** | `NB:sect3` |
| 3.1 | Lensing by a Singular Isothermal Sphere (SIS) | ρ∝r⁻², flat rotation curve, θ_E, image positions/magnifications | `NB:subsect3.1` |
| 3.2 | Effective Lensing Potential | ψ, α=∇ψ, ∇²ψ=2κ, Jacobian 𝒜, convergence κ, shear γ₁,γ₂, magnification | `NB:subsect3.2` |
| 3.3 | Gravitational Lensing via Fermat's Principle | time-delay function t(θ), stationary points = images, image types I/II/III, critical lines/caustics, odd-image theorem | `NB:subsect3.3` (+3.3.1, 3.3.2) |
| 3.4 | Circularly Symmetric Lens Models | table of ψ,α for point mass/SIS/softened SIS/const. sheet; tangential vs radial critical curves; image configs | `NB:subsect3.4` |
| 3.5 | Non-Circularly-Symmetric Lens Models | elliptical density model, elliptical potential model, external shear, image configurations (cross, fold, cusp, ring) | `NB:subsect3.5` (+3.5.1-3.5.3) |
| 3.6 | Studies of Galaxy Lensing | individual model fitting (Ring Cycle, LensClean, LensMEM), statistical population modeling | `NB:subsect3.6` (+3.6.1, 3.6.2) |
| 3.7 | Astrophysical Results from Galaxy Lensing | galaxy structure/core radii, ellipticity vs. shear degeneracy, galaxy evolution, CDM constraints, H0 via time delay, cosmological constant | `NB:subsect3.7` (+3.7.1-3.7.5) |
| 4 | Lensing by Galaxy Clusters and Large-Scale Structure | not the audience's regime; noted briefly below | `NB:sect4` |
| 4.1 | Strong Lensing by Clusters — Giant Arcs | basic optics, mass inside arc, asphericity, core radii, radial profile, substructure | `NB:subsect4.1` (+4.1.1-4.1.7) |
| 4.2 | Weak Lensing by Clusters — Arclets | Kaiser & Squires algorithm, **mass-sheet degeneracy** (§4.2, eq 4.16-4.18), breaking it via magnification, source redshifts, finite-field methods, results | `NB:subsect4.2` (+4.2.1-4.2.6) |
| 4.3 | Weak Lensing by Large-Scale Structure | cosmic shear, QSO-galaxy correlations, CMB lensing, outlook | `NB:subsect4.3` (+4.3.1-4.3.4) |

## PART B — Results worth having at hand

**Deflection, lens equation (§2.1.1-2.1.3).**
Point mass: α̂ = 4GM/(c²b) (eq 2.8). Thin-screen, extended lens:
α̂(ξ) = (4G/c²) ∫ Σ(ξ')(ξ-ξ')/|ξ-ξ'|² d²ξ' (eq 2.11). Reduced deflection
α = (D_ds/D_s) α̂ (eq 2.14). **Lens equation**: β = θ - α(θ) (eq 2.15).

**Effective potential, convergence, shear, Jacobian, magnification (§3.2).**
ψ(θ) = (D_ds/D_dD_s)(2/c²)∫Φ dz (eq 3.11); ∇_θψ = α (eq 3.12);
∇²_θψ = 2κ, κ ≡ Σ/Σ_cr (eq 3.13); ψ(θ) = (1/π)∫κ(θ')ln|θ-θ'|d²θ' (eq 3.14).
Σ_cr = (c²/4πG)(D_s/D_dD_ds) (eq 2.18).
Jacobian 𝒜 = ∂β/∂θ = δ_ij - ψ_ij = ℳ⁻¹ (eq 3.16). κ = ½tr(ψ_ij) (eq 3.19).
Shear: γ₁ = ½(ψ_11-ψ_22) = γcos2φ, γ₂ = ψ_12 = γsin2φ (eq 3.20). So
𝒜 = (1-κ)·I - γ·[[cos2φ, sin2φ],[sin2φ, -cos2φ]] (eq 3.21).
Image of unit circle → ellipse with axes (1-κ-γ)⁻¹, (1-κ+γ)⁻¹ (eq 3.22).
**Magnification** μ = detℳ = 1/detA = 1/[(1-κ)²-γ²] (eq 3.23; circular-lens
special case μ=θ dθ/(β dβ), eq 2.27).

**Critical curves vs. caustics (§3.3.2, item 4).** Critical lines = loci in
θ-space where detA=0 (images merge/diverge, μ→∞ locally); caustics = their
images in β-space under the lens equation. The paper is explicit that these
are different spaces: critical lines live in the image (θ) plane, caustics in
the source (β) plane. Crossing a caustic (in source position) changes the
image multiplicity by ±2; crossing model space near a critical curve (in image
position) is where magnification is large. Tangential critical curve ↔ outer,
produces tangentially-elongated arcs; radial critical curve ↔ inner, produces
radially-elongated images (§3.4, discussion after eq 3.23 fig. 3.6).

**Odd-number-of-images theorem (§3.3.2, item 5).** "New extrema are always
created in pairs... the total number of extrema, and thus the number of images
of a generic (non-singular) lens, is odd (Burke 1981)." Assumption: the lens
is *non-singular* (no infinite-density point) and the source is off any
caustic. Observed lenses usually show an *even* number because one image sits
at/near θ=0 (the lens center) where a singular density profile (point mass,
SIS, or any cuspy profile) drives its magnification to zero — it is
"technically" present with zero flux (§3.1, after eq 3.9) or "usually very
faint... vanishes if the lens has a singular core" (§3.4, after eq 3.23, fig
3.5 discussion) — hence unobserved, leaving an apparently even count.

**Image parity / type (§3.3.2, item 2-3).** Via Fermat's principle, images sit
at stationary points of t(θ); type set by Hessian 𝒯 ∝ A eigenvalues: Type I =
minimum (detA>0, trA>0); Type II = **saddle point** (detA<0); Type III =
maximum (detA>0, trA<0). μ = 1/detA, so Type II images have negative
magnification ⇒ **flipped parity**; Types I, III positive. Saddle images are
generically dimmer/first perturbed side and, per property 1, vary *later*
(farther down the time-delay surface) unless closest to lens; brightness
grows as an image approaches a critical curve, where detA→0 (property 4).

**SIS (§3.1).** ρ(r) = σ_v²/(2πG r²) (eq 3.4); flat rotation curve v_rot²=2σ_v²
(eq 3.5); Σ(ξ)=σ_v²/(2Gξ) (eq 3.6); α̂ = 4πσ_v²/c² = 1.4″(σ_v/220 km/s)²,
**constant with ξ** (eq 3.7); θ_E = 4π(σ_v²/c²)(D_ds/D_s) = α̂·D_ds/D_s (eq
3.8). Two images (if β<θ_E): θ_± = β ± θ_E (eq 3.9), collinear with lens and
source. μ_± = θ_±/β = 1 ± θ_E/β (eq 3.10). Third (central) image formally at
θ=0 with zero flux; finite only if the singular core is softened.

**Circularly symmetric lens table (§3.4, Table 3.1)**: ψ(θ), α(θ) given for
point mass, SIS, softened-core isothermal sphere (θ_c), constant-density
sheet: softened isothermal ψ = (D_ds/D_s)(4πσ²/c²)√(θ_c²+θ²), α =
(D_ds/D_s)(4πσ²/c²)·θ/√(θ_c²+θ²) (used for the central/5th image discussion).

**Elliptical galaxy and external shear (§3.5).**
Elliptical *density* model (eq 3.28): Σ(θ1,θ2) = Σ0/[θ_c²+(1-ε)θ1²+(1+ε)θ2²]^{1/2};
for θ_c=0, closed-form α1, α2, μ⁻¹ given (eq after 3.28, no number) via
arctan/arctanh of φ. Simpler, more common in practice: elliptical
*potential* model (Blandford & Kochanek 1987, eq 3.29):
ψ(θ1,θ2) = (D_ds/D_s)(4πσ_v²/c²)[θ_c² + (1-ε)θ1² + (1+ε)θ2²]^{1/2}.
Caution stated explicitly: for large ε the elliptical-*potential* model gives
unphysical dumbbell-shaped isodensity contours — "for small ε it is a
perfectly viable lens model" (§3.5.1, after eq 3.29). [note: this is the
theoretical reason power-law-elliptical-mass profiles are generally
parameterized in *convergence*, not potential, once ε is not small.]
**External shear** (eq 3.30): ψ = (κ/2)(θ1²+θ2²) + (γ/2)(θ1²-θ2²) in shear's
principal-axis frame, κ,γ locally constant. Explicitly: "An external shear
breaks the circular symmetry of a lens and therefore it often has the same
effect as introducing ellipticity... It is frequently possible to model the
same system either with an elliptical potential or with a circular potential
plus an external shear" (§3.5.2) — i.e., shear and mass ellipticity are
partially degenerate in image configuration alone.

**Time delays (§3.3.1, §3.7.4).** t(θ) = [(1+z_d)/c](D_dD_s/D_ds)[½(θ-β)² -
ψ(θ)] = t_geom + t_grav (eq 3.26); t_grav is the Shapiro delay (eq 2.4) with a
(1+z_d) factor. Images occur where ∇_θ t=0 (Fermat's principle, eq 3.24-3.25).
H0 determination (§3.7.4): the lens equation is dimensionless (image
positions/magnifications carry no absolute scale), but Δt scales as H0⁻¹
(both t_geom, t_grav do), so H0·Δτ (eq 3.36) depends only on the dimensionless
lens model/geometry — measuring Δτ plus a good model of image positions and
magnifications yields H0 (Refsdal 1964).

**Mass-sheet degeneracy (§4.2.1-4.2.3, derived for weak-lensing/shear
reconstruction but generally true).** Measured ellipticity only constrains
g ≡ γ/(1-κ) (eq 4.15), which is invariant under 𝒜 → λ𝒜, i.e.
1-κ' = λ(1-κ), γ' = λγ (eq 4.16-4.17) — equivalently
κ → λκ + (1-λ) (eq 4.18): adding a uniform sheet of convergence (1-λ) and
rescaling. Traced to Falco, Gorenstein & Shapiro (1985) in the *galaxy*-lensing
context, not just clusters. Broken by measuring absolute magnification
(μ∝λ⁻², eq 4.19, §4.2.3), since shape/shear-ratio data alone cannot fix λ.
[note: for a parametric power-law+shear fit to image positions and flux
ratios, this degeneracy is the reason an unconstrained large-scale
convergence sheet (e.g. from an unmodelled group/cluster halo) trades off
against the overall mass normalization; time delays or absolute flux/size
constraints are needed to break it, per §4.2.3 and §3.7.4.]

## PART C — Physical intuition and diagnostics

**Source position vs. caustic → image configuration (§3.4-3.5, figs 3.5-3.8).**
For a non-singular circular lens there are two caustics: an outer, point-like
one at the very center of the source plane, and an inner one that is a circle
(§3.4, before fig 3.5). Ellipticity/shear inflates the point caustic into a
diamond ("astroid"/tangential caustic) with 4 cusps (§3.5.3, after eq 3.30).
Reading off image count from source position relative to caustics (§3.5.3,
figs 3.7-3.8):
- Source outside all caustics → 1 image.
- Source inside the astroid (diamond) caustic but outside the outer caustic →
  3 images ("two of the four outer images move toward each other, merge, and
  disappear as the source... crosses the astroid caustic. Three images remain
  until the source crosses the radial caustic").
- Source well inside the diamond, near its center → 5 images (4 around the
  Einstein ring + 1 faint central image): "A source behind the center of the
  lens has five images because it is enclosed by two caustics. One image
  appears at the lens center, and the four others form a cross-shaped
  pattern" (§3.5.3) — this is the **cross/quad** configuration.
- Source moving toward a **fold** (a smooth edge segment of the caustic, "an
  inter-cusp segment"): two images merge and vanish there (left panel, fig
  3.7).
- Source moving onto a **cusp** point of the caustic: three images merge into
  one (right panel, fig 3.7) — the classic 3-image cusp merger seen as a
  bright close triplet in real quads.
- Extended source astride a fold or cusp of the tangential caustic → giant
  **arc** (2 or 3 merging images respectively, fig 3.8 top panels).
- Extended source covering most of the diamond caustic → near-complete
  **Einstein ring** from 4 merging images (fig 3.8 bottom panels; example:
  radio ring MG 1131+0456, §3.5.3).
Caution: "in each panel... source plane with caustics on the right, image
configuration with critical curves on the left" — always reason source→image
via the caustic, since the mapping is many-to-one only near caustics.

**Einstein radius ↔ enclosed mass (§2.1.4).** θ_E marks where the *mean*
convergence within θ_E equals Σ_cr exactly — "the mean surface mass density
inside the Einstein radius is just the critical density Σ_cr" (§2.1.4, after
eq 2.21). θ_E ≈ (4GM(θ_E)/c²)(D_ds/(D_dD_s)) (eq 2.21), so a measured θ_E is a
direct, near-model-independent measurement of the projected mass enclosed
within that radius — this holds even for non-circular/elliptical lenses to
good approximation since θ_E is set essentially by "the Einstein radius of
the lens" (§3.6.1: "the angular separation between images determines the
Einstein radius... and therefore gives the mass M or velocity dispersion
σ_v"). Also: image separations are of order 2θ_E, and sources within ~θ_E of
the lens are strongly magnified while sources well outside are magnified
little (§2.1.4) — a quick visual check that a quad's image-to-image spacing
should be ~2× the fitted θ_E.

**Fifth/central image (§3.1, §3.4, §3.7.1).** For any circularly symmetric
lens there is technically a central image at θ=0 co-located with the lens
light. It carries essentially zero flux and *vanishes identically* if the
lens density is singular at the center (point mass, pure SIS, or any
power-law cusp) — "this image vanishes if the lens has a singular core (the
curvature of the time delay function then becomes infinite)" (§3.4). Its
observability is controlled entirely by the **core radius**: "The central
image is missing in virtually every known multiply-imaged quasar. The lensing
galaxies in these cases must therefore have very small core radii,
r_c < 200 pc" (§3.7.1). So for a power-law elliptical fit: absence of a
central image is itself data, constraining the profile away from a large
softened core.

**Satellites / interlopers.** [note: this 1996 paper does not treat a
satellite or interloper galaxy as a distinct topic — the closest analogue is
§3.5.2's statement that external shear and galaxy ellipticity partly trade
off in producing the same image configurations, and §3.7.1's remark that
"the mass ellipticity required to fit the images is larger than the
ellipticity of the galaxy isophotes, suggesting either that the dark matter
is more asymmetric than the luminous matter or that there is a significant
contribution from external shear" (§3.7.1) — i.e.
an unmodelled nearby perturber shows up observationally as excess shear/
ellipticity relative to the light, which is diagnostically ambiguous within
image positions alone (ties to the mass-sheet-adjacent degeneracy in Part B).]

