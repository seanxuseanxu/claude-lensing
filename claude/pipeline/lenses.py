"""The seam between the lens-independent pipeline and one specific lens system.

Every number that describes a particular lens -- its files, its pixel scale, its noise
level, the flux scale its priors are centred on -- lives in ``claude/<LENS>/config.py`` as
a single ``CONFIG = LensConfig(...)``. Nothing in ``claude/pipeline/`` may name a lens.

The split between what belongs here and what belongs in ``priors.py`` is not a matter of
taste. Checkpoints under ``runs/`` store *unconstrained* samples; a stored value maps back
to a physical one through the prior's bijector, which depends on the prior's **support**.
So:

- **Supports live in the pipeline**, keyed by prior version. A ``Uniform``'s bounds, a
  ``TruncatedNormal``'s limits -- these are the functional form of the prior, they are
  versioned, and a config cannot touch them. This makes the versioning invariant hold by
  construction rather than by documentation.
- **Location and scale live in the config.** A ``LogNormal``'s median, a ``Normal``'s
  sigma. Changing these moves prior mass without changing the bijector, so an old
  checkpoint still decodes to the same physical value.

If a new lens genuinely needs a different bound, that is a new prior version in
``priors.py``, not a new config field.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple

# claude/pipeline/lenses.py -> claude/
CLAUDE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SersicPrior:
    """Location-scale parameters for one Sersic component's prior.

    Deliberately carries no bounds. The ``n_sersic`` Uniform support and the ``e1``/``e2``
    TruncatedNormal limits are fixed by the prior version in ``priors.py`` -- see this
    module's docstring for why.

    Attributes:
        R_sersic_median: median of the ``R_sersic`` LogNormal, in arcsec.
        R_sersic_logsigma: its sigma, in log space (i.e. in dex-ish units).
        ellip_sigma: sigma of the ``e1`` and ``e2`` TruncatedNormals.
        center_sigma: sigma of the ``center_x`` and ``center_y`` Normals, in arcsec.
        Ie_median: median of the ``Ie`` LogNormal, in the simulator's surface-brightness
            units. Note that ``LensSimulator`` multiplies its output by
            ``det(transform_pix2angle) = delta_pix**2``, so a unit ``Ie`` renders far
            fainter than the data -- this number is typically of order 10-100, not 1, and
            must be set from a measured render rather than guessed.
        Ie_logsigma: its sigma, in log space.
    """

    R_sersic_median: float
    R_sersic_logsigma: float
    ellip_sigma: float
    center_sigma: float
    Ie_median: float
    Ie_logsigma: float


@dataclass(frozen=True)
class MassPrior:
    """Location-scale parameters for the EPL + external-shear deflector prior.

    Same rule as ``SersicPrior``: no bounds here. ``gamma``'s truncation at (1, 3) and the
    ellipticity limits belong to the prior version.

    Attributes:
        theta_E_median: median of the Einstein-radius LogNormal, in arcsec.
        theta_E_logsigma: its sigma in log space. Together these must put real prior mass
            across the full radial span of the images the model is asked to reproduce.
        gamma_center: centre of the ``gamma`` (radial power-law slope) TruncatedNormal.
        gamma_sigma: its sigma.
        ellip_sigma: sigma of the deflector ``e1``/``e2`` Normals.
        center_sigma: sigma of the deflector centre Normals, in arcsec, relative to the
            crop centre.
        shear_width: sigma of the external-shear ``gamma1``/``gamma2`` Normals.
    """

    theta_E_median: float
    theta_E_logsigma: float
    gamma_center: float
    gamma_sigma: float
    ellip_sigma: float
    center_sigma: float
    shear_width: float


@dataclass(frozen=True)
class LensConfig:
    """Everything the pipeline needs to know about one lens system.

    Attributes:
        name: the directory name under ``claude/``, e.g. ``"DESI238"``. Used for figure
            titles, ``metadata.json``, and to resolve ``root``.
        cutout_file: science cutout filename, relative to ``root``. ``.npy`` or ``.fits``.
        psf_file: PSF filename, relative to ``root``.
        delta_pix: pixel scale in arcsec, at the *science image's* grid. Derive it from the
            parent frame's CD matrix as ``sqrt(|det CD|) * 3600`` rather than trusting a
            header keyword.
        num_pix: side length of the crop actually fitted, in pixels. Must equal the size
            ``crop`` produces, which ``data.load_image`` asserts.
        supersample: simulator supersampling factor.
        exp_time: exposure time in seconds, used for the Poisson term of the noise model.
            Correct only to first order when the frame is a mosaic of several exposures
            with varying coverage; the drizzle ``WHT`` extension is the proper fix.
        background_rms: sky sigma in the image's own units. Measure it from blank patches
            in the *parent* frame -- ``sigma_clipped_stats`` over a cutout that is mostly
            galaxy clips galaxy light, not sky, and overestimates badly.
        crop: ``(row_slice, col_slice)`` into the cutout, or ``None`` to fit it whole.
        centroid_pix: deflector centroid ``(row, col)`` within the crop, in pixels.
        mass_prior: deflector prior location-scale parameters.
        source_prior: prior for each Sersic source component. Components share it, which
            makes them formally exchangeable so the data rather than the prior chooses the
            morphology.
        lens_light_priors: keyed by component role. ``"single"`` for the one-component
            model; ``"bulge"`` and ``"disk"`` for the two-component model, which get
            deliberately *different* priors because they mean distinct physical things and
            identical ones would only add a label-switching mode.
        clumps: ``{name: (r_arcsec, pa_deg)}`` for the observed images. **The PA convention
            is array-space** -- measured from the +row axis toward +col within the crop --
            not sky PA. Converting these through the image WCS the standard way (North
            through East) puts the markers on empty sky.
        companion_max: upper bound on the brightest pixel in each of the crop's four
            corners, in the image's units. gigalens has no pixel masking by default (though
            see ``mask_circles`` below), so a companion galaxy inside the frame is fitted as
            if it were part of the lens system unless masked out. ``verify.py`` asserts the
            worst corner is below this. ``None`` skips the check, which is the honest
            default for a system whose field has not been inspected.
        mask_circles: discs to EXCLUDE from the likelihood, as ``(row, col, radius)`` in
            **crop** pixel coordinates -- i.e. indices into the array ``data.load_image``
            returns, after ``crop`` has already been applied, not into the parent frame. A
            pixel is excluded when its centre falls within ``radius`` of ``(row, col)``.
            ``None`` (the default) fits every pixel in the crop, which is unchanged pipeline
            behaviour. Use this for a foreground interloper that no square crop can exclude
            without also clipping the arcs; a crop that alone suffices should still be
            preferred over a mask.
        svi_guard_reference: ``{run_label: should_pass}`` naming stored runs under
            ``runs/`` whose SVI covariance conditioning is *known* to be healthy or
            diverged. ``verify.py`` asserts the guard classifies each correctly. This is
            what makes the guard a tested thing rather than a hopeful one -- an earlier
            version of it tested positive definiteness and silently passed a run that was
            in fact diverged. Empty means the check reports itself skipped.
        parent_frame: filename of the full drizzled frame, if kept alongside. The pipeline
            never reads it; recorded so the provenance of the cutout is traceable.
        notes: free text shown by ``verify.py``.
    """

    name: str
    cutout_file: str
    psf_file: str
    delta_pix: float
    num_pix: int
    exp_time: float
    background_rms: float
    mass_prior: MassPrior
    source_prior: SersicPrior
    lens_light_priors: Mapping[str, SersicPrior]
    supersample: int = 2
    crop: Optional[Tuple[slice, slice]] = None
    centroid_pix: Optional[Tuple[float, float]] = None
    clumps: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    companion_max: Optional[float] = None
    mask_circles: Optional[Tuple[Tuple[float, float, float], ...]] = None
    svi_guard_reference: Mapping[str, bool] = field(default_factory=dict)
    parent_frame: Optional[str] = None
    notes: str = ""

    @property
    def root(self) -> Path:
        """The lens's own directory, ``claude/<name>/``."""
        return CLAUDE_DIR / self.name

    @property
    def runs_dir(self) -> Path:
        """Where fits for this lens are written, ``claude/<name>/runs/``."""
        return self.root / "runs"

    @property
    def clump_radii(self):
        """``{name: r_arcsec}``, the radii alone. Used to sanity-check ``theta_E``."""
        return {name: r for name, (r, _) in self.clumps.items()}


def available():
    """Names of every lens directory under ``claude/`` that defines a ``config.py``."""
    return sorted(
        p.parent.name
        for p in CLAUDE_DIR.glob("*/config.py")
        if not p.parent.name.startswith(".")
    )


def load(name) -> LensConfig:
    """Return the ``CONFIG`` defined by ``claude/<name>/config.py``.

    Imported by file path rather than by package name so a lens directory needs no
    ``__init__.py`` and never has to be on ``PYTHONPATH``. The module is registered under
    a namespaced key (``claude_lens_<name>``) so two lenses loaded in one process cannot
    collide on the bare name ``config``.
    """
    path = CLAUDE_DIR / name / "config.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"no config for lens {name!r}: expected {path}. "
            f"Known lenses: {', '.join(available()) or '(none)'}"
        )
    spec = importlib.util.spec_from_file_location(f"claude_lens_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        cfg = module.CONFIG
    except AttributeError:
        raise AttributeError(f"{path} defines no CONFIG") from None
    if cfg.name != name:
        raise ValueError(
            f"{path} defines CONFIG.name = {cfg.name!r} but lives in directory {name!r}; "
            f"the two must agree or run outputs land in the wrong place"
        )
    return cfg


def add_argument(parser, required=True):
    """Add the standard ``--lens`` flag to an ``argparse`` parser."""
    parser.add_argument(
        "--lens",
        type=str,
        required=required,
        default=None,
        help=f"Lens system to operate on. Known: {', '.join(available()) or '(none)'}",
    )


def from_run_dir(run_dir):
    """Infer the lens name for an existing run directory.

    Prefers the ``"lens"`` key in the run's ``metadata.json``. Runs written before that key
    existed fall back to the grandparent directory name, which is correct for the
    ``claude/<LENS>/runs/<label>/`` layout.
    """
    import json

    meta_path = Path(run_dir) / "metadata.json"
    if meta_path.is_file():
        try:
            name = json.loads(meta_path.read_text()).get("lens")
        except (ValueError, OSError):
            name = None
        if name:
            return name
    return Path(run_dir).resolve().parent.parent.name
