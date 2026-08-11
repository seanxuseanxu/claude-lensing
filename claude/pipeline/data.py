"""Load and condition a lens system's science cutout and PSF.

Nothing here is specific to any one lens. Every number -- pixel scale, crop, exposure
time, background level -- comes from the ``LensConfig`` passed in. What stays here are the
two conditioning steps that are universal:

- **Cast to native-endian float32.** FITS and drizzle products are routinely big-endian
  (``>f4``, ``>f8``), and JAX rejects those outright with ``TypeError: Error interpreting
  argument as a JAX value``. Cast before slicing, not after.
- **Normalise the PSF to unit sum.** A PSF that sums to anything else rescales every
  simulated image by that factor, silently biasing all the amplitudes.

Neither step is optional, and neither is a judgement call, which is why they live in the
pipeline rather than in a config.
"""

import numpy as np
from astropy.io import fits

from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.simulator import SimulatorConfig


def load_image(cfg):
    """Return the science crop as native-endian float32, in the image's own units.

    No sky is subtracted. Whether that is right is a per-lens judgement recorded in the
    config; the pipeline's job is not to guess. If the outer ring of a crop is deflector
    wings rather than sky, subtracting it removes real galaxy light and biases the
    lens-light ``R_sersic`` and ``n_sersic`` low -- the Sersic profile is there to model
    those wings.
    """
    path = cfg.root / cfg.cutout_file
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = fits.getdata(path)
    arr = np.asarray(arr).astype(np.float32)
    if cfg.crop is not None:
        arr = arr[cfg.crop]
    if arr.shape != (cfg.num_pix, cfg.num_pix):
        raise ValueError(
            f"{cfg.name}: crop produced {arr.shape} but num_pix={cfg.num_pix} implies "
            f"({cfg.num_pix}, {cfg.num_pix}). The simulator grid and the data would not "
            f"line up, and gigalens would not raise."
        )
    return np.ascontiguousarray(arr)


def load_psf(cfg):
    """Return the PSF, native-endian float32, normalised to unit sum.

    Assumes the PSF is odd-sized, already centred, and sampled at the same pixel scale as
    the science image. None of those are checkable from a bare PSF file -- most carry no
    WCS or scale keywords -- so they are assumptions about the input, stated here rather
    than silently relied on. Given the same scale, the PSF is not oversampled relative to
    the science grid and lenstronomy's ``subgrid_kernel`` (called by ``SceneSimulator``
    when ``supersample > 1``) handles the rest.
    """
    psf = np.asarray(fits.getdata(cfg.root / cfg.psf_file)).astype(np.float32)
    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        raise ValueError(
            f"{cfg.name}: PSF is {psf.shape}, but an even-sized kernel has no centre "
            f"pixel and cannot be centred on the grid."
        )
    return np.ascontiguousarray(psf / psf.sum())


def build_mask(cfg, shape):
    """Return a boolean keep-mask (True = keep) from ``cfg.mask_circles``, or ``None``.

    ``None`` when ``cfg.mask_circles`` is ``None`` -- the unchanged-default-behaviour case,
    so ``ImageData(mask=build_mask(cfg, image.shape))`` is a no-op for every existing
    config. Each ``(row, col, radius)`` in ``cfg.mask_circles`` is in **crop** pixel
    coordinates, i.e. indices into the array this shape describes; a pixel is excluded when
    its centre falls within ``radius`` of ``(row, col)``.

    Raises if a disc lies entirely outside ``shape`` (a config typo, not a no-op -- a mask
    that silently excludes nothing would hide the mistake) or if the resulting mask keeps
    no pixels at all.
    """
    if cfg.mask_circles is None:
        return None
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    mask = np.ones(shape, dtype=bool)
    for row, col, radius in cfg.mask_circles:
        disc = (rows - row) ** 2 + (cols - col) ** 2 <= radius**2
        if not disc.any():
            raise ValueError(
                f"{cfg.name}: mask circle (row={row}, col={col}, radius={radius}) lies "
                f"entirely outside the {shape} crop -- this is a config typo, not a no-op."
            )
        mask &= ~disc
    if not mask.any():
        raise ValueError(
            f"{cfg.name}: mask_circles {cfg.mask_circles} exclude every pixel in the "
            f"{shape} crop; nothing would be left to fit."
        )
    return mask


def load(cfg):
    """Return ``(image, psf, sim_config)`` ready for ``SceneSimulator`` and ``ImageData``.

    ``likelihood_precision`` is set explicitly rather than left to default. The default is
    ``None``, which resolves to float64 and warns (then silently degrades to float32)
    unless ``JAX_ENABLE_X64=1`` is set. float32 is what every run in ``runs/`` was fitted
    at, so stating it keeps the port a change of API and not of arithmetic. Raising the
    pipeline to float64 is a separate, measurable decision -- it roughly doubles forward
    memory and compute, and it needs x64 enabled process-wide.
    """
    image = load_image(cfg)
    psf = load_psf(cfg)
    sim_config = SimulatorConfig(
        delta_pix=cfg.delta_pix,
        num_pix=cfg.num_pix,
        supersample=cfg.supersample,
        kernel=psf,
        likelihood_precision="float32",
    )
    return image, psf, sim_config


def build_prob_model(cfg, model, image, sim_config, mask):
    """Wrap ``image`` and ``model`` into the ``ProbModel`` the fit actually optimises.

    The one place this construction happens. ``run_fit.py`` (which fits), ``verify.py``
    (which pre-flights) and ``diagnose.rebuild_model`` (which reconstructs a finished run)
    each arrive here by a different route -- argparse, fixed defaults, and a run's recorded
    ``metadata.json`` respectively -- but the likelihood they build must be the same one,
    or a pre-flight check passes a model that is not the one fitted and a diagnosis reads a
    finished run against a likelihood it never saw. The three differ in what feeds ``model``
    and ``mask``, and that is the whole of the intended difference.

    ``ImageData`` derives ``sigma = sqrt(background_rms^2 + clip(observed, 0, inf)/exp_time)``
    ONCE, from the observed image. The pre-port ``ForwardProbModel`` recomputed it from the
    *simulated* image every step, making the variance parameter-dependent. This is the old
    ``BackwardProbModel`` convention and the only one the scene API offers; it is a real
    change to the likelihood, not a refactor.
    """
    ds = ImageData(
        image,
        sim_config,
        background_rms=cfg.background_rms,
        exp_time=cfg.exp_time,
        mask=mask,
        sees="all",
    )
    return ProbModel(model, ds, mode="forward")
