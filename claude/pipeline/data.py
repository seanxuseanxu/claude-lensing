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
