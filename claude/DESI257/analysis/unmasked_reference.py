"""Score round 2's posterior draws under the UNMASKED likelihood.

Round 3's MAP has to clear the best draw of the posterior it is meant to seed, but that
posterior does not exist yet. Round 2's draws are physical parameter vectors under the
same 36-parameter model and the same priors, so scoring them through the new (unmasked)
ProbModel gives a valid lower bound on what round 3's MAP must beat, before round 3 is run.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/global/u1/s/seanjx/claude-lensing/claude/pipeline")

import numpy as np
import jax.numpy as jnp

import data
import lenses
import priors
from gigalens.jax.scene_prob_model import ImageData, ProbModel

cfg = lenses.load("DESI257")
print("config mask_circles:", cfg.mask_circles)

image, psf, sim_config = data.load(cfg)
model = priors.build_model(
    cfg, theta_E_center=1.7, theta_E_width=0.35, n_source=2, n_lens_light=2
)
mask = data.build_mask(cfg, image.shape)
print("mask:", None if mask is None else int(mask.sum()))
ds = ImageData(image, sim_config, background_rms=cfg.background_rms,
               exp_time=cfg.exp_time, mask=mask, sees="all")
prob = ProbModel(model, ds, mode="forward")
print("event_size:", prob.event_size)

z = np.load("/global/u1/s/seanjx/claude-lensing/claude/DESI257/runs/r2_s2l2/hmc.npz")["samples"]
print("r2 samples:", z.shape)
flat = z.reshape(-1, z.shape[-1])
flat = flat[:: max(1, len(flat) // 4000)]
print("scoring", flat.shape[0], "draws")

lps = []
for i in range(0, len(flat), 500):
    lp, _ = prob.log_prob(jnp.asarray(flat[i:i + 500]))
    lps.append(np.asarray(lp))
lps = np.concatenate(lps)
print(f"UNMASKED log posterior over r2 draws: max {lps.max():.3f}  "
      f"median {np.median(lps):.3f}  min {lps.min():.3f}")

# Same draws under the round-2 (masked) likelihood, for the offset between the two
# likelihoods -- the extra 403 pixels change the normalisation, so the two numbers are not
# on the same scale and the comparison must not be made across them by accident.
import dataclasses
mask2 = data.build_mask(
    dataclasses.replace(cfg, mask_circles=((8.0, 15.0, 12.0),)), image.shape
)
ds2 = ImageData(image, sim_config, background_rms=cfg.background_rms,
                exp_time=cfg.exp_time, mask=mask2, sees="all")
prob2 = ProbModel(model, ds2, mode="forward")
lps2 = []
for i in range(0, len(flat), 500):
    lp, _ = prob2.log_prob(jnp.asarray(flat[i:i + 500]))
    lps2.append(np.asarray(lp))
lps2 = np.concatenate(lps2)
print(f"MASKED   log posterior over r2 draws: max {lps2.max():.3f}  "
      f"median {np.median(lps2):.3f}")
print(f"median offset (unmasked - masked): {np.median(lps) - np.median(lps2):.3f}")
