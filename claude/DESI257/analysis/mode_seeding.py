"""Where were round 2's chains seeded? The 73.4% 'dominant mode' fraction is only a
measurement of mode weight if the chains were not started inside one mode."""
import sys
sys.path.insert(0, "/global/u1/s/seanjx/claude-lensing/claude/pipeline")
import numpy as np
import diagnose

RUN = "/global/u1/s/seanjx/claude-lensing/claude/DESI257/runs/r2_s2l2"
rm = diagnose.rebuild_model(RUN, quiet=True)
gi = diagnose.column(rm.model, "gamma", kind="mass")

svi = np.load(f"{RUN}/svi.npz")
mean_con = diagnose.to_constrained(rm.model, np.asarray(svi["mean"]))
print(f"SVI surrogate mean gamma: {float(mean_con[0, gi]):.4f}")

mp = np.load(f"{RUN}/map.npz", allow_pickle=True)
map_con = diagnose.to_constrained(rm.model, np.asarray(mp["best_z"]).reshape(-1))
print(f"MAP best-particle gamma:  {float(map_con[0, gi]):.4f}")

z = np.load(f"{RUN}/hmc.npz")["samples"]
first = diagnose.to_constrained(rm.model, z[:, 0, :])[:, gi]
print(f"gamma at post-burn-in draw 0, over {z.shape[0]} chains: "
      f"min {first.min():.4f} max {first.max():.4f} median {np.median(first):.4f}")
print(f"chains starting with gamma < 2.66: {int((first < 2.66).sum())}/{len(first)}")

con = diagnose.to_constrained(rm.model, z.reshape(-1, z.shape[-1]))[:, gi]
print(f"draws with gamma < 2.66: {100*float((con < 2.66).mean()):.1f}%")
