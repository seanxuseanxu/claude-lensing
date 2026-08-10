"""Depth of round 2's two modes, under the likelihood they were fitted with.

Depth is a density, not a mass. This measures the density; it does not measure the weights.
"""
import sys
sys.path.insert(0, "/global/u1/s/seanjx/claude-lensing/claude/pipeline")
import numpy as np, jax.numpy as jnp
import diagnose

RUN = "/global/u1/s/seanjx/claude-lensing/claude/DESI257/runs/r2_s2l2"
rm = diagnose.rebuild_model(RUN, quiet=True)
gi = diagnose.column(rm.model, "gamma", kind="mass")
ti = diagnose.column(rm.model, "theta_E", kind="mass")

z = np.load(f"{RUN}/hmc.npz")["samples"]
flat = z.reshape(-1, z.shape[-1])
sub = flat[:: max(1, len(flat) // 20000)]
con = diagnose.to_constrained(rm.model, sub)
below = con[:, gi] < 2.66
print(f"thinned to {len(sub)} draws; gamma<2.66 fraction {100*below.mean():.1f}%")

lp, chi2 = [], []
for i in range(0, len(sub), 500):
    a, b = rm.prob.log_prob(jnp.asarray(sub[i:i+500]))
    lp.append(np.asarray(a)); chi2.append(np.asarray(b))
lp = np.concatenate(lp); chi2 = np.concatenate(chi2)
prior = np.asarray(rm.prob.log_prior(jnp.asarray(sub)))

n_kept = int(np.asarray(rm.mask).sum())
for name, sel in (("mode A (gamma<2.66)", below), ("mode B (gamma>2.66)", ~below)):
    print(f"{name}: n={sel.sum()}  median logpost {np.median(lp[sel]):.1f}  "
          f"median log prior {np.median(prior[sel]):.1f}  "
          f"median chi2_nu {np.median(chi2[sel]):.4f}  "
          f"median theta_E {np.median(con[sel, ti]):.4f}  "
          f"median gamma {np.median(con[sel, gi]):.4f}")
d = np.median(chi2[~below]) - np.median(chi2[below])
print(f"delta chi2 (B - A) over {n_kept} kept px: {d*n_kept:.1f}")
print(f"delta median log posterior (B - A): {np.median(lp[~below])-np.median(lp[below]):.1f}")
print(f"delta median log prior     (B - A): {np.median(prior[~below])-np.median(prior[below]):.1f}")
for nm in ("e1","e2"):
    c = diagnose.column(rm.model, nm, kind="mass")
    print(f"host {nm}: A {np.median(con[below,c]):.4f}  B {np.median(con[~below,c]):.4f}")
g1 = diagnose.column(rm.model, "gamma1", kind="mass"); g2 = diagnose.column(rm.model, "gamma2", kind="mass")
sh = np.hypot(con[:,g1], con[:,g2])
print(f"|shear|: A {np.median(sh[below]):.4f}  B {np.median(sh[~below]):.4f}")
print(f"gamma max overall: {con[:,gi].max():.4f}  mode B 98th pct {np.percentile(con[~below,gi],98):.4f}")
