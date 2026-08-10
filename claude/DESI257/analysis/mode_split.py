"""Step 1: reproduce RESULTS_v2.md section 3's mode split.

Per-draw membership by constrained gamma < 2.66; chains kept if frac_below > 0.98.
"""
import sys
sys.path.insert(0, '/global/u1/s/seanjx/claude-lensing/claude/pipeline')
sys.path.insert(0, '/global/u1/s/seanjx/claude-lensing/claude/DESI257')

import numpy as np
import diagnose

RUN_DIR = '/global/u1/s/seanjx/claude-lensing/claude/DESI257/runs/r2_s2l2'

rm = diagnose.rebuild_model(RUN_DIR, quiet=True)
model = rm.model

d = np.load(f'{RUN_DIR}/hmc.npz', allow_pickle=True)
samples = d['samples']  # (n_chains, n_steps, n_params) unconstrained
n_chains, n_steps, n_params = samples.shape
print(f'samples shape: {samples.shape}')

gamma_idx = diagnose.column(model, 'gamma', kind='mass')
theta_e_idx = diagnose.column(model, 'theta_E', kind='mass')
print('gamma_idx', gamma_idx, 'theta_e_idx', theta_e_idx)

flat = samples.reshape(-1, n_params)
constrained_flat = diagnose.to_constrained(model, flat)
constrained = constrained_flat.reshape(n_chains, n_steps, n_params)

gamma = constrained[:, :, gamma_idx]  # (n_chains, n_steps)
theta_E = constrained[:, :, theta_e_idx]

below = gamma < 2.66
frac_below = below.mean(axis=1)  # per chain
print('frac_below per chain, sorted:')
print(np.sort(frac_below))

kept = frac_below > 0.98
print(f'kept chains: {kept.sum()} / {n_chains}')
print('kept chain indices:', np.where(kept)[0].tolist())
print('rejected-chain frac_below range:', frac_below[~kept].min(), frac_below[~kept].max())
print('kept-chain frac_below range:', frac_below[kept].min(), frac_below[kept].max())

dominant_theta_E = theta_E[kept]
minority_theta_E = theta_E[~kept]
print(f'dominant-mode median theta_E = {np.median(dominant_theta_E):.4f}')
print(f'dominant-mode 16/84 pct theta_E = {np.percentile(dominant_theta_E, [16,84])}')
print(f'minority-mode median theta_E = {np.median(minority_theta_E):.4f}')

# fraction of draws (over all 50 chains) with gamma<2.66
draw_frac_below = below.mean()
print(f'overall per-draw frac gamma<2.66 = {draw_frac_below:.4f} (paper: 0.734 dominant)')

# dominant-mode median for FULL parameter vector, in unconstrained space then constrained
# Use unconstrained median (as point_estimate() does), restricted to kept chains
dom_unc = samples[kept].reshape(-1, n_params)
z_median_unc = np.median(dom_unc, axis=0)
z_median_con = diagnose.to_constrained(model, z_median_unc[None, :])[0]

np.save('/tmp/claude-105387/-global-u1-s-seanjx-claude-lensing/6c331f7b-f28d-47a9-974e-6a49ca61ce98/scratchpad/z_median_unc.npy', z_median_unc)
np.save('/tmp/claude-105387/-global-u1-s-seanjx-claude-lensing/6c331f7b-f28d-47a9-974e-6a49ca61ce98/scratchpad/kept_mask.npy', kept)

print('median param vector (constrained), theta_E, gamma:',
      z_median_con[theta_e_idx], z_median_con[gamma_idx])

for i, n in enumerate(model.z_param_names):
    print(i, n, z_median_con[i])
