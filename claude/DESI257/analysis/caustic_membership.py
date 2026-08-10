"""Steps 2 and 3 at the dominant-mode median parameter vector."""
import sys, time
sys.path.insert(0, '/global/u1/s/seanjx/claude-lensing/claude/pipeline')
sys.path.insert(0, '/global/u1/s/seanjx/claude-lensing/claude/DESI257')

import numpy as np
import diagnose
from gigalens.jax.analysis import LensSolver
from matplotlib.path import Path

RUN_DIR = '/global/u1/s/seanjx/claude-lensing/claude/DESI257/runs/r2_s2l2'
SCRATCH = '/tmp/claude-105387/-global-u1-s-seanjx-claude-lensing/6c331f7b-f28d-47a9-974e-6a49ca61ce98/scratchpad'

rm = diagnose.rebuild_model(RUN_DIR, quiet=True)
model = rm.model

z_median_unc = np.load(f'{SCRATCH}/z_median_unc.npy')
params = model.constrained(np.asarray(z_median_unc))
print('params keys:', list(params.keys()) if hasattr(params, 'keys') else type(params))

solver = LensSolver(rm.prob)
print('trace_mode', solver.trace_mode, 'default_plane', solver.default_plane())

t0 = time.time()
critical, caustics = solver.critical_and_caustics(params, plane=1)
print(f'critical_and_caustics took {time.time()-t0:.2f}s')
print(f'n critical curves = {len(critical)}, n caustics = {len(caustics)}')
for i, (xs, ys) in enumerate(caustics):
    closure_gap = np.hypot(xs[0]-xs[-1], ys[0]-ys[-1])
    print(f'caustic {i}: n_vertices={len(xs)}, closure_gap={closure_gap:.6f} arcsec, '
          f'extent x=[{xs.min():.4f},{xs.max():.4f}] y=[{ys.min():.4f},{ys.max():.4f}]')

# theta_E validation
gamma_idx = diagnose.column(model, 'gamma', kind='mass')
theta_e_idx = diagnose.column(model, 'theta_E', kind='mass')
z_median_con = diagnose.to_constrained(model, z_median_unc[None, :])[0]
theta_E = z_median_con[theta_e_idx]
gamma = z_median_con[gamma_idx]
center_idx_x = diagnose.column(model, 'center_x', kind='mass')
center_idx_y = diagnose.column(model, 'center_y', kind='mass')
lens_center = (z_median_con[center_idx_x], z_median_con[center_idx_y])
print(f'theta_E={theta_E:.4f}, gamma={gamma:.4f}, lens_center={lens_center}')
diagnose.check_theta_E(critical, lens_center, theta_E, gamma=gamma)

# --- component centroids ---
spans = rm.spans  # [(22,29),(29,36)] source component spans
names = model.z_param_names
def comp_params(span):
    idxs = list(range(*span))
    d = {}
    for i in idxs:
        n = names[i]
        key = n.rsplit('/', 1)[-1]
        d[key] = z_median_con[i]
    return d

src_params = [comp_params(s) for s in spans]
for i, sp in enumerate(src_params):
    print(f'src{i} params:', sp)

# Use the single tangential caustic (gamma>2 -> exactly one curve expected)
assert len(caustics) == 1, f'expected exactly 1 caustic (gamma>2), got {len(caustics)}'
xs, ys = caustics[0]
verts = np.stack([xs, ys], axis=-1)
path = Path(verts)

def signed_distance(px, py, verts):
    """Signed distance (arcsec) from point to polyline; sign from containment.
    Positive = inside, negative = outside."""
    # distance to each segment
    p = np.array([px, py])
    v = verts
    n = len(v)
    dmin = np.inf
    for i in range(n - 1):
        a, b = v[i], v[i+1]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-300), 0.0, 1.0)
        proj = a + t * ab
        d = np.hypot(*(p - proj))
        dmin = min(dmin, d)
    # closing segment
    a, b = v[-1], v[0]
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-300), 0.0, 1.0)
    proj = a + t * ab
    d = np.hypot(*(p - proj))
    dmin = min(dmin, d)
    inside = path.contains_point((px, py))
    return (dmin if inside else -dmin), inside

print()
print('=== Step 2: centroid inside/outside tangential caustic ===')
for i, sp in enumerate(src_params):
    cx, cy = sp['center_x'], sp['center_y']
    dist, inside = signed_distance(cx, cy, verts)
    print(f'src{i}: centroid=({cx:.4f},{cy:.4f}), inside={inside}, signed_dist={dist:.4f} arcsec')

print()
print('=== Step 3: point-source image count at each centroid ===')
for i, sp in enumerate(src_params):
    cx, cy = sp['center_x'], sp['center_y']
    imgs = solver.solve_images(params, (cx, cy), plane=1)
    print(f'src{i}: {imgs.shape[1]} images at {[tuple(round(v,4) for v in c) for c in imgs.T]}')

np.save(f'{SCRATCH}/caustic_verts_median.npy', verts)
