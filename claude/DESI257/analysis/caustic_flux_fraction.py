"""Steps 4 and 5: extended-source flux fraction inside the tangential caustic,
at the median and propagated over posterior draws.
"""
import sys, time
sys.path.insert(0, '/global/u1/s/seanjx/claude-lensing/claude/pipeline')
sys.path.insert(0, '/global/u1/s/seanjx/claude-lensing/claude/DESI257')

import numpy as np
import jax.numpy as jnp
import diagnose
from gigalens.jax.analysis import LensSolver
from matplotlib.path import Path

RUN_DIR = '/global/u1/s/seanjx/claude-lensing/claude/DESI257/runs/r2_s2l2'
SCRATCH = '/tmp/claude-105387/-global-u1-s-seanjx-claude-lensing/6c331f7b-f28d-47a9-974e-6a49ca61ce98/scratchpad'

rm = diagnose.rebuild_model(RUN_DIR, quiet=True)
model = rm.model
names = model.z_param_names
spans = rm.spans

solver = LensSolver(rm.prob)
profiles = [comp.profile for comp in model.planes[1].light]


def comp_params_from_z(z_con, span):
    idxs = list(range(*span))
    d = {}
    for i in idxs:
        n = names[i]
        key = n.rsplit('/', 1)[-1]
        d[key] = float(z_con[i])
    return d


def ellip_to_qphi(e1, e2):
    phi = np.arctan2(e2, e1) / 2
    c = np.hypot(e1, e2)
    q = (1 - c) / (1 + c)
    return q, phi


def flux_grid_and_total(profile, p, k=18.0, n=400):
    """Grid over the component's own light out to k*R_sersic (major-axis-adjusted),
    return (X, Y, flux, total_flux_on_grid)."""
    q, phi = ellip_to_qphi(p['e1'], p['e2'])
    half = k * p['R_sersic'] / np.sqrt(q)  # covers major axis
    cx, cy = p['center_x'], p['center_y']
    xs = np.linspace(cx - half, cx + half, n)
    ys = np.linspace(cy - half, cy + half, n)
    X, Y = np.meshgrid(xs, ys)
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    flux = np.asarray(profile.light(
        Xj, Yj, R_sersic=p['R_sersic'], n_sersic=p['n_sersic'],
        e1=p['e1'], e2=p['e2'], center_x=p['center_x'], center_y=p['center_y'],
        Ie=p['Ie']))
    dpix = (2 * half / (n - 1)) ** 2
    total = flux.sum() * dpix
    return X, Y, flux, total, dpix


def inside_flux_fraction(profile, p, caustic_verts, k=18.0, n=400):
    X, Y, flux, total, dpix = flux_grid_and_total(profile, p, k=k, n=n)
    path = Path(caustic_verts)
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    inside = path.contains_points(pts).reshape(X.shape)
    inside_flux = (flux * inside).sum() * dpix
    return inside_flux / total, total


# ---------------- Step 4: at the median ----------------
z_median_unc = np.load(f'{SCRATCH}/z_median_unc.npy')
z_median_con = diagnose.to_constrained(model, z_median_unc[None, :])[0]
caustic_verts = np.load(f'{SCRATCH}/caustic_verts_median.npy')

print('=== Step 4: flux fraction inside caustic (median), convergence check ===')
for i, span in enumerate(spans):
    p = comp_params_from_z(z_median_con, span)
    for k in (10.0, 18.0, 30.0):
        frac, total = inside_flux_fraction(profiles[i], p, caustic_verts, k=k, n=400)
        print(f'src{i} k={k}: inside_flux_fraction={frac:.5f}  total_flux(grid)={total:.4f}')

print()
print('=== Step 5: posterior propagation over dominant-mode draws ===')
kept = np.load(f'{SCRATCH}/kept_mask.npy')
d = np.load(f'{RUN_DIR}/hmc.npz', allow_pickle=True)
samples = d['samples']  # (50, 10000, 36)
dom = samples[kept]  # (17, 10000, 36)
dom_flat = dom.reshape(-1, dom.shape[-1])

rng = np.random.default_rng(0)
N_DRAWS = 300
idx = rng.choice(dom_flat.shape[0], size=N_DRAWS, replace=False)
draws_unc = dom_flat[idx]

theta_e_idx = diagnose.column(model, 'theta_E', kind='mass')
gamma_idx = diagnose.column(model, 'gamma', kind='mass')

inside_centroid = {0: [], 1: []}
flux_fracs = {0: [], 1: []}
n_curves_hist = []
n_images_hist = {0: [], 1: []}
gammas = []
t0 = time.time()
for draw_i, z_unc in enumerate(draws_unc):
    z_con = diagnose.to_constrained(model, z_unc[None, :])[0]
    params = model.constrained(np.asarray(z_unc))
    gam = float(z_con[gamma_idx])
    gammas.append(gam)
    critical, caustics = solver.critical_and_caustics(params, plane=1)
    n_curves_hist.append(len(caustics))
    # tangential = curve with largest mean radius from lens center in IMAGE plane
    theta_e_c_x = diagnose.column(model, 'center_x', kind='mass')
    theta_e_c_y = diagnose.column(model, 'center_y', kind='mass')
    lens_center = (float(z_con[theta_e_c_x]), float(z_con[theta_e_c_y]))
    if len(caustics) == 0:
        for c in (0, 1):
            inside_centroid[c].append(np.nan)
            flux_fracs[c].append(np.nan)
            n_images_hist[c].append(np.nan)
        continue
    radii = [diagnose.curve_radius(cc, lens_center) for cc in critical]
    tang_i = int(np.argmax(radii))
    xs, ys = caustics[tang_i]
    verts = np.stack([xs, ys], axis=-1)
    path = Path(verts)
    for c, span in enumerate(spans):
        p = comp_params_from_z(z_con, span)
        inside = bool(path.contains_point((p['center_x'], p['center_y'])))
        inside_centroid[c].append(inside)
        frac, _ = inside_flux_fraction(profiles[c], p, verts, k=18.0, n=250)
        flux_fracs[c].append(frac)
        imgs = solver.solve_images(params, (p['center_x'], p['center_y']), plane=1)
        n_images_hist[c].append(imgs.shape[1])
    if draw_i % 25 == 0:
        print(f'  draw {draw_i}/{N_DRAWS}  elapsed {time.time()-t0:.1f}s  gamma={gam:.3f} '
              f'n_caustics={len(caustics)}')

print(f'Total time for {N_DRAWS} draws: {time.time()-t0:.1f}s')
print('n_curves histogram:', np.unique(n_curves_hist, return_counts=True))

np.savez(f'{SCRATCH}/posterior_results.npz',
         inside_centroid_0=np.array(inside_centroid[0]),
         inside_centroid_1=np.array(inside_centroid[1]),
         flux_frac_0=np.array(flux_fracs[0]),
         flux_frac_1=np.array(flux_fracs[1]),
         n_images_0=np.array(n_images_hist[0]),
         n_images_1=np.array(n_images_hist[1]),
         gammas=np.array(gammas),
         n_curves_hist=np.array(n_curves_hist))

for c in (0, 1):
    ic = np.array(inside_centroid[c], dtype=float)
    ff = np.array(flux_fracs[c], dtype=float)
    ni = np.array(n_images_hist[c], dtype=float)
    valid = ~np.isnan(ic)
    print(f'--- src{c} ---')
    print(f'  P(centroid inside caustic) = {np.nanmean(ic):.4f}  ({int(np.nansum(ic))}/{valid.sum()} draws)')
    print(f'  flux fraction inside: median={np.nanmedian(ff):.4f}, '
          f'16-84pct=[{np.nanpercentile(ff,16):.4f}, {np.nanpercentile(ff,84):.4f}]')
    print(f'  n_images: unique/counts = {np.unique(ni[~np.isnan(ni)], return_counts=True)}')
