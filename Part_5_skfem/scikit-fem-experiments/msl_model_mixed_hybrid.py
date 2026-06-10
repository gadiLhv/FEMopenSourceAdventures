import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import eigs
from skfem import Basis, ElementTriN1, ElementTriP1, ElementTriP0, ElementVector, BilinearForm
from skfem.io import from_meshio
from skfem.helpers import curl, dot

# =============================================================================
# Hybrid revision: FAST sparse shift-invert + physical-mode FILTERING.
#
# The lesson learned from the dense (msl_model_mixed_direct.py) experiment:
# the robustness we wanted never came from the dense solve itself -- it came
# from FILTERING the spectrum for the physical guided mode. The original
# fragility was returning a handful of *unfiltered* modes and trusting
# "Mode 1".
#
# So here we keep the cheap sparse solver (scipy.sparse.linalg.eigs, seconds)
# but:
#   * use ONE fixed, neutral shift sigma (no per-run guessing),
#   * ask for MANY modes (k=NUM_MODES) so the physical one is surely captured,
#   * apply the same physical-window filter (1 < e_eff < eps_r).
# The shift sits in the negative-lambda region, far from the lambda~0 spurious
# swarm, so the physical mode is not crowded out. Result: dense-quality answer
# at sparse speed, insensitive to the exact shift.
# =============================================================================

# --- 0. Configuration & Physical Tags ---
ptag_gnd = 1
ptag_line = 2
ptag_air = 3
ptag_fr4 = 4
ptag_bounds = 5

fr4_epsr = 4.2
f0 = 2e9

k0 = 2 * np.pi * f0 / 3e8  # Operating wavenumber (sets your analysis frequency)

# Solver controls
NUM_MODES = 30  # modes to pull near the shift
# e_eff_shift = 0.5*(1 + fr4_epsr)            # neutral shift guess (NOT tuned per run)
e_eff_shift = 1.5
sigma_shift = -(k0 ** 2) * e_eff_shift  # lambda = -kz^2, so shift is negative

# 1. Load Mesh via Meshio
msh = meshio.read("MSL_XY.msh")
mesh = from_meshio(msh)

pIdx_surf = msh.cell_data_dict['gmsh:physical']['triangle']
pIdx_edge = msh.cell_data_dict['gmsh:physical']['line']

# --- 2. Build Dielectric Profile ---
eps_elements = np.ones(mesh.t.shape[1])
eps_elements[pIdx_surf == ptag_fr4] = fr4_epsr
eps_elements[pIdx_surf == ptag_air] = 1.0

# --- 3. Define Both Discrete Function Spaces ---
element_t = ElementTriN1()  # Edge elements for transverse fields (Et)
basis_t = Basis(mesh, element_t)

element_z = ElementTriP1()  # Nodal Lagrange elements for longitudinal fields (Ez)
basis_z = Basis(mesh, element_z)

# Project material properties to the respective quadrature points
basis0_t = basis_t.with_element(ElementTriP0())
eps_quad_t = basis0_t.interpolate(eps_elements)

basis0_z = basis_z.with_element(ElementTriP0())
eps_quad_z = basis0_z.interpolate(eps_elements)


# --- 4. Weak Formulations (The Sub-Blocks) ---
@BilinearForm
def att_form(u, v, w):
    return curl(u) * curl(v) - (k0 ** 2) * w.epsilon * dot(u, v)


@BilinearForm
def btt_form(u, v, w):
    return dot(u, v)


@BilinearForm
def btz_form(u, v, w):
    return dot(u, v.grad)  # u is vector (Et), v.grad is vector gradient of scalar (Ez)


@BilinearForm
def bzt_form(u, v, w):
    return dot(u.grad, v)  # u.grad is vector gradient of scalar, v is vector


@BilinearForm
def bzz_form(u, v, w):
    return dot(u.grad, v.grad) - (k0 ** 2) * w.epsilon * u * v


# Assemble individual sub-blocks
A_tt = att_form.assemble(basis_t, basis_t, epsilon=eps_quad_t)
B_tt = btt_form.assemble(basis_t, basis_t)
B_tz = btz_form.assemble(basis_t, basis_z)
B_zt = bzt_form.assemble(basis_z, basis_t)
B_zz = bzz_form.assemble(basis_z, basis_z, epsilon=eps_quad_z)

# --- 5. Enforce PEC Boundary Conditions ---
# Find all meshio lines that are metallic boundaries
pec_mask = np.isin(pIdx_edge, [ptag_gnd, ptag_line, ptag_bounds])
meshio_pec_lines = msh.cells_dict['line'][pec_mask]

# A. Transverse Edges (Et = 0)
meshio_pec_lines_sorted = np.sort(meshio_pec_lines, axis=1)
pec_set = {tuple(row) for row in meshio_pec_lines_sorted}
skfem_facets = mesh.facets.T
pec_facet_indices = np.array([
    i for i, facet in enumerate(skfem_facets) if tuple(facet) in pec_set
])
pec_dofs_t = basis_t.get_dofs(facets=pec_facet_indices).all()
free_t = basis_t.complement_dofs(pec_dofs_t)

# B. Longitudinal Nodes (Ez = 0)
pec_nodes = np.unique(meshio_pec_lines)
pec_dofs_z = basis_z.get_dofs(nodes=pec_nodes).all()
free_z = basis_z.complement_dofs(pec_dofs_z)

# --- 6. Crop and Glue Blocks into Global System ---
# Swap B_zt and B_tz to align with their true test/trial (row/column) dimensions

A_cropped = bmat([
    [A_tt[free_t,:][:, free_t], None],
    [None, csr_matrix((len(free_z), len(free_z)))]
], format='csr')

B_cropped = bmat([
    [B_tt[free_t,:][:, free_t], B_zt[free_t,:][:, free_z]],  # Top-Right: Row=t, Col=z
    [B_tz[free_z,:][:, free_t], B_zz[free_z,:][:, free_z]]  # Bottom-Left: Row=z, Col=t
], format='csr')

# --- 7. Solve (sparse shift-invert, many modes, ONE fixed shift) ---
n = A_cropped.shape[0]
ncv = min(n - 1, max(4 * NUM_MODES, 60))  # generous Krylov subspace for robustness
print(f"Sparse shift-invert: {NUM_MODES} modes near e_eff~{e_eff_shift:.2f} "
      f"(matrix {n}x{n})...", flush=True)
vals, vecs = eigs(A_cropped, M=B_cropped, k=NUM_MODES,
                  sigma=sigma_shift, ncv=ncv, which='LM')

kz2_all = -vals  # complex; physical guided modes are ~real
e_eff_all = kz2_all / k0 ** 2  # effective Dk = kz^2 / k0^2

# --- 7a. Filter for the physical guided modes (this is what gives robustness) ---
e_eff_re = np.real(e_eff_all)

finite = np.isfinite(e_eff_all)
realish = np.abs(np.imag(e_eff_all)) < 1e-6 * np.maximum(np.abs(e_eff_re), 1.0)
in_window = (e_eff_re > 1.0) & (e_eff_re < fr4_epsr)

physical = finite & realish & in_window
idx_phys = np.where(physical)[0]
idx_phys = idx_phys[np.argsort(-e_eff_re[idx_phys])]  # most-bound first


def fc_of(kz2):
    kc2 = k0 ** 2.0 - np.real(kz2)
    kc = np.sqrt(np.abs(kc2)) * np.sign(kc2)
    return kc * 3e8 / (2 * np.pi)


print("\nPhysical guided modes (1 < e_eff < eps_r), most-bound first:", flush=True)
if idx_phys.size == 0:
    print("  (none in window -- showing nearest real candidates; "
          "consider widening NUM_MODES or the shift)", flush=True)
    cand = np.where(finite & realish & (e_eff_re > 0.0))[0]
    idx_phys = cand[np.argsort(-e_eff_re[cand])][:8]

for rank, i in enumerate(idx_phys):
    print(f"  Mode {rank+1}: e_eff = {e_eff_re[i]:.6f}   "
          f"kz^2 = {np.real(kz2_all[i]):.4f}   "
          f"fc = {fc_of(kz2_all[i])/1e9:.6f} GHz", flush=True)

# --- 8. Reconstruct and Plot Et field of the fundamental physical mode ---
mode_idx = idx_phys[0]
print(f"\nPlotting fundamental mode: e_eff = {e_eff_re[mode_idx]:.6f}", flush=True)

len_free_t = len(free_t)
vec_t_free = vecs[:len_free_t, mode_idx]

full_vec_t = np.zeros(basis_t.N, dtype=complex)
full_vec_t[free_t] = vec_t_free

# NOTE: basis_t.interpolator() / probes() is broken for vector (edge) elements
# in skfem 11.x -- it builds a mismatched coo_matrix and raises ValueError.
# Instead, project the Nedelec field onto a nodal P1 vector basis and read the
# nodal x/y components. Project real and imaginary parts separately because
# Basis.project() silently discards the imaginary part of a complex vector.
basis_viz = basis_t.with_element(ElementVector(ElementTriP1()))
viz_vec = (basis_viz.project(basis_t.interpolate(full_vec_t.real))
           +1j * basis_viz.project(basis_t.interpolate(full_vec_t.imag)))

ux_nodes = viz_vec[basis_viz.nodal_dofs[0]]
uy_nodes = viz_vec[basis_viz.nodal_dofs[1]]
mag_nodes = np.sqrt(np.abs(ux_nodes) ** 2 + np.abs(uy_nodes) ** 2)

# --- 9. Set Up the Plot System ---
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.tripcolor(
    mesh.p[0], mesh.p[1], mesh.t.T,
    mag_nodes,
    shading='gouraud',
    cmap='turbo'
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Absolute Field Magnitude $|E_t|$ (Linear Scale)', rotation=270, labelpad=15)
ax.set_title(f"Microstrip Fundamental Mode: |E|  (e_eff = {e_eff_re[mode_idx]:.3f})")
ax.set_xlabel("X Coordinate [m]")
ax.set_ylabel("Y Coordinate [m]")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
