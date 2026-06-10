import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.sparse import bmat, csr_matrix
from scipy.linalg import eig  # dense, full-spectrum generalized eigensolver
from scipy.sparse.linalg import eigs  # used ONLY to recover one eigenvector at a known eigenvalue
import time
from skfem import Basis, ElementTriN1, ElementTriP1, ElementTriP0, ElementVector, BilinearForm
from skfem.io import from_meshio
from skfem.helpers import curl, dot

# =============================================================================
# Direct-solve revision of msl_model_mixed.py (1st-order Nedelec / P1).
#
# Motivation: the sparse shift-invert path (scipy.sparse.linalg.eigs with a
# `sigma` target) is hypersensitive to the e_eff_target guess. That is a known
# pathology of this mixed Et-Ez formulation: the pencil is indefinite and full
# of spurious near-null modes, and ARPACK happily converges to whatever cluster
# sits nearest the shift.
#
# This revision sidesteps the guess entirely: scipy.linalg.eig computes the
# ENTIRE spectrum (no sigma), and we filter for the physical guided modes
# afterwards. For this mesh (~2k DOF) the dense O(n^3) solve is sub-second.
# This stays feasible up to ~10k DOF; beyond that, move to SLEPc.
# =============================================================================

# --- 0. Configuration & Physical Tags ---
ptag_gnd = 1
ptag_line = 2
ptag_air = 3
ptag_fr4 = 4
ptag_bounds = 5

fr4_epsr = 4.2
f0 = 2e9


k0 = 2*np.pi*f0/3e8  # Operating wavenumber (sets your analysis frequency)

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
element_t = ElementTriN1()   # Edge elements for transverse fields (Et)
basis_t = Basis(mesh, element_t)

element_z = ElementTriP1()   # Nodal Lagrange elements for longitudinal fields (Ez)
basis_z = Basis(mesh, element_z)

# Project material properties to the respective quadrature points
basis0_t = basis_t.with_element(ElementTriP0())
eps_quad_t = basis0_t.interpolate(eps_elements)

basis0_z = basis_z.with_element(ElementTriP0())
eps_quad_z = basis0_z.interpolate(eps_elements)

# --- 4. Weak Formulations (The Sub-Blocks) ---
@BilinearForm
def att_form(u, v, w):
    return curl(u) * curl(v) - (k0**2) * w.epsilon * dot(u, v)

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
    return dot(u.grad, v.grad) - (k0**2) * w.epsilon * u * v

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
    [A_tt[free_t, :][:, free_t], None],
    [None, csr_matrix((len(free_z), len(free_z)))]
], format='csr')

B_cropped = bmat([
    [B_tt[free_t, :][:, free_t], B_zt[free_t, :][:, free_z]], # Top-Right: Row=t, Col=z
    [B_tz[free_z, :][:, free_t], B_zz[free_z, :][:, free_z]]  # Bottom-Left: Row=z, Col=t
], format='csr')

# --- 7. Solve the System (dense, full spectrum, NO target guess) ---
# eig returns ALL eigenvalues of A x = lambda B x. Our eigenvalue is lambda = -kz^2.
#
# SOUL RELIEF: this dense solve is HEAVY (~70-120 s for ~2k DOF). Without the
# prints below it looks frozen -- especially under a debugger, where stdout is
# buffered. We also ask for eigenVALUES ONLY (right=False), which is ~40%
# faster than also computing all 2k eigenvectors. We then recover the single
# eigenvector we actually plot via a targeted solve (section 8).
n = A_cropped.shape[0]
print(f"[1/3] Dense eig on {n}x{n} (eigenvalues only). "
      f"This takes ~1-2 min -- not frozen, just LAPACK grinding...", flush=True)
_t0 = time.time()
vals = eig(A_cropped.toarray(), B_cropped.toarray(), right=False)
print(f"      ...done in {time.time()-_t0:.1f} s.", flush=True)

kz2_all = -vals                       # complex; physical guided modes are ~real
e_eff_all = kz2_all / k0**2           # effective Dk = kz^2 / k0^2

# --- 7a. Filter for the physical guided modes ---
# The mixed formulation produces a swarm of spurious modes:
#   * lambda ~ 0  ->  e_eff ~ 0   (gradient / static null space of the zero block)
#   * complex / huge eigenvalues  (indefinite pencil junk)
# A genuine quasi-TEM microstrip mode is real with 1 < e_eff < eps_r.
e_eff_re = np.real(e_eff_all)

finite = np.isfinite(e_eff_all)
# nearly-real: imaginary part negligible vs magnitude
realish = np.abs(np.imag(e_eff_all)) < 1e-6 * np.maximum(np.abs(e_eff_re), 1.0)
# physical guided-mode window
in_window = (e_eff_re > 1.0) & (e_eff_re < fr4_epsr)

physical = finite & realish & in_window
idx_phys = np.where(physical)[0]
# Sort most-bound first (highest e_eff = fundamental quasi-TEM at the top)
idx_phys = idx_phys[np.argsort(-e_eff_re[idx_phys])]

# Cutoff-frequency style quantities, consistent with the other scripts
def fc_of(kz2):
    kc2 = k0**2.0 - np.real(kz2)
    kc = np.sqrt(np.abs(kc2)) * np.sign(kc2)
    return kc * 3e8 / (2*np.pi)

print("\n[2/3] Physical guided modes (1 < e_eff < eps_r), most-bound first:", flush=True)
if idx_phys.size == 0:
    print("  (none found in window -- showing nearest real candidates instead)", flush=True)
    # Fallback: nearest real/finite modes by e_eff so you can see what came out
    cand = np.where(finite & realish & (e_eff_re > 0.0))[0]
    idx_phys = cand[np.argsort(-e_eff_re[cand])][:8]

for rank, i in enumerate(idx_phys):
    print(f"  Mode {rank+1}: e_eff = {e_eff_re[i]:.6f}   "
          f"kz^2 = {np.real(kz2_all[i]):.4f}   "
          f"fc = {fc_of(kz2_all[i])/1e9:.6f} GHz", flush=True)

# --- 8. Reconstruct and Plot Et field of the fundamental physical mode ---
mode_idx = idx_phys[0]  # most-bound physical mode
lambda_phys = vals[mode_idx]            # EXACT eigenvalue from the dense solve
print(f"\n[3/3] Recovering eigenvector at the known eigenvalue "
      f"(e_eff = {e_eff_re[mode_idx]:.6f})...", flush=True)

# We computed eigenvalues only, so recover the single eigenvector we plot via a
# targeted shift-invert AT the exact eigenvalue. This is NOT a guess -- sigma is
# the eigenvalue the dense solve already found -- so k=1 converges instantly.
_, vecs_one = eigs(A_cropped, M=B_cropped, k=1, sigma=lambda_phys, which='LM')

# Extract only the Transverse entries from the combined solution vector
len_free_t = len(free_t)
vec_t_free = vecs_one[:len_free_t, 0]

# Map back to full sizing including zeroed PEC boundaries
full_vec_t = np.zeros(basis_t.N, dtype=complex)
full_vec_t[free_t] = vec_t_free

# NOTE: basis_t.interpolator() / probes() is broken for vector (edge) elements
# in skfem 11.x -- it builds a mismatched coo_matrix and raises ValueError.
# Instead, project the Nedelec field onto a nodal P1 vector basis and read the
# nodal x/y components. Project real and imaginary parts separately because
# Basis.project() silently discards the imaginary part of a complex vector.
basis_viz = basis_t.with_element(ElementVector(ElementTriP1()))
viz_vec = (basis_viz.project(basis_t.interpolate(full_vec_t.real))
           + 1j * basis_viz.project(basis_t.interpolate(full_vec_t.imag)))

ux_nodes = viz_vec[basis_viz.nodal_dofs[0]]
uy_nodes = viz_vec[basis_viz.nodal_dofs[1]]
mag_nodes = np.sqrt(np.abs(ux_nodes)**2 + np.abs(uy_nodes)**2)

# --- 9. Set Up the Plot System ---
fig, ax = plt.subplots(figsize=(8, 6))

# Smooth, edge-free background on a linear scale ('turbo' matches classic MATLAB)
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
