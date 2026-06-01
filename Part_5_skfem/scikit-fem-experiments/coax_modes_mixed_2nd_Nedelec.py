import numpy as np
import meshio
import matplotlib.pyplot as plt
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import eigs
from skfem import Basis, ElementTriN1, ElementTriP1, ElementTriP0, BilinearForm, ElementTriN2, ElementTriP2
from skfem.io import from_meshio
from skfem.helpers import curl, dot
from skfem.visuals.matplotlib import plot
from skfem import Functional

# --- 0. Configuration & Physical Tags ---
ptag_gnd = 1
ptag_line = 2
ptag_teflon = 3 
ptag_air = 4
fill_epsr = 2.1

f0 = 2e9
num_modes = 8
mode_idx = 0 # Plot the first discovered mode

k0 = 2*np.pi*f0/3e8  # Operating wavenumber (sets your analysis frequency)
# e_eff_target = fill_epsr
e_eff_target = 1.5


###############
kz2_target = (k0**2)*e_eff_target

# 1. Load Mesh via Meshio
msh = meshio.read("Coax.msh")
mesh = from_meshio(msh)

pIdx_surf = msh.cell_data_dict['gmsh:physical']['triangle']
pIdx_edge = msh.cell_data_dict['gmsh:physical']['line']

# --- 2. Build Dielectric Profile ---
eps_elements = np.ones(mesh.t.shape[1])
eps_elements[pIdx_surf == ptag_teflon] = fill_epsr
eps_elements[pIdx_surf == ptag_air] = 1.0

# --- 3. Define Both Discrete Function Spaces ---
element_t = ElementTriN2()   # 2nd-order edge elements for Et
basis_t = Basis(mesh, element_t)

element_z = ElementTriP2()   # 2nd-order nodal elements for Ez
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
pec_mask = np.isin(pIdx_edge, [ptag_gnd, ptag_line])
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
# pec_nodes = np.unique(meshio_pec_lines)
# pec_dofs_z = basis_z.get_dofs(nodes=pec_nodes).all()
# free_z = basis_z.complement_dofs(pec_dofs_z)
pec_dofs_z = basis_z.get_dofs(facets=pec_facet_indices).all()
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

# --- 7. Solve the System ---


sigma_target = -kz2_target
# Shift-invert target mode search (sigma handles the zero-diagonal blocks gracefully)
vals, vecs = eigs(A_cropped, M=B_cropped, k=num_modes, sigma=sigma_target, which='LM')

kz2 = -np.real(vals)
kc2 = k0**2.0 - kz2
kc = np.sqrt(np.abs(np.real(kc2)))*np.sign(kc2)

fc = kc*3e8/(2*np.pi)

print("\nCalculated Propagation Constants Squared (beta^2):")
for i, val in enumerate(vals):
    # print(f"Mode {i+1}: {np.real(val):.6f}")
    print(f"Mode {i+1}: {(fc[i]/1e9):.6f}")
# --- 8. Reconstruct and Plot Et field ---

# Extract only the Transverse entries from the combined solution vector
len_free_t = len(free_t)
vec_t_free = vecs[:len_free_t, mode_idx]

# Map back to full sizing including zeroed PEC boundaries
full_vec_t = np.zeros(basis_t.N, dtype=complex)
full_vec_t[free_t] = vec_t_free

Et_I = basis_t.interpolator(full_vec_t)
val_nodes = Et_I(mesh.p)

mag_nodes = np.sqrt(np.abs(val_nodes[0])**2 + np.abs(val_nodes[1])**2)
mag_nodes = np.resize(mag_nodes,mag_nodes.size)

# --- 3. Set Up the Plot System ---
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
ax.set_title(f"Microstrip Mode 1: |E|")
ax.set_xlabel("X Coordinate [m]")
ax.set_ylabel("Y Coordinate [m]")
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
