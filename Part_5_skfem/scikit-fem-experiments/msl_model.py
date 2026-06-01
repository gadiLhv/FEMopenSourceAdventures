import numpy as np
import meshio
from skfem import Basis, ElementTriN1, ElementTriP0, BilinearForm
from skfem.io import from_meshio
from skfem.helpers import curl, dot
import matplotlib.pyplot as plt

ptag_gnd = 1
ptag_line = 2
ptag_air = 3 
ptag_fr4 = 4
ptag_bounds = 5

fr4_epsr = 4.5

# 1. Load the mesh using meshio and convert to skfem
msh = meshio.read("MSL_XY.msh")
mesh = from_meshio(msh)

# 2. Extract the raw numeric IDs for all triangle elements
# This returns a 1D NumPy array matching the triangles in mesh.t
pIdx_surf = msh.cell_data_dict['gmsh:physical']['triangle']
pIdx_edge = msh.cell_data_dict['gmsh:physical']['line']

# 3. Initialize your dielectric array (defaulting to 1.0 for vacuum/air)
eps_elements = np.ones(mesh.t.shape[1])

# 4. Assign specific dielectrics by matching the index
# Change these numbers (e.g., 1, 2) to match your actual Gmsh physical surface IDs
# Let's assume ID 2 is your microstrip substrate (e.g., FR4 = 4.4)
eps_elements[pIdx_surf == ptag_fr4] = fr4_epsr 
# Let's assume ID 3 is another dielectric material if present
eps_elements[pIdx_surf == ptag_air] = 1 

# 5. Define Finite Element Basis and Weak Formulations
element = ElementTriN1() 
basis = Basis(mesh, element)

@BilinearForm
def stiffness_form(u, v, w):
    return curl(u) * curl(v)

@BilinearForm
def mass_form(u, v, w):
    return w.epsilon * dot(u, v)

# 6. Project element-wise constants to quadrature integration points
basis0 = basis.with_element(ElementTriP0())
eps_quadrature = basis0.interpolate(eps_elements)

# 7. Assemble the global matrices
A = stiffness_form.assemble(basis)
B = mass_form.assemble(basis, epsilon=eps_quadrature)


#######################################
# Start detecting boundary conditions #
#######################################

# 8. Identify the meshio line elements matching your metallic tags (1, 2, and 5)
pec_mask = np.isin(pIdx_edge, [ptag_gnd, ptag_line, ptag_bounds])
meshio_pec_lines = msh.cells_dict['line'][pec_mask]

# Sort node pairs horizontally to match skfem's sorted facet convention
meshio_pec_lines_sorted = np.sort(meshio_pec_lines, axis=1)
pec_set = {tuple(row) for row in meshio_pec_lines_sorted}

# skfem stores facets as a (2, Nfacets) array where each column is already sorted
skfem_facets = mesh.facets.T 

# Map the meshio lines to skfem facet indices
pec_facet_indices = np.array([
    i for i, facet in enumerate(skfem_facets) if tuple(facet) in pec_set
])

# 9. Get the exact DOFs associated with these boundary facets
pec_dofs = basis.get_dofs(facets=pec_facet_indices).all()

# Isolate the remaining free (unconstrained) internal degrees of freedom
free_dofs = basis.complement_dofs(pec_dofs)

#######################################
# Solve equation system #
#######################################

from scipy.sparse.linalg import eigs

# 10. Solve the Generalized Eigenvalue Problem
num_modes = 6

# # Using a shift (sigma=0.1) bypasses the huge static nullspace (0 eigenvalues)
# vals, vecs = eigs(
#     A[free_dofs, :][:, free_dofs], 
#     M=B[free_dofs, :][:, free_dofs], 
#     k=num_modes, 
#     sigma=0.1,  
#     which='LM'
# )

# Using a shift (sigma=0.1) bypasses the huge static nullspace (0 eigenvalues)
vals, vecs = eigs(
    A[free_dofs, :][:, free_dofs], 
    M=B[free_dofs, :][:, free_dofs], 
    k=num_modes, 
    sigma=3000,  
    which='LM'
)

# 11. Reconstruct full vector fields (padding back the 0s on the PEC boundaries)
full_vecs = np.zeros((basis.N, num_modes), dtype=complex)
full_vecs[free_dofs, :] = vecs

print("Calculated Waveguide Eigenvalues (k0^2 / Cutoff Wavenumbers Squared):")
for i, val in enumerate(vals):
    print(f"Mode {i+1}: {np.real(val):.6f}")



# Plotting the first fundamental mode field magnitude
mode_idx = 0
from skfem.visuals.matplotlib import plot
fig, ax = plt.subplots(figsize=(6,6))
plot(basis, np.abs(full_vecs[:, mode_idx]), ax=ax, shading='gouraud')
ax.set_title(f"Coaxial Mode {mode_idx+1} (From Gmsh)")
plt.show()