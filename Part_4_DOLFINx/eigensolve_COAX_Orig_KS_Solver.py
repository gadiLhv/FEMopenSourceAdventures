import sys

from mpi4py import MPI

import numpy as np


import dolfinx

print(f"DOLFINx version: {dolfinx.__version__} based on GIT commit: {dolfinx.git_commit_hash} of https://github.com/FEniCS/dolfinx/")

import scipy

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_scalar_type, fem, io, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmshio


try:
    import pyvista
    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

try:
    from slepc4py import SLEPc
except ModuleNotFoundError:
    print("slepc4py is required for this demo")
    sys.exit(0)

try:
    from petsc4py import PETSc
except ModuleNotFoundError:
    print("petsc4py is required for this demo")
    sys.exit(0)

####################
# Input parameters #
####################

f0 = 2e9

er_teflon = 2.5
er_air = 1.0

numSolutions = 2
eigenMode_num = 0

vector_basisFunction_order = 2
scalar_basisFunction_order = 3

# Import mesh file
msh, cell_tags, facet_tags = gmshio.read_from_msh( "COAX.msh", MPI.COMM_WORLD, gdim=2)

# My hello world
print("Import Good!! Number of dimensions: {}".format(msh.topology.dim))

# Scalar and vector function spaces
FS_V = element("N1curl", msh.basix_cell(), vector_basisFunction_order)
FS_S = element("Lagrange", msh.basix_cell(), scalar_basisFunction_order)
combined_space = fem.functionspace(msh, mixed_element([FS_V, FS_S]))

# Define trial and test functions
Et_bf, Ez_bf = ufl.TrialFunctions(combined_space)
Et_tf, Ez_tf = ufl.TestFunctions(combined_space)

m0 = scipy.constants.mu_0
e0 = scipy.constants.epsilon_0
c0 = scipy.constants.speed_of_light

k0 = 2.0*np.pi*f0/c0

# Find tags for relevant facets and cells
facets_gnd = facet_tags.find(1)
facets_line = facet_tags.find(2)

# Tags for dielectric fill
cells_teflon = facet_tags.find(3)

# Concatenate metals
facet_metals = np.concatenate((facets_gnd, facets_line))

# Set a function space for the epsilon and mu values (for interpolation)
DG0 = fem.FunctionSpace(msh, ("DG", 0))

# Function for relative epsilon
e_r  = fem.Function(DG0)
m_r = fem.Function(DG0)

# The 'x' here doesn't mean only x. This is the class that returns
# the "Function degree-of-freedom vector".
e_r.x.array[cells_teflon] = np.full_like(cells_teflon, er_teflon, dtype=default_scalar_type)

m_r.x.array[cells_teflon] = np.full_like(cells_teflon, 1.0, dtype=default_scalar_type)


# Formulate FEM problem
a_tt = ((1/m_r)*ufl.inner(ufl.curl(Et_bf), ufl.curl(Et_tf)) - (k0**2) * e_r * ufl.inner(Et_bf, Et_tf)) * ufl.dx
b_tt = (1/m_r)*ufl.inner(Et_bf, Et_tf) * ufl.dx
b_tz = (1/m_r)*ufl.inner(Et_bf, ufl.grad(Ez_tf)) * ufl.dx
b_zt = ufl.inner(ufl.grad(Ez_bf), Et_tf) * ufl.dx
b_zz = ((1/m_r)*ufl.inner(ufl.grad(Ez_bf), ufl.grad(Ez_tf)) - (k0**2) * e_r * ufl.inner(Ez_bf, Ez_tf)) * ufl.dx

Aform = fem.form(a_tt)
Bform = fem.form(b_tt + b_tz + b_zt + b_zz)

# Add dirichlet boundary conditions
bc_dofs = fem.locate_dofs_topological(combined_space, msh.topology.dim - 1, facet_metals)
PEC_bc = fem.Function(combined_space)
with PEC_bc.vector.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(PEC_bc, bc_dofs)

# Now we can solve the problem with SLEPc. First of all, we need to 
# assemble our and matrices with PETSc in this way:
A = assemble_matrix(Aform, bcs=[bc])
A.assemble()
B = assemble_matrix(Bform, bcs=[bc])
B.assemble()



# Now, we need to create the eigenvalue problem in SLEPc. Our problem is a 
# linear eigenvalue problem, that in SLEPc is solved with the EPS module. 
# We can initialize this solver in the following way:
eps = SLEPc.EPS().create(msh.comm)

# We can pass to EPS our matrices by using the setOperators routine:
eps.setOperators(A, B)

############################
# Regular old shift invert #
############################
tol = 1e-9
eps.setTolerances(tol=tol)

# Set solver type
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

# Get ST context from eps
st = eps.getST()

# Set shift-and-invert transformation
st.setType(SLEPc.ST.Type.SINVERT)

# Target only real values
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setDimensions(nev = numSolutions)

# Target value:
bt2 = k0*k0*er_teflon

# Target the magnitude of the eigenvalues
eps.setTarget(-bt2);
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE);

eps.solve()
eps.view()
eps.errorView()