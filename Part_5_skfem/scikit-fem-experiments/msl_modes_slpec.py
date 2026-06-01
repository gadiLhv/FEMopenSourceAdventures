# --- 7. Solve the System (SLEPc / PETSc Backend) ---
import sys
# Initialize PETSc/SLEPc before importing them
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc

num_modes = 8
kz2_target = (k0**2) * e_eff_target  # You can target the exact shift now!
sigma_target = -kz2_target

def scipy_to_petsc(scipy_csr):
    """Converts a SciPy CSR matrix to a PETSc Mat."""
    petsc_mat = PETSc.Mat().create()
    petsc_mat.setSizes(scipy_csr.shape)
    petsc_mat.setType("aij")
    petsc_mat.setUp()
    
    indptr = scipy_csr.indptr.astype(np.int32)
    indices = scipy_csr.indices.astype(np.int32)
    
    petsc_mat.setValuesCSR(indptr, indices, scipy_csr.data)
    petsc_mat.assemblyBegin()
    petsc_mat.assemblyEnd()
    return petsc_mat

print("Converting matrices to PETSc...")
A_petsc = scipy_to_petsc(A_cropped)
B_petsc = scipy_to_petsc(B_cropped)

# Create the Generalized Eigenvalue Problem Solver (EPS)
eps = SLEPc.EPS().create()
eps.setOperators(A_petsc, B_petsc)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # Generalized Non-Hermitian
eps.setDimensions(nev=num_modes)

# Configure Shift-and-Invert
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(sigma_target)

# Force the underlying linear solver (KSP) to use MUMPS for robust exact shifts
ksp = st.getKSP()
ksp.setType('preonly')
pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps')

print(f"Solving with SLEPc (Shift: {sigma_target:.4f})...")
eps.solve()

# --- Extract Results ---
nconv = eps.getConverged()
print(f"Number of converged eigenpairs: {nconv}")

if nconv == 0:
    print("Solver failed to converge on any modes.")
    sys.exit()

vals = []
vecs_list = []

# Create empty PETSc vectors to hold the eigenvectors
vr, vi = A_petsc.createVecs()

for i in range(min(nconv, num_modes)):
    val = eps.getEigenpair(i, vr, vi)
    vals.append(val)
    # vr.getArray() returns a view, so we copy it to safely store it
    vecs_list.append(vr.getArray().copy())

# Convert back to standard NumPy arrays for the rest of your script
vals = np.array(vals)
vecs = np.column_stack(vecs_list)

kz2 = -np.real(vals)
kc2 = k0**2.0 - kz2
kc = np.sqrt(np.abs(np.real(kc2))) * np.sign(kc2)
fc = kc * 3e8 / (2 * np.pi)

print("\nCalculated Propagation Constants Squared (beta^2):")
for i, val in enumerate(vals):
    print(f"Mode {i+1}: {(fc[i]/1e9):.6f} GHz")