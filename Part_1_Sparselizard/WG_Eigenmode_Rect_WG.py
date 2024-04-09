from spylizard import *
import numpy as np

import gmsh
import sys

##########
# Inputs #
##########

# Tags for various materials
skin = 1
air = 2
wholedomain = 2

# Frequency to search cutoff frequencies around
f_search = 1e9;

# Number of modes to calculate
numModesToCalc = 5

# Eigenmode to display
monToLoad = 0

mymesh = mesh("rect_WG_2D.msh")

#######################
# Initial definitions #
#######################

# Edge shape functions 'hcurl' for the electric field E.
# Fields x and y are the x and y coordinate fields.
E = field("hcurl") 
x = field("x") 
y = field("y")

# Use interpolation order 2 on the whole domain:
E.setorder(wholedomain, 2)

# Some coefficients
m0 = getmu0()
e0 = getepsilon0()
C = 1/np.sqrt(e0*m0)

k_search = 2*np.pi*f_search/C

# Electric permittivity:
er = parameter()
mr = parameter()

er.setvalue(air, 1)
mr.setvalue(air, 1)


# The waveguide is a perfect conductor. We thus force all
# tangential components of E to 0 on the waveguide skin.
E.setconstraint(skin)

##################################
# Generate weak-form formulation #
##################################

maxwell_curlE = formulation()
maxwell_EE = formulation()

# This is the weak formulation for electromagnetic waves:
maxwell_curlE += integral(wholedomain, (1/mr)*curl(dof(E))*curl(tf(E)))
maxwell_EE += integral(wholedomain, er*dof(E)*tf(E))

maxwell_curlE.generate()
maxwell_EE.generate()


# Get the stiffness and mass matrix:
A = maxwell_curlE.A()
B = maxwell_EE.A()

#############################
# Solve Eigenvalue problem  #
#############################

# Create the object to solve the generalized eigenvalue problem K*x = lambda*M*x :
eig = eigenvalue(A, B)

# Compute the 10 eigenvalues closest to the target magnitude 0.0 (i.e. the 10 first ones):
eig.compute(numModesToCalc, k_search**2.0)

lambda_r = np.array(eig.geteigenvaluerealpart())
lambda_i = np.array(eig.geteigenvalueimaginarypart())

# lam = np.array(lambda_r) + 1j*np.array(lambda_i)
lam = lambda_r
w_eig = np.sqrt(lam)*C
f_eig = w_eig/(2.0*np.pi) 


# Print the eigenfrequencies:
for fIdx in range(len(lam)):
    print("Eigenmode #{} has a cutoff frequency of {} GHz".format(fIdx + 1,f_eig[fIdx]/1e9))


# The eigenvectors are real thus we only need the real part:
Eeig_r = eig.geteigenvectorrealpart() 

fc = f_eig[monToLoad] 
kc = (2*np.pi*fc)/C

E.setdata(wholedomain, Eeig_r[monToLoad])
E.write(wholedomain, "Er.pos", 2)

Erx = compx(E)
Ery = compy(E)

# Initialize gmsh:
gmsh.initialize()

gmsh.open("Er.pos")
gmsh.merge("rect_WG_2D.msh")

# Creates graphical user interface
if 'close' not in sys.argv:
    gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()


