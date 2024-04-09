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

# Fundamental frequency for harmonics calculation
f0 = 2.0e9

# Cutoff frequency to search around
fc_search = 1.0e9

# Number of modes to calculate
numModesToCalc = 5

# Eigenmode to display
monToLoad = 0


mymesh = mesh("rect_WG_2D.msh")

# Edge shape functions 'hcurl' for the electric field E.
# Fields x and y are the x and y coordinate fields.
x = field('x')
y = field('y')
et = field("hcurl")
ez = field("h1")

# Use interpolation order 2 on the whole domain:
et.setorder(wholedomain, 2)
ez.setorder(wholedomain, 2)

# The cutoff frequency for a 0.2 m width is freq = 0.75 GHz in theory. 
# With this code and a fine enough mesh you will get the same value.

m0 = getmu0()
e0 = getepsilon0()
C = 1/np.sqrt(e0*m0)  

k0 = 2.0*getpi()*f0/C 
w0 = 2*getpi()*f0

# Electric permittivity:
er = parameter()
mr = parameter()

# Set Dirichlet Et = 0 boundary conditions on metalic shell
er.setvalue(air, 1)
mr.setvalue(air, 1)


# The waveguide is a perfect conductor. We thus force all
# tangential components of E to 0 on the waveguide skin.
et.setconstraint(skin)
ez.setconstraint(skin)


#########################
# Weak form formulation #
#########################


# Operators grad() and curl() in the transverse plane:
dtdtgradez = expression(3,1,[dtdt(dx(dof(ez))), dtdt(dy(dof(ez))), 0])
gradtfez = expression(3,1,[dx(tf(ez)), dy(tf(ez)), 0])

mode = formulation()

mode += integral(wholedomain, curl(dof(et))*curl(tf(et)) - k0*k0*mr*er*(dof(et))*tf(et))
mode += integral(wholedomain, dtdtgradez*tf(et) + dtdt(dof(et))*tf(et))

mode += integral(wholedomain, dtdtgradez*gradtfez + dtdt(dof(et))*gradtfez)
mode += integral(wholedomain, -k0*k0*mr*er*dtdt(dof(ez))*tf(ez))    

mode.generate()

# Get the stiffness matrix K and mass matrix M:
K = mode.K()
M = mode.M()

# Create the object to solve the generalized eigenvalue problem K*x = lambda*M*x :
eig = eigenvalue(K, M)

kc_search = 2.0*np.pi*fc_search/C
kz2_search = k0**2.0 - kc_search**2.0

# Compute the 10 eigenvalues closest to the target magnitude 0.0 (i.e. the 10 first ones):
eig.compute(numModesToCalc, -kz2_search)

kz2_found = -np.array(eig.geteigenvaluerealpart())

kc_found = np.sqrt(k0**2.0 - kz2_found)
fc_found = kc_found*C/(2.0*np.pi)

# Print the eigenfrequencies:
for fIdx in range(len(fc_found)):
    print("Eigenmode #{} has a cutoff frequency of {} GHz".format(fIdx + 1,fc_found[fIdx]/1e9))
 
 
kz = np.sqrt(kz2_found[monToLoad])

# Output mode
Eeig_r = eig.geteigenvectorrealpart() 
et.setdata(wholedomain, Eeig_r[monToLoad])

Et = et/kz
Et.write(wholedomain, "Er.pos", 2)

# Initialize gmsh:
gmsh.initialize()

gmsh.open("Er.pos")
gmsh.merge("rect_WG_2D.msh")

# Creates graphical user interface
if 'close' not in sys.argv:
    gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()
