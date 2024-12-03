from spylizard import *
import numpy as np

import gmsh
import sys

##########
# Inputs #
##########

# Tags for various materials
gnd = 1
line = 2
teflon = 3 
air = 4
fill_epsr = 2.5

# Fundamental frequency for harmonics calculation
f0 = 2.0e9

# Number of modes to calculate
numModesToCalc = 5

# Eigenmode to display
monToLoad = 0

# Choose to display in-phase or quadrature
whichPol = 1

mymesh = mesh("Coax.msh")

# Edge shape functions 'hcurl' for the electric field E.
# Fields x and y are the x and y coordinate fields.
x = field('x')
y = field('y')
et = field("hcurl",[2,3])
ez = field("h1",[2,3])

wholedomain = selectunion([teflon])
metals = selectunion([gnd,line])

# Use interpolation order 2 on the whole domain:
et.setorder(wholedomain, 2)
ez.setorder(wholedomain, 3)

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
er.setvalue(teflon, fill_epsr)
mr.setvalue(teflon, 1.0)

# The waveguide is a perfect conductor. We thus force all
# tangential components of E to 0 on the waveguide skin.
et.setconstraint(metals)
ez.setconstraint(metals)


#########################
# Weak form formulation #
#########################

# Set the fundamental frequency for the multi-harmonic calculation
setfundamentalfrequency(f0)

gradez = expression(3,1,[dx(dof(ez)), dy(dof(ez)), 0])
gradtfez = expression(3,1,[dx(tf(ez)), dy(tf(ez)), 0])

mode_1 = formulation()
mode_2 = formulation()

iC2 = 1/(C*C)

mode_1 += integral(wholedomain, (1/mr)*curl(dof(et))*curl(tf(et)) + iC2*er*dtdt(dof(et))*tf(et))
mode_1 += integral(wholedomain, 1*((1/mr)*gradez*tf(et) + (1/mr)*dof(et)*tf(et) + iC2*er*dtdt(dof(ez))*tf(ez) + (1/mr)*gradez*gradtfez + (1/mr)*dof(et)*gradtfez))
mode_1 += integral(wholedomain, -1*((1/mr)*gradez*tf(et) + (1/mr)*dof(et)*tf(et) + iC2*er*dtdt(dof(ez))*tf(ez) + (1/mr)*gradez*gradtfez + (1/mr)*dof(et)*gradtfez))

mode_2 += integral(wholedomain, 1*((1/mr)*curl(dof(et))*curl(tf(et)) + iC2*er*dtdt(dof(et))*tf(et)))
mode_2 += integral(wholedomain, -1*((1/mr)*curl(dof(et))*curl(tf(et)) + iC2*er*dtdt(dof(et))*tf(et)))
mode_2 += integral(wholedomain, (1/mr)*(gradez*gradtfez + dof(et)*gradtfez))
mode_2 += integral(wholedomain, iC2*er*dtdt(dof(ez))*tf(ez))
mode_2 += integral(wholedomain, (1/mr)*(gradez*tf(et) + dof(et)*tf(et)))

mode_1.generate()
mode_2.generate()

A = mode_1.A()
B = mode_2.A()

# Create the object to solve the generalized eigenvalue problem K*x = lambda*M*x :
eig = eigenvalue(A, B)

kz2_search = (k0**2)*fill_epsr

# Compute the 10 eigenvalues closest to the target magnitude 0.0 (i.e. the 10 first ones):
eig.compute(numModesToCalc, -kz2_search)

kz2_found = -np.array(eig.geteigenvaluerealpart())


kc2_found = k0**2.0 - kz2_found
kc_found_sign = np.sign(kc2_found)
kc_found = np.sqrt(np.abs(kc2_found))*kc_found_sign
fc_found = kc_found*C/(2.0*np.pi)

# Print the eigenfrequencies:
for fIdx in range(len(fc_found)):
    print("Eigenmode #{} has an cutoff frequency of {} GHz".format(fIdx + 1,fc_found[fIdx]/1e9))

 
kz = np.sqrt(kz2_found[monToLoad])

########################
# Calculate eigenmodes #
########################

Evec = eig.geteigenvectorrealpart()
et.setdata(wholedomain,Evec[monToLoad])
ez.setdata(wholedomain,Evec[monToLoad])

print("Drawing mode with fc = {} GHz\n".format(fc_found[monToLoad]/1e9))

pol1 = (whichPol == 0)*2 + (whichPol == 1)*3 
pol2 = (whichPol == 0)*3 + (whichPol == 1)*2

# One way to phrase all three components of the E-field
E_r = array3x1(compx(et.harmonic(pol1))/kz,compy(et.harmonic(pol1))/kz,ez.harmonic(pol2))

# Can also disect it to it's transverse and normal components.
Et_r = array3x1(compx(et.harmonic(pol1))/kz,compy(et.harmonic(pol1))/kz,0)
Ez_r = array3x1(0,0,ez.harmonic(pol2))

# And now calculate the H-field, finally
Ht_r = (1/(w0*m0*mr))*array3x1(
            -(compy(et.harmonic(pol1)) + dy(ez.harmonic(pol1))),
             (compx(et.harmonic(pol1)) + dx(ez.harmonic(pol1))),
              0)
Hz_r = array3x1(0,0,compz(-(1/(w0*m0*mr*kz))*curl(et.harmonic(pol2))))
H_r = array3x1(compx(Ht_r),compy(Ht_r),compz(Hz_r)) 

E_r.write(wholedomain, "Er.pos", 2)
H_r.write(wholedomain, "Hr.pos", 2)


# Initialize gmsh:
gmsh.initialize()

gmsh.open("Er.pos")
gmsh.merge("Coax.msh")

# Creates graphical user interface
if 'close' not in sys.argv:
    gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()
