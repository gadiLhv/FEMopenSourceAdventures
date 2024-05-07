from spylizard import *
import numpy as np

import gmsh
import sys

# Initialize gmsh:
gmsh.initialize()

mymesh = mesh("Coax.msh")

gnd = 1
line = 2
teflon = 3 
air = 4
fill_epsr = 2.5

wholedomain = selectunion([teflon,gnd,line])
metals = selectunion([gnd,line])

# Frequency to calculate
m0 = getmu0()
e0 = getepsilon0()
C = 1/np.sqrt(e0*m0)

# Edge shape functions 'hcurl' for the electric field E.
# Fields x and y are the x and y coordinate fields.
PHI = field("h1"); 
x = field("x"); 
y = field("y");

# Same solved for vacuum
PHI = field("h1")

allCells = selectall()

# Use interpolation order 2 on the whole domain:
PHI.setorder(allCells, 2)

# The cutoff frequency for a 0.2 m width is freq = 0.75 GHz in theory. 
# With this code and a fine enough mesh you will get the same value. 

# Electric permittivity:
eps = parameter()
mu = parameter()

eps.setvalue(air, 1.0*e0)
mu.setvalue(air, 1.0)
eps.setvalue(teflon, fill_epsr*e0)
mu.setvalue(teflon, 1.0)

V = port()
Q = port()

PHI.setport(line, V, Q)

# The waveguide is a perfect conductor. We thus force all
# tangential components of E to 0 on the waveguide skin.
PHI.setconstraint(gnd)
PHI.setconstraint(line)

# We force an electric field in the y direction on region 'left'
# that is 0 on the exterior of 'left' and one sine period inside.
laplace = formulation()

# Set a Qi charge per unit depth on the electrode:
laplace +=  Q - Qi;
laplace += integral(wholedomain, -eps * grad(dof(PHI)) * grad(tf(PHI)));
laplace.solve()

E = -grad(PHI)

E.write(wholedomain, "Er.pos", 2)

Erx = compx(E)
Ery = compy(E)

gmsh.open("Er.pos")
gmsh.merge("Coax.geo")
gmsh.merge("Coax.msh")
# Initialize gmsh:
gmsh.initialize()

# Creates graphical user interface
if 'close' not in sys.argv:
    gmsh.fltk.run()

# It finalize the Gmsh API
gmsh.finalize()







