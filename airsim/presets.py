import numpy

import dirs
from airfoil import Airfoil,AirfoilElement
from mesh import MeshScheme

# The s1223 airfoil shape; good for high downforce and low reynolds #s
s1223 = numpy.loadtxt(open(dirs.res_path("s1223.dat"), "rb"), skiprows=1)
s1223 = numpy.flipud(s1223)
# Cut of trailing edge (causes meshing issues)
s1223 = s1223[2:-2,:]
# If you want to subsample the airfoil
#s1223 = s1223[::5]

# A basic symmetric airfoil
nacam2 = numpy.loadtxt(open(dirs.res_path("nacam2.dat"), "rb"), skiprows=1)
nacam2 = numpy.flipud(nacam2)

# A manually designed sample s1223 wing
_a1_elems = [AirfoilElement([-0.14, 0],     0.076,  -12),
             AirfoilElement([0,     0],     0.457,  8),
             AirfoilElement([0.43,  0.09],  0.191,  30),
             AirfoilElement([0.59,  0.20],  0.114,  52),
             AirfoilElement([0.65,  0.30],  0.076,  60)];
a1 = Airfoil(_a1_elems)
a1.set_profile_coords(s1223)

# A test wing with a single s1223 element
_a2_elems = [AirfoilElement(s1223, [0,0], 1, 45)];
a2 = Airfoil(_a2_elems)
a2.set_profile_coords(s1223)

# The same wing but with nacam2 elements
_a3_elems = [AirfoilElement([-0.14,    0],     0.076, -12),
             AirfoilElement([0,        0],     0.457,  8),
             AirfoilElement([0.43,     0.09],  0.191,  30),
             AirfoilElement([0.59,     0.20],  0.114,  52),
             AirfoilElement([0.65,     0.30],  0.076,  60)];
a3 = Airfoil(_a3_elems)
a3.set_profile_coords(nacam2)

b1 = numpy.array(
        [[-1,   -2],    [10.0, -2],
         [10.0, 5],     [-1,   5]]
    )

b2 = numpy.array(
        [[-5,   -5],    [8.0, -5],
         [8.0, 5],     [-5,   5]]
    )


# Meshes equally finely for all the airfoils and coarsely for surroundings
def basic_mesh_scheme(airfoil):
    return MeshScheme([0.005] * len(airfoil.elems), 0.35, b1)

# Meshes extra fine around the leading element
def fine_lead_mesh_scheme(airfoil):
    return MeshScheme([0.003] + ([0.003] * (len(airfoil.elems) - 1)),
                      0.35, b1)

# Coarse mesh
def coarse_mesh_scheme(airfoil):
    return MeshScheme([0.005] * len(airfoil.elems), 0.5, b2)
