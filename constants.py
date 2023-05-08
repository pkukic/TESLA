import fenics as fe
import mshr as ms
from dolfin import Point

# All lengths are in angstroms
DISCARDED_WIDTH_EACH_SIDE = 100.0

HEIGHT = 50.0
WIDTH = 500.0 + DISCARDED_WIDTH_EACH_SIDE * 2

X_BOTTOM_LEFT, Y_BOTTOM_LEFT = (0.0, 0.0)
X_TOP_RIGHT, Y_TOP_RIGHT = (WIDTH, HEIGHT)

# The potentials are in volts
TOP_POTENTIAL = 1.0
BOTTOM_POTENTIAL = 0.0

# Electrical conductivity of oxide
# In sieverts/angstrom
SIGMA_HRS = 3.0 * 1e7

# Electrical conductivity of defect
# In sieverts/angstrom
SIGMA_LRS = 3.5 * 1e14

# Defect formation energy
# In electron volts
E_A = 5.9

# Bond polarization factor
# In electron angstroms
B = 91.8

# Defect radius
# In angstroms
R = 1.4


# This is some 0-dimensional magic number
MESH_RESOLUTION = 500

# The Lagrangian elements are of degree 2
FS_DEGREE = 2



domain_vertices = [
    Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT),
    Point(X_TOP_RIGHT, Y_BOTTOM_LEFT),
    Point(X_TOP_RIGHT, Y_TOP_RIGHT), 
    Point(400.0, Y_TOP_RIGHT), 
    Point(350.0, 25.0),
    Point(300.0, Y_TOP_RIGHT),
    Point(X_BOTTOM_LEFT, Y_TOP_RIGHT),
    Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT)
]

domain = ms.Polygon(domain_vertices)
mesh = ms.generate_mesh(domain, MESH_RESOLUTION)

x_left = DISCARDED_WIDTH_EACH_SIDE
x_right = WIDTH - DISCARDED_WIDTH_EACH_SIDE
submesh_vertices = [
    Point(x_left, Y_BOTTOM_LEFT),
    Point(x_right, Y_BOTTOM_LEFT),
    Point(x_right, Y_TOP_RIGHT), 
    Point(400.0, Y_TOP_RIGHT), 
    Point(350.0, 25.0),
    Point(300.0, Y_TOP_RIGHT),
    Point(x_left, Y_TOP_RIGHT),
    Point(x_left, Y_BOTTOM_LEFT)
]   

subdomain = ms.Polygon(submesh_vertices)
mesh_sub = ms.generate_mesh(subdomain, MESH_RESOLUTION)

