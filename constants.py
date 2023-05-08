import fenics as fe
import mshr as ms
from dolfin import Point

import utils

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

# Pre-exponential factor
# 1 / (seconds * angstrom**3)
G_0 = 1.0

# Area factor
# In angstroms
A_F = 500.0

# Defect formation energy
# In electron volts
E_A = 5.9

# Bond polarization factor
# In electron angstroms
B = 91.8

# Defect radius
# In angstroms
R = 1.4

# Boltzmann constant
# in electron volts / kelvin
K_B = 8.617 * 1e-5

# The temperature
# In kelvin
T = 300

# Time simulation step
# In seconds
DELTA_T = 0.1

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

mesh_avg_areas = utils.areas_from_mesh(mesh)
mesh_sub_avg_areas = utils.areas_from_mesh(mesh_sub)

