import fenics as fe
import mshr as ms
from dolfin import Point, parameters

import utils

import pickle
import functools
import os

parameters["reorder_dofs_serial"] = False

# All lengths are in angstroms
DISCARDED_WIDTH_EACH_SIDE = 100.0

HEIGHT = 50.0
WIDTH = 500.0 + DISCARDED_WIDTH_EACH_SIDE * 2

X_BOTTOM_LEFT, Y_BOTTOM_LEFT = (0.0, 0.0)
X_TOP_RIGHT, Y_TOP_RIGHT = (WIDTH, HEIGHT)

# The potentials are in volts
INITIAL_TOP_POTENTIAL = 0.0
BOTTOM_POTENTIAL = 0.0

# Electrical conductivity of oxide
# In sieverts/angstrom
SIGMA_HRS = 3.0 * 1e-13

# Electrical conductivity of defect
# In sieverts/angstrom
SIGMA_LRS = 3.5 * 1e-6

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
T = 300.0

# Time simulation step
# In seconds
DELTA_T = 0.0001

# Initial time
# In seconds
INITIAL_TIME = 0.0

# Compliance current
# In amperes
I_CC = 1 * 1e-9

# Voltage ramp-up rate
# In volts
VR = 1.0

# This is some 0-dimensional magic number
INITIAL_MESH_RESOLUTION = 300

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

@functools.lru_cache
def mesh():
    m = ms.generate_mesh(domain, INITIAL_MESH_RESOLUTION)

    # Define the mesh density function
    def mesh_density(x):
        if x[0] >= 10 and x[0] <= 40:
            return True
        else:
            return False

    # Define the mesh function
    mf = fe.MeshFunction(value_type="bool", mesh=m, dim=2)
    mf.set_all(False)
    for cell in fe.cells(m):
        if mesh_density(cell.midpoint()):
            mf[cell] = True

    m = fe.refine(m, mf)

    def mesh_density(x):
        if x[0] >= 20 and x[0] <= 30:
            return True
        else:
            return False

    mf = fe.MeshFunction(value_type="bool", mesh=m, dim=2)
    mf.set_all(False)
    for cell in fe.cells(m):
        if mesh_density(cell.midpoint()):
            mf[cell] = True

    m = fe.refine(m, mf)

    return m

@functools.lru_cache
def mesh_sub():
    m = ms.generate_mesh(subdomain, INITIAL_MESH_RESOLUTION)

    # Define the mesh density function
    def mesh_density(x):
        if x[0] >= 10 and x[0] <= 40:
            return True
        else:
            return False

    # Define the mesh function
    mf = fe.MeshFunction(value_type="bool", mesh=m, dim=2)
    mf.set_all(False)
    for cell in fe.cells(m):
        if mesh_density(cell.midpoint()):
            mf[cell] = True

    m = fe.refine(m, mf)

    def mesh_density(x):
        if x[0] >= 20 and x[0] <= 30:
            return True
        else:
            return False

    mf = fe.MeshFunction(value_type="bool", mesh=m, dim=2)
    mf.set_all(False)
    for cell in fe.cells(m):
        if mesh_density(cell.midpoint()):
            mf[cell] = True

    m = fe.refine(m, mf)

    return m   

@functools.lru_cache
def mesh_avg_areas():
    if os.path.exists('mesh_avg_areas.pickle'):
        with open('mesh_avg_areas.pickle', 'rb') as f:
            mesh_avg_areas = pickle.load(f)
        return mesh_avg_areas
    else:
        mesh_avg_areas = utils.areas_from_mesh(mesh())
        with open('mesh_avg_areas.pickle', 'wb+') as f:
            pickle.dump(mesh_avg_areas, f, pickle.HIGHEST_PROTOCOL)
        return mesh_avg_areas

@functools.lru_cache
def mesh_sub_avg_areas():
    if os.path.exists('mesh_sub_avg_areas.pickle'):
        with open('mesh_sub_avg_areas.pickle', 'rb') as f:
            mesh_sub_avg_areas = pickle.load(f)
        return mesh_sub_avg_areas
    else:
        mesh_sub_avg_areas = utils.areas_from_mesh(mesh_sub())
        with open('mesh_sub_avg_areas.pickle', 'wb+') as f:
            pickle.dump(mesh_sub_avg_areas, f, pickle.HIGHEST_PROTOCOL)
        return mesh_sub_avg_areas

@functools.lru_cache
def lagrange_function_space_second_order():
    return fe.FunctionSpace(
            mesh(),
            'CG',
            FS_DEGREE
    )

@functools.lru_cache
def lagrange_function_sub_space_second_order():
    return fe.FunctionSpace(
        mesh_sub(),
        'CG', 
        FS_DEGREE
    )

@functools.lru_cache
def lagrange_vector_sub_space_second_order():
    return fe.VectorFunctionSpace(
            mesh_sub(), 
            'CG', 
            FS_DEGREE
    )