import fenics as fe
import mshr as ms
from dolfin import Point, plot

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import poisson
import constants
import utils

def G(electric_field_values):
    return constants.G_0 * np.exp(-(constants.E_A - constants.B * electric_field_values) / (constants.K_B * constants.T))

def P_c(g_values, avg_areas):
    return 1 - np.exp(- g_values * avg_areas * constants.A_F * constants.DELTA_T)

if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh)

    lagrange_function_sub_space_second_order = fe.FunctionSpace(
        constants.mesh_sub,
        'CG', 
        constants.FS_DEGREE
    )

    sigma_sub = fe.Function(lagrange_function_sub_space_second_order)
    sigma_sub.assign(fe.interpolate(sigma, lagrange_function_sub_space_second_order))
    
    e = poisson.E_from_sigma(sigma, constants.mesh, constants.mesh_sub)

    coords = constants.mesh_sub.coordinates()
    e_values = e.compute_vertex_values()
    g_values = G(e_values)
    pc_values = P_c(g_values, constants.mesh_sub_avg_areas)

    print(e_values)
    print(g_values)
    print(pc_values)

    print(coords[np.argmax(e_values)], np.max(e_values))
    print(coords[np.argmax(g_values)], np.max(g_values))
    print(coords[np.argmax(pc_values)], np.max(pc_values))
    