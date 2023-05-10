import fenics as fe
from dolfin import plot, parameters

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import poisson
import constants
import functions

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True

if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh())
    top_potential = constants.INITIAL_TOP_POTENTIAL
    time = constants.INITIAL_TIME
    current = 0.0

    while (current < constants.I_CC):
        e_vect = poisson.E_from_sigma(sigma, top_potential)
        e = poisson.magnitude_of_E(e_vect)

        sigma_sub = fe.Function(constants.lagrange_function_sub_space_second_order())
        sigma_sub.assign(fe.interpolate(sigma, constants.lagrange_function_sub_space_second_order()))
        sigma_sub_values = sigma_sub.compute_vertex_values()

        coords = constants.mesh_sub().coordinates()
        e_values = e.compute_vertex_values()
        g_values = functions.G(e_values)
        pc_values = functions.P_c(g_values, constants.mesh_sub_avg_areas())
        new_sigma_values = functions.new_sigma_from_pc(pc_values, coords, sigma_sub_values)
        new_sigma = functions.sigma_f_from_vals(new_sigma_values)

        c = plot(e, cmap='inferno')
        plt.gca().set_aspect('equal')
        plt.colorbar(c, fraction=0.047*1/10)
        plt.show()
        
        c = plot(new_sigma, cmap='inferno')
        plt.gca().set_aspect('equal')
        plt.colorbar(c, fraction=0.047*1/10)
        plt.show()

        current = functions.I(new_sigma, e_vect)
        print(f'Current at iteration: {current}')

        time += constants.DELTA_T
        top_potential += constants.DELTA_T * constants.VR
        sigma = new_sigma

    print(f'Final current: {current}')
    print(f'Final potential: {top_potential}')


