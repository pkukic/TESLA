import fenics as fe
from dolfin import parameters
import numpy as np

import poisson
import constants
import functions

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True

def single_sim(height_arr):
    ps = poisson.PoissonSolver(height_arr)
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=ps.mesh)

    time = constants.INITIAL_TIME
    top_potential = constants.INITIAL_TOP_POTENTIAL
    current = constants.INITAL_CURRENT

    while (current < constants.I_CC):
        ps = poisson.PoissonSolver(height_arr)
        e_vect = ps.E_from_sigma(sigma, top_potential)
        e = ps.magnitude_of_E(e_vect)
        e_values = e.compute_vertex_values()

        fs = functions.FunctionSolver(height_arr)

        sigma_sub = fe.Function(fs.lagrange_function_sub_space)
        sigma_sub.assign(fe.interpolate(sigma, fs.lagrange_function_sub_space))
        sigma_sub_values = sigma_sub.compute_vertex_values()

        g_values = functions.FunctionSolver.G(e_values)    
        pc_values = fs.P_c(g_values)
        new_sigma_values = fs.new_sigma_from_pc(pc_values, sigma_sub_values)
        new_sigma = fs.sigma_f_from_vals(new_sigma_values)

        current = fs.I(new_sigma, e_vect)
        # print(f'Current at iteration: {current}')

        time += constants.DELTA_T
        top_potential += constants.DELTA_T * constants.VR
        sigma = new_sigma

    return top_potential


def error_from_n_sims(n, height_arr):
    potentials = []
    for i in range(n):
        print(f"Iteration {i}")
        p = single_sim(height_arr)
        potentials.append(p)
    parr = np.array(potentials)
    return np.var(parr) / np.mean(parr)


if __name__ == '__main__':
    height_arr = tuple([2*i for i in range(6)] + [20-2*i for i in range(6, 11)])
    print(height_arr)
    N = 20
    err = error_from_n_sims(N, height_arr)
    print(err)