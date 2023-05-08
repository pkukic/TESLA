import fenics as fe
from dolfin import plot, parameters

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import poisson
import constants

parameters["reorder_dofs_serial"] = False

def G(electric_field_values):
    return constants.G_0 * np.exp(-(constants.E_A - constants.B * electric_field_values) / (constants.K_B * constants.T))


def P_c(g_values, avg_areas):
    return 1 - np.exp(- g_values * avg_areas * constants.A_F * constants.DELTA_T)


def new_sigma_from_pc(pc_values, coords, old_sigma_values):
    new_sigma_values = np.copy(old_sigma_values)
    p = np.random.uniform(0, 1, pc_values.size)

    ind = np.argwhere(p < pc_values)
    c_zero = coords[ind]
    for ci in c_zero:
        dist_vector = np.linalg.norm(coords - ci)
        close_to_ci = np.argwhere(dist_vector < constants.R)
        ind = np.union1d(ind, close_to_ci)

    new_sigma_values[ind] = constants.SIGMA_LRS
    return new_sigma_values


def sigma_f_from_vals(sigma_vals):
    new_sigma_function = fe.Function(constants.lagrange_function_sub_space_second_order())
    new_sigma_function.vector().set_local(sigma_vals)
    new_sigma_function.set_allow_extrapolation(True)

    new_sigma_wider = fe.Function(constants.lagrange_function_space_second_order())
    new_sigma_wider.assign(fe.interpolate(new_sigma_function, constants.lagrange_function_space_second_order()))

    return new_sigma_wider


# boundary_markers = fe.MeshFunction("size_t", constants.mesh_sub(), constants.mesh_sub().topology().dim()-1)
# boundary_markers.set_all(0)
# def bottom(x, on_boundary):
#     return on_boundary and fe.near(x[1], constants.Y_BOTTOM_LEFT)

# def I(sigma, e_vect):
#     sigma_sub = fe.Function(constants.lagrange_function_sub_space_second_order())
#     sigma_sub.assign(fe.interpolate(sigma, constants.lagrange_function_sub_space_second_order()))
#     fe.AutoSubDomain(bottom).mark(boundary_markers, 1)
#     dss = fe.Measure("ds", subdomain_data=boundary_markers)
#     n = fe.FacetNormal(constants.mesh_sub())
#     return fe.dot((sigma_sub * e_vect), n) * dss(1) * fe.Constant(constants.A_F)


if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh())
    e_vect = poisson.E_from_sigma(sigma)
    e = poisson.magnitude_of_E(e_vect)

    sigma_sub = fe.Function(constants.lagrange_function_sub_space_second_order())
    sigma_sub.assign(fe.interpolate(sigma, constants.lagrange_function_sub_space_second_order()))
    sigma_sub_values = sigma_sub.compute_vertex_values()

    coords = constants.mesh_sub().coordinates()
    e_values = e.compute_vertex_values()
    g_values = G(e_values)
    pc_values = P_c(g_values, constants.mesh_sub_avg_areas())
    new_sigma_values = new_sigma_from_pc(pc_values, coords, sigma_sub_values)

    print(e_values)
    print(g_values)
    print(pc_values)
    print(new_sigma_values)

    print(coords[np.argmax(e_values)], np.max(e_values))
    print(coords[np.argmax(g_values)], np.max(g_values))
    print(coords[np.argmax(pc_values)], np.max(pc_values))
    print(coords[np.argmax(new_sigma_values)], np.max(new_sigma_values))

    new_sigma = sigma_f_from_vals(new_sigma_values)
    
    # i = I(new_sigma, e_vect)
    # print(i)

    c = plot(new_sigma, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()
