import fenics as fe
from dolfin import plot, parameters

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import poisson
import constants

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True

def G(electric_field_values):
    electric_field_values = np.array(electric_field_values, np.float128)
    return constants.G_0 * np.exp(-(constants.E_A - constants.B * electric_field_values) / (constants.K_B * constants.T))


def P_c(g_values, avg_areas):
    g_values = np.array(g_values, np.float128)
    avg_areas = np.array(avg_areas, np.float128)
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
    new_sigma_function = fe.Function(constants.lagrange_function_sub_space())
    new_sigma_function.vector().set_local(sigma_vals)
    new_sigma_function.set_allow_extrapolation(True)

    # print('Plotting new_sigma')
    # c = plot(new_sigma_function, cmap='inferno')
    # plt.gca().set_aspect('equal')
    # plt.colorbar(c, fraction=0.047*1/10)
    # plt.show()

    # f1 = fe.Expression("abs(x[0] - midline)", midline=constants.WIDTH/2, degree=1)

    # f = fe.project(
    #     f1,
    #     constants.lagrange_function_space()
    # )W
    # condition = fe.lt(f, fe.Constant(constants.WIDTH/2))

    # new_sigma_wider = fe.Function(constants.lagrange_function_space())
    # new_sigma_wider = fe.project(new_sigma_function, constants.lagrange_function_space())
    # new_sigma_wider = fe.conditional(
    #     condition, 
    #     new_sigma_function, 
    #     fe.Constant(constants.SIGMA_HRS)
    # )
    
    class SigmaExpr(fe.UserExpression):
        def __init__(self, *args, **kwargs):
            super().__init__(**kwargs)
            
        def eval(self, values, x):
            if x[0] >= constants.DISCARDED_WIDTH_EACH_SIDE and x[0] <= constants.WIDTH - constants.DISCARDED_WIDTH_EACH_SIDE:
                values[0] = abs(new_sigma_function(x[0], x[1]))
            else:
                values[0] = constants.SIGMA_HRS
            return

        def value_shape(self):
            return ()

    # new_sigma_wider = fe.Function(constants.lagrange_function_space())
    new_sigma_wider = fe.Function(constants.lagrange_function_space(), mesh=constants.mesh())
    new_sigma_wider.assign(fe.project(v=SigmaExpr(), V=constants.lagrange_function_space()))
    new_sigma_wider.set_allow_extrapolation(True)
    # print(new_sigma_wider(50, 10))
    # print(new_sigma_wider(350, 25))

    # print('Plotting new_sigma_wider')
    # c = plot(new_sigma_wider, cmap='inferno')
    # plt.gca().set_aspect('equal')
    # plt.colorbar(c, fraction=0.047*1/10)
    # plt.show()

    return new_sigma_wider


def I(sigma, e_vect):
    sigma_sub = fe.Function(constants.lagrange_function_sub_space())
    sigma_sub.assign(fe.interpolate(sigma, constants.lagrange_function_sub_space()))

    boundaries = fe.MeshFunction("size_t", constants.mesh_sub(), constants.mesh_sub().topology().dim()-1)
    default_boundary_marker = 0
    bottom_boundary_marker = 1

    boundaries.set_all(default_boundary_marker)

    def bottom(x, on_boundary):
        return on_boundary and fe.near(x[1], constants.Y_BOTTOM_LEFT)
    

    fe.AutoSubDomain(bottom).mark(boundaries, bottom_boundary_marker)
    ds = fe.Measure("ds", subdomain_data=boundaries, subdomain_id=bottom_boundary_marker, domain=constants.mesh_sub())

    n = fe.FacetNormal(constants.mesh_sub())

    return constants.A_F * abs(fe.assemble(sigma * fe.dot(e_vect, n) * ds))


if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh())
    top_potential = 1.0

    print('Solving Poisson equation...')

    e_vect = poisson.E_from_sigma(sigma, top_potential)
    e = poisson.magnitude_of_E(e_vect)

    print('Poisson equation solved.')

    print('Computing G, P_c and sigma...')

    sigma_sub = fe.Function(constants.lagrange_function_sub_space())
    sigma_sub.assign(fe.interpolate(sigma, constants.lagrange_function_sub_space()))
    sigma_sub_values = sigma_sub.compute_vertex_values()

    coords = constants.mesh_sub().coordinates()
    e_values = e.compute_vertex_values()
    g_values = G(e_values)
    pc_values = P_c(g_values, constants.mesh_sub_avg_areas())
    new_sigma_values = new_sigma_from_pc(pc_values, coords, sigma_sub_values)

    print('Computed.')

    print(e_values)
    print(g_values)
    print(pc_values)
    print(new_sigma_values)

    print(coords[np.argmax(e_values)], np.max(e_values))
    print(coords[np.argmax(g_values)], np.max(g_values))
    print(coords[np.argmax(pc_values)], np.max(pc_values))
    print(coords[np.argmax(new_sigma_values)], np.max(new_sigma_values))

    print('Computing new sigma...')

    new_sigma = sigma_f_from_vals(new_sigma_values)

    print('Computed.')

    i = I(new_sigma, e_vect)
    print(i)


    c = plot(e, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()
    

    c = plot(new_sigma, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()
