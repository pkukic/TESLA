import fenics as fe
from dolfin import plot, parameters

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import poisson
import constants
import domain

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True

class SigmaExpr(fe.UserExpression):
    def __init__(self, new_sigma_function, **kwargs):
        super().__init__(**kwargs)
        self.new_sigma_function = new_sigma_function
        
    def eval(self, values, x):
        if x[0] >= constants.DISCARDED_WIDTH_EACH_SIDE and x[0] <= constants.WIDTH - constants.DISCARDED_WIDTH_EACH_SIDE:
            values[0] = abs(self.new_sigma_function(x[0], x[1]))
        else:
            values[0] = constants.SIGMA_HRS
        return

    def value_shape(self):
        return ()


class FunctionSolver:
    def __init__(self, height_arr):
        self.height_arr = height_arr
        
        self.lagrange_function_space = domain.lagrange_function_space(height_arr)
        self.lagrange_function_sub_space = domain.lagrange_function_sub_space(height_arr)
        self.lagrange_vector_sub_space = domain.lagrange_vector_sub_space(height_arr)
        
        self.mesh = domain.mesh(height_arr)
        self.mesh_sub = domain.mesh_sub(height_arr)
        self.mesh_sub_avg_areas = domain.mesh_sub_avg_areas(height_arr)
        self.mesh_sub_coords = self.mesh_sub.coordinates()


    @staticmethod
    def G(electric_field_values):
        electric_field_values = np.array(electric_field_values, np.float128)
        return constants.G_0 * np.exp(-(constants.E_A - constants.B * electric_field_values) / (constants.K_B * constants.T))
    

    def new_sigma_from_pc(self, pc_values, old_sigma_values):
        new_sigma_values = np.copy(old_sigma_values)
        p = np.random.uniform(0, 1, pc_values.size)

        ind = np.argwhere(p < pc_values)
        c_zero = self.mesh_sub_coords[ind]
        for ci in c_zero:
            dist_vector = np.linalg.norm(self.mesh_sub_coords - ci)
            close_to_ci = np.argwhere(dist_vector < constants.R)
            ind = np.union1d(ind, close_to_ci)

        new_sigma_values[ind] = constants.SIGMA_LRS
        return new_sigma_values


    def P_c(self, g_values):
        g_values = np.array(g_values, np.float128)
        avg_areas = np.array(self.mesh_sub_avg_areas, np.float128)
        return 1 - np.exp(- g_values * avg_areas * constants.A_F * constants.DELTA_T)


    def sigma_f_from_vals(self, sigma_vals):
        new_sigma_function = fe.Function(self.lagrange_function_sub_space)
        new_sigma_function.vector().set_local(sigma_vals)
        new_sigma_function.set_allow_extrapolation(True)

        # new_sigma_wider = fe.Function(constants.lagrange_function_space())
        new_sigma_wider = fe.Function(self.lagrange_function_space, mesh=self.mesh)
        new_sigma_wider.assign(fe.project(v=SigmaExpr(new_sigma_function), V=self.lagrange_function_space))
        new_sigma_wider.set_allow_extrapolation(True)

        return new_sigma_wider


    def I(self, sigma, e_vect):
        sigma_sub = fe.Function(self.lagrange_function_sub_space)
        sigma_sub.assign(fe.interpolate(sigma, self.lagrange_function_sub_space))

        boundaries = fe.MeshFunction("size_t", self.mesh_sub, self.mesh_sub.topology().dim()-1)
        default_boundary_marker = 0
        bottom_boundary_marker = 1
        boundaries.set_all(default_boundary_marker)

        def bottom(x, on_boundary):
            return on_boundary and fe.near(x[1], constants.Y_BOTTOM_LEFT)
        
        fe.AutoSubDomain(bottom).mark(boundaries, bottom_boundary_marker)
        ds = fe.Measure("ds", subdomain_data=boundaries, subdomain_id=bottom_boundary_marker, domain=self.mesh_sub)
        n = fe.FacetNormal(self.mesh_sub)

        return constants.A_F * abs(fe.assemble(sigma * fe.dot(e_vect, n) * ds))


if __name__ == '__main__':
    height_arr = tuple([2*i for i in range(6)] + [20-2*i for i in range(6, 11)])
    ps = poisson.PoissonSolver(height_arr)

    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=ps.mesh)
    top_potential = 1.0

    print('Solving Poisson equation...')

    e_vect = ps.E_from_sigma(sigma, top_potential)
    e = ps.magnitude_of_E(e_vect)
    e_values = e.compute_vertex_values()

    print('Poisson equation solved.')

    print('Computing G, P_c and sigma...')

    fs = FunctionSolver(height_arr)

    sigma_sub = fe.Function(fs.lagrange_function_sub_space)
    sigma_sub.assign(fe.interpolate(sigma, fs.lagrange_function_sub_space))
    sigma_sub_values = sigma_sub.compute_vertex_values()

    g_values = FunctionSolver.G(e_values)    
    pc_values = fs.P_c(g_values)
    new_sigma_values = fs.new_sigma_from_pc(pc_values, sigma_sub_values)
    new_sigma = fs.sigma_f_from_vals(new_sigma_values)

    print('Computed.')

    print(e_values)
    print(g_values)
    print(pc_values)
    print(new_sigma_values)

    print(fs.mesh_sub_coords[np.argmax(e_values)], np.max(e_values))
    print(fs.mesh_sub_coords[np.argmax(g_values)], np.max(g_values))
    print(fs.mesh_sub_coords[np.argmax(pc_values)], np.max(pc_values))
    print(fs.mesh_sub_coords[np.argmax(new_sigma_values)], np.max(new_sigma_values))

    print('Computing current...')

    i = fs.I(new_sigma, e_vect)

    print('Computed.')

    print(i)

    c = plot(e, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()
    

    c = plot(new_sigma, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()
