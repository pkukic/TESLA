import fenics as fe
from dolfin import plot, parameters

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import constants
import domain

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True

class PoissonSolver:

    def __init__(self, height_arr):
        self.height_arr = height_arr
        self.lagrange_function_space = domain.lagrange_function_space(height_arr)
        self.lagrange_function_sub_space = domain.lagrange_function_sub_space(height_arr)
        self.lagrange_vector_sub_space = domain.lagrange_vector_sub_space(height_arr)
        self.mesh = domain.mesh(height_arr)
        self.mesh_sub = domain.mesh_sub(height_arr)


    def E_from_sigma(self, sigma, top_potential):
        # Boundary conditions
        def left_bottom_right_boundary(x, on_boundary):
            return on_boundary and (
                fe.near(x[0], constants.X_BOTTOM_LEFT) or
                fe.near(x[0], constants.X_TOP_RIGHT) or
                fe.near(x[1], constants.Y_BOTTOM_LEFT)
            )
        
        def top_boundary(x, on_boundary):
            return on_boundary and not left_bottom_right_boundary(x, on_boundary)
        
        homogenous_left_bottom_right_BC = fe.DirichletBC(
            self.lagrange_function_space,
            fe.Constant(constants.BOTTOM_POTENTIAL),
            left_bottom_right_boundary
        )

        homogenous_top_BC = fe.DirichletBC(
            self.lagrange_function_space,
            fe.Constant(top_potential),
            top_boundary
        )

        boundary_conditions = [homogenous_left_bottom_right_BC, homogenous_top_BC]

        # Trial and test functions
        u_trial = fe.TrialFunction(self.lagrange_function_space)
        v_test = fe.TestFunction(self.lagrange_function_space)

        # Weak form
        n = fe.FacetNormal(self.mesh)
        weak_form_lhs = (
            sigma * fe.dot(fe.grad(u_trial) * v_test, n) * fe.ds
            -
            sigma * fe.dot(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
            +
            v_test * fe.dot(fe.grad(u_trial), fe.grad(sigma)) * fe.dx
        )
        weak_form_rhs = fe.Constant(0.0) * v_test * fe.dx

        # Solution
        u_solution = fe.Function(self.lagrange_function_space)
        fe.solve(
            weak_form_lhs == weak_form_rhs,
            u_solution,
            boundary_conditions,
        )

        u_solution.set_allow_extrapolation(True)

        # Restrict solution to submesh
        u_sub = fe.Function(self.lagrange_function_sub_space)
        u_sub.assign(fe.interpolate(u_solution, self.lagrange_function_sub_space))
        u_sub.set_allow_extrapolation(True)

        grad_u_sub = fe.project(fe.grad(u_sub), self.lagrange_vector_sub_space)
        return grad_u_sub


    def magnitude_of_E(self, e_vect):
        magnitude_e_vect = fe.project(fe.sqrt(fe.dot(e_vect, e_vect)), self.lagrange_function_sub_space)
        return magnitude_e_vect


if __name__ == '__main__':
    height_arr = tuple([2*i for i in range(6)] + [20-2*i for i in range(6, 11)])
    ps = PoissonSolver(height_arr)

    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=ps.mesh)
    top_potential = 1.0

    e_vect = ps.E_from_sigma(sigma, top_potential)
    e = ps.magnitude_of_E(e_vect)

    c = plot(e, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()


