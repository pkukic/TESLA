import fenics as fe
import mshr as ms
from dolfin import Point, plot, parameters

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import constants

parameters["reorder_dofs_serial"] = False

def E_from_sigma(sigma):
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
        constants.lagrange_function_space_second_order(),
        fe.Constant(constants.BOTTOM_POTENTIAL),
        left_bottom_right_boundary
    )

    homogenous_top_BC = fe.DirichletBC(
        constants.lagrange_function_space_second_order(),
        fe.Constant(constants.TOP_POTENTIAL),
        top_boundary
    )

    boundary_conditions = [homogenous_left_bottom_right_BC, homogenous_top_BC]

    # Trial and test functions
    u_trial = fe.TrialFunction(constants.lagrange_function_space_second_order())
    v_test = fe.TestFunction(constants.lagrange_function_space_second_order())

    # Weak form
    
    n = fe.FacetNormal(constants.mesh())
    weak_form_lhs = (
        sigma * fe.dot(fe.grad(u_trial) * v_test, n) * fe.ds
        -
        sigma * fe.dot(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
        +
        v_test * fe.dot(fe.grad(u_trial), fe.grad(sigma)) * fe.dx
    )
    weak_form_rhs = fe.Constant(0.0) * v_test * fe.dx

    # Solution
    u_solution = fe.Function(constants.lagrange_function_space_second_order())
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        boundary_conditions
    )

    u_solution.set_allow_extrapolation(True)

    # Restrict solution to submesh
    u_sub = fe.Function(constants.lagrange_function_sub_space_second_order())
    u_sub.assign(fe.interpolate(u_solution, constants.lagrange_function_sub_space_second_order()))
    u_sub.set_allow_extrapolation(True)

    grad_u_sub = fe.project(fe.grad(u_sub), constants.lagrange_vector_sub_space_second_order())
    return grad_u_sub

def magnitude_of_E(e_vect):
    magnitude_e_vect = fe.project(fe.sqrt(fe.dot(e_vect, e_vect)), constants.lagrange_function_sub_space_second_order())
    return magnitude_e_vect

if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh())

    e = magnitude_of_E(E_from_sigma(sigma))

    c = plot(e, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()



