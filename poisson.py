import fenics as fe
import mshr as ms
from dolfin import Point, plot

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import constants

def E_from_sigma(sigma, mesh, mesh_sub):
    # Function space
    lagrange_function_space_second_order = fe.FunctionSpace(
        mesh,
        'CG',
        constants.FS_DEGREE
    )

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
        lagrange_function_space_second_order,
        fe.Constant(constants.BOTTOM_POTENTIAL),
        left_bottom_right_boundary
    )

    homogenous_top_BC = fe.DirichletBC(
        lagrange_function_space_second_order,
        fe.Constant(constants.TOP_POTENTIAL),
        top_boundary
    )

    boundary_conditions = [homogenous_left_bottom_right_BC, homogenous_top_BC]

    # Trial and test functions
    u_trial = fe.TrialFunction(lagrange_function_space_second_order)
    v_test = fe.TestFunction(lagrange_function_space_second_order)

    # Weak form
    
    n = fe.FacetNormal(mesh)
    weak_form_lhs = (
        sigma * fe.dot(fe.grad(u_trial) * v_test, n) * fe.ds
        -
        sigma * fe.dot(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
        +
        v_test * fe.dot(fe.grad(u_trial), fe.grad(sigma)) * fe.dx
    )
    weak_form_rhs = fe.Constant(0.0) * v_test * fe.dx

    # Solution
    u_solution = fe.Function(lagrange_function_space_second_order)
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        boundary_conditions
    )

    u_solution.set_allow_extrapolation(True)

    # Restrict solution to submesh
    lagrange_function_sub_space_second_order = fe.FunctionSpace(
        mesh_sub,
        'CG', 
        constants.FS_DEGREE
    )

    u_sub = fe.Function(lagrange_function_sub_space_second_order)
    u_sub.assign(fe.interpolate(u_solution, lagrange_function_sub_space_second_order))
    u_sub.set_allow_extrapolation(True)

    lagrange_vector_sub_space_second_order = fe.VectorFunctionSpace(
        mesh_sub, 
        'CG', 
        constants.FS_DEGREE
    )

    grad_u_sub = fe.project(fe.grad(u_sub), lagrange_vector_sub_space_second_order)
    magnitude_grad_u_sub = fe.project(fe.sqrt(fe.dot(grad_u_sub, grad_u_sub)), lagrange_function_sub_space_second_order)

    return magnitude_grad_u_sub


if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh)

    e = E_from_sigma(sigma, constants.mesh, constants.mesh_sub)

    c = plot(e, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()



