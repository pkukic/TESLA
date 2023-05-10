import fenics as fe
from dolfin import plot, parameters

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import constants

# import time

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True

def E_from_sigma(sigma, top_potential):
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
        constants.lagrange_function_space(),
        fe.Constant(constants.BOTTOM_POTENTIAL),
        left_bottom_right_boundary
    )

    homogenous_top_BC = fe.DirichletBC(
        constants.lagrange_function_space(),
        fe.Constant(top_potential),
        top_boundary
    )

    boundary_conditions = [homogenous_left_bottom_right_BC, homogenous_top_BC]

    # Trial and test functions
    u_trial = fe.TrialFunction(constants.lagrange_function_space())
    v_test = fe.TestFunction(constants.lagrange_function_space())

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
    u_solution = fe.Function(constants.lagrange_function_space())
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        boundary_conditions,
    )

    u_solution.set_allow_extrapolation(True)

    # Restrict solution to submesh
    u_sub = fe.Function(constants.lagrange_function_sub_space())
    u_sub.assign(fe.interpolate(u_solution, constants.lagrange_function_sub_space()))
    u_sub.set_allow_extrapolation(True)

    grad_u_sub = fe.project(fe.grad(u_sub), constants.lagrange_vector_sub_space())
    return grad_u_sub

def magnitude_of_E(e_vect):
    magnitude_e_vect = fe.project(fe.sqrt(fe.dot(e_vect, e_vect)), constants.lagrange_function_sub_space())
    return magnitude_e_vect

if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh())
    top_potential = 1.0

    e = magnitude_of_E(E_from_sigma(sigma, top_potential))

    c = plot(e, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()


    # forbidden_pairs = [
    #     ('bicgstab', 'icc'),
    #     ('bicgstab', 'ilu'),
    #     ('bicgstab', 'jacobi'),
    #     ('cg', 'amg'),
    #     ('cg', 'default'),
    #     ('gmres', 'hyper_parasails'),
    #     ('gmres', 'icc'),
    #     ('gmres', 'ilu'),
    #     ('gmres', 'jacobi'),
    #     ('gmres', 'none'),
    #     ('gmres', 'petsc_amg'),
    # ]

    # forbidden_units = [
    #     'bicgstab', # done
    #     'cg',
    #     'tmfqr'
    # ]

    # for i, solver in enumerate(['bicgstab', 'cg', 'default', 'gmres', 'minres', 'mumps', 'petsc', 'richardson', 'superlu', 'superlu_dist', 'tmfqr', 'umfpack']):
    #     for preconditioner in ['amg', 'default', 'hypre_amg', 'hypre_euclid', 'hypre_parasails', 'icc', 'ilu', 'jacobi', 'none', 'petsc_amg', 'sor']:
    #         SOLVER_DICT = {'linear_solver': solver, 'preconditioner': preconditioner}
    #         if solver in forbidden_units:
    #             continue
    #         if (solver, preconditioner) in forbidden_pairs:
    #             continue
    #         if i < 10:
    #             continue
    #         time_start = time.time()
    #         print(SOLVER_DICT)
    #         e = magnitude_of_E(E_from_sigma(sigma))
    #         e = magnitude_of_E(E_from_sigma(sigma))
    #         time_end = time.time()
    #         print(SOLVER_DICT, (time_end - time_start) / 2)


