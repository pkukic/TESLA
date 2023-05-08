import fenics as fe
import mshr as ms
from dolfin import Point, plot

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

DISCARDED_WIDTH_EACH_SIDE = 10.0

HEIGHT = 5.0
WIDTH = 50.0 + DISCARDED_WIDTH_EACH_SIDE * 2

X_BOTTOM_LEFT, Y_BOTTOM_LEFT = (0.0, 0.0)
X_TOP_RIGHT, Y_TOP_RIGHT = (WIDTH, HEIGHT)

MESH_RESOLUTION = 500
FS_DEGREE = 2

TOP_POTENTIAL = 1.0
BOTTOM_POTENTIAL = 0.0

SIGMA_HRS = 3.0 * 1e6 # S/nm


def E_from_sigma(sigma, mesh, mesh_sub):
    # Function space
    lagrange_function_space_second_order = fe.FunctionSpace(
        mesh,
        'CG',
        FS_DEGREE
    )

    # Boundary conditions
    def left_bottom_right_boundary(x, on_boundary):
        return on_boundary and (
            fe.near(x[0], X_BOTTOM_LEFT) or
            fe.near(x[0], X_TOP_RIGHT) or
            fe.near(x[1], Y_BOTTOM_LEFT)
        )
    
    def top_boundary(x, on_boundary):
        return on_boundary and not left_bottom_right_boundary(x, on_boundary)
    
    homogenous_left_bottom_right_BC = fe.DirichletBC(
        lagrange_function_space_second_order,
        fe.Constant(BOTTOM_POTENTIAL),
        left_bottom_right_boundary
    )

    homogenous_top_BC = fe.DirichletBC(
        lagrange_function_space_second_order,
        fe.Constant(TOP_POTENTIAL),
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
        FS_DEGREE
    )

    u_sub = fe.Function(lagrange_function_sub_space_second_order)
    u_sub.assign(fe.interpolate(u_solution, lagrange_function_sub_space_second_order))
    u_sub.set_allow_extrapolation(True)

    lagrange_vector_sub_space_second_order = fe.VectorFunctionSpace(
        mesh_sub, 
        'CG', 
        FS_DEGREE
    )

    grad_u_sub = fe.project(fe.grad(u_sub), lagrange_vector_sub_space_second_order)
    magnitude_grad_u_sub = fe.project(fe.sqrt(fe.dot(grad_u_sub, grad_u_sub)), lagrange_function_sub_space_second_order)

    return magnitude_grad_u_sub


if __name__ == '__main__':
    domain_vertices = [
        Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_TOP_RIGHT), 
        Point(40.0, Y_TOP_RIGHT), 
        Point(35.0, 2.5),
        Point(30.0, Y_TOP_RIGHT),
        Point(X_BOTTOM_LEFT, Y_TOP_RIGHT),
        Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT)
    ]

    domain = ms.Polygon(domain_vertices)
    mesh = ms.generate_mesh(domain, MESH_RESOLUTION)

    x_left = DISCARDED_WIDTH_EACH_SIDE
    x_right = WIDTH - DISCARDED_WIDTH_EACH_SIDE
    submesh_vertices = [
        Point(x_left, Y_BOTTOM_LEFT),
        Point(x_right, Y_BOTTOM_LEFT),
        Point(x_right, Y_TOP_RIGHT), 
        Point(40.0, Y_TOP_RIGHT), 
        Point(35.0, 2.5),
        Point(30.0, Y_TOP_RIGHT),
        Point(x_left, Y_TOP_RIGHT),
        Point(x_left, Y_BOTTOM_LEFT)
    ]   

    subdomain = ms.Polygon(submesh_vertices)
    mesh_sub = ms.generate_mesh(subdomain, MESH_RESOLUTION)

    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=SIGMA_HRS, domain=mesh)

    e = E_from_sigma(sigma, mesh, mesh_sub)

    c = plot(e, cmap='inferno')
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/10)
    plt.show()



