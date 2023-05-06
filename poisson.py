import fenics as fe
import mshr as ms
from dolfin import Point, plot

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

HEIGHT = 5.0
WIDTH = 50.0

X_BOTTOM_LEFT, Y_BOTTOM_LEFT = (0.0, 0.0)
X_TOP_RIGHT, Y_TOP_RIGHT = (WIDTH, HEIGHT)

MESH_RESOLUTION = 500
FS_DEGREE = 2

TOP_POTENTIAL = 1.0
BOTTOM_POTENTIAL = 0.0

SIGMA_HRS = 3.0 * 1e6 # S/nm

def main():

    # Domain
    domain_vertices = [
        Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_TOP_RIGHT), 
        Point(30.0, Y_TOP_RIGHT), 
        Point(25.0, 2.5),
        Point(20.0, Y_TOP_RIGHT),
        Point(X_BOTTOM_LEFT, Y_TOP_RIGHT),
        Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT)
    ]

    domain = ms.Polygon(domain_vertices)
    mesh = ms.generate_mesh(domain, MESH_RESOLUTION)

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
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=SIGMA_HRS, domain=mesh)
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
    
    # c = plot(u_solution)
    # plt.gca().set_aspect('equal')
    # plt.colorbar(c, fraction=0.047*1/5)
    # plt.show()

    # Create submesh
    # Generate submesh
    x_left = 5.0
    x_right = 45.0
    submesh_vertices = [
        Point(x_left, Y_BOTTOM_LEFT),
        Point(x_right, Y_BOTTOM_LEFT),
        Point(x_right, Y_TOP_RIGHT), 
        Point(30.0, Y_TOP_RIGHT), 
        Point(25.0, 2.5),
        Point(20.0, Y_TOP_RIGHT),
        Point(x_left, Y_TOP_RIGHT),
        Point(x_left, Y_BOTTOM_LEFT)
    ]

    subdomain = ms.Polygon(submesh_vertices)
    mesh_sub = ms.generate_mesh(subdomain, MESH_RESOLUTION)

    # Restrict solution to submesh
    u_sub = fe.Function(lagrange_function_space_second_order)
    u_sub.assign(fe.interpolate(u_solution, fe.FunctionSpace(mesh_sub, 'CG', FS_DEGREE)))


    # TODO: this fails with some compilation error - fenics bug
    # V = u_sub.function_space()
    # mesh = V.mesh()
    # degree = V.ufl_element().degree()
    # W = fe.VectorFunctionSpace(mesh, 'CG', degree)

    # grad_u_sub = fe.project(fe.grad(u_sub), W)

    c = plot(u_sub)
    plt.gca().set_aspect('equal')
    plt.colorbar(c, fraction=0.047*1/5)
    plt.show()

    return

if __name__ == '__main__':
    main()
