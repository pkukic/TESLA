import fenics as fe
import mshr as ms
from dolfin import Point, plot

import numpy as np
from poisson import E_from_sigma

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

DISCARDED_WIDTH_EACH_SIDE = 100.0

HEIGHT = 50.0
WIDTH = 500.0 + DISCARDED_WIDTH_EACH_SIDE * 2

X_BOTTOM_LEFT, Y_BOTTOM_LEFT = (0.0, 0.0)
X_TOP_RIGHT, Y_TOP_RIGHT = (WIDTH, HEIGHT)

MESH_RESOLUTION = 500
FS_DEGREE = 2

TOP_POTENTIAL = 1.0
BOTTOM_POTENTIAL = 0.0

SIGMA_HRS = 3.0 * 1e6 # S/nm

G_0 = 1e30
# E_a = 

def G(e):
    pass

if __name__ == '__main__':
    domain_vertices = [
        Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_TOP_RIGHT), 
        Point(400.0, Y_TOP_RIGHT), 
        Point(350.0, 25.0),
        Point(300.0, Y_TOP_RIGHT),
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
        Point(400.0, Y_TOP_RIGHT), 
        Point(350.0, 25.0),
        Point(300.0, Y_TOP_RIGHT),
        Point(x_left, Y_TOP_RIGHT),
        Point(x_left, Y_BOTTOM_LEFT)
    ]   

    subdomain = ms.Polygon(submesh_vertices)
    mesh_sub = ms.generate_mesh(subdomain, MESH_RESOLUTION)

    # plot(mesh_sub)
    # plt.show()

    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=SIGMA_HRS, domain=mesh)

    e = E_from_sigma(sigma, mesh, mesh_sub)

    coords = mesh_sub.coordinates()
    values = e.compute_vertex_values()

    print(coords[np.argmax(values)])

    # c = plot(e, cmap='inferno')
    # plt.gca().set_aspect('equal')
    # plt.colorbar(c, fraction=0.047*1/10)
    # plt.show()