from dolfin import *
import mshr

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

MESH_RESOLUTION = 500

if __name__ == '__main__':
    domain_vertices = [
        Point(0.0, 0.0),
        Point(50.0, 0.0),
        Point(50.0, 10.0), 
        Point(30.0, 10.0), 
        Point(25.0, 6.0),
        Point(20.0, 10.0),
        Point(0.0, 10.0),
        Point(0.0, 0.0)
    ]

    domain = mshr.Polygon(domain_vertices)
    m = mshr.generate_mesh(domain, MESH_RESOLUTION)

    plot(m)
    plt.gca().set_aspect('equal')
    plt.show()


