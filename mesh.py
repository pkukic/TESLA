from dolfin import *
import mshr

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

INITIAL_MESH_RESOLUTION = 60

parameters["reorder_dofs_serial"] = False

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
    m = mshr.generate_mesh(domain, INITIAL_MESH_RESOLUTION)

    # Define the mesh density function
    def mesh_density(x):
        if x[0] >= 10 and x[0] <= 40:
            return True
        else:
            return False

    # Define the mesh function
    mf = MeshFunction(value_type="bool", mesh=m, dim=2)
    mf.set_all(False)
    for cell in cells(m):
        if mesh_density(cell.midpoint()):
            mf[cell] = True

    m = refine(m, mf)

    def mesh_density(x):
        if x[0] >= 20 and x[0] <= 30:
            return True
        else:
            return False

    mf = MeshFunction(value_type="bool", mesh=m, dim=2)
    mf.set_all(False)
    for cell in cells(m):
        if mesh_density(cell.midpoint()):
            mf[cell] = True

    m = refine(m, mf)

    plot(m)
    plt.gca().set_aspect('equal')
    plt.show()


