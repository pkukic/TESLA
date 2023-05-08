import fenics as fe
import mshr as ms
from dolfin import Point, plot

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import poisson
import constants


if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh)

    e = poisson.E_from_sigma(sigma, constants.mesh, constants.mesh_sub)

    # c = plot(e, cmap='inferno')
    # plt.gca().set_aspect('equal')
    # plt.colorbar(c, fraction=0.047*1/10)
    # plt.show()

    coords = constants.mesh_sub.coordinates()
    values = e.compute_vertex_values()

    print(coords[np.argmax(values)])

    # c = plot(e, cmap='inferno')
    # plt.gca().set_aspect('equal')
    # plt.colorbar(c, fraction=0.047*1/10)
    # plt.show()