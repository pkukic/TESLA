import fenics as fe
import mshr as ms
from dolfin import Point, plot

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

import poisson
import constants

def G(electric_field_values):
    return constants.G_0 * np.exp(-(constants.E_A - constants.B * electric_field_values) / (constants.K_B * constants.T))


if __name__ == '__main__':
    sigma = fe.Expression('SIGMA_HRS', degree=1, SIGMA_HRS=constants.SIGMA_HRS, domain=constants.mesh)

    e = poisson.E_from_sigma(sigma, constants.mesh, constants.mesh_sub)

    coords = constants.mesh_sub.coordinates()
    e_values = e.compute_vertex_values()

    print(coords[np.argmax(e_values)])
    print(G(e_values))

    