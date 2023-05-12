from dolfin import parameters, Point

import fenics as fe
import mshr as ms

import pickle
import functools
import os

import utils

parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True

from constants import *


@functools.lru_cache
def domain(height_arr):
    start = [
        Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_BOTTOM_LEFT),
        Point(X_TOP_RIGHT, Y_TOP_RIGHT), 
    ]

    middle = [Point(400 - 10*i, 50 - 2.5*h) for i, h in enumerate(height_arr)]

    end = [
        Point(X_BOTTOM_LEFT, Y_TOP_RIGHT),
        Point(X_BOTTOM_LEFT, Y_BOTTOM_LEFT),
    ]

    domain_vertices = start + middle + end
    return ms.Polygon(domain_vertices)


@functools.lru_cache
def subdomain(height_arr):
    x_left = DISCARDED_WIDTH_EACH_SIDE
    x_right = WIDTH - DISCARDED_WIDTH_EACH_SIDE
    start = [
        Point(x_left, Y_BOTTOM_LEFT),
        Point(x_right, Y_BOTTOM_LEFT),
        Point(x_right, Y_TOP_RIGHT), 
    ]

    middle = [Point(400 - 10*i, 50 - 2.5*h) for i, h in enumerate(height_arr)]

    end = [
        Point(x_left, Y_TOP_RIGHT),
        Point(x_left, Y_BOTTOM_LEFT)
    ]

    subdomain_vertices = start + middle + end
    return ms.Polygon(subdomain_vertices)


@functools.lru_cache
def refine_function(mesh, x_start, x_end):
    # Define the mesh density function
    def mesh_density(x):
        if x[0] >= x_start and x[0] <= x_end:
            return True
        else:
            return False

    # Define the mesh function
    mf = fe.MeshFunction(value_type="bool", mesh=mesh, dim=2)
    mf.set_all(False)
    for cell in fe.cells(mesh):
        if mesh_density(cell.midpoint()):
            mf[cell] = True

    return mf


@functools.lru_cache
def mesh(height_arr):
    m = ms.generate_mesh(domain(height_arr), INITIAL_MESH_RESOLUTION)

    mf_1 = refine_function(m, 10, 40)
    m = fe.refine(m, mf_1)

    mf_2 = refine_function(m, 20, 30)
    m = fe.refine(m, mf_2)

    return m


@functools.lru_cache
def mesh_sub(height_arr):
    m = ms.generate_mesh(subdomain(height_arr), INITIAL_MESH_RESOLUTION)

    mf_1 = refine_function(m, 10, 40)
    m = fe.refine(m, mf_1)

    mf_2 = refine_function(m, 20, 30)
    m = fe.refine(m, mf_2)

    return m   


@functools.lru_cache
def mesh_avg_areas(height_arr):
    return utils.areas_from_mesh(mesh(height_arr))


@functools.lru_cache
def mesh_sub_avg_areas(height_arr):
    return utils.areas_from_mesh(mesh_sub(height_arr))


@functools.lru_cache
def lagrange_function_space(height_arr):
    return fe.FunctionSpace(
            mesh(height_arr),
            'CG',
            FS_DEGREE
    )


@functools.lru_cache
def lagrange_function_sub_space(height_arr):
    return fe.FunctionSpace(
        mesh_sub(height_arr),
        'CG', 
        FS_DEGREE
    )


@functools.lru_cache
def lagrange_vector_sub_space(height_arr):
    return fe.VectorFunctionSpace(
            mesh_sub(height_arr), 
            'CG', 
            FS_DEGREE
    )