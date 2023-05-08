import numpy as np
from dolfin import cells

import functools

@functools.lru_cache
def triangle_area(a, b, c):
    return abs(0.5 * np.cross(b - a, c - a))

@functools.lru_cache
def areas_from_mesh(mesh):
    vertex_to_cell = {}
    for cell in cells(mesh):
        for vertex in cell.entities(0):
            if vertex not in vertex_to_cell:
                vertex_to_cell[vertex] = []
            vertex_to_cell[vertex].append(cell.index())

    # Compute the average area for each vertex
    avg_areas = np.zeros(mesh.num_vertices())
    for vertex in range(mesh.num_vertices()):
        # Get the cell indices adjacent to this vertex
        cell_indices = np.array(vertex_to_cell[vertex])
        # Get the coordinates of the vertices of the adjacent triangles
        triangle_vertices = mesh.coordinates()[mesh.cells()[cell_indices], :]
        # Compute the area of all adjacent triangles at once using NumPy vector operations
        areas = triangle_area(triangle_vertices[:, 0, :], triangle_vertices[:, 1, :], triangle_vertices[:, 2, :])
        # Compute the average area of all adjacent triangles
        avg_area = np.mean(areas)
        # Store the average area for this vertex
        avg_areas[vertex] = avg_area
    return avg_areas