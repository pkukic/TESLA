from dolfin import *
import mshr

import domain

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

# plt.style.use('seaborn') # I personally prefer seaborn for the graph style, but you may choose whichever you want.
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

INITIAL_MESH_RESOLUTION = 20

parameters["reorder_dofs_serial"] = False

if __name__ == '__main__':
    # domain_vertices = [
    #     Point(0.0, 0.0),
    #     Point(50.0, 0.0),
    #     Point(50.0, 10.0), 
    #     Point(30.0, 10.0), 
    #     Point(25.0, 6.0),
    #     Point(20.0, 10.0),
    #     Point(0.0, 10.0),
    #     Point(0.0, 0.0)
    # ]

    # domain = mshr.Polygon(domain_vertices)
    # m = mshr.generate_mesh(domain, INITIAL_MESH_RESOLUTION)

    # # Define the mesh density function
    # def mesh_density(x):
    #     if x[0] >= 10 and x[0] <= 40:
    #         return True
    #     else:
    #         return False

    # # Define the mesh function
    # mf = MeshFunction(value_type="bool", mesh=m, dim=2)
    # mf.set_all(False)
    # for cell in cells(m):
    #     if mesh_density(cell.midpoint()):
    #         mf[cell] = True

    # m = refine(m, mf)

    # def mesh_density(x):
    #     if x[0] >= 20 and x[0] <= 30:
    #         return True
    #     else:
    #         return False

    # mf = MeshFunction(value_type="bool", mesh=m, dim=2)
    # mf.set_all(False)
    # for cell in cells(m):
    #     if mesh_density(cell.midpoint()):
    #         mf[cell] = True

    # m = refine(m, mf)

    height_arr = tuple([2*i for i in range(4)] + [16-2*i for i in range(4, 9)])
    mesh = domain.mesh(height_arr)

    plt.figure().set_figwidth(16)
    plot(mesh)
    ax = plt.gca()
    ax.set_aspect(1.6)
    ax.set_xlim([0, 700])
    ax.set_ylim([0, 50])
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_xticks([0, 100, 150, 200, 250, 300, 400, 450, 500, 550, 600, 700])
    extra_lines = [150, 250, 450, 550]
    for line in extra_lines:
        plt.axvline(x=line, color='black')
    plt.xlabel(r"x [$\AA$]")
    plt.ylabel(r"y [$\AA$]")
    plt.savefig("mesh_full.png",bbox_inches='tight', dpi=1200)
    plt.show()
   


