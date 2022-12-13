from fenics import *
from fenics_adjoint import *

base_path_mesh = "meshes/"

def get_boundaries(mesh, boundaries_path):
    mvc_boundaries = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
    f = XDMFFile(mesh.mpi_comm(), boundaries_path)
    f.read(mvc_boundaries)
    boundaries = MeshFunction("size_t", mesh, mvc_boundaries)
    return boundaries

def init_meshes_3D_forward():
    dim = 3
    mesh = Mesh(base_path_mesh+ "3D_mesh_fine.xml")
    mesh_coarse = Mesh(base_path_mesh + "3D_mesh.xml")
    boundaries = MeshFunction("size_t", mesh, base_path_mesh+ "3D_boundaries_fine.xml")
    boundary_mesh_f = get_top_mesh(mesh, dim)
    boundary_mesh_c = get_top_mesh(mesh_coarse, dim)
    return mesh, mesh_coarse, boundaries, boundary_mesh_f, boundary_mesh_c

def init_meshes_3D_forward_gmsh():
    dim = 3
    mesh = Mesh()
    with XDMFFile(mesh.mpi_comm(), base_path_mesh + "3D_mesh_fine.xdmf") as file:
        file.read(mesh)

    mesh_coarse = Mesh()
    with XDMFFile(mesh_coarse.mpi_comm(), base_path_mesh + "3D_mesh.xdmf") as file:
        file.read(mesh_coarse)

    boundaries = get_boundaries(mesh, base_path_mesh + "3D_boundaries_fine.xdmf")

    #boundaries = MeshFunction("size_t", mesh, base_path_mesh+ "3D_boundaries_fine.xml")
    boundary_mesh_f = get_top_mesh(mesh, dim)
    boundary_mesh_c = get_top_mesh(mesh_coarse, dim)
    return mesh, mesh_coarse, boundaries, boundary_mesh_f, boundary_mesh_c


def init_meshes_3D_inverse():
    mesh = Mesh()
    with XDMFFile(mesh.mpi_comm(), base_path_mesh + "3D_mesh.xdmf") as file:
        file.read(mesh)
    boundaries = get_boundaries(mesh,base_path_mesh + "3D_boundaries.xdmf" )
    return mesh, boundaries

def init_meshes_2D_forward():
    mesh = Mesh(base_path_mesh +"2D_meshfine.xml")
    mesh_coarse = Mesh()
    f = XDMFFile(mesh_coarse.mpi_comm(), base_path_mesh + "2D_mesh.xdmf")
    f.read(mesh_coarse)
    boundaries = MeshFunction("size_t", mesh, base_path_mesh + "2D_boundariesfine.xml")
    return mesh, mesh_coarse, boundaries


def init_meshes_2D_inverse():
    mesh = Mesh()
    f = XDMFFile(mesh.mpi_comm(),base_path_mesh +  "2D_mesh.xdmf")
    f.read(mesh)
    boundaries = get_boundaries(mesh, base_path_mesh +  "2D_boundaries.xdmf")
    return mesh, boundaries




def get_top_mesh(mesh, dim):
    bmesh = BoundaryMesh(mesh, "exterior")
    mapping = bmesh.entity_map(dim-1)
    part_of_bot = MeshFunction("size_t", bmesh, dim-1)
    for cell in cells(bmesh):
        curr_facet_normal = Facet(mesh, mapping[cell.index()]).normal()
        if near(curr_facet_normal.y(), 1):  # On bot boundary
            part_of_bot[cell] = 1
    bot_boundary = SubMesh(bmesh, part_of_bot, 1)
    return bot_boundary