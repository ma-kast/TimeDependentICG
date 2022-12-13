from dolfin import *
base_path = "meshes/"
left_p = -2.5
right_p = 2.5
up_p = 2.5
down_p = -1.5
back_p = -2.5
front_p = 2.5


is_fine = True
suffix = ""

if is_fine:
    suffix= "_fine"

mesh = Mesh()
f = XDMFFile(mesh.mpi_comm(), base_path + "3D_mesh" + suffix +".xdmf")
f.read(mesh)


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] -left_p)  < DOLFIN_EPS


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] -right_p)  < DOLFIN_EPS

class Front(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[2] -front_p)  < DOLFIN_EPS

class Back(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[2] -back_p) < DOLFIN_EPS

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] -up_p)  < DOLFIN_EPS

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] -down_p) < DOLFIN_EPS

boundaries.set_all(0)
left = Left()
left.mark(boundaries ,1)
right = Right()
right.mark(boundaries, 2)

front = Front()
front.mark(boundaries, 3)
back = Back()
back.mark(boundaries, 4)

top = Top()
top.mark(boundaries, 5)
bottom = Bottom()
bottom.mark(boundaries, 6)


#File(base_path + "3D_mesh" + suffix+".xml") << mesh
#File(base_path + "3D_boundaries" + suffix+".xml") << boundaries
XDMFFile(base_path + "3D_mesh" + suffix+".xdmf").write(mesh)
XDMFFile(base_path + "3D_boundaries" + suffix+".xdmf").write(boundaries)
