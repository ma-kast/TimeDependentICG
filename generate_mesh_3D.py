from dolfin import *
base_path = "meshes/"

is_fine = True
suffix = ""
if is_fine:
    resolution_sides = 6
    resolution_depth = 6
    suffix= "_fine"
else:
    resolution_sides = 4
    resolution_depth = 4
left_p = -2.5
right_p = 2.5
up_p = 2.5
down_p = -1.5
back_p = -2.5
front_p = 2.5
#box = Rectangle(Point(left_p,down_p), Point(right_p, up_p))


mesh = BoxMesh(Point(left_p,down_p, back_p), Point(right_p, up_p, front_p), resolution_sides, resolution_depth,resolution_sides) # generate_mesh(box, resolution)


class BorderTop(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]>2.25

class BorderMid(SubDomain):
        def inside(self, x, on_boundary):
            return x[1] > 1.25

borderT = BorderTop()
borderM = BorderMid()

# Number of refinements
nor = 2

for i in range(nor):
    edge_markers = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    borderM.mark(edge_markers, True)

    mesh= refine(mesh, edge_markers)

nor=1
for i in range(nor):
    edge_markers = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    borderT.mark(edge_markers, True)

    mesh= refine(mesh, edge_markers)


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
        return on_boundary and abs(x[1] -back_p) < DOLFIN_EPS

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


File(base_path + "3D_mesh" + suffix+".xml") << mesh
File(base_path + "3D_boundaries" + suffix+".xml") << boundaries
XDMFFile(base_path + "3D_mesh" + suffix+".xdmf").write(mesh)
XDMFFile(base_path + "3D_boundaries" + suffix+".xdmf").write(boundaries)
