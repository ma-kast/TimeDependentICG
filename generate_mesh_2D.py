from dolfin import *
#from mshr import *
base_path = "meshes/"

is_fine = False
suffix= ''
if is_fine:
    resolution = 50
    suffix = 'fine'
else:
    resolution = 40


left_p = -2.5
right_p = 2.5
up_p = 2.5
down_p = -1.5
#box = Rectangle(Point(left_p,down_p), Point(right_p, up_p))
if is_fine:
    mesh = RectangleMesh(Point(left_p,down_p), Point(right_p, up_p), resolution, resolution, 'crossed') # generate_mesh(box, resolution)
else:

    mesh = RectangleMesh.create( [Point(left_p,down_p), Point(right_p, up_p) ], [resolution, resolution], CellType.Type.quadrilateral)

class Border(SubDomain):
    def inside(self, x, on_boundary):
        return x[1]>2.25

Border = Border()

# Number of refinements
nor = 0

for i in range(nor):
    edge_markers = MeshFunction("bool", mesh, mesh.topology().dim() - 1)
    Border.mark(edge_markers, True)

    mesh= refine(mesh, edge_markers)



boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] -left_p)  < DOLFIN_EPS


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] -right_p)  < DOLFIN_EPS



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

top = Top()
top.mark(boundaries, 5)
bottom = Bottom()
bottom.mark(boundaries, 6)

if is_fine:
    File(base_path + "2D_mesh" + suffix+".xml") << mesh
    File(base_path + "2D_boundaries" + suffix+".xml") << boundaries
XDMFFile(base_path + "2D_mesh" + suffix+".xdmf").write(mesh)
XDMFFile(base_path + "2D_boundaries" + suffix+".xdmf").write(boundaries)
