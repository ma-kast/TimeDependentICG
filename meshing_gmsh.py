import pygmsh


is_fine = True
base_path = "meshes/"


suffix = ""

left_p = -2.5
right_p = 2.5
up_p = 2.5
down_p = -1.5
back_p = -2.5
front_p = 2.5

dx= right_p - left_p
dy = up_p - down_p
dz= front_p - back_p


fine_factor =0.8
scale_factor = 0.8
if is_fine:
    alpha = fine_factor
    suffix= "_fine"
else:
    alpha = 1


with pygmsh.occ.Geometry() as geom:


    def callback_func_p2(dim, tag, x,y,z, lc= None):


        if y > 2.3:
            lengthscale =  0.1
        elif y>1.5:
            lengthscale =  0.2
        else:
            lengthscale =  0.8

        return alpha *lengthscale

    def callback_func_coarse(dim, tag, x,y,z, lc= None):


        if y > 2.2:
            lengthscale =  0.075
        elif y>2:
            lengthscale =  0.2
        elif y> 1.5:
            lengthscale =  0.3
        elif y>0:
            lengthscale =  0.5
        else:
            lengthscale =  0.8

        return alpha *lengthscale*scale_factor

    geom.characteristic_length_min = 0.01
    geom.characteristic_length_max = 0.8
    domain = geom.add_box([left_p, down_p, back_p], [dx,dy,dz])
    print(domain)
    geom.set_mesh_size_callback(
      callback_func_coarse
    )
    mesh = geom.generate_mesh()

print(mesh.points)

print(mesh.cells[2])

mesh.write(base_path + "3D_mesh" + suffix+".xdmf", )


import meshio

for cell in mesh.cells:

    if cell.type == "tetra":
        tetra_cells = cell.data

pruned_mesh = meshio.Mesh(points=mesh.points, cells={"tetra": tetra_cells})

meshio.write(base_path + "3D_mesh" + suffix+".xdmf", pruned_mesh)