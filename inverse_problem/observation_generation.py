from fenics import *
from fenics_adjoint import *
import numpy as np


def obtain_surface_value_field(u, boundary_mesh, boundary_mesh_coarse, mesh_coarse):

    # Interpolate to mesh on boundary
    V_b = FunctionSpace(boundary_mesh, 'CG',1)
    u_b = interpolate(u, V_b)
    # Compute constant values
    V_b_const = FunctionSpace(boundary_mesh, 'DG', 0)
    u_const_b = project(u_b, V_b_const)
    # Interpolate constant values onto coarse mesh
    V_b_coarse_const = FunctionSpace(boundary_mesh_coarse, 'DG', 0)
    u_const_b_coarse = interpolate(u_const_b, V_b_coarse_const)
    # Extrapolate back to full mesh for compatibility
    V_coarse = FunctionSpace(mesh_coarse, "DG", 0)
    u_const_b_coarse.set_allow_extrapolation(True)
    final_field = interpolate(u_const_b_coarse, V_coarse)
    return final_field


def add_noise_to_field(u, noise_level, is_rel = False):
    noise = Function(u.function_space())
    n_vals = len( noise.vector()[:])
    noise_vals = np.random.normal(0, noise_level, (n_vals,))

    if is_rel:
        noise.vector()[:] = noise_vals * u.vector()[:]
    else:
        noise.vector()[:] = noise_vals
    u.vector().axpy(1, noise.vector())


def create_artificial_observation_callback(h_file,factor_t ,boundary_mesh, boundary_mesh_coarse, mesh_coarse,noise_level, is_rel= True, is_3D= False):

    if not is_3D:
        V_coarse = FunctionSpace(mesh_coarse, "CG", 1)

    def callback(c,u,v,n):

        print("saving solution for n=", n," observation=", int((n + 1) / factor_t - 1))
        if is_3D:
            final_field = obtain_surface_value_field(v, boundary_mesh, boundary_mesh_coarse, mesh_coarse)
        else:
            final_field = interpolate(v, V_coarse)
        add_noise_to_field(final_field, noise_level, is_rel)

        write_to_h_file(h_file, final_field, "v_noise", int((n + 1) / factor_t - 1))

    return callback, factor_t


def write_to_h_file(h_file, field, field_name, index):

    h_file.write_checkpoint(field, field_name, int(index), XDMFFile.Encoding.HDF5, True)


def create_solution_out_callback(save_path_solution_base, mesh, factor_t):
    names = ["c", "u", "v", "rhs_for_v"]
    h_files =  []
    for name in names:
        h_file = XDMFFile(mesh.mpi_comm(), save_path_solution_base+name + ".xdmf")
        h_file.write(mesh)
        h_files.append(h_file)

    def callback(c, u, v, n):
        write_to_h_file(h_files[0], c,"c",(n + 1) / factor_t - 1)
        write_to_h_file(h_files[1], u, "u", (n + 1) / factor_t - 1)
        write_to_h_file(h_files[2], v, "v", (n + 1) / factor_t - 1)

        rhs_for_v = project(u * c, c.function_space())
        write_to_h_file(h_files[3], rhs_for_v, "rhs_for_v", (n + 1) / factor_t - 1)

    return callback, factor_t


def create_extract_points_v_callback(num_steps, points, data_block):

    def callback(c, u, v, n):
        for index, point in enumerate(points):
            val = v(point)
            data_block[n, index] = val

    return callback, 1
