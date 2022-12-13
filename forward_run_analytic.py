from fenics import *
from fenics_adjoint import *
import numpy as np

from inverse_problem.mesh_handling import init_meshes_2D_forward, init_meshes_3D_forward, init_meshes_3D_forward_gmsh

from inverse_problem.problem_definition import forward_run, forward_run_analytic

from inverse_problem.observation_generation import create_artificial_observation_callback, create_solution_out_callback,\
    create_extract_points_v_callback_3D

set_log_level(30)
np.random.seed(42)
is_3D = True
is_constant_source = True
with_env_ICG = False
suffix = ""
simu_type = ""


if is_3D:
    suffix = '3D'
save_path_tumor_loc = 'forward_run_analytic'+suffix + '/parameter_setting.pvd'


if is_constant_source:
    simu_type = simu_type + "_const"
if with_env_ICG:
    simu_type = simu_type + "_env"


save_path_observations = 'forward_run_analytic' + suffix + '/v_obs' + simu_type + 'noise.xdmf'
save_path_solution_base = 'forward_run_analytic' + suffix + "/solution" + simu_type
factor_t = 1
noise_level = 0.03

if is_3D:
    suffix = "3D"
    T = 20 +20
    num_steps = 5 +5
    dim = 3
    mesh, mesh_coarse, boundaries, boundary_mesh_f, boundary_mesh_c = init_meshes_3D_forward_gmsh()

else:
    T = 20
    num_steps = 5
    dim = 2
    mesh, mesh_coarse, boundaries = init_meshes_2D_forward()
    boundary_mesh_f = None
    boundary_mesh_c = None

element_order = 1
V = FunctionSpace(mesh, 'CG', element_order)
dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Set up solution callbacks for recording things
h_file_obs = XDMFFile(mesh_coarse.mpi_comm(), save_path_observations)
h_file_obs.write(mesh_coarse)
obs_callback_tup = create_artificial_observation_callback(h_file_obs, factor_t, ds, boundary_mesh_f, boundary_mesh_c,
                                                          mesh_coarse, element_order, noise_level= noise_level, is_rel=True,
                                                          is_3D=is_3D)
solution_callback = create_solution_out_callback(save_path_solution_base, mesh, factor_t)

callbacks = [solution_callback, obs_callback_tup ]

# Define ground truth
if is_3D:
    tumor_term = Expression('pow(x[0]-0.5 ,2) + pow(x[1] - 2,2) + pow(x[2] ,2) < pow(0.3,2)', degree=2)
else:
    tumor_term = Expression('pow(x[0]-0.3 ,2) + pow(x[1] - 2.0,2) < pow(0.3,2)', degree=2)

tumor = interpolate(tumor_term, V)

# Save ground truth to file
tumor.rename("tumor", "")


points = [[-2.5, 2.5,-2.5],[0.5,2.5, 0], [0.0,2.5, 0],  [2.5, 2.5, 0] ]
n_points = len(points)
data_block = np.zeros((num_steps,n_points))
point_callback = create_extract_points_v_callback_3D(num_steps, points, data_block)
callbacks.append(point_callback)
vtkfile = File(save_path_tumor_loc)
vtkfile << (tumor, 0)


#Perform forward run and save observations
forward_run_analytic(tumor, dx, ds,V, num_steps, T, is_3D, is_constant_source, with_env_ICG, callbacks)

np.save("point_data_no_ICG.npy",data_block)