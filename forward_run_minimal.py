from fenics import *
from fenics_adjoint import *
import numpy as np

from inverse_problem.mesh_handling import init_meshes_2D_forward, init_meshes_3D_forward, init_meshes_3D_forward_gmsh

from inverse_problem.problem_definition import forward_run, forward_run_analytic

from inverse_problem.observation_generation import create_artificial_observation_callback, create_solution_out_callback

set_log_level(30)
np.random.seed(42)
is_3D = True
is_constant_source = True
uses_diffusion = False
suffix = ""



if is_3D:
    suffix = '3D'
save_path_tumor_loc = 'forward_run'+suffix + '/parameter_setting.pvd'
save_path_solution_base = 'forward_run' + suffix + "/solution_"

if is_constant_source:
    save_path_observations = 'forward_run'+suffix + '/v_obs_const.xdmf'
else:
    save_path_observations = 'forward_run' + suffix + '/v_obs.xdmf'

factor_t = 2
noise_level = 0.1

if is_3D:
    suffix = "3D"
    T = 20
    num_steps = 5*factor_t # 100*factor_t
    dim = 3
    mesh, mesh_coarse, boundaries, boundary_mesh_f, boundary_mesh_c = init_meshes_3D_forward_gmsh()
    noise_level = 0.01
else:
    T = 30
    num_steps = 120*factor_t
    dim = 2
    mesh, mesh_coarse, boundaries = init_meshes_2D_forward()
    boundary_mesh_f = None
    boundary_mesh_c = None

element_order = int(1)
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



# V = FunctionSpace(mesh, 'Lagrange', element_order)

# Define ground truth
if is_3D:
    tumor_term = Expression('pow(x[0]-0.5 ,2) + pow(x[1] - 2,2) + pow(x[2] ,2) < pow(0.3,2)', degree=2)
else:
    tumor_term = Expression('pow(x[0]-0.3 ,2) + pow(x[1] - 2.0,2) < pow(0.3,2)', degree=2)

tumor = interpolate(tumor_term, V)

# Save ground truth to file
tumor.rename("tumor", "")
vtkfile = File(save_path_tumor_loc)
vtkfile << (tumor, 0)


#Perform forward run and save observations
#forward_run(tumor, dx, ds,V, num_steps, T, is_3D,is_constant_source, callbacks, uses_diffusion)
forward_run_analytic(tumor, dx, ds,V, num_steps, T, is_3D, is_constant_source, callbacks)