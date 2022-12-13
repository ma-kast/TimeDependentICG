from fenics import *
from fenics_adjoint import *

from inverse_problem.mesh_handling import init_meshes_2D_inverse, init_meshes_3D_inverse

from inverse_problem.problem_definition import forward_run_c

from inverse_problem.loss_optimization import get_loss_callback, get_regularization_term_bilaplace,\
    get_optimization_callback

is_3D = True
is_constant_source = True
with_env_ICG = False
simu_type = ""
suffix = ""
set_log_level(30)
element_order = 1

if is_3D:
    suffix = "3D"
    mesh, boundaries = init_meshes_3D_inverse()
    T =20 #1.5/12
    num_steps = 5 # 60 #60
    dim = 3

else:

    mesh, boundaries = init_meshes_2D_inverse()
    T = 30
    num_steps = 120
    dim = 2

# Regularization parameters

lengthscale = 0.03
sigma2 = 1

dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
V= FunctionSpace(mesh, "CG", element_order)

if is_constant_source:
    simu_type = simu_type + "_const"
if with_env_ICG:
    simu_type = simu_type + "_env"


v_file_str = 'forward_run_analytic' + suffix + '/v_obs' + simu_type + '.xdmf'

v_file = XDMFFile(v_file_str)

v_file = XDMFFile(v_file_str)

V_param = FunctionSpace(mesh, 'CG', 1)

ICG = Function(V_param)
ICG.rename("ICG", "")

loss = [0]

# Computes the loss during the forward run
loss_callback = get_loss_callback(v_file, loss, mesh, boundaries, is_3D,  element_order)

n_snapshots_inv = 1
loss_callback = (loss_callback[0], n_snapshots_inv) # only consider 5 snapshots

#forward_run(tumor, dx, ds, V, num_steps, T, is_3D,is_constant_source, [loss_callback], uses_diffusion) # Has to be run for taping purposes.
forward_run_c(ICG, dx, ds,V, is_3D, is_constant_source, [loss_callback])

total_loss = loss[0]

control_var = Control(ICG)

Jhat = ReducedFunctional(total_loss,[control_var])

# create a callback, so we can follow the optimization.
vtkfile = File('inverse_run_analytic'+ suffix +'/result' + simu_type+ '_ICG.pvd')
opt_callback = get_optimization_callback(control_var, vtkfile)

g_opt = minimize(Jhat, method ="L-BFGS-B", tol = 1e-8,
                       options = {'disp': True, "maxiter":50 }, callback = opt_callback)




