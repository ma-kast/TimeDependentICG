from fenics import *
from fenics_adjoint import *

from inverse_problem.mesh_handling import init_meshes_2D_inverse, init_meshes_3D_inverse

from inverse_problem.problem_definition import forward_run

from inverse_problem.loss_optimization import get_loss_callback, get_regularization_term_bilaplace,\
    get_optimization_callback

is_3D = True
suffix = ""
set_log_level(30)

noise_level = 1e-3 # Technically, we need to guess this for the inverse problem
if is_3D:
    suffix = "3D"
    mesh, boundaries = init_meshes_3D_inverse()
    T = 30
    num_steps = 120
    dim = 3
else:

    mesh, boundaries = init_meshes_2D_inverse()
    T = 30
    num_steps = 120
    dim = 2

# Regularization parameters

lengthscale = 1
sigma2 = 1

dx = Measure('dx', domain=mesh)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)


# set data to load
v_file_str = "forward_run" + suffix + "/v_obs.xdmf"
v_file = XDMFFile(v_file_str)

V_param = FunctionSpace(mesh, 'CG', 1)

tumor = Function(V_param)
tumor.rename("tumor", "")

loss = [0]

# Computes the loss during the forward run
loss_callback = get_loss_callback(v_file, loss, mesh, boundaries, is_3D, noise_level)
forward_run(tumor, mesh, boundaries, num_steps, T, is_3D, [loss_callback]) # Has to be run for taping purposes.

reg_operator_bilaplace = get_regularization_term_bilaplace(lengthscale, sigma2, dim, V_param, dx, ds)


total_loss = loss[0] + reg_operator_bilaplace(tumor)

control_var = Control(tumor)

Jhat = ReducedFunctional(total_loss,[control_var])

# create a callback, so we can follow the optimization.
vtkfile = File('inverse_run'+ suffix +'/result.pvd')
opt_callback = get_optimization_callback(control_var, vtkfile)

g_opt = minimize(Jhat, method ="L-BFGS-B", tol = 1e-8,
                       options = {'disp': True, "maxiter":50 }, callback = opt_callback)




