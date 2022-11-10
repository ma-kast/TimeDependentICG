from fenics import *
from fenics_adjoint import *

import numpy as np
import scipy.special


def get_loss_callback(v_file, loss, mesh, boundaries, is_3D, noise_level):

    if is_3D:
        V = FunctionSpace(mesh, 'DG', 0)
    else:
        V = FunctionSpace(mesh, 'CG', 1)
    v_obs = Function(V)
    ds = Measure('ds', subdomain_data=boundaries)

    def callback(c, u, v, n):
        v_file.read_checkpoint(v_obs, "v_noise", n)
        observation_operator = (1/noise_level**2) * (v - v_obs) * (v - v_obs) * ds(5)
        loss[0] = loss[0] + assemble(observation_operator)

    return callback, 1


def compute_reg_constants(lengthscale, var, dim):
    # Computes the different weights for the terms in the bilaplacian.
    nu = 2 - 0.5 * dim  # From the Matern random field definition. We use A^2, so we can solve for nu.

    kappa = np.sqrt(8 * nu) / lengthscale

    # Normalizing constant from spectral density
    s = np.sqrt(var) * np.power(kappa, nu) * np.sqrt(np.power(4. * np.pi, 0.5 * dim)) / scipy.special.gamma(nu)

    scaling = Constant(1. / s)
    mass_term = Constant(np.power(kappa, 2))

    beta_robin = Constant(kappa / np.sqrt(2))  # magical number, designed to reduce boundary artifacts
    return scaling, mass_term, beta_robin


def get_regularization_term_bilaplace(lengthscale, var, dim, V_param, dx, ds):

    scaling, mass_term, beta_robin = compute_reg_constants(lengthscale, var, dim)

    MinvA_mu = Function(V_param)
    a_ = TestFunction(V_param)
    a = TrialFunction(V_param)

    M_form = a*a_ * dx

    def operator(trial_func, test_func):
        rhs_for_reg = scaling * (mass_term * trial_func * test_func * dx + dot(grad(trial_func), grad(test_func))
                                 * dx + beta_robin * trial_func * test_func * ds)
        return rhs_for_reg

    def apply_regularizer(mu, output):
        # Here, we compute the operator and then project back onto the function space
        rhs_for_reg = operator(mu, a_)  # The test function here is important, so we can project in variational form.
        solve(M_form == rhs_for_reg, output)

    def compute_bilaplace_reg(result):

        apply_regularizer(result, MinvA_mu)
        reg_term = MinvA_mu * MinvA_mu * dx
        return assemble(reg_term)

    return compute_bilaplace_reg


def compute_laplace_reg(trial_func, test_func, dx):
    return assemble(1e-1 * dot(grad(trial_func), grad(test_func)) * dx)


def get_optimization_callback(control_var, vtkfile):
    V = control_var.function_space()
    parameter_out = Function(V)
    parameter_out.rename("tumor","")

    optimization_iterations = 0
    def cb(*args, **kwargs):
        nonlocal optimization_iterations
        optimization_iterations += 1
        current_param = control_var.tape_value()
        parameter_out.assign(current_param)
        vtkfile << (parameter_out, optimization_iterations)


    return cb

