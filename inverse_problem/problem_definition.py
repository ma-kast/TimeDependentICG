from fenics import *
from fenics_adjoint import *


def define_varf_forms_photon(u_funcs, v_funcs, c_n, light_source, dx_loc, ds_loc):

    u, u_, u_out = u_funcs
    v, v_, v_out = v_funcs

    # Physical parameters
    mu_axi = Constant(0.023)
    mu_sxp = Constant(9.84)
    # mu_axf will be replaced by the concentration!
    mu_axf = c_n
    mu_ami = Constant(0.0289)
    mu_amf = Constant(0)
    mu_smp = Constant(9.84)

    g = Constant(2.5156)
    # ICG quantum efficiency
    Gm = Constant(0.016)

    # diffusion coefficient
    dm = 1 / (3 * (mu_ami + mu_amf + mu_smp))
    # absorption coefficient
    km = mu_ami + mu_amf

    Dx = 1 / (3 * (mu_axi + mu_sxp))  # Here we made an approximation by dropping the term in mu_axf
    # absorption coefficient
    kx = mu_axi + mu_axf

    lhs_v = dm * dot(grad(v), grad(v_)) * dx_loc + km * v * v_ * dx_loc + 0.5 * g * v * v_ * ds_loc(5)

    rhs_v = Gm * mu_axf * dot(u_out, v_) * dx_loc

    lhs_u = Dx * dot(grad(u), grad(u_)) * dx_loc + kx * u * u_ * dx_loc + 0.5 * g * u * u_ * ds_loc(5)
    rhs_u = -0.5 * light_source * u_ * ds_loc(5)

    return lhs_u, rhs_u, rhs_v, lhs_v


def define_varf_c(c_funcs, c_n, source_term, reaction_term,  diffusion_coefficient, dt, dx_loc, ds_loc):

    c, c_, c_out = c_funcs
    F_c = 1. / dt * (c - c_n) * c_ * dx_loc + diffusion_coefficient * dot(grad(c), grad(
        c_)) * dx_loc + source_term * c_ * dx_loc - reaction_term * c * c_ * dx_loc

    lhs_c, rhs_c = lhs(F_c), rhs(F_c)
    return lhs_c, rhs_c


def get_FEM_functions(V, name= None):
    a_trial = TrialFunction(V)
    a_test = TestFunction(V)
    a = Function(V)
    if name is not None:
        a.rename(name, "")

    return a_trial, a_test, a


def setup_c_problem_terms(deviation, is_3D, is_constant_source = False):

    # These are user defined terms, we might have to update these based on the physical realities.
    diffusion_coefficient = Constant(1e-3)
    source_term_base = Expression('(t<20)*(-b*0.01*(20-t))', degree=1, b=1, t=0)
    reaction_term_base = Constant(-0.006)
    source_term = source_term_base * (Constant(1.0) + 5 * deviation)
    reaction_term = reaction_term_base * (Constant(1.0) - 0.5 * deviation)

    if is_constant_source:
        light_source = Expression(('-10'), degree=2)
    else:
        if is_3D:
            light_source = Expression(('-10*exp(-pow(x[0]-sin(t)*1.5,2))'), degree=1, t=0)
        else:
            light_source = Expression(('-10*exp(-pow(x[0]-sin(t)*1.5,2))'), degree=1, t=0)

    return diffusion_coefficient, source_term, source_term_base, reaction_term, light_source


def forward_run(tumor, mesh, boundaries, num_steps, T, is_3D, callbacks = None):
    # Setup the problem
    if callbacks is None:
        callbacks = []
    dx = Measure('dx', domain=mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

    V = FunctionSpace(mesh, 'CG', 1)

    c_funcs = get_FEM_functions(V, "c")
    u_funcs = get_FEM_functions(V, "u")
    v_funcs = get_FEM_functions(V, "v")
    c_n = Function(V)
    c_n.rename("c_n", "")
    u_out = u_funcs[2]
    v_out = v_funcs[2]
    c_out = c_funcs[2]

    diffusion_coefficient, source_term, source_term_base, reaction_term, light_source = setup_c_problem_terms(tumor,
                                                                                                              is_3D)
    # Time-stepping
    dt_val = T / num_steps
    dt = Constant(0)
    dt.assign(dt_val)

    lhs_c, rhs_c = define_varf_c(c_funcs, c_n, source_term, reaction_term, diffusion_coefficient, dt, dx, ds)
    lhs_u, rhs_u, rhs_v, lhs_v = define_varf_forms_photon(u_funcs, v_funcs, c_n, light_source, dx, ds)

    # Define LU solver for V, we should be able to reuse the factor
    A = assemble(lhs_v, PETScMatrix())
    v_solver = LUSolver(A)

    for n in range(num_steps):

        t_cur = (n + 1) * dt_val
        source_term_base.t = t_cur
        light_source.t = t_cur

        # Solve variational problem for time step
        solve(lhs_c == rhs_c, c_out)
        # Update previous solution
        c_n.assign(c_out)
        solve(lhs_u == rhs_u, u_out)
        f_v = assemble(rhs_v)
        v_solver.solve(v_out.vector(), f_v)

        for callback_tup in callbacks:

            callback, factor = callback_tup
            if (n + 1) % factor == 0:  # Check if we want to do something at this time step for that callback
                callback(c_out, u_out, v_out, n)