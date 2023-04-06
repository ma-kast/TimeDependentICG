from fenics import *
from fenics_adjoint import *


def define_varf_forms_photon(u_funcs, v_funcs, c_n, light_source, dx_loc, ds_loc):

    u, u_, u_out = u_funcs
    v, v_, v_out = v_funcs

    # Physical parameters
    mu_axi = Constant(0.023)
    mu_sxp = Constant(9.84)
    # mu_axf will be replaced by the concentration!
    mu_axf =  c_n
    mu_ami = Constant(0.0289)
    mu_amf = Constant(0.0)
    mu_smp = Constant(9.84)

    g = Constant(2.5156)
    # ICG quantum efficiency
    Gm = Constant(0.016)

    # diffusion coefficient
    dm = 1.0 / (3.0 * (mu_ami + mu_amf + mu_smp))
    # absorption coefficient
    km = mu_ami + mu_amf

    Dx = 1.0 / (3.0 * (mu_axi + mu_sxp))  # Here we made an approximation by dropping the term in mu_axf
    # absorption coefficient
    kx = mu_axi + mu_axf

    lhs_v = dm * dot(grad(v), grad(v_)) * dx_loc + km * v * v_ * dx_loc + 0.5 * g * v * v_ * ds_loc(5)

    rhs_v = Gm * mu_axf * dot(u_out, v_) * dx_loc

    lhs_u = Dx * dot(grad(u), grad(u_)) * dx_loc + kx * u * u_ * dx_loc + 0.5 * g * u * u_ * ds_loc(5)
    rhs_u =  -0.5 * light_source * u_ * ds_loc(5)

    return lhs_u, rhs_u, rhs_v, lhs_v


def define_varf_c(c_funcs, c_n, source_term, reaction_term,  diffusion_coefficient, dt, dx_loc, ds_loc, uses_diffusion = True):

    c, c_, c_out = c_funcs
    F_c = 1. / dt * (c - c_n) * c_ * dx_loc +  source_term * c_ * dx_loc - reaction_term * c * c_ * dx_loc
    if uses_diffusion:
        F_c = F_c + diffusion_coefficient * dot(grad(c), grad(c_)) * dx_loc
    lhs_c, rhs_c = lhs(F_c), rhs(F_c)
    return lhs_c, rhs_c


def get_FEM_functions(V, name= None):
    a_trial = TrialFunction(V)
    a_test = TestFunction(V)
    a = Function(V)
    if name is not None:
        a.rename(name, "")

    return a_trial, a_test, a


def setup_c_problem_terms(deviation):

    # These are user defined terms, we might have to update these based on the physical realities.
    diffusion_coefficient = Constant(1e-3) #Constant(1e-3)
    source_term_base = Expression('(t<20)*(-b*0.01*(20-t))', degree=1, b=1, t=0)
    #source_term_base = Expression('(t<15)*(-b*0.01*(15-t))', degree=1, b=1, t=0)
    reaction_term_base = Constant(-0.006)
    source_term = source_term_base * (Constant(1.0) + 5 * deviation)
    reaction_term = reaction_term_base * (Constant(1.0) - 0.5 * deviation)

    return diffusion_coefficient, source_term, source_term_base, reaction_term


def get_light_source(is_3D, element_order, is_constant_source):
    if is_constant_source:
        if is_3D:

            light_source = Expression(('-10*exp(- (pow(x[0],2) +pow(x[2],2))/10)'), degree=element_order)
        else:
            light_source = Expression(('-10*exp(-pow(x[0],2)/10)'), degree=element_order)

    else:
        if is_3D:
            light_source = Expression(('-10*exp(-pow(x[0]-sin(t)*1.5,2))'), degree=element_order, t=0)
        else:
            light_source = Expression(('-10*exp(-pow(x[0]-sin(t)*1.5,2))'), degree=element_order, t=0)

    return light_source

def forward_run_analytic(tumor, dx, ds,V, num_steps, T, is_3D, is_constant_source, with_env_ICG,  callbacks = None):
    # Setup the problem
    if callbacks is None:
        callbacks = []

    u_funcs = get_FEM_functions(V, "u")
    v_funcs = get_FEM_functions(V, "v")
    c_n = Function(V)
    c_n.rename("c_n", "")
    u_out = u_funcs[2]
    v_out = v_funcs[2]

    light_source = get_light_source(is_3D, 1,is_constant_source)


    # Time-stepping
    dt_val = T / num_steps
    dt = Constant(0)
    dt.assign(dt_val)
    t_cur = Constant(0)
    k= Constant(0.006) * (Constant(1.0) - 0.5 *tumor)
    a= Constant(20)
    if with_env_ICG:
        f = (Constant(1.0) + 5* tumor) *0.001
    else:
        f =  6 * tumor * 0.001
    kt = k * t_cur
    kt_change = k*a
    #c_n = -f* exp(-kt)
    is_before = Expression("t<t_change", degree=1, t= t_cur, t_change=a )
    is_after = Expression("t>=t_change", degree=1, t= t_cur, t_change=a)
    val_trans = - f * exp(-kt_change)*(exp(kt_change) * (- a * k + kt_change - Constant(1)) + a * k + Constant(1)) / (k * k)
    c_n = - f * exp(-kt)*(exp(kt) * (- a * k + kt - Constant(1)) + a * k + Constant(1)) / (k * k)  * is_before +\
          is_after * exp(-k*(t_cur-a)) *val_trans
    rhs_c = c_n * u_funcs[1]* dx(metadata = {"quadrature_degree": 2})
    lhs_c = u_funcs[0]*u_funcs[1]*dx(metadata = {"quadrature_degree": 2})
    c_out = Function(V) #project(c_n, V)



    lhs_u, rhs_u, rhs_v, lhs_v = define_varf_forms_photon(u_funcs, v_funcs, c_out, light_source, dx, ds)

    problem_u = LinearVariationalProblem(lhs_u, rhs_u, u_out, [])
    solver_u = LinearVariationalSolver(problem_u)
    solver_u.parameters["linear_solver"]= "gmres"
    solver_u.parameters["krylov_solver"]["absolute_tolerance"]= 1e-12
    solver_u.parameters["krylov_solver"]["maximum_iterations"] =1000
    solver_u.parameters["krylov_solver"]["relative_tolerance"] = 1e-12

    problem_v = LinearVariationalProblem(lhs_v, rhs_v, v_out, [])
    solver_v = LinearVariationalSolver(problem_v)
    solver_v.parameters["linear_solver"]= "gmres"
    solver_v.parameters["krylov_solver"]["absolute_tolerance"]= 1e-12
    solver_v.parameters["krylov_solver"]["relative_tolerance"] = 1e-12


    for n in range(num_steps):

        t_cur.assign( (n + 1) * dt_val)
        print((n + 1) * dt_val)
        solve(lhs_c == rhs_c, c_out, solver_parameters={"linear_solver": "gmres"})
        solver_u.solve()
        solver_v.solve()

        for callback_tup in callbacks:
            callback, factor = callback_tup
            if (n + 1) % factor == 0:  # Check if we want to do something at this time step for that callback
                callback(c_out, u_out, v_out, n)

def forward_run_c(c_n, dx, ds,V, is_3D, is_constant_source, callbacks = None):
    # Setup the problem
    if callbacks is None:
        callbacks = []

    u_funcs = get_FEM_functions(V, "u")
    v_funcs = get_FEM_functions(V, "v")

    u_out = u_funcs[2]
    v_out = v_funcs[2]

    light_source = get_light_source(is_3D,1,  is_constant_source)

    lhs_u, rhs_u, rhs_v, lhs_v = define_varf_forms_photon(u_funcs, v_funcs, c_n, light_source, dx, ds)

    problem_u = LinearVariationalProblem(lhs_u, rhs_u, u_out, [])
    solver_u = LinearVariationalSolver(problem_u)
    solver_u.parameters["linear_solver"]= "gmres"
    solver_u.parameters["krylov_solver"]["absolute_tolerance"]= 1e-12
    solver_u.parameters["krylov_solver"]["maximum_iterations"] =1000
    solver_u.parameters["krylov_solver"]["relative_tolerance"] = 1e-12

    problem_v = LinearVariationalProblem(lhs_v, rhs_v, v_out, [])
    solver_v = LinearVariationalSolver(problem_v)
    solver_v.parameters["linear_solver"]= "gmres"
    solver_v.parameters["krylov_solver"]["absolute_tolerance"]= 1e-12
    solver_v.parameters["krylov_solver"]["relative_tolerance"] = 1e-12

    for n in range(1):
        solver_u.solve()
        solver_v.solve()

        for callback_tup in callbacks:
            callback, factor = callback_tup
            if (n + 1) % factor == 0:  # Check if we want to do something at this time step for that callback
                callback(c_n, u_out, v_out, n)


def forward_run(tumor, dx, ds,V, num_steps, T, is_3D, is_constant_source, callbacks = None, uses_diffusion = True):
    # Setup the problem
    if callbacks is None:
        callbacks = []

    c_funcs = get_FEM_functions(V, "c")
    u_funcs = get_FEM_functions(V, "u")
    v_funcs = get_FEM_functions(V, "v")
    c_n = Function(V)
    c_n.rename("c_n", "")
    u_out = u_funcs[2]
    v_out = v_funcs[2]
    c_out = c_funcs[2]

    diffusion_coefficient, source_term, source_term_base, reaction_term = setup_c_problem_terms(tumor)
    light_source = get_light_source(is_3D, 1, is_constant_source)

    # Time-stepping
    dt_val = T / num_steps
    dt = Constant(0)
    dt.assign(dt_val)

    lhs_c, rhs_c = define_varf_c(c_funcs, c_n, source_term, reaction_term, diffusion_coefficient, dt, dx, ds, uses_diffusion)
    lhs_u, rhs_u, rhs_v, lhs_v = define_varf_forms_photon(u_funcs, v_funcs, c_n, light_source, dx, ds)

    problem_u = LinearVariationalProblem(lhs_u, rhs_u, u_out, [])
    solver_u = LinearVariationalSolver(problem_u)
    solver_u.parameters["linear_solver"]= "gmres"
    solver_u.parameters["krylov_solver"]["absolute_tolerance"]= 1e-12
    solver_u.parameters["krylov_solver"]["maximum_iterations"] =1000
    solver_u.parameters["krylov_solver"]["relative_tolerance"] = 1e-12

    problem_v = LinearVariationalProblem(lhs_v, rhs_v, v_out, [])
    solver_v = LinearVariationalSolver(problem_v)
    solver_v.parameters["linear_solver"]= "gmres"
    solver_v.parameters["krylov_solver"]["absolute_tolerance"]= 1e-12
    solver_v.parameters["krylov_solver"]["relative_tolerance"] = 1e-12

    for n in range(num_steps):

        t_cur = (n + 1) * dt_val
        source_term_base.t = t_cur
        light_source.t = t_cur

        # Solve variational problem for time step
        solve(lhs_c == rhs_c, c_out, solver_parameters = {"linear_solver": "gmres"})
        # Update previous solution
        c_n.assign(c_out)
        solver_u.solve()
        solver_v.solve()

        for callback_tup in callbacks:
            callback, factor = callback_tup
            if (n + 1) % factor == 0:  # Check if we want to do something at this time step for that callback
                callback(c_out, u_out, v_out, n)