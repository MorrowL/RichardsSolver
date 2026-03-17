import numpy as np
from RichardsSolver import *


def test_mass_balance_small_domain():

    grid_points = 25
    dt = Constant(100)
    function_space = "DQ"
    time_integrator = "BackwardEuler"
    time_final = 2e05

    for polynomial_degree in range(3):
        compute_mass_balance(grid_points,
                    dt,
                    time_final,
                    polynomial_degree,
                    time_integrator,
                    function_space)


def compute_mass_balance(grid_points: int,
                time_step: Constant = 10.0,
                t_final: float = 1.0e5,
                polynomial_degree: int = 1,
                time_integrator: str = "BackwardEuler",
                function_space: str = 'DG1'):
    
    mesh = UnitSquareMesh(grid_points, grid_points, quadrilateral=True)
    V = FunctionSpace(mesh, function_space, polynomial_degree)
    W = VectorFunctionSpace(mesh, function_space, polynomial_degree)

    # --- Simple soil model ---
    soil_curves = HaverkampCurve(
        theta_r = 0.05,
        theta_s = 0.40,
        Ks      = 1e-5,
        alpha   = 0.5,
        beta    = 1.3,
        A       = 0.01,
        gamma   = 1.5,
        Ss      = 0.0,
    )

    # Boundary conditions: no flux everywhere
    bcs = {
        1: {"flux": 0}, # Left
        2: {"flux": 0}, # Right
        3: {"flux": 0}, # Bottom
        4: {"flux": 1e-06}, # Top
        }

    # Initial condition
    h     = Function(V).assign(-1.0)
    h_old = Function(V).assign(h)
    q     = Function(W)

    moisture_content = soil_curves.moisture_content
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))

    # Richards equation object 
    eq = RichardsSolver(
        V=V,
        W=W,
        mesh=mesh,
        soil_curves=soil_curves,
        bcs=bcs,
        solver_parameters="direct",
        time_integrator=time_integrator,
    )

    time_var = Constant(0.0)

    richards_solver = richardsSolver(h, h_old, time_var, time_step, eq)

    initial_mass = assemble(theta * eq.dx)
    previous_mass = initial_mass
    exterior_flux = 0.0
    mass_error = 0

    # Run a few timesteps
    while float(time_var) < t_final:

        time_var.assign(time_var + float(time_step))
        h_old.assign(h)
        richards_solver.solve()

        theta.interpolate(moisture_content(h))
        exterior_flux = float(time_step)*1e-06
        current_mass = assemble(theta * eq.dx)

        mass_error += abs(abs(current_mass - previous_mass) - abs(exterior_flux))
        previous_mass = current_mass

    # Check mass/loss
    PETSc.Sys.Print("Mass loss/gain", {float(mass_error)})

    assert np.isclose(mass_error, 0.0, atol=1e-6), \
        f"Mass imbalance too large: {mass_error}"
