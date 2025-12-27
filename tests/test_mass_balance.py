import firedrake as fd
import numpy as np
from RichardsSolver import *


def test_mass_balance_small_domain():

    mesh = fd.UnitSquareMesh(25, 25)
    V = fd.FunctionSpace(mesh, "DG", 2)
    W = fd.VectorFunctionSpace(mesh, "DG", 2)

    # --- Simple soil model ---
    soil = HaverkampCurve(
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
        1: {"flux": 0}, 
        2: {"flux": 0}, 
        3: {"flux": 0}, 
        4: {"flux": 0}
        }

    # Initial condition
    h     = fd.Function(V).assign(-1.0)
    h_old = fd.Function(V).assign(h)
    q     = fd.Function(W)

    moisture = soil.moisture_content
    theta = fd.Function(V).interpolate(moisture(h))

    # Richards equation object 
    eq = RichardsSolver(
        V=V,
        W=W,
        mesh=mesh,
        soil_curves=soil,
        bcs=bcs,
        solver_parameters="direct",
        time_integrator="BackwardEuler",
    )

    dt       = fd.Constant(10.0)
    time_var = fd.Constant(0.0)

    solver = richardsSolver(h, h_old, time_var, dt, eq)

    initial_mass = fd.assemble(theta * eq.dx)
    exterior_flux = 0.0

    # Run a few timesteps
    for step in range(100):
        time_var.assign(step * float(dt))
        h_old.assign(h)

        h, q, snes = advance_solution(eq, h, h_old, solver)

        theta.interpolate(moisture(h))
        exterior_flux += fd.assemble(dt * fd.dot(q, -eq.n) * eq.ds)

    final_mass = fd.assemble(theta * eq.dx)

    # Check mass/loss
    mass_error = final_mass - initial_mass - exterior_flux

    assert np.isclose(mass_error, 0.0, atol=1e-6), \
        f"Mass imbalance too large: {mass_error}"
