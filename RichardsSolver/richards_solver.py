import firedrake as fd
from RichardsSolver.utilities import interior_penalty_factor
import numpy as np
import ufl

"""
Richards equation solver in mixed form:

    ∂θ/∂t = ∇·(K(h) ∇(h + z))

where:
    h  = pressure head
    S  = effective saturation
    C  = specific moisture capacity
    K  = hydraulic conductivity
    z  = vertical coordinate

The solver constructs the weak residual F(h, v) = 0 for a given test function v.
"""


def richardsSolver(h: fd.Function, h_old: fd.Function, time: fd.Constant, time_step: fd.Constant, eq):

    relative_conductivity = eq.soil_curves.relative_conductivity

    # Choose the time integration method
    match eq.time_integrator:
        case "Picard":
            hDiff = h
        case "SemiImplicit":
            hDiff = h
            K = relative_conductivity(h_old)
        case "ImplicitMidpoint":
            hDiff = (h + h_old)/2
            K = relative_conductivity(hDiff)
        case 'BackwardEuler':
            hDiff = h
            K = relative_conductivity(h)
        case 'CrankNicolson':
            hDiff = (h + h_old)/2
            K = (relative_conductivity(h)+relative_conductivity(h_old))/2
        case _:
            raise ValueError(f"Unknown time integrator '{eq.time_integrator}'")
    K_old = relative_conductivity(h_old)

    # --- Residual ---
    F = mass_term(h, h_old, time_step, eq) \
        + gravity_advection(K, eq) \
        + diffusion_term(hDiff, K, K_old, eq)

    problem = fd.NonlinearVariationalProblem(F, h)

    # Jacobian not needed for SemiImplicit
    solver_parameters = eq.solver_parameters

    solverRichardsNonlinear = fd.NonlinearVariationalSolver(
                                problem,
                                solver_parameters=solver_parameters
                                )

    return solverRichardsNonlinear


def mass_term(h: fd.Function, h_old: fd.Function, time_step: fd.Constant, eq):

    theta_new = eq.soil_curves.moisture_content(h)
    theta_old = eq.soil_curves.moisture_content(h_old)

    if eq.time_integrator == "SemiImplicit":
        C = eq.soil_curves.water_retention(h_old)
        F = fd.inner(C*(h - h_old)/time_step, eq.test_function) * eq.dx
    else:
        F = fd.inner((theta_new - theta_old)/time_step, eq.test_function) * eq.dx

    return F


def diffusion_term(hDiff: fd.Function, K: fd.Function, K_old: fd.Function, eq):

    v = eq.test_function
    grad_v = fd.grad(v)
    bcs = eq.bcs

    # Volume integral
    F = fd.inner(grad_v, K * fd.grad(hDiff)) * eq.dx

    # SIPG
    sigma = interior_penalty_factor(eq, shift=0)
    sigma_int = sigma * fd.avg(fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh))

    jump_v = fd.jump(v, eq.n) 
    jump_h = fd.jump(hDiff, eq.n) 
    avg_K  = fd.avg(K_old)

    F += sigma_int * fd.inner(jump_v, avg_K * jump_h) * eq.dS 
    F -= fd.inner(fd.avg(K_old * grad_v), jump_h) * eq.dS 
    F -= fd.inner(jump_v, fd.avg(K_old * fd.grad(hDiff))) * eq.dS

    # Impose bcs within the weak formulation
    for bc_idx, bc_info in bcs.items():
        boundaryInfo = bc_info
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]
        if boundaryType == 'h':
            sigma_ext = sigma * fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh)
            diff = hDiff - boundaryValue

            F += 2 * sigma_ext * v * K * diff * eq.ds(bc_idx)
            F -= fd.inner(K * grad_v, eq.n) * diff * eq.ds(bc_idx)
            F -= fd.inner(v * eq.n, K * fd.grad(hDiff)) * eq.ds(bc_idx)
        elif boundaryType == 'flux':
            F -= boundaryValue * eq.test_function * eq.ds(bc_idx)
        else:
            raise ValueError("Unknown boundary type, must be 'h' or 'flux'")

    return F


def gravity_advection(K, eq): 

    v = eq.test_function 
    x = fd.SpatialCoordinate(eq.mesh) 

    # Vertical unit vector
    z = x[eq.dim - 1]
    vertical = fd.grad(z)

    q = K * vertical 
    qn = 0.5 * (fd.dot(q, eq.n) + abs(fd.dot(q, eq.n))) 

    F = fd.inner(q, fd.grad(v)) * eq.dx   # Main volume integral
    F -= fd.jump(v) * (qn("+") - qn("-")) * eq.dS  # Upwinding

    return F
