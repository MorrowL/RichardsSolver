import firedrake as fd
from RichardsSolver.utilities import interior_penalty_factor
import numpy as np
import ufl

"""
This model provides a solver solving solving Richards equation in pressure-head form:
$$(C + S_sS) \frac{\parial h}{\partial t} = \nabla (K \grad(h + z))
where:
- $h$ is the pressure head
- $S_s$ is the specific storage coefficient
- $S(h) = (\theta - \theta_r)/(\theta_s - \theta_r)$ is the effective saturation
- $C(h) = d\theta/dh$ is the specific moisture capacity
- $K(h)$ is the hydraulic conductivity
- $z$ is the vertical coordinate (gravity term)
This equation is expressed in variational (weak) form and expressied in the residual form $0 = F(h, v)$ where $v$ is the test function
INPUTS:
    h : solution at current time step to be solved for
    h_hold : solution at previous time step
    time : curent time
    time_step : size of timestep
    eq : dumb
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
            raise ValueError("Temporal integration method not recognised")
    K_old = relative_conductivity(h_old)

    # Richards equation in variational residual form F = 0
    F = mass_term(h, h_old, time_step, eq) \
        + gravity_advection(K, eq) \
        + diffusion_term(hDiff, K, K_old, eq)

    problem = fd.NonlinearVariationalProblem(F, h)

    # Jacobian not needed for SemiImplicit
    solver_parameters = eq.solver_parameters

    solverRichardsNonlinear = fd.NonlinearVariationalSolver(problem,
                                solver_parameters=solver_parameters)

    return solverRichardsNonlinear


def mass_term(h: fd.Function, h_old: fd.Function, time_step: fd.Constant, eq):

    theta_new = eq.soil_curves.moisture_content(h)
    theta_old = eq.soil_curves.moisture_content(h_old)

    if eq.time_integrator == "SemiImplicit":
        C = eq.soil_curves.water_retention(h_old)
        F = fd.inner(C*(h - h_old)/time_step, eq.v) * eq.dx
    else:
        F = fd.inner((theta_new - theta_old)/time_step, eq.v) * eq.dx

    return F


def diffusion_term(hDiff: fd.Function, K: fd.Function, K_old: fd.Function, eq):

    bcs = eq.bcs

    # Volume integral
    F = fd.inner(fd.grad(eq.v), K * fd.grad(hDiff)) * eq.dx

    # SIPG
    sigma = interior_penalty_factor(eq, shift=0)
    sigma_int = sigma * fd.avg(fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh))

    F += sigma_int*fd.inner(fd.jump(eq.v, eq.n), fd.avg(K_old) * fd.jump(hDiff, eq.n)) * eq.dS
    F -= fd.inner(fd.avg(K_old * fd.grad(eq.v)), fd.jump(hDiff, eq.n)) * eq.dS
    F -= fd.inner(fd.jump(eq.v, eq.n), fd.avg(K_old * fd.grad(hDiff))) * eq.dS

    # Impose bcs within the weak formulation
    for bc_idx, bc_info in bcs.items():
        boundaryInfo = bc_info
        boundaryType = next(iter(boundaryInfo))
        boundaryValue = boundaryInfo[boundaryType]
        if boundaryType == 'h':
            sigma_ext = sigma * fd.FacetArea(eq.mesh) / fd.CellVolume(eq.mesh)
            F += 2 * sigma_ext * eq.v * K * (hDiff - boundaryValue) * eq.ds(bc_idx)
            F -= fd.inner(K * fd.grad(eq.v), eq.n) * (hDiff - boundaryValue) * eq.ds(bc_idx)
            F -= fd.inner(eq.v * eq.n, K * fd.grad(hDiff)) * eq.ds(bc_idx)
        elif boundaryType == 'flux':
            F -= boundaryValue * eq.v * eq.ds(bc_idx)
        else:
            raise ValueError("Unknown boundary type, must be 'h' or 'flux'")

    return F


def gravity_advection(K: fd.Function, eq):

    # Gravity driven volumetric flux

    v = eq.v
    x = fd.SpatialCoordinate(eq.mesh)

    nDown = fd.grad(x[eq.dimen-1])

    q  = K * nDown
    qn = 0.5*(fd.dot(q, eq.n) + abs(fd.dot(q, eq.n)))

    # Main volume integral
    F = fd.inner(q, fd.grad(eq.v))*eq.dx
    
    # Upwinding
    F -= fd.jump(v)*(qn('+') - qn('-'))*eq.dS

    return F
