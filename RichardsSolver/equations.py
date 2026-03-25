import firedrake as fd
from firedrake.petsc import PETSc


def advance_solution(eq, 
                     h : fd.Function, 
                     h_old : fd.Function, 
                     richards_solver,
                     h_star = 0
                     ):

    mesh = eq.mesh
    x = fd.SpatialCoordinate(mesh)

    if eq.time_integrator == "Picard":
        max_it = 30
    else:
        max_it = 1

    nonlinear_iteration = 0
    linear_iterations   = 0

    relative_conductivity = eq.soil_curves.relative_conductivity
    q = eq.q

    richards_solver.solve()

    snes = richards_solver.snes
    nonlinear_iteration += snes.getIterationNumber()
    linear_iterations   += snes.ksp.getIterationNumber()

    match eq.time_integrator:
        case "BackwardEuler":
            K = relative_conductivity(h)
            q.interpolate(-K*fd.grad(h + x[eq.dim-1]))
        case "ImplicitMidpoint":
            K = relative_conductivity((h+h_old)/2)
            q.interpolate(-K*fd.grad((h+h_old)/2 + x[eq.dim-1]))
        case "CrankNicolson":
            K = (relative_conductivity(h) + relative_conductivity(h_old))/2
            q.interpolate(-K*fd.grad((h+h_old)/2 + x[eq.dim-1]))
        case "Picard":
            K = relative_conductivity(h)
            q.interpolate(-K*fd.grad(h + x[eq.dim-1]))
        case "SemiImplicit":
            K = relative_conductivity(h_old)
            q.interpolate(-K*fd.grad(h + x[eq.dim-1]))

    return h, q, nonlinear_iteration, linear_iterations

