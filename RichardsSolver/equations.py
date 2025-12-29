import firedrake as fd
from firedrake.petsc import PETSc


def advance_solution(eq, h, h_old, richards_solver):

    mesh = eq.mesh
    x = fd.SpatialCoordinate(mesh)

    relative_conductivity = eq.soil_curves.relative_conductivity
    q = eq.q

    richards_solver.solve()
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

    snes = richards_solver.snes

    return h, q, snes

