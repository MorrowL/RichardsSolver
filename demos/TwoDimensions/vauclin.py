from RichardsSolver import *
from ufl import tanh, SpatialCoordinate
from firedrake import Function, FunctionSpace, VectorFunctionSpace, Constant, RectangleMesh, assemble, dot
from firedrake.petsc import PETSc

"""
Recharge of a two-dimensional water table
=========================================
Here we reproduce the test case presented in:
    Vauclin, Khanju, and Vachaud, Water Resources Research, 1979
    Experimental and numerical study of a transient, two-dimensional unsaturated-saturated water table recharge problem
    https://doi.org/10.1029/WR015i005p01089
The simulation is performed in a domain of 3 x 2 metres, and the initial
condition is chosen such that the region z <= 0.65 m is fully saturated ($\theta =
\theta_s$), $h(t=0) = z - 0.65$ m. For the boundary conditions, the bottom and left boundary are no flux ($q cdot n = 0$), the right boundary fixed the height of the water table ($h = z - 0.65$ m). For the top boundary, water is injected at a rate of 14.8 cm/hour  in the region where x <= 0.5 m and 0 otherwise. The simulation is concluded after 8 hours
"""


def setup_mesh_and_spaces():

    """Defines the computational mesh and function spaces."""

    nodes_x, nodes_z, L_x, L_z = 91, 61, 3.0, 2.0
    mesh = RectangleMesh(nodes_x, nodes_z, L_x, L_z, name="mesh", quadrilateral=True)

    poly_degree = 2
    V = FunctionSpace(mesh, "DQ", poly_degree)   # Function space for pressure head
    W = VectorFunctionSpace(mesh, "DQ", poly_degree)  # Function space for volumetric flux (doesn't influence solution)
    PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

    return mesh, V, W


def define_time_parameters():

    """Sets simulation time and time-stepping constants."""

    t_final_hours = 8.0
    t_final = t_final_hours * 60 * 60  # 8 hours in seconds
    dt = Constant(200)  # s

    time_integrator = "BackwardEuler"

    return t_final, dt, time_integrator


def define_soil_curves():

    # Specify the hydrological parameters
    soil_curves = HaverkampCurve(
        theta_r=0.00,   # Residual water content [-]
        theta_s=0.37,   # Saturated water content [-]
        Ks=9.722e-05,   # Saturated hydraulic conductivity [m/s]
        alpha=0.44,     # Fitting parameter [m]
        beta=1.2924,    # Fitting parameter [-]
        A=0.0104,       # Fitting parameter [m]
        gamma=1.5722,   # Fitting parameter [-]
        Ss=0e-00,       # Specific storage coefficient [1/m]
    )

    return soil_curves


def setup_boundary_conditions(mesh, time_var):

    """Defines the boundary conditions dictionary."""

    x = SpatialCoordinate(mesh)
    X, Z = x[0], x[1]

    recharge_rate = Constant(4.11e-05) # m/s 

    # Define the recharge region (0 <= x <= 0.5 m) using tanh smoothing
    left_edge = 0.5 * (1 + tanh(10 * (X + 0.50))) 
    right_edge = 0.5 * (1 + tanh(10 * (X - 0.50))) 
    recharge_region_indicator = left_edge - right_edge

    top_flux = tanh(0.000125 * time_var) * recharge_rate * recharge_region_indicator

    richards_bcs = {
        1: {'flux': 0.0},       # Left boundary
        2: {'h': 0.65 - Z},     # Right boundary: fixed water table at z = 0.65 m
        3: {'flux': 0.0},       # Bottom boundary
        4: {'flux': top_flux},  # Top boundary
    }

    return richards_bcs


def main():

    mesh, V, W = setup_mesh_and_spaces()
    t_final, dt, time_integrator = define_time_parameters()
    soil_curves = define_soil_curves()

    time_var = Constant(0.0)

    x = SpatialCoordinate(mesh)
    X, Z = x[0], x[1]

    # Initial condition of water table at z = 0.65 m
    initial_head = Function(V, name="InitialCondition").interpolate(0.65 - Z)

    h     = Function(V, name="PressureHead").assign(initial_head)
    h_old = Function(V, name="OldSolution").assign(h)
    q     = Function(W, name='VolumetricFlux')

    moisture_content = soil_curves.moisture_content
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))

    richards_bcs = setup_boundary_conditions(mesh, time_var)

    # Solver Instantiation
    eq = RichardsSolver(V=V,
                        W=W,
                        mesh=mesh,
                        soil_curves=soil_curves,
                        bcs=richards_bcs,
                        solver_parameters='direct',
                        time_integrator=time_integrator
                        )

    # Solver Instantiation
    richards_solver = richardsSolver(h, h_old, time_var, dt, eq)

    time = 0.0

    output = VTKFile("vauclin_recharge.pvd")
    output.write(h, theta, q, time=time)
    exterior_flux = 0
    initial_mass = assemble(theta*eq.dx)

    while time < t_final:

        time_var.assign(time)

        h_old.assign(h)
        h, q, snes = advance_solution(eq, h, h_old, richards_solver)
        time += float(dt)

        exterior_flux += assemble(dt*dot(q, -eq.n)*eq.ds)

        theta.interpolate(moisture_content(h))
        output.write(h, theta, q, time=time)

    final_mass = assemble(theta*eq.dx)
    PETSc.Sys.Print(f"Initial mass: {initial_mass}")
    PETSc.Sys.Print(f"Final mass: {final_mass}")
    PETSc.Sys.Print(f"Exterior flux: {exterior_flux}")


if __name__ == "__main__":
    main()
