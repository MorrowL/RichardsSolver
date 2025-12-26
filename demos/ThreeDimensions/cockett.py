from RichardsSolver import *

from ufl import tanh, sin, exp, SpatialCoordinate
from firedrake import Function, FunctionSpace, VectorFunctionSpace, Constant, RectangleMesh, assemble, dot, ExtrudedMesh
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
import time

"""
Three-dimensional infultration of water into a heterogeneous soil
=================================================================
Here we present an example of the infultration into a heterogeneous column of soil composed of a mixture of sand and loamy sand, as described by
    Cockett, Heagy, and Haber, Computers and Geosciences, 2018, Efficient 3D inversions using the Richards equation
    https://doi.org/10.1016/j.cageo.2018.04.006
Simulations are formed in a rectangular prism of side length 2.0 x 2.0 x 2.6 m. No flux is imposed on all the boundaries except the top where $h = -0.1$ m.
"""


def setup_mesh_and_spaces():

    """Defines the computational mesh and function spaces."""

    nodes_x, nodes_y = round(1.0*151), round(1.0*151)
    nodes_z = round(1.3*nodes_x)
    L_x, L_y, L_z = 2, 2, 2.6
    mesh2D = RectangleMesh(nodes_x, nodes_y, L_x, L_y, quadrilateral=True)
    mesh = ExtrudedMesh(mesh2D, nodes_z, layer_height=L_z/nodes_z, name='mesh')

    poly_degree = 2
    # Function space for pressure head
    V = FunctionSpace(mesh, "DQ", poly_degree)
    # Volumetric flux post-processed from h
    W = VectorFunctionSpace(mesh, "DQ", poly_degree)

    PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

    asdf

    return mesh, V, W


def define_time_parameters():

    """Sets simulation time and time-stepping constants."""

    t_final_hours = 16.0
    t_final = t_final_hours * 60 * 60  # 8 hours in seconds
    dt = Constant(300)  # s

    time_integrator = "BackwardEuler"

    return t_final, dt, time_integrator


def define_soil_curves(mesh):

    """Construct heterogeneuous soil."""

    x = SpatialCoordinate(mesh)
    X, Y, Z = x[0], x[1], x[2]

    epsilon = 1/500
    r = [0.0729, 0.0885, 0.7984, 0.9430, 0.6837, 0.1321, 0.7227, 0.1104, 0.1175, 0.6407]
    I = sin(3*(X-r[0])) + sin(3*(Y-r[1])) + sin(3*(Z-r[2])) + sin(3*(X-r[3])) + sin(3*(Y-r[4])) + sin(3*(Z-r[5]))+sin(3*(X-r[6])) + sin(3*(Y-r[7])) + sin(3*(Z-r[8]))
    I = 0.5*(1 + tanh(I/epsilon))

    # Specify the hydrological parameters
    soil_curves = VanGenuchtenCurve(
        theta_r=0.02*I + 0.035*(1-I),    # Residual water content [-]
        theta_s=0.417*I + 0.401*(1-I),   # Saturated water content [-]
        Ks=5.82e-05*I + 1.69e-05*(1-I),  # Saturated hydraulic conductivity [m/s]
        alpha=13.8*I + 11.5*(1-I),       # Related to inverse of air entry [1/m]
        n=1.592*I + 1.474*(1-I),         # Measure of pore distribution [-]
        Ss=0,                            # Specific storage coefficient [1/m]
    )

    return soil_curves


def setup_boundary_conditions(mesh, time_var):

    """Defines the boundary conditions dictionary."""

    top_bc = -0.1
    bottom_bc = -0.3
    
    richards_bcs = {
        1: {'flux': 0},               # Left
        2: {'flux': 0},               # Right
        3: {'flux': 0},               # Front
        4: {'flux': 0},               # Back
        'bottom': {'h': bottom_bc},   # Top
        'top': {'h': top_bc},         # Bottom
    }

    return richards_bcs


def main():

    mesh, V, W = setup_mesh_and_spaces()
    t_final, dt, time_integrator = define_time_parameters()
    soil_curves = define_soil_curves(mesh)

    time_var = Constant(0.0)

    x = SpatialCoordinate(mesh)
    X, Z = x[0], x[2]

    # Initial condition
    initial_head = Function(V, name="InitialCondition").interpolate(-0.3 + 0.2*exp(5*(Z-2.6)))
    h = Function(V, name="PressureHead").assign(initial_head)
    h_old = Function(V, name="OldSolution").assign(h)

    moisture_content = soil_curves.moisture_content
    theta = Function(V, name='MoistureContent').interpolate(moisture_content(h))

    richards_bcs = setup_boundary_conditions(mesh, time_var)

    solver_parameters = {'ksp_type': 'gmres', 'mat_type': 'aij', 'pc_type': 'python', 'pc_python_type': 'firedrake.AssembledPC', 'assembled_pc_type': 'bjacobi', 'assembled_sub_pc_type': 'ilu', 'ksp_rtol': 1e-05}

    PETSc.Sys.Print(solver_parameters)  

    # Solver Instantiation
    eq = RichardsSolver(V=V,
                        W=W,
                        mesh=mesh,
                        soil_curves=soil_curves,
                        bcs=richards_bcs,
                        solver_parameters=solver_parameters,
                        time_integrator=time_integrator,
                        quad_degree=1)

    # Solver Instantiation
    richards_solver = richardsSolver(h, h_old, time_var, dt, eq)

    current_time = 0.0
    exterior_flux = 0
    timestep_number = 0
    nonlinear_iteration = 0
    linear_iterations = 0
    initial_mass = assemble(theta*eq.dx)
    start_time = time.perf_counter()

    while current_time < t_final:

        time_var.assign(current_time)

        h_old.assign(h)
        h, q, snes = advance_solution(eq, h, h_old, richards_solver)
        current_time += float(dt)

        timestep_number += 1
        exterior_flux += assemble(dt*dot(q, -eq.n)*eq.ds)
        nonlinear_iteration += snes.getIterationNumber()
        linear_iterations += snes.ksp.getIterationNumber()
        theta.interpolate(moisture_content(h))

        PETSc.Sys.Print(f"Current time [h]: {current_time/3600}")   
        PETSc.Sys.Print(f"Nonlinear iterations: {snes.getIterationNumber()}")
        PETSc.Sys.Print(f"Linear iterations: {snes.ksp.getIterationNumber()}")
        PETSc.Sys.Print(f"Water content: {assemble(theta*eq.dx)}")
        PETSc.Sys.Print("")

        
        #output.write(h, theta, q, time=time)

    end_time = time.perf_counter()
    final_mass = assemble(theta*eq.dx)

    PETSc.Sys.Print(f"Execution time: {end_time-start_time} seconds")
    PETSc.Sys.Print("")
    PETSc.Sys.Print(f"Number of timesteps: {timestep_number}")
    PETSc.Sys.Print(f"Total number of nonlinear iterations: {nonlinear_iteration}")
    PETSc.Sys.Print(f"Total number of linear iterations: {linear_iterations}")
    PETSc.Sys.Print("")
    PETSc.Sys.Print(f"Initial mass: {initial_mass}")
    PETSc.Sys.Print(f"Final mass: {final_mass}")
    PETSc.Sys.Print(f"Exterior flux: {exterior_flux}")


if __name__ == "__main__":
    main()
