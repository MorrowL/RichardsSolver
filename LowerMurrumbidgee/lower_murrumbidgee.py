from RichardsSolver import *
from surface_mesh import *
from ufl import *
from firedrake import *
from firedrake.petsc import PETSc


def setup_mesh_and_spaces():

    """Defines the computational mesh and function spaces."""

    # Function space of horizontal and vertical
    horiz_elt = FiniteElement("DG", triangle, 1)
    vert_elt = FiniteElement("DG", interval, 1)
    horizontal_resolution = 3000
    number_layers = 300

    # Details of mesh extrusion
    surface_mesh(horizontal_resolution)
    layer_height = 1/number_layers
    mesh2D = Mesh('MurrumbidgeeMeshSurface.msh')
    mesh = ExtrudedMesh(mesh2D, number_layers, layer_height=layer_height, extrusion_type='uniform', name='mesh')

    W = VectorFunctionSpace(mesh, 'CG', 1)
    X = assemble(interpolate(mesh.coordinates, W))
    mesh_coords = X.dat.data_ro

    # Transform the z coordinate such that top and bottom are the points given by elevation_data.csv and bedrock_data.csv
    z = mesh_coords[:, 2]
    bedrock = data_2_function(mesh_coords, 'bedrock_data.csv')
    elevation = data_2_function(mesh_coords, 'elevation_data.csv')
    mesh.coordinates.dat.data[:, 2] = bedrock*z + elevation - bedrock

    elt = TensorProductElement(horiz_elt, vert_elt)
    V = FunctionSpace(mesh, elt)
    W = VectorFunctionSpace(mesh, elt)

    PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())
    PETSc.Sys.Print("Horizontal resolution:", horizontal_resolution)
    PETSc.Sys.Print("Number of layers:", number_layers)

    ihbiub

    return mesh, V, W


def define_time_parameters():

    """Sets simulation time and time-stepping constants."""

    t_final_years = 7.0
    t_final = t_final_years * 3.156e+7  # in seconds

    dt_days = 7.0
    dt = Constant(dt_days * 86400)  # s

    time_var = Constant(0.0)
    time_integrator = "BackwardEuler"

    return t_final, dt, time_integrator


def define_soil_curves(mesh, V):

    Vcg = FunctionSpace(mesh, 'CG', 1)

    x = SpatialCoordinate(mesh)
    X, Y, Z = x[0], x[1], x[2]

    # Construct some functions from external data
    elevation_cg = Function(Vcg)
    elevation_cg.dat.data[:] = data_2_function(mesh_coords, 'elevation_data.csv')
    elevation = Function(V, name='elevation').interpolate(elevation_cg)

    shallowLayer_cg = Function(Vcg)
    shallowLayer_cg.dat.data[:] = data_2_function(mesh_coords, 'shallow_layer.csv')
    shallowLayer = Function(V, name='ShallowLayer').interpolate(shallowLayer_cg)

    lowerLayer_cg = Function(Vcg)
    lowerLayer_cg.dat.data[:] = data_2_function(mesh_coords, 'lower_layer.csv')
    lowerLayer = Function(V, name='LowerLayer').interpolate(lowerLayer_cg)

    bedrock_cg = Function(Vcg)
    bedrock_cg.dat.data[:] = data_2_function(mesh_coords, 'bedrock_data.csv')
    bedrock = Function(V, name='Bedrock').interpolate(bedrock_cg)

    depth = Function(V, name='depth').interpolate(elevation - Z)

    # Indicator functions of where each layer is
    delta = 1
    I1 = 0.5*(1 + tanh(delta*(shallowLayer - depth)))
    I2 = 0.5*(1 + tanh(delta*(lowerLayer - depth)))
    I3 = 0.5*(1 + tanh(delta*(bedrock - depth)))

    S_depth = 1/((1 + 0.000071*depth)**5.989)     # Depth dependent porosity
    K_depth = (1 - depth / (58 + 1.02*depth))**3  # Depth dependent conductivity
    Ks = Function(V, name='SaturatedConductivity').interpolate(K_depth*(2.5e-05*I1 + 1e-03*(1 - I1)*I2 + 5e-04*(1-I2)))

    # Specify the hydrological parameters
    soil_curve = HaverkampCurve(
        theta_r=0.025,         # Residual water content [-]
        theta_s=0.40*S_depth,  # Saturated water content [-]
        Ks=Ks,                 # Saturated hydraulic conductivity [m/s]
        alpha=0.44,            # Fitting parameter [m]
        beta=1.2924,           # Fitting parameter [-]
        A=0.0104,              # Fitting parameter [m]
        gamma=1.5722,          # Fitting parameter [-]
        Ss=0,              # Specific storage coefficient [1/m]
    )

    return soil_curve


def setup_boundary_conditions(mesh, time_var):

    """Defines the boundary conditions dictionary."""

    x = SpatialCoordinate(mesh)
    X, Y, Z = x[0], x[1], x[2]

    watertable_cg = Function(Vcg)
    watertable_cg.dat.data[:] = data_2_function(mesh_coords, 'water_table.csv')
    watertable = Function(Vdq, name='WaterTable').interpolate(watertable_cg)

    rainfall_cg = Function(Vcg)
    rainfall_cg.dat.data[:] = data_2_function(mesh_coords, 'rainfall_data.csv')
    rainfall = Function(Vdq, name='Rainfall').interpolate(watertable_cg)

    # Extraction points
    spread = 50000000
    Ind = 0
    xPts = np.array([1.7e05, 2.2e05, 2.4e05, 2.0e05, 1.6e05, 1.7e05, 2.2e05, 2.5e05, 2.0e05, 1.5e05, 2.3e05, 1.0e05, 2.0e05, 1.9e05, 1.9e05, 1.9e05, 1.6e05, 2.6e05, 1.2e05, 2.5e05])
    yPts = np.array([8.0e04, 4.3e04, 4.2e04, 7.3e04, 3.5e04, 9.3e04, 6.0e04, 6.5e04, 6.0e04, 5.0e04, 9.0e04, 6.5e04, 2.0e04, 1.0e05, 8.5e04, 4.4e04, 6.0e04, 5.5e04, 6.5e04, 2.2e04])
    for ii in range(20):
        Ind = Ind + exp((-(x[0]-xPts[ii])**2-(x[1]-yPts[ii])**2)/spread)


    richards_bcs = {
        1: {'h': depth - watertable},
        'bottom': {'flux': 3e-9*Ind},
        'top': {'flux': 0.14*3.171e-11*rainfall},
    }

    return richards_bcs


def main():

    mesh, V, W = setup_mesh_and_spaces()
    t_final, dt, time_integrator = define_time_parameters()
    soil_curves = define_soil_curves(mesh)

    time_var = Constant(0.0)

    x = SpatialCoordinate(mesh)
    X, Z = x[0], x[1]

    # Initial condition of water table at z = 0.65 m
    Vcg = FunctionSpace(mesh, 'CG', 1)

    watertable_cg = Function(Vcg)
    watertable_cg.dat.data[:] = data_2_function(mesh_coords, 'water_table.csv')
    watertable = Function(Vdq, name='WaterTable').interpolate(watertable_cg)

    elevation_cg = Function(Vcg)
    elevation_cg.dat.data[:] = data_2_function(mesh_coords, 'elevation_data.csv')
    elevation = Function(V, name='elevation').interpolate(elevation_cg)

    depth = Function(V, name='depth').interpolate(elevation - Z)

    initial_head = Function(V, name="InitialCondition").interpolate(depth - watertable)
    h = Function(V, name="PressureHead").assign(initial_head)
    h_old = Function(V, name="OldSolution").assign(h)
    q = Function(W, name='VolumetricFlux')

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
                        quad_degree=3)

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

    final_mass = assemble(theta*eq.dx)
    PETSc.Sys.Print(f"Initial mass: {initial_mass}")
    PETSc.Sys.Print(f"Final mass: {final_mass}")
    PETSc.Sys.Print(f"Exterior flux: {exterior_flux}")


if __name__ == "__main__":
    main()
