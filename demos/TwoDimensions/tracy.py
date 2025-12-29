from RichardsSolver import *

"""
Comparison to Tracy's two dimensional exact solution
====================================================
Here we compare numerical solutions with the exact solution derived in
    Tracy, 2006, Water Resources Research, Clean two- and three-dimensional solutions of Richards' equation for testing numerical solvers
    https://doi.org/10.1029/2005WR004638
The simulation is performed on a square of side length L = 15.24 metres. Dirichlet boundaries conditions are imposed on the bottom, left and right boundaries $h = -L$. For the top boundary, we have
    $$h(x,z=L,t) = (1/alpha)*ln(exp(alpha*h_r) + h_0*(sin(pi*x/L)))$$
where $\alpha=0.25$, $hr=-L$, and $h_0 =  1 - exp(alpha*h_r)$. For the initial condition, we use $h$ from Tracy's exact solution at $t=2000$. We compute the L2 norm of h_{numerical}-h_{exact}

"""

# Global parameters
L     = 15.24 # Domain length [m]
alpha = 0.25  # Fitting parameter
hr    = -L
h0    = 1 - exp(alpha*hr)


def setup_mesh_and_spaces():

    """Create a square domain and DQ function spaces."""

    #L     = 15.24  # 
    nodes = 51    # Number of grid points in each direction

    mesh = RectangleMesh(nodes, nodes, L, L, name="mesh", quadrilateral=True)

    poly_degree = 2
    V = FunctionSpace(mesh, "DQ", poly_degree)       # Pressure head
    W = VectorFunctionSpace(mesh, "DQ", poly_degree) # Volumetric flux (post-processing)

    PETSc.Sys.Print("The number of degrees of freedom is:", V.dim())

    return mesh, V, W


def define_soil_curves():

    """Exponential soil model used in Tracy (2006)."""

    soil_curves = ExponentialCurve(
        theta_r=0.15,  # Residual water content [-]
        theta_s=0.45,  # Saturated water content [-]
        Ks=1.00e-05,   # Saturated hydraulic conductivity [m/s]
        alpha=alpha,    # Fitting parameter [1/m]
        Ss=0.00,       # Specific storage coefficient [1/m]
    )

    return soil_curves


def exact_solution(X, t, soil_curves):

    # Exact solution from Tracy 2006 (https://doi.org/10.1029/2005WR004638, page 4)

    beta = sqrt(alpha**2/4 + (pi/L)**2)
    hss = h0*sin(pi*X[0]/L)*exp((alpha/2)*(L - X[1]))*sinh(beta*X[1])/sinh(beta*L)
    c = alpha*(soil_curves.parameters["theta_s"] - soil_curves.parameters["theta_r"])/soil_curves.parameters["Ks"]

    phi = 0
    for k in range(1, 200):
        lambdak = k*pi/L
        gamma = (beta**2 + lambdak**2)/c
        phi = phi + ((-1)**k)*(lambdak/gamma)*sin(lambdak*X[1])*exp(-gamma*t)
    phi = phi*((2*h0)/(L*c))*sin(pi*X[0]/L)*exp(alpha*(L-X[1])/2)

    hBar = hss + phi

    hExact = ((1/alpha)*ln(exp(alpha*hr) + hBar))

    return hExact

def define_time_parameters():

    """Sets simulation time and time-stepping constants."""

    dt      = Constant(5000)
    t_final = 2e06

    time_integrator = "BackwardEuler"

    return t_final, dt, time_integrator


def setup_boundary_conditions(X, soil_curves):

    """Dirichlet BCs from Tracy (2006)."""

    top_bc = (1/alpha)*ln(exp(alpha*hr) + (h0)*(sin(pi*X[0]/L)))
    richards_bcs = {
        1: {'h': hr},
        2: {'h': hr},
        3: {'h': hr},
        4: {'h': top_bc},
    }

    return richards_bcs


def main():

    mesh, V, W = setup_mesh_and_spaces()
    t_final, dt, time_integrator = define_time_parameters()
    soil_curves = define_soil_curves()

    time_var = Constant(0.0)
    x = SpatialCoordinate(mesh)

    # Initial condition
    offset = 2000
    h_initial = exact_solution(x, offset, soil_curves)

    h     = Function(V, name="PressureHead").interpolate(h_initial)
    h_old = Function(V, name="OldSolution").assign(h)

    richards_bcs = setup_boundary_conditions(x, soil_curves)

    # Solver Instantiation
    eq = RichardsSolver(V=V,
                        W=W,
                        mesh=mesh,
                        soil_curves=soil_curves,
                        bcs=richards_bcs,
                        solver_parameters='direct',
                        time_integrator=time_integrator)

    richards_solver = richardsSolver(h, h_old, time_var, dt, eq)

    current_time        = 0.0
    nsteps = int(round(t_final / float(dt)))

    for timestep_number in range(nsteps):

        time_var.assign(current_time)

        h_old.assign(h)
        h, q, snes = advance_solution(eq, h, h_old, richards_solver)
        current_time += float(dt)

    # Compute L2 norm of error
    hExact = exact_solution(x, t_final+offset, soil_curves)
    PETSc.Sys.Print("L2 error: ", sqrt(assemble((h - hExact)**2 * eq.dx)))


if __name__ == "__main__":
    main()
