def RichardsSolver( h0, V, v, mesh, solverParameters, boundaryCondition, C, K, moistureContent ):

    from gadopt import Measure, Function, DirichletBC, dx, VTKFile, CheckpointFile, Constant, interpolate, assemble, sqrt, SpatialCoordinate
    import numpy as np
    from src.ProblemDefinition import ProblemDefinitionNonlinear, ProblemDefinitionLinear
    from src.timeStepper import chooseTimeStep
    from scipy.io import savemat

    dx = Measure("dx", domain=mesh, metadata={"quadrature_degree": 3})
    ds = Measure("ds", domain=mesh, metadata={"quadrature_degree": 3})
    dimen = mesh.topological_dimension()
    x     = SpatialCoordinate(mesh)

    # Do some unpacking
    finalTime = solverParameters["finalTime"]
    maxdt     = solverParameters["maxdt"]
    mindt     = solverParameters["mindt"]
    imexParameter = solverParameters["imexParameter"]

    h     = Function(V, name="Pressure Head");           h.assign(h0)
    hOld  = Function(V, name="PressureHeadOld");        hOld.assign(h0)
    hStar = Function(V, name="ApproximateSolution");   hStar.assign(h0)
    q     = Function(V, name="ApproximateSolution");     q.interpolate( -K(hOld, x)*(hOld.dx(0) + 1) )
    theta = Function(V, name = "Moisture Conent")

    DiffX = Constant( 0.1*sqrt(assemble((q)**2 * dx)) )
    timeStep = mindt; dt = Constant(timeStep)
    currentTime = 0
    iterations = 0

    solverRichardsNonlinear = ProblemDefinitionNonlinear( h, hOld, DiffX, currentTime, dt, v, V, imexParameter, K, C, boundaryCondition, mesh, dx, ds )
    solverRichardsLinear    = ProblemDefinitionLinear( h, hStar, hOld, DiffX, currentTime, timeStep, v, V, imexParameter, K, C, boundaryCondition, mesh, dx, ds )

    # Save the solution
    tStore = np.zeros(102)
    xStore = np.linspace(0, solverParameters["domainWidth"], solverParameters["nodesWidth"])
    zStore = np.linspace(0, solverParameters["domainDepth"], solverParameters["nodesDepth"])
    hStore = np.zeros((solverParameters["nodesWidth"], solverParameters["nodesDepth"], 102))
    thetaStore = np.zeros((solverParameters["nodesWidth"], solverParameters["nodesDepth"], 102))


    xx, zz = np.meshgrid(xStore, zStore)
    
    tNext = 0
    tInterval = finalTime / 100
    plotIdx = 0
    trig = 0

    outfile = VTKFile("output.pvd")

    while currentTime < finalTime:

        theta.interpolate(moistureContent(h, x))

        # Solve the system
        hOld.assign(h); hStar.assign(h)
        if solverParameters["solverType"] == "Newton":
            solverRichardsNonlinear.solve()
        elif solverParameters["solverType"] == "Picard":
            solverRichardsLinear.solve()

        currentTime = np.round(currentTime + timeStep, 8)
        iterations += 1

        # Save the solution
        if currentTime >= tNext:

            tStore[plotIdx] = currentTime
            outfile.write(h, theta, time=currentTime)
            for X in range(len(xStore)):
                for Z in range(len(zStore)):
                    hStore[X, Z, plotIdx] = np.array( h.at(xStore[X], zStore[Z], dont_raise=True) )
                    thetaStore[X, Z, plotIdx] = np.array( theta.at(xStore[X], zStore[Z], dont_raise=True) )

            plotIdx += 1
            tNext = tNext + tInterval

            saveVariables = {'tStore' : tStore, 'z' : zStore, 'x' : xStore,  'pressureHead' : hStore, 'moistureContent' : thetaStore}
            savemat("test4.mat", saveVariables)

            # Update timestep
        if iterations % 50 == 0:

            timeStep = chooseTimeStep( h, hOld, timeStep, solverParameters, dx )
            dt.assign(timeStep)

            print(currentTime)
            print(timeStep)

            q.interpolate( -K(hOld, x)*(hOld.dx(dimen-1) + 1) )
            DiffX.assign( 0*sqrt(assemble((q)**2 * dx)) )

        if (currentTime + timeStep) >= finalTime:
            timeStep = finalTime - currentTime
            timeStep = np.minimum(timeStep, maxdt)
            timeStep = np.maximum(timeStep, 1e-05)
            dt.assign(timeStep)

    for X in range(len(xStore)):
        for Z in range(len(zStore)):
            hStore[X, Z, plotIdx]     = np.array( h.at(xStore[X], zStore[Z], dont_raise=True) )
            thetaStore[X, Z, plotIdx] = np.array( theta.at(xStore[X], zStore[Z], dont_raise=True) )

    return h, tStore, hStore, thetaStore, zStore, xStore
