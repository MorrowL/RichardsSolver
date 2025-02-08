def RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions ):

    import firedrake as fd
    import numpy as np
    import ufl
    from ProblemDefinition import ProblemDefinitionNonlinear, ProblemDefinitionLinear
    from modelTypes import relativePermeability, moistureContent
    from timeStepper import chooseTimeStep
    from scipy.io import savemat

    dx = fd.Measure("dx", domain=mesh, metadata={"quadrature_degree": 3})
    ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": 3})
    dimen = mesh.topological_dimension()
    x     = fd.SpatialCoordinate(mesh)

    # Do some unpacking
    finalTime     = timeParameters["finalTime"]
    maxdt         = solverParameters["maxdt"]
    mindt         = solverParameters["mindt"]

    h     = fd.Function(V, name="Pressure Head");         h.assign(h0)
    hOld  = fd.Function(V, name="PressureHeadOld");       hOld.assign(h0)
    hStar = fd.Function(V, name="ApproximateSolution");   hStar.assign(h0)
    q     = fd.Function(V, name="ApproximateSolution");   q.interpolate( -relativePermeability(modelParameters, hOld, x, 0)*(hOld.dx(0) - 1) )
    theta = fd.Function(V, name = "Moisture Conent")

    DiffX = fd.Constant( solverParameters["Regularisation"] )
    timeStep = timeParameters["timeStepSize"]; 
    dt = fd.Constant(timeStep)
    currentTime = 0
    timeConstant = fd.Constant(currentTime)
    iterations = 0

    solverRichardsNonlinear = ProblemDefinitionNonlinear( h, hOld, DiffX, timeConstant, dt, v, V, timeParameters['timeIntegration'], modelParameters, setBoundaryConditions, mesh, dx, ds )

    # Save the solution
    tStore = np.zeros(solverParameters["numberPlots"]+1)

    if dimen == 1:

        xStore = 0
        zStore = np.linspace(0, solverParameters["domainDepth"], solverParameters["nodesDepth"])
        hStore = np.zeros((solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))
        thetaStore = np.zeros((solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))

    elif dimen == 2:

        xStore = np.linspace(0, solverParameters["domainWidth"], solverParameters["nodesWidth"])
        zStore = np.linspace(0, solverParameters["domainDepth"], solverParameters["nodesDepth"])
        hStore = np.zeros((solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))
        thetaStore = np.zeros((solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))

    else:

        outfile = fd.VTKFile(solverParameters["fileName"])
    
    tNext = 0
    tInterval = finalTime / solverParameters["numberPlots"]
    plotIdx = 0

    while currentTime <= finalTime:

        theta.interpolate(moistureContent(modelParameters, h, x, currentTime))

        # Save the solution
        if currentTime >= tNext:

            print("Time ", currentTime)

            tStore[plotIdx] = currentTime

            if dimen == 3:

                outfile.write(h, theta, time=currentTime)

            else:

                if dimen == 1:

                    for Z in range(len(zStore)):
                        hStore[Z, plotIdx] = np.array( h.at(zStore[Z], dont_raise=True) )
                        thetaStore[Z, plotIdx] = np.array( theta.at( zStore[Z], dont_raise=True) )

                elif dimen == 2:

                    for X in range(len(xStore)):
                        for Z in range(len(zStore)):
                            hStore[X, Z, plotIdx] = np.array( h.at(xStore[X], zStore[Z], dont_raise=True) )
                            thetaStore[X, Z, plotIdx] = np.array( theta.at(xStore[X], zStore[Z], dont_raise=True) )


                saveVariables = {'tStore' : tStore, 'z' : zStore, 'x' : xStore,  'pressureHead' : hStore, 'moistureContent' : thetaStore}
                savemat(solverParameters["fileName"], saveVariables)

            plotIdx += 1
            tNext = tNext + tInterval

        # Solve the system
        hOld.assign(h); hStar.assign(h);
        solverRichardsNonlinear.solve()
        if timeParameters['timeIntegration'] == 'modifiedEuler':
            hOld.assign( h )
            solverRichardsNonlinear.solve()
            h.assign( 0.5*(h + hStar))

        currentTime = np.round(currentTime + timeStep, 8)
    
        iterations += 1
        timeStep = timeParameters["timeStepSize"]
        dt.assign( timeStep ) 
        timeConstant.assign( currentTime )

        if (currentTime + timeStep) >= tNext:
            timeStep = tNext - currentTime
            timeStep = np.minimum(timeStep, maxdt)
            timeStep = np.maximum(timeStep, 1e-05)
            dt.assign(timeStep)
