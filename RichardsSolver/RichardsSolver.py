def RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions ):

    import firedrake as fd
    from firedrake.petsc import PETSc
    import numpy as np
    from ProblemDefinition import ProblemDefinitionNonlinear, ProblemDefinitionLinear
    from modelTypes import relativePermeability, moistureContent
    from updateTimeStep import updateTimeStep
    from scipy.io import savemat

    VecSpace = fd.VectorFunctionSpace(mesh, "CG", 1)
    R        = fd.FunctionSpace(mesh, 'R', 0)

    dx = fd.Measure("dx", domain=mesh, metadata={"quadrature_degree": 3})
    ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": 3})
    dimen = mesh.topological_dimension()
    x     = fd.SpatialCoordinate(mesh)

    # Do some unpacking
    finalTime     = timeParameters["finalTime"]

    h     = fd.Function(V, name="Pressure Head");         h.assign(h0)
    hOld  = fd.Function(V, name="PressureHeadOld");       hOld.assign(h0)
    hStar = fd.Function(V, name="ApproximateSolution");   hStar.assign(h0)
    q     = fd.Function(VecSpace, name='volumetricFlux'); q.assign( fd.project(fd.as_vector(fd.grad( h )), VecSpace) )
    theta = fd.Function(V, name = "Moisture Conent")
    K     = fd.Function(V, name = "Relative Permeability")

    timeStep = timeParameters["timeStepSize"]; 

    timeStep    =  fd.Function(R, name="delta_t").assign( 1 )
    currentTime = fd.Function(R, name="time").assign( 0 )
    DiffX       = fd.Function(R, name="smoothing").assign( 0 )

    iterations = 0
    timeIntegrator = timeParameters["timeIntegration"]

    #solverRichardsLinear    = ProblemDefinitionNonlinear( h, hOld, DiffX, timeConstant, dt, v, V, "forwardEuler", "pressureHead", modelParameters, setBoundaryConditions, mesh, dx, ds )
    solverRichardsNonlinear = ProblemDefinitionNonlinear( h, hOld, DiffX, currentTime, timeStep, v, V, timeIntegrator, "mixed", modelParameters, setBoundaryConditions, mesh, dx, ds )

    # Save the solution
    tStore = np.zeros(solverParameters["numberPlots"]+1)

    if dimen == 1:

        xStore = 0
        zStore = np.linspace(0, solverParameters["domainDepth"], solverParameters["nodesDepth"])
        hStore = np.zeros((solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))
        thetaStore = np.zeros((solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))

#    elif dimen == 2:

#        xStore = np.linspace(0, solverParameters["domainWidth"], solverParameters["nodesWidth"])
#        zStore = np.linspace(0, solverParameters["domainDepth"], solverParameters["nodesDepth"])
#        hStore = np.zeros((solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))
#        thetaStore = np.zeros((solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["numberPlots"]+1))

    else:

        outfile = fd.VTKFile(solverParameters["fileName"])
    
    tNext = 0
    tInterval = finalTime / solverParameters["numberPlots"]
    plotIdx = 0

    while float(currentTime) <= finalTime:

        theta.interpolate(moistureContent(modelParameters, h, x, currentTime))
        K.interpolate(relativePermeability(modelParameters, h, x, currentTime))
        q.assign( fd.project(fd.as_vector(-K*fd.grad( h + x[dimen-1])), VecSpace) )

        # Save the solution
        if float(currentTime) >= tNext:

            PETSc.Sys.Print(float(currentTime))

            tStore[plotIdx] = float(currentTime)

    #        udiffabs = fd.Function(V).interpolate(abs((h - hStar)))
    #        with udiffabs.dat.vec_ro as v:
    #            absoluteError = v.max()[1]
    #        PETSc.Sys.Print(absoluteError)

            if dimen >= 2:

                outfile.write(h, theta, K, q, time=float(currentTime))

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
        hOld.assign(h); 
#        solverRichardsLinear.solve()
#        hStar.assign(h);
        solverRichardsNonlinear.solve()


        if timeParameters['timeIntegration'] == 'modifiedEuler':
            hOld.assign( h )
            solverRichardsNonlinear.solve()
            h.assign( 0.5*(h + hStar))

        currentTime.assign( currentTime + timeStep )
        iterations += 1

        # Update timestep
        if iterations % 5:
            timeStep = updateTimeStep( h, hOld, timeStep, timeParameters, V )

            qNorm = fd.Function(V).interpolate( fd.sqrt(fd.dot(q, q)) ) 
            with qNorm.dat.vec_ro as v:
                qMax = v.max()[1]


            laplacian = fd.Function(V).interpolate(abs( h.dx(0) )) 
            with laplacian.dat.vec_ro as v:
                lapMax = v.max()[1]

            DiffX.assign( 1e-7*lapMax * qMax); #PETSc.Sys.Print(float(DiffX))

        if float(currentTime + timeStep) > tNext:
            timeStepNew = tNext - float(currentTime)
            timeStepNew = np.maximum(timeStepNew, 1)
            timeStep.assign( timeStepNew )

    
    PETSc.Sys.Print("Total number of iterations", iterations)