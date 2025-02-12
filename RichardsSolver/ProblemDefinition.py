def ProblemDefinitionNonlinear( h, hOld, DiffX, timeConstant, timeStep, v, V, timeIntegrator, modelFormulation, modelParameters, setBoundaryConditions, mesh, dx, ds ):
    
    # Returns the variational problem for solving Richards equation
    import firedrake as fd
    import numpy as np
    from modelTypes import relativePermeability, waterRetention, moistureContent

    dimen = mesh.topological_dimension()
    x     = fd.SpatialCoordinate(mesh)

    #timeIntegrator   = timeParameters['timeIntegration']
    #modelFormulation = timeParameters['modelFormulation']

    boundaryCondition = setBoundaryConditions(timeConstant)

    bottomBC = boundaryCondition[1];    topBC = boundaryCondition[2]
    keyB     = next(iter(bottomBC));    keyT  = next(iter(topBC))
    valB = bottomBC[keyB];              valT = topBC[keyT]

    if timeIntegrator == 'backwardEuler':
        hBar = h; hDiff = h
    elif timeIntegrator == 'crankNicolson':
        hBar = 0.5*(h + hOld); hDiff = hBar
    elif timeIntegrator == 'modifiedEuler':
        hBar = hOld; hDiff = h
    else:
        hBar = hOld; hDiff = h


    C = waterRetention( modelParameters, hBar, x, timeConstant )
    K = relativePermeability( modelParameters, hBar, x, timeConstant )

    if dimen == 1:
        gravity = fd.as_vector([ K ])
    elif dimen == 2:
        gravity = fd.as_vector([0,  K ]);
    else:
        gravity = fd.as_vector([0, 0, K ])

    normalVector = fd.FacetNormal(mesh)

    # Define problem
    if modelFormulation == 'mixed':

        thetaOld = moistureContent( modelParameters, hOld, x, timeConstant)
        thetaNew = moistureContent( modelParameters, h, x, timeConstant)

        F = ( fd.inner( (thetaNew - thetaOld )/timeStep, v) +
            fd.inner( K*fd.grad( hDiff ), fd.grad(v) )  -
        fd.inner( K.dx(dimen-1), v )
        + fd.inner( DiffX*fd.grad( h), fd.grad(v) ) )*dx

    else:

        F = ( fd.inner( C*(h - hOld)/timeStep, v) +
            fd.inner( K*fd.grad( hDiff ), fd.grad(v) )  -
        fd.inner( K.dx(dimen-1), v )
        + fd.inner( DiffX*fd.grad( h), fd.grad(v) ) )*dx

    strongBCS = [];

    for index in range(len(boundaryCondition)):
        boundaryInfo  = boundaryCondition[index+1]
        boundaryType  = next(iter(boundaryInfo));
        boundaryValue = boundaryInfo[boundaryType]; 
    
        if boundaryType == "h":
            strongBCS.append(fd.DirichletBC(V, boundaryValue, index+1))
        else:
            F = F - ( -( fd.dot( normalVector , gravity ) - boundaryValue) ) * v * ds(index+1)

    problem = fd.NonlinearVariationalProblem(F, h, bcs = strongBCS)
   
    solverRichardsNonlinear  = fd.NonlinearVariationalSolver(problem,
                                        solver_parameters={
                                        'mat_type': 'aij',
                                        'snes_type': 'newtonls',
                                        'ksp_type': 'preonly',
                                        'pc_type': 'lu',
                                        })
    
    return solverRichardsNonlinear
    
def ProblemDefinitionLinear( h, hStar, hOld, DiffX, currentTime, timeStep, v, V, imexParameter, K, C, boundaryCondition, mesh, dx, ds ):

    from gadopt import lhs, rhs, TrialFunction, Function, inner, grad, LinearVariationalProblem, LinearVariationalSolver, SpatialCoordinate, DirichletBC

    hTemp = TrialFunction(V)

    dimen = mesh.topological_dimension()
    x     = SpatialCoordinate(mesh)

    bottomBC = boundaryCondition[1];    topBC = boundaryCondition[2]
    keyB     = next(iter(bottomBC));    keyT  = next(iter(topBC))
    valB = bottomBC[keyB];              valT = topBC[keyT]

    if keyT == "h":
        A0, B0, C0 = 1, 0, valT
    else:
        A0, B0, C0 = 0, 1, -(K(h, x) - valT)

    if keyB == "h":
        Ab, Bb, Cb = 1, 0, valB
    else:
        Ab, Bb, Cb = 0, 1, K(h, x)

    # Impose strong Diriclet boundary condition
    if B0 == 0:
        bcTop    = DirichletBC(V, C0/A0, 4)   # BC at top
    if Bb == 0:
        bcBottom = DirichletBC(V, Cb/Ab, 3)   # BC at bottom

    # Define problem
    F = ( inner( C( 0.5*(hOld + hOld), x )*(hTemp - hOld)/timeStep, v) +
        inner( K( 0.5*(hOld + hOld), x )*grad( 0.5*(hTemp + hOld)), grad(v) )  +
    inner( K( 0.5*(hOld + hOld), x ).dx(dimen-1), v ) )*dx

        # Impose Neumann boundary conditions
    if B0 != 0:
        F = F -  (C0/B0) * v * ds(4)
    if Bb != 0:
        F = F -  (Cb/Bb) * v * ds(3)

    a, L = lhs(F), rhs(F)

    if (B0 == 0) and (Bb == 0):
        problem = LinearVariationalProblem(a, L, h, bcs = [bcBottom, bcTop])
    elif (B0 == 0):
        problem = LinearVariationalProblem(a, L, h, bcs = [bcTop])
    elif (Bb == 0):
        problem = LinearVariationalProblem(a, L, h, bcs = [bcBottom])
    else:
        problem = LinearVariationalProblem(a, L, h)

    solverRichardsLinear  = LinearVariationalSolver(problem,
                                        solver_parameters={
                                        'mat_type': 'aij',
                                        'ksp_type': 'preonly',
                                        'pc_type': 'lu'})
    
    return solverRichardsLinear

def hMid(h, hOld, omega):
  return( omega*h + (1 - omega)*hOld )



