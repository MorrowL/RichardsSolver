def ProblemDefinitionNonlinear( h, hOld, DiffX, currentTime, timeStep, v, V, imexParameter, K, C, boundaryCondition, mesh, dx, ds ):
    
    # Returns the variational problem for solving Richards equation

    from gadopt import sqrt, dot, grad, inner, NonlinearVariationalProblem, NonlinearVariationalSolver, DirichletBC
    from firedrake import assemble, Function, interpolate, assign, SpatialCoordinate

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
    F = ( inner( C( hMid(h, hOld, imexParameter), x )*(h - hOld)/timeStep, v) +
        inner( K( hMid(h, hOld, imexParameter), x )*grad( 0.5*(h + hOld) ), grad(v) )  -
    inner( K( hMid(h, hOld, imexParameter), x ).dx(dimen-1), v )
    + inner( C(h, x)*DiffX*grad( h), grad(v) ) )*dx

    # Impose Neumann boundary conditions
    if B0 != 0:
        F = F -  (C0/B0) * v * ds(4)
    if Bb != 0:
        F = F -  (Cb/Bb) * v * ds(3)

    if (B0 == 0) and (Bb == 0):
        problem = NonlinearVariationalProblem(F, h, bcs = [bcBottom, bcTop])
    elif (B0 == 0):
        problem = NonlinearVariationalProblem(F, h, bcs = [bcTop])
    elif (Bb == 0):
        problem = NonlinearVariationalProblem(F, h, bcs = [bcBottom])
    else:
        problem = NonlinearVariationalProblem(F, h)
   
    solverRichardsNonlinear  = NonlinearVariationalSolver(problem,
                                        solver_parameters={
                                        'mat_type': 'aij',
                                        'snes_type': 'newtonls',
                                        'ksp_type': 'preonly',
                                        'pc_type': 'lu'})
    
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



