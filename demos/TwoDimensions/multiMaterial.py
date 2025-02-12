import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
from firedrake import *
from RichardsSolver import *
import time

timeParameters = {
    "finalTime"    : 150000,
    "timeStepType" : "adaptive",
    "timeStepSize" : 30,
    "modelFormulation" : 'mixed',
    "timeIntegration" : 'crankNicolson'
}

solverParameters = {
  "domainDepth"  : 3.00,   # depth of domain (cms)
  "nodesDepth"   : 150,
  "domainWidth"  : 5.0,   # depth of domain (cms)
  "nodesWidth"   : 300,
  "Regularisation" : 1e-07, 
  "fileName"   : "multiMaterial.pvd",
  "numberPlots"   : 30
}

mesh = RectangleMesh( solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["domainWidth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

gridSpace = solverParameters['domainDepth'] / solverParameters['nodesDepth']
eps = 25;
Ix = (1 + 0.5*tanh(eps*(x[0]-1)) ) - (1 + 0.5*tanh(eps*(x[0]-4)) )
Iz = 0.5*(1 + tanh(eps*(x[1]-1)))  - 0.5*(1 + tanh(eps*(x[1] - 3)));
I3 = 0.5*(1 + tanh(eps*(1.5 - x[1] - 0.75*cos(x[0]+0.6))));
I = 1 - Ix*Iz*I3

modelParameters = {
   "modelType" : "VanGenuchten",
   "thetaR"    : 0.10,
   "thetaS"    : 0.38,
   "alpha"     : 3.40,
   "n"         : 2.00,
   "Ks"        : 5.8300e-05*I + 5.8300e-07*(1 - I)
}


V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

h0   = Function(V, name="InitialCondition")

h0.interpolate( -2.5 )

def setBoundaryConditions(timeConstant):

    leftBC, rightBC, bottomBC, topBC = 1,2,3,4
    boundaryCondition = {
    leftBC   : {'q' : 0 },
    rightBC  : {'q' : 0 },
    bottomBC : {'q' : 0 },    # cm/s
    topBC    : {'q' : 2.5e-06},
    }

    return boundaryCondition

start = time.time()
RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions )
end = time.time()
print(end - start)
