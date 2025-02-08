## Reproduces the two-dimensional test problem by Kirkland et al. 

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
    "finalTime"    : 86400,
    "timeStepType" : "constant",
    "timeStepSize" : 60,
    "timeIntegration" : 'crankNicolson'
}

solverParameters = {
  "domainDepth"  : 100,   # depth of domain (cms)
  "nodesDepth"   : 1000,
  "maxdt"        : 5.0,
  "mindt"        : 0.01,
  "Regularisation" : 0.0, 
  "dtTol"        : 0.10,
  "fileName"   : "Celia2.mat",
  "numberPlots"   : 30
}

mesh = IntervalMesh( solverParameters["nodesDepth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

modelParameters = {
   "modelType" : "VanGenuchten",
   "thetaR"    : 0.102,
   "thetaS"    : 0.368,
   "alpha"     : 0.0335,
   "n"         : 2.00,
   "Ks"        : 0.00922,
}

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

# Initial condition
h0   = Function(V, name="InitialCondition")
h0.interpolate( -1000 * (x[0] < 90) + (92.5*x[0] - 9325) * (x[0] >= 90))

def setBoundaryConditions(timeConstant):

    bottomBC, topBC = 1,2
    boundaryCondition = {
    bottomBC : {'h' : -1000 },    # cm/s
    topBC    : {'h' : -75.0},
    }

    return boundaryCondition

start = time.time()
RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions )
end = time.time()
print(end - start)
