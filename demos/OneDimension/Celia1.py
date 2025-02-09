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
    "finalTime"    : 360,
    "timeStepType" : "constant",
    "timeStepSize" : 0.001,
    "timeIntegration" : 'crankNicolson'
}

solverParameters = {
  "domainDepth"  : 40.00,   # depth of domain (cms)
  "nodesDepth"   : 300,
  "maxdt"        : 2.50,
  "mindt"        : 0.10,
  "Regularisation" : 0.0, 
  "dtTol"        : 0.00,
  "fileName"   : "Celia1.mat",
  "numberPlots"   : 10
}

mesh = IntervalMesh( solverParameters["nodesDepth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

modelParameters = {
   "modelType" : "Haverkamp",
   "thetaR"    : 0.075,
   "thetaS"    : 0.287,
   "A"         : 1.175e06,
   "gamma"     : 4.74,
   "alpha"     : 1.611e06,
   "beta"      : 3.96,
   "Ks"        : 9.44e-03,
}

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

# Initial condition
h0   = Function(V, name="InitialCondition")
h0.interpolate( -61.50 * (x[0] < 35) + (7.96*x[0] - 340.1) * (x[0] >= 35))

def setBoundaryConditions(timeConstant):

    bottomBC, topBC = 1,2
    boundaryCondition = {
    bottomBC : {'h' : -61.50 },    # cm/s
    topBC    : {'h' : -20.70},
    }

    return boundaryCondition


start = time.time()
RichardsSolver( h0, V, v, mesh, solverParameters, modelParameters, timeParameters, setBoundaryConditions )
end = time.time()
print("Simulation time ", end - start)
