## Reproduces the two-dimensional test problem by Kirkland et al. 

import os, sys
sys.path.insert(0, '../../RichardsSolver')
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
from gadopt import *
from RichardsSolver import *
import time

solverParameters = {
  "domainDepth"  : 1.25,   # depth of domain (m)
  "nodesDepth"   : 125,
  "domainWidth"  : 1.0,   # depth of domain (m)
  "nodesWidth"   : 100,
  "finalTime"    : 1000,       # final time (s)
  "maxdt"        : 5.0,
  "Regularisation" : 0.00, 
  "mindt"        : 0.000001,
  "dtTol"        : 0.10,
  "imexParameter" : 0.0,
  "solverType" : "Newton",    # Picard or Newton
  "fileName"   : "TracySolution.mat",
  "numberPlots"   : 10
}

mesh = RectangleMesh( solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["domainWidth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

alpha = 0.25;
hr = -10.0;
h0 = 1 - exp(alpha*hr)

# Dirichlet boundary conditions
bTop     = -75.0   # Pressure head value at z = 0
bBottom     = -1000    # Pressure head value at z = L

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

# Initial condition
h0   = Function(V, name="InitialCondition")
h0.interpolate( hr )

def setBoundaryConditions(timeConstant):

    alpha = 0.25;
    hr = -10.0;
    h0 = 1 - exp(alpha*hr)

    leftBC, rightBC, bottomBC, topBC = 1,2,3,4
    boundaryCondition = {
    leftBC   : {'h' : hr},
    rightBC  : {'h' : hr},
    bottomBC : {'h' : hr },    # cm/s
    topBC    : {'h' : (1/alpha)*ln(exp(alpha*hr) + h0*sin(pi*x[0]))},
    }

    return boundaryCondition

def C(h, x):

    hg = 2
    thetaS = 0.45
    thetaR = 0.15

    return( (thetaS - thetaR) * exp(h * alpha) * alpha )

def K(h, x):

    Ks = 1e-05    # m/s
    alpha = 0.25        # m

    return Ks*exp(h*alpha)

def moistureContent(h, x):

    alpha = 0.25        # m
    thetaS = 0.45
    thetaR = 0.15

    return( thetaR + (thetaS - thetaR) * exp(h * alpha) )

start = time.time()
h, tStore, hStore, thetaStore, zStore, xStore = RichardsSolver( h0, V, v, mesh, solverParameters, setBoundaryConditions, C, K, moistureContent )
end = time.time()
print(end - start)

saveVariables = {'tStore' : tStore, 'z' : zStore, 'x' : xStore,  'pressureHead' : hStore, 'moistureContent' : thetaStore}
savemat(solverParameters["fileName"], saveVariables)
