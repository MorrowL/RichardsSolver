## Reproduces the two-dimensional test problem by Kirkland et al. 

import os, sys
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
from gadopt import *
from src.RichardsSolver import *
import time

solverParameters = {
  "domainDepth"  : 40.00,   # depth of domain (cms)
  "nodesDepth"   : 300,
  "domainWidth"  : 20.0,   # depth of domain (cms)
  "nodesWidth"   : 150,
  "finalTime"    : 2.0*360,       # final time (days)
  "maxdt"        : 2.0*2.50,
  "mindt"        : 0.10,
  "dtTol"        : 0.10,
  "imexParameter" : 0.00,
  "solverType" : "Newton"    # Picard or Newton
}

mesh = RectangleMesh( solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["domainWidth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

bottomBC, topBC = 1,2
boundaryCondition = {
   bottomBC : {'q' : 0.00},    # cm/s
   topBC    : {'q' : 5.0e-03 + 0.00001*x[0]},
}

# Dirichlet boundary conditions
bTop     = -75.0   # Pressure head value at z = 0
bBottom     = -1000    # Pressure head value at z = L

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

h0   = Function(V, name="InitialCondition")

bTop, bBottom = -20.70, -61.50
h0.interpolate( -61.0 )

def C(h, x):

  alpha  = 1.611e06
  beta   = 3.96
  thetaR = 0.075
  thetaS = 0.287

  return( -sign(h)*alpha*beta*(thetaS - thetaR)*pow(abs(h),beta-1) ) / ( pow(alpha + pow(abs(h),beta),2) )

def K(h, x):

  import numpy  

  xPoints = [20, 10, 35, 25]
  zPoints = [30, 25, 27, 15]

  I = 0;

  for ii in range(4):

    R = ((x[0]-xPoints[ii])**2 + (x[1]-zPoints[ii])**2)**0.5
    I1 = 0.5*(1 + tanh(2*(R - 5))); 
    I1 = 1 - I1;
    I = I + I1;

  A      = 1.175e06
  gamma  = 4.74
  Ks     = abs(9.44e-03*(1 - I) + 0*1e-7*I)

  return Ks*A / ( A + pow(abs(h), gamma) )

def moistureContent(h, x):

  alpha  = 1.611e06
  beta   = 3.96
  thetaR = 0.075
  thetaS = 0.287

  return( thetaR + alpha*(thetaS - thetaR) / (alpha + abs(h)**beta) )

start = time.time()
h, tStore, hStore, zStore, xStore = RichardsSolver( h0, V, v, mesh, solverParameters, boundaryCondition, C, K, moistureContent )
end = time.time()
print(end - start)

theta        = np.array(moistureContent(hStore, x))
permeability = np.array(K(hStore, x))

saveVariables = {'tStore' : tStore, 'z' : zStore, 'x' : xStore,  'pressureHead' : hStore, 'moistureContent' : theta}
savemat("test1.mat", saveVariables)
