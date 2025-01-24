## Reproduces the second benchmark test from Celia et al. 1990 Figure 3 (doi: 10.1029/WR026i007p01483)

import os, sys
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from scipy.io import savemat
import numpy as np
from gadopt import *
from src.RichardsSolver import *
import time

solverParameters = {
  "domainLength" : 100.00,   # length of domain (cms)
  "nodes"        : 400,
  "finalTime"    : 86400,       # final time (secs)
  "maxdt"        : 30,
  "mindt"        : 0.01,
  "dtTol"        : 0.10,
  "imexParameter" : 0.5,
  "solverType" : "Newton"    # Picard or Newton
}

# Dirichlet boundary conditions
bTop     = -75.0   # Pressure head value at z = 0
bBottom     = -1000    # Pressure head value at z = L

mesh  = IntervalMesh(solverParameters["nodes"], -solverParameters["domainLength"], 0, name='MeshA')
mesh.cartesian = True
x     = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

h0   = Function(V, name="InitialCondition")
h0.interpolate( bTop - (bBottom - bTop)*tanh(1*x[0]) )

def C(h):

    n = 2
    m = 0.5
    alpha = 0.0335 
    thetaS = 0.368
    thetaR = 0.102

    C =  -(thetaS - thetaR) * n * m * h * (alpha**n) * (abs(h)**(n-2)) * (( alpha**n * abs(h)**n + 1)**(-m-1) )

    return( C )

def K(h):

    n = 2
    m = 0.5
    Ks = 0.00922
    alpha = 0.0335 
    return Ks * (1 - (alpha*abs(h))**(n-1)* (1 + pow(alpha*abs(h), n))**(-m))**2 / (( 1 + pow(alpha*abs(h), n) )**(m/2))

def moistureContent(h):

    n = 2
    m = 0.5
    alpha = 0.0335 
    thetaS = 0.368
    thetaR = 0.102

    return( thetaR + (thetaS - thetaR) / (1 + abs(alpha*h)**n)**m )

def boundaryConditionTop():
   
   # Robin BC of the form
   #  A*u + B*dhdz = C
   # at z = 0

  A = 1
  B = 0
  C = -75
   
  return A, B, C

def boundaryConditionBottom():
   
   # Robin BC of the form
   #  A*u + B*dhdz = C
   # at z = -L

  A = 1
  B = 0
  C = -1000
   
  return A, B, C

h, tStore, zStore, hStore = RichardsSolver( h0, V, v, mesh, solverParameters, boundaryConditionTop, boundaryConditionBottom, C, K )

theta        = moistureContent(hStore)
permeability = K(hStore)

saveVariables = {'tStore' : tStore, 'z' : zStore, 'pressureHead' : hStore, 'moistureContent' : theta, 'hydraulicConductivity' : permeability}
savemat("test2.mat", saveVariables)
