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
  "domainDepth"  : 300.00,   # depth of domain (cms)
  "nodesDepth"   : 120,
  "domainWidth"  : 500.0,   # depth of domain (cms)
  "nodesWidth"   : 200,
  "finalTime"    : 24*3600,       # final time (s)
  "maxdt"        : 24*3600/200,
  "mindt"        : 0.01,
  "dtTol"        : 0.10,
  "imexParameter" : 0.50,
  "solverType" : "Newton"    # Picard or Newton
}

mesh = RectangleMesh( solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["domainWidth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

bottomBC, topBC = 1,2
boundaryCondition = {
   bottomBC : {'q' : 0.0},    # cm/s
   topBC    : {'q' : 5e-04*((1 + 0.5*tanh(0.125*(x[0]-150)) ) - (1 + 0.5*tanh(0.125*(x[0]-350)) ))},
}

# Dirichlet boundary conditions
bTop     = -75.0   # Pressure head value at z = 0
bBottom     = -1000    # Pressure head value at z = L

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

h0   = Function(V, name="InitialCondition")

eps = 0.125;
Ix = (1 + 0.5*tanh(eps*(x[0]-100)) ) - (1 + 0.5*tanh(eps*(x[0]-400)) )
Iz = 0.5*(1 + tanh(eps*(x[1]-200)))
I = Ix*Iz

h0.interpolate( -200 )

def C(h, x):

    eps = 0.125;
    Ix = (1 + 0.5*tanh(eps*(x[0]-100)) ) - (1 + 0.5*tanh(eps*(x[0]-400)) )
    Iz = 0.5*(1 + tanh(eps*(x[1]-200)))
    I = Ix*Iz

    n = 2.2390*I + 2.2390*(1-I)
    m = 1 - 1/n
    Ks = 0.006262 *I + 1.516e-04 *(1 - I)
    alpha = 0.0280*I + 0.0280*(1-I) 
    thetaS = 0.3658*I + 0.3658*(1-I)
    thetaR = 0.100*I + 0.100*(1-I)

    C =  -(thetaS - thetaR) * n * m * h * (alpha**n) * (abs(h)**(n-2)) * (( alpha**n * abs(h)**n + 1)**(-m-1) )

    return( C )

def K(h, x):

    eps = 0.125;
    Ix = (1 + 0.5*tanh(eps*(x[0]-100)) ) - (1 + 0.5*tanh(eps*(x[0]-400)) )
    Iz = 0.5*(1 + tanh(eps*(x[1]-200)))
    I = Ix*Iz

    n = 2.2390*I + 2.2390*(1-I)
    m = 1 - 1/n
    Ks = 0.006262 *I + 1.516e-04 *(1 - I)
    alpha = 0.0280*I + 0.0280*(1-I) 
    thetaS = 0.3658*I + 0.3658*(1-I)
    thetaR = 0.100*I + 0.100*(1-I)

    return Ks * (1 - (alpha*abs(h))**(n-1)* (1 + pow(alpha*abs(h), n))**(-m))**2 / (( 1 + pow(alpha*abs(h), n) )**(m/2))

def moistureContent(h, x):

    eps = 0.125;
    Ix = (1 + 0.5*tanh(eps*(x[0]-100)) ) - (1 + 0.5*tanh(eps*(x[0]-400)) )
    Iz = 0.5*(1 + tanh(eps*(x[1]-200)))
    I = Ix*Iz

    n = 2.2390*I + 2.2390*(1-I)
    m = 1 - 1/n
    Ks = 0.006262 *I + 1.516e-04 *(1 - I)
    alpha = 0.0280*I + 0.0280*(1-I) 
    thetaS = 0.3658*I + 0.3658*(1-I)
    thetaR = 0.100*I + 0.100*(1-I)

    return( thetaR + (thetaS - thetaR) / (1 + abs(alpha*h)**n)**m )

start = time.time()
h, tStore, hStore, thetaStore, zStore, xStore = RichardsSolver( h0, V, v, mesh, solverParameters, boundaryCondition, C, K, moistureContent )
end = time.time()
print(end - start)

saveVariables = {'tStore' : tStore, 'z' : zStore, 'x' : xStore,  'pressureHead' : hStore, 'moistureContent' : thetaStore}
savemat("test4.mat", saveVariables)
