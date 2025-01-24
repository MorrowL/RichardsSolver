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
  "domainDepth"  : 125.00,   # depth of domain (cms)
  "nodesDepth"   : 150,
  "domainWidth"  : 125.0,   # depth of domain (cms)
  "nodesWidth"   : 150,
  "finalTime"    : 2*24*3600,       # final time (s)
  "maxdt"        : 2*24*3600/200,
  "mindt"        : 0.01,
  "dtTol"        : 0.10,
  "imexParameter" : 0.50,
  "solverType" : "Newton"    # Picard or Newton
}

mesh = Mesh('t1.msh')

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

import matplotlib.pyplot as plt
fig, axes = plt.subplots()
firedrake.triplot(mesh, axes=axes)
axes.legend();

plt.savefig('Mesh.png', dpi=300)

bottomBC, topBC = 1,2
boundaryCondition = {
   bottomBC : {'q' : 0},    # cm/s
   topBC    : {'q' : 5.0e-05},
}

# Dirichlet boundary conditions
bTop     = -75.0   # Pressure head value at z = 0
bBottom     = -1000    # Pressure head value at z = L

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

h0   = Function(V, name="InitialCondition")

h0.interpolate( -50.0 )

def C(h, x):

    I = 1;

    n = 1.592*I + 1.474*(1-I)
    m = 1 - 1/n
    Ks = 0.00583*I + 0.0*(1 - I)
    alpha = 0.138*I + 0.115*(1-I) 
    thetaS = 0.417*I + 0.417*(1-I)
    thetaR = 0.0286*I + 0.0286*(1-I)

    C =  -(thetaS - thetaR) * n * m * h * (alpha**n) * (abs(h)**(n-2)) * (( alpha**n * abs(h)**n + 1)**(-m-1) )

    return( C )

def K(h, x):

    
    I = 1;

    n = 1.592*I + 1.474*(1-I)
    m = 1 - 1/n
    Ks = 0.00583*I + 0.0*(1 - I)
    alpha = 0.138*I + 0.115*(1-I) 
    thetaS = 0.417*I + 0.417*(1-I)
    thetaR = 0.0286*I + 0.0286*(1-I)

    return Ks * (1 - (alpha*abs(h))**(n-1)* (1 + pow(alpha*abs(h), n))**(-m))**2 / (( 1 + pow(alpha*abs(h), n) )**(m/2))

def moistureContent(h, x):

    I = 1;

    n = 1.592*I + 1.474*(1-I)
    m = 1 - 1/n
    Ks = 0.00583*I + 0.0*(1 - I)
    alpha = 0.138*I + 0.115*(1-I) 
    thetaS = 0.417*I + 0.417*(1-I)
    thetaR = 0.0286*I + 0.0286*(1-I)

    return( thetaR + (thetaS - thetaR) / (1 + abs(alpha*h)**n)**m )

start = time.time()
h, tStore, hStore, thetaStore, zStore, xStore = RichardsSolver( h0, V, v, mesh, solverParameters, boundaryCondition, C, K, moistureContent )
end = time.time()
print(end - start)

saveVariables = {'tStore' : tStore, 'z' : zStore, 'x' : xStore,  'pressureHead' : hStore, 'moistureContent' : thetaStore}
savemat("test3.mat", saveVariables)
