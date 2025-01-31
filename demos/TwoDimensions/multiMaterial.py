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

solverParameters = {
  "domainDepth"  : 3.00,   # depth of domain (cms)
  "nodesDepth"   : 300,
  "domainWidth"  : 5.0,   # depth of domain (cms)
  "nodesWidth"   : 500,
  "finalTime"    : 100*24*3600,       # final time (s)
  "maxdt"        : 100*24*3600/101,
  "Regularisation" : 0.0, 
  "mindt"        : 0.000001,
  "dtTol"        : 0.10,
  "imexParameter" : 0.0,
  "solverType" : "Newton"    # Picard or Newton
}

mesh = RectangleMesh( solverParameters["nodesWidth"], solverParameters["nodesDepth"], solverParameters["domainWidth"], solverParameters["domainDepth"])

mesh.cartesian = True
x     = SpatialCoordinate(mesh)

# Dirichlet boundary conditions
bTop     = -75.0   # Pressure head value at z = 0
bBottom     = -1000    # Pressure head value at z = L

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

h0   = Function(V, name="InitialCondition")

h0.interpolate( -2.5 )

def setBoundaryConditions(timeConstant):

    c = 0.5*(1 - tanh(timeConstant - 86000))

    leftBC, rightBC, bottomBC, topBC = 1,2,3,4
    boundaryCondition = {
    leftBC   : {'q' : 0 },
    rightBC  : {'q' : 0 },
    bottomBC : {'q' : 0 },    # cm/s
    topBC    : {'q' : c*10e-07},
    }

    return boundaryCondition

def C(h, x):

    eps = 25;
    Ix = (1 + 0.5*tanh(eps*(x[0]-1)) ) - (1 + 0.5*tanh(eps*(x[0]-4)) )
    Iz = 0.5*(1 + tanh(eps*(x[1]-1)))  - 0.5*(1 + tanh(eps*(x[1] - 3)));
    I3 = 0.5*(1 + tanh(eps*(1.5 - x[1] - 0.75*cos(x[0]+0.6))));
    I = 1 - Ix*Iz*I3
    
    n = 2
    m = 1 - 1/n
    Ks = 0.0000583
    alpha = 3.40
    thetaS = 0.38
    thetaR = 0.1

    I = 0.5*(1 + tanh(-1000*h))

    C =  -(thetaS - thetaR) * n * m * h * (alpha**n) * (abs(h)**(n-2)) * (( alpha**n * abs(h)**n + 1)**(-m-1) ) * I

    return C

def K(h, x):

    eps = 25;
    Ix = (1 + 0.5*tanh(eps*(x[0]-1)) ) - (1 + 0.5*tanh(eps*(x[0]-4)) )
    Iz = 0.5*(1 + tanh(eps*(x[1]-1)))  - 0.5*(1 + tanh(eps*(x[1] - 3)));
    I3 = 0.5*(1 + tanh(eps*(1.5 - x[1] - 0.75*cos(x[0]+0.6))));
    I = 1 - Ix*Iz*I3

    n = 2
    m = 1 - 1/n
    Ks = 5.8300e-05*I + 5.8300e-07*(1 - I)
    alpha = 3.40
    thetaS = 0.38
    thetaR = 0.1

    I = 0.5*(1 + tanh(-1000*h))

    K =  Ks * (1 - (alpha*abs(h))**(n-1)* (1 + pow(alpha*abs(h), n))**(-m))**2 / (( 1 + pow(alpha*abs(h), n) )**(m/2))

    return K

def moistureContent(h, x):  

    eps = 25;
    Ix = (1 + 0.5*tanh(eps*(x[0]-1)) ) - (1 + 0.5*tanh(eps*(x[0]-4)) )
    Iz = 0.5*(1 + tanh(eps*(x[1]-1)))  - 0.5*(1 + tanh(eps*(x[1] - 3)));
    I3 = 0.5*(1 + tanh(eps*(1.5 - x[1] - 0.75*cos(x[0]+0.6))));
    I = 1 - Ix*Iz*I3

    n = 2
    m = 1 - 1/n
    Ks = 0.0000583
    alpha = 3.40
    thetaS = 0.38
    thetaR = 0.1

    return( thetaR + (thetaS - thetaR) / (1 + abs(alpha*h)**n)**m )

start = time.time()
h, tStore, hStore, thetaStore, zStore, xStore = RichardsSolver( h0, V, v, mesh, solverParameters, setBoundaryConditions, C, K, moistureContent )
end = time.time()
print(end - start)

saveVariables = {'tStore' : tStore, 'z' : zStore, 'x' : xStore,  'pressureHead' : hStore, 'moistureContent' : thetaStore}
savemat("test4.mat", saveVariables)
