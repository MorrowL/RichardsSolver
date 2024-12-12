## Reproduces the first benchmark test from Celia 1990 (Figure 1)

import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from gadopt import *
from RichardsSolver import *
import time

domainLength = 40.00    # length of domain (cms)
nodes        = 200      # number of spatial nodes

solverParameters = {
  "finalTime" : 360,
  "maxdt"     : 2.5,
  "mindt"     : 0.01,
  "imexParameter" : 0.5,
  "solverType" : "Newton"    # Picard or Newton
}

# Dirichlet boundary conditions
b0     = -20.70    # Pressure head value at z = 0
bL     = -61.50    # Pressure head value at z = L

def C(h):

  alpha  = 1.611e06
  beta   = 3.96
  thetaS = 0.287
  thetaR = 0.075

  return( -sign(h)*alpha*beta*(thetaS - thetaR)*pow(abs(h),beta-1) ) / ( pow(alpha + pow(abs(h),beta),2) )

def K(h):

  A      = 1.175e06
  gamma  = 4.74
  Ks     = 9.44e-03

  return Ks*A / ( A + pow(abs(h), gamma) )

def moistureContent(h):

  alpha  = 1.611e06
  beta   = 3.96
  thetaS = 0.287
  thetaR = 0.075

  return( thetaR + alpha*(thetaS - thetaR) / (alpha + abs(h)**beta) )

mesh  = IntervalMesh(nodes, domainLength)
x     = SpatialCoordinate(mesh)
xCont = np.linspace(0, domainLength, nodes)

V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)

h0   = Function(V, name="InitialCondition")
h0.interpolate( b0 + (bL - b0)*tanh(1*x[0]) )

start = time.time()
hPicard = RichardsSolverNewton( h0, V, v, mesh, solverParameters, [b0, bL], C, K )
end = time.time(); print("Picard solver " + str(round(end-start,1)) + 'sec')

start = time.time()
solverParameters.update(solverType='Newton')
hNewton = RichardsSolverNewton( h0, V, v, mesh, solverParameters, [b0, bL], C, K )
end = time.time(); print("Newton solver " + str(round(end-start,1)) + 'sec')

hInit = np.array( h0.at(xCont) )
hNewton = np.array( hNewton.at(xCont) )
hPicard = np.array( hPicard.at(xCont) )

fig, ax = plt.subplots()
plt.plot(xCont, hNewton)
plt.plot(xCont, hPicard)
plt.plot(xCont, hInit, 'red')
plt.savefig('Solution.png', bbox_inches='tight')