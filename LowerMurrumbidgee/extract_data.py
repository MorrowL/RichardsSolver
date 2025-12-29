from firedrake import *
import numpy as np
from surface_mesh import *

dx, layers = 3000, 400

with CheckpointFile(f"DG11_dx={dx}_layers={layers}.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh")
    h     = afile.load_function(mesh, "PressureHead")
    theta = afile.load_function(mesh, "MoistureContent")
    q     = afile.load_function(mesh, "VolumetricFlux")
    depth = afile.load_function(mesh, "depth")

outfile = VTKFile("lower_murrumbidgee.pvd")
outfile.write(h, theta, q, depth)
