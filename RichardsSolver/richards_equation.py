import firedrake as fd

import ufl
from abc import ABC, abstractmethod
from typing import Dict, Any
from RichardsSolver.utilities import CombinedSurfaceMeasure


class RichardsSolver(ABC):

    """
    Base class for Richards equation solvers. 
    Handles: 
       - mesh and function space setup 
       - facet/volume measures (extruded or not) 
       - solver parameter selection 
    """

    def __init__(self,
                 V: fd.FunctionSpace,
                 W: fd.VectorFunctionSpace,
                 mesh: fd.mesh,
                 soil_curves: Dict,
                 bcs: Dict,
                 solver_parameters='iterative',
                 time_integrator="BackwardEuler",
                 quad_degree=0):

        self.mesh = mesh
        self.trial_space = V
        self.test_function = fd.TestFunction(V)

        self.dim = mesh.topological_dimension()
        self.n = fd.FacetNormal(mesh)

        self.q = fd.Function(W, name="VolumetricFlux")
        self.time_integrator = time_integrator
        self.soil_curves = soil_curves
        self.bcs = bcs

        if quad_degree == 0:
            
            degree = V.ufl_element().degree()
            if not isinstance(degree, int):
                degree = max(degree)
            quad_degree = 2 * max(V.ufl_element().degree()) + 1
        self.quad_degree = quad_degree

        # Measures (extruded vs non-extruded)
        measure_kwargs = {"domain": self.mesh, "degree": quad_degree}
        self.dx = fd.dx(**measure_kwargs)

        if self.trial_space.extruded:
            self.ds = CombinedSurfaceMeasure(**measure_kwargs)
            self.dS = fd.dS_v(**measure_kwargs) + fd.dS_h(**measure_kwargs)
        else:
            self.dS = fd.Measure("dS", domain=mesh, metadata={"quadrature_degree": quad_degree})
            self.ds = fd.Measure("ds", domain=mesh, metadata={"quadrature_degree": quad_degree})

        if solver_parameters == "direct":
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": 'preonly',
                "pc_type": 'lu',
                "pc_factor_mat_solver_type": "mumps"
                }
            
        elif solver_parameters == "iterative":
            self.solver_parameters = {
                "mat_type": "aij",
                "ksp_type": 'gmres',
                "pc_type": 'bjacobi',
                }
            
        else:
            self.solver_parameters = solver_parameters
