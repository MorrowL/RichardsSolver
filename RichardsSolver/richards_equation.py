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
                 solver_parameters='default',
                 time_integrator="BackwardEuler",
                 source_term=0,
                 quad_degree=0):

        self.mesh = mesh
        self.trial_space = V
        self.test_function = fd.TestFunction(V)

        self.dim = mesh.topological_dimension()
        self.n = fd.FacetNormal(mesh)

        self.q = fd.Function(W, name="VolumetricFlux")
        self.h_star = fd.Function(V, name='ApproximateSolution')

        accepted_solvers = ['BackwardEuler', 'CrankNicolson', 'Picard', 'ImplicitMidpoint', 'SemiImplicit']
        if time_integrator in accepted_solvers:
            self.time_integrator = time_integrator
        else:
            TypeError('Time Integrator not recognised')
        self.soil_curves = soil_curves
        self.bcs = bcs

        self.source_term = source_term

        if quad_degree == 0:
            
            degree = V.ufl_element().degree()
            if not isinstance(degree, int):
                degree = max(degree)
            quad_degree = 2 * degree + 1
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

        if solver_parameters == "default":
            if mesh.topological_dimension() <= 2:
                solver_parameters = 'direct'
            else:
                solver_parameters = 'iterative'

        if solver_parameters == 'direct':
            if time_integrator == 'SemiImplicit':
                self.solver_parameters = {
                    "mat_type": "aij",
                    "ksp_type": 'preonly',
                    "pc_type": 'lu',
                    "pc_factor_mat_solver_type": "mumps",
                    'snes_type': 'ksponly',
                    }
            else:
                self.solver_parameters = {
                    "mat_type": "aij",
                    "ksp_type": 'preonly',
                    "pc_type": 'lu',
                    "pc_factor_mat_solver_type": "mumps",
                    'snes_type': 'newtonls'
                    }
        elif solver_parameters == "iterative":
            if time_integrator == 'SemiImplicit':
                self.solver_parameters = {
                    "mat_type": "aij",
                    "ksp_type": 'gmres',
                    "pc_type": 'bjacobi',
                    'snes_type': 'ksponly',
                    }
            else:
                self.solver_parameters = {
                    "mat_type": "aij",
                    "ksp_type": 'gmres',
                    "pc_type": 'bjacobi',
                    'snes_type': 'newtonls'
                    }
        # User specified solver parameters
        else:
            self.solver_parameters = solver_parameters
