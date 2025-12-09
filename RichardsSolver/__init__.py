from firedrake import *
from firedrake.output import VTKFile

from .richards_solver import (
    richardsSolver,
    mass_term,
    diffusion_term,
    gravity_advection
)

from .richards_equation import (
    RichardsSolver
)

from .soil_curves import (
    SoilCurve,
    HaverkampCurve,
    VanGenuchtenCurve,
    ExponentialCurve,
)

from .utility import (
    data_2_function,
    interior_penalty_factor,
    cell_edge_integral_ratio,
    get_boundary_ids
)
