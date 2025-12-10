import numpy as np
import scipy.io
import firedrake as fd
from firedrake.petsc import PETSc
from scipy.interpolate import griddata
import ufl

def data_2_function(mesh_coords, file_name):
    # Takes a data set that defines a value defined at the surface of the mesh and defines a firedrake function from this data

    x_coord = mesh_coords[:, 0]
    y_coord = mesh_coords[:, 1]
    elevation = x_coord*0
    distance = elevation + 100000

    mat = scipy.io.loadmat(file_name)
    x = mat.get('x')
    x_surface = x.flatten()
    y = mat.get('y')
    y_surface = y.flatten()
    z = mat.get('z')
    z_surface = z.flatten()

    points = np.vstack((x_surface, y_surface))
    points = points.T

    elevation = griddata(points, z_surface, (x_coord, y_coord), method='linear')

    return elevation


def interior_penalty_factor(eq, *, shift: int = 0) -> float:
    """Interior Penalty method

    For details on the choice of sigma, see
    https://www.researchgate.net/publication/260085826

    We use Equations (3.20) and (3.23). Instead of getting the maximum over two adjacent
    cells (+ and -), we just sum (i.e. 2 * avg) and have an extra 0.5 for internal
    facets.
    """
    degree = eq.trial_space.ufl_element().degree()
    if not isinstance(degree, int):
        degree = max(degree)

    if degree == 0:  # probably only works for orthogonal quads and hexes
        sigma = 1.0
    else:
        # safety factor: 1.0 is theoretical minimum
        alpha = getattr(eq, "interior_penalty", 2.0)
        num_facets = eq.mesh.ufl_cell().num_facets()
        sigma = alpha * cell_edge_integral_ratio(eq.mesh, degree + shift) * num_facets

    return sigma


def cell_edge_integral_ratio(mesh: fd.MeshGeometry, p: int) -> int:
    r"""
    Ratio C such that \int_f u^2 <= C Area(f)/Volume(e) \int_e u^2 for facets f,
    elements e, and polynomials u of degree p.

    See Equation (3.7), Table 3.1, and Appendix C from Hillewaert's thesis:
    https://www.researchgate.net/publication/260085826
    """

    match cell_type := mesh.ufl_cell().cellname():
        case "triangle":
            return (p + 1) * (p + 2) / 2.0
        case "quadrilateral" | "interval * interval":
            return (p + 1) ** 2
        case "triangle * interval":
            return (p + 1) ** 2
        case "quadrilateral * interval" | "hexahedron":
            # if e is a wedge and f is a triangle: (p+1)**2
            # if e is a wedge and f is a quad: (p+1)*(p+2)/2
            # here we just return the largest of the the two (for p>=0)
            return (p + 1) ** 2
        case "tetrahedron":
            return (p + 1) * (p + 3) / 3
        case _:
            raise NotImplementedError(f"Unknown cell type in mesh: {cell_type}")
        

class CombinedSurfaceMeasure(ufl.Measure):
    """
    A surface measure that combines ds_v, the integral over vertical boundary facets, and ds_t and ds_b,
    the integral over horizontal top and bottom facets. The vertical boundary facets are identified with
    the same surface ids as ds_v. The top and bottom surfaces are identified via the "top" and "bottom" ids."""

    def __init__(self, domain, degree):
        self.ds_v = fd.ds_v(domain=domain, degree=degree)
        self.ds_t = fd.ds_t(domain=domain, degree=degree)
        self.ds_b = fd.ds_b(domain=domain, degree=degree)

    def __call__(self, subdomain_id, **kwargs):
        if subdomain_id == 'top':
            return self.ds_t(**kwargs)
        elif subdomain_id == 'bottom':
            return self.ds_b(**kwargs)
        else:
            return self.ds_v(subdomain_id, **kwargs)

    def __rmul__(self, other):
        """This is to handle terms to be integrated over all surfaces in the form of other*ds.
        Here the CombinedSurfaceMeasure ds is not called, instead we just split it up as below."""
        return other*self.ds_v + other*self.ds_t + other*self.ds_b

