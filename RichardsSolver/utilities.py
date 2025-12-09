import numpy as np
import scipy.io
import firedrake as fd
from firedrake.petsc import PETSc
from scipy.interpolate import griddata

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


def interior_penalty_factor(eq: Equation, *, shift: int = 0) -> float:
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
        num_facets = eq.mesh.ufl_cell().num_facets
        sigma = alpha * cell_edge_integral_ratio(eq.mesh, degree + shift) * num_facets

    return sigma


def cell_edge_integral_ratio(mesh: fd.MeshGeometry, p: int) -> int:
    r"""
    Ratio C such that \int_f u^2 <= C Area(f)/Volume(e) \int_e u^2 for facets f,
    elements e, and polynomials u of degree p.

    See Equation (3.7), Table 3.1, and Appendix C from Hillewaert's thesis:
    https://www.researchgate.net/publication/260085826
    """
    match cell_type := mesh.ufl_cell().cellname:
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


def get_boundary_ids(mesh) -> BoundaryIDNamespace:
    # PETSc creates these labels when loading meshes from files, Firedrake imitates it
    # in its own mesh creation functions.

    if mesh.topology_dm.hasLabel("Face Sets"):
        axis_extremes_order = [["left", "right"], ["bottom", "top"]]
        dim = mesh.geometric_dimension
        plex_dim = mesh.topology_dm.getCoordinateDim()  # In an extruded mesh, this is different to the
        # firedrake-assigned geometric_dimension
        if dim == 3:
            # For 3D meshes, we label dim[1] (y) as "front", "back" and dim[2] (z) as "bottom","top"
            axis_extremes_order.insert(1, ["front", "back"])
        bounding_box = mesh.topology_dm.getBoundingBox()
        boundary_tol = [abs(dim[1] - dim[0]) * 1e-6 for dim in bounding_box]
        if dim > 3:
            raise ValueError(f"Cannot handle {dim} dimensional mesh")
        # Recover the boundary ids Firedrake has inserted
        coords = mesh.topology_dm.getCoordinatesLocal()
        coord_sec = mesh.topology_dm.getCoordinateSection()
        mesh.topology_dm.markBoundaryFaces("TEMP_LABEL", 1)
        if mesh.topology_dm.getStratumSize("TEMP_LABEL", 1) > 0:
            boundary_cells = [
                (i, mesh.topology_dm.getLabelValue("Face Sets", i))
                for i in mesh.topology_dm.getStratumIS("TEMP_LABEL", 1).getIndices()
            ]
        else:
            boundary_cells = []
        identified = dict.fromkeys([i[1] for i in boundary_cells])
        nvert = -1
        for cell, face_id in boundary_cells:
            if identified[face_id] is None:
                face_coords = mesh.topology_dm.vecGetClosure(coord_sec, coords, cell)
                if nvert == -1:
                    nvert = len(face_coords) // plex_dim
                # Flattened version of axis_extremes_order
                tmp_bdys = [
                    axis_extremes_order[idim][ibdy]
                    for idim in range(plex_dim)
                    for ibdy in range(2)
                    if all(
                        abs(face_coords[idim + plex_dim * i] - bounding_box[idim][ibdy]) < boundary_tol[idim]
                        for i in range(nvert)
                    )
                ]
                # Found a cell unambiguously on one boundary
                if len(tmp_bdys) == 1:
                    identified[face_id] = tmp_bdys[0]
        # Not every MPI rank has every boundary of the mesh, gather all boundaries
        # seen by all MPI ranks
        gathered_boundaries = mesh.comm.allgather(identified)
        mesh.topology_dm.removeLabel("TEMP_LABEL")
        kwargs = dict(
            [
                (proc_bdy.get(face_id), face_id)  # invert boundary mapping
                for proc_bdy in reversed(gathered_boundaries)  # keep value on lowest comm rank
                for face_id in set().union(*gathered_boundaries)  # all gathered face_ids
            ]
        )
        kwargs.pop(None, None)  # remove None entry
        # Get remaining dimensions (if any)
        for idim in range(plex_dim, dim):
            for axis_label in axis_extremes_order[idim]:
                kwargs[axis_label] = axis_label
    # Integer subdomain meshes are only assigned by petsc or the firedrake utility mesh
    # module. If the "Face Sets" label is missing from the mesh, it was not created by
    # either of these and therefore will only have the "bottom" and "top" labels
    # utilised by 'CombinedSurfaceMeasure' above
    else:
        kwargs = {"bottom": "bottom", "top": "top"}
    return BoundaryIDNamespace(**kwargs)
