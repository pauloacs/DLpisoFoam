"""
Data interpolation, normalization, and grid operations.
"""

import os
import numpy as np
from scipy.spatial import qhull
from scipy.spatial import cKDTree
import sklearn
from numba import njit, prange


def interp_weights(
    xyz: np.ndarray,
    uvw: np.ndarray,
    d: int=3,
    interp_method='IDW'):
    """
    Get interpolation weights and vertices using barycentric interpolation.

    This function calculates the interpolation weights and vertices for interpolating values from the original grid to the target grid.
    The interpolation is performed using Delaunay triangulation.

    Args:
        xyz (ndarray): Coordinates of the original grid.
        uvw (ndarray): Coordinates of the target grid.
        d (int, optional): Number of dimensions. Default is 3.
        interp_method (str): Interpolation method, 'IDW' or 'barycentric'

    Returns:
        ndarray: Vertices of the simplices that contain the target grid points.
        ndarray: Interpolation weights for each target grid point.
    """
    # For 3D data, baricentric interpolation is very slow - so IDW is the default

    if interp_method == "IDW":
        tree = cKDTree(xyz)
        nndist, nni = tree.query(np.array(uvw), k=4, workers=-1)
        vertices = nni
        #vertices = list(nni)
        # IDW interpolation weights - two options:
        # 1. Linear distance (similar to barycentric interpolation - smoother, more balanced)
        wts = (1./np.maximum(nndist, 1e-6)) / (1./np.maximum(nndist, 1e-6)).sum(axis=-1)[:,None]
        #wts = list(wts)
        # 2. Squared distance (more peaked - gives stronger weight to nearest point)
        # wts = list((1./np.maximum(nndist**2, 1e-6)) / (1./np.maximum(nndist**2, 1e-6)).sum(axis=-1)[:,None])

    elif interp_method == "barycentric":
        tri = qhull.Delaunay(xyz)
        simplex = tri.find_simplex(uvw)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        valid = ~(simplex == -1)

        import dask.array as da
        from scipy.spatial import Delaunay

        # Convert input arrays to dask arrays
        xyz_dask = da.from_array(xyz, chunks='auto')
        uvw_dask = da.from_array(uvw, chunks='auto')

        # Perform Delaunay triangulation
        tri = Delaunay(xyz)

        # Find the simplex containing each point in uvw
        simplex = da.map_blocks(tri.find_simplex, uvw_dask, dtype=int)
        vertices = da.map_blocks(np.take, tri.simplices, simplex, axis=0, dtype=int)
        temp = da.map_blocks(np.take, tri.transform, simplex, axis=0, dtype=float)
        delta = uvw_dask - temp[:, d]
        bary = da.einsum('njk,nk->nj', temp[:, :d, :], delta)
        wts = da.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        valid = ~(simplex == -1)

        # Compute the results
        vertices = vertices.compute()
        wts = wts.compute()
        valid = valid.compute()

        # Fill out-of-bounds points with Inverse-Distance Weighting
        if (~valid).any():
            tree = sklearn.neighbors.KDTree(xyz, leaf_size=40)
            nndist, nni = tree.query(np.array(uvw)[~valid], k=3)
            invalid = np.flatnonzero(~valid)
            vertices[invalid] = list(nni)
            wts[invalid] = list((1./np.maximum(nndist**2, 1e-6)) / (1./np.maximum(nndist**2, 1e-6)).sum(axis=-1)[:,None])

    return vertices, wts


def trilinear_interp_weights(grid_points, mesh_points):
    axes = [np.unique(grid_points[:, component]) for component in range(3)]
    if any(axis.size < 2 for axis in axes):
        raise ValueError('Trilinear interpolation requires at least two nodes per axis.')
    node_shape = tuple(axis.size for axis in axes)
    if np.prod(node_shape) != len(grid_points):
        raise ValueError('Trilinear interpolation requires a complete Cartesian grid.')
    grid_indices = tuple(
        np.searchsorted(axis, grid_points[:, component])
        for component, axis in enumerate(axes)
    )
    node_lookup = np.empty(node_shape, dtype=np.int64)
    node_lookup[grid_indices] = np.arange(len(grid_points))
    lower_indices = []
    fractions = []
    for component, axis in enumerate(axes):
        coordinates = np.clip(mesh_points[:, component], axis[0], axis[-1])
        lower = np.clip(np.searchsorted(axis, coordinates, side='right') - 1, 0, axis.size - 2)
        lower_indices.append(lower)
        fractions.append((coordinates - axis[lower]) / (axis[lower + 1] - axis[lower]))
    vertices = []
    weights = []
    for offset_x in (0, 1):
        for offset_y in (0, 1):
            for offset_z in (0, 1):
                offsets = (offset_x, offset_y, offset_z)
                vertices.append(node_lookup[tuple(
                    lower + offset for lower, offset in zip(lower_indices, offsets)
                )])
                weight = np.ones(len(mesh_points), dtype=np.float64)
                for fraction, offset in zip(fractions, offsets):
                    weight *= fraction if offset else 1.0 - fraction
                weights.append(weight)
    return np.column_stack(vertices), np.column_stack(weights)


def trilinear_scatter_weights(mesh_points, grid_points, cell_volumes):
    """Build oracle-style volume-weighted eight-node restriction; unsupported nodes return zero."""
    cell_volumes = np.asarray(cell_volumes, dtype=np.float64)
    if cell_volumes.shape != (len(mesh_points),) or not np.all(np.isfinite(cell_volumes)) or np.any(cell_volumes <= 0):
        raise ValueError('Trilinear scatter requires finite positive cell volumes for every mesh cell.')
    nodes, weights = trilinear_interp_weights(grid_points, mesh_points)
    destinations = nodes.ravel()
    sources = np.repeat(np.arange(len(mesh_points)), 8)
    contributions = (weights * cell_volumes[:, None]).ravel()
    occupied = contributions > 0
    destinations, sources, contributions = destinations[occupied], sources[occupied], contributions[occupied]
    order = np.argsort(destinations, kind='stable')
    destinations, sources, contributions = destinations[order], sources[order], contributions[order]
    counts = np.bincount(destinations, minlength=len(grid_points))
    totals = np.bincount(destinations, weights=contributions, minlength=len(grid_points))
    slots = np.arange(len(destinations)) - np.repeat(np.cumsum(counts) - counts, counts)
    vertices = np.zeros((len(grid_points), max(1, int(counts.max()))), dtype=np.int64)
    normalized = np.zeros(vertices.shape, dtype=np.float64)
    vertices[destinations, slots] = sources
    normalized[destinations, slots] = contributions / totals[destinations]
    return vertices, normalized


def mesh_to_grid_weights(mesh_points, grid_points, cell_volumes):
    """Use trilinear scatter with linear mesh interpolation/extrapolation for unsupported nodes."""
    from scipy.spatial import Delaunay

    vertices, weights = trilinear_scatter_weights(mesh_points, grid_points, cell_volumes)
    missing = ~np.any(weights > 0, axis=1)
    if not np.any(missing):
        return vertices, weights
    triangulation = Delaunay(mesh_points)
    targets = grid_points[missing]
    simplices = triangulation.find_simplex(targets)
    valid = np.all(np.isfinite(triangulation.transform), axis=(1, 2))
    outside = simplices < 0
    if np.any(outside):
        boundary = np.flatnonzero(np.any(triangulation.neighbors < 0, axis=1) & valid)
        if not len(boundary):
            raise ValueError('No finite boundary tetrahedra available for mesh extrapolation.')
        centres = mesh_points[triangulation.simplices[boundary]].mean(axis=1)
        nearest = cKDTree(centres).query(targets[outside])[1]
        simplices[outside] = boundary[nearest]
    if not np.all(valid[simplices]):
        raise ValueError('Mesh interpolation selected a degenerate tetrahedron.')
    transform = triangulation.transform[simplices]
    barycentric = np.einsum('nij,nj->ni', transform[:, :3], targets - transform[:, 3])
    fill_weights = np.column_stack((barycentric, 1. - barycentric.sum(axis=1)))
    if vertices.shape[1] < 4:
        padding = ((0, 0), (0, 4 - vertices.shape[1]))
        vertices, weights = np.pad(vertices, padding), np.pad(weights, padding)
    vertices[missing] = 0
    weights[missing] = 0.
    vertices[missing, :4] = triangulation.simplices[simplices]
    weights[missing, :4] = fill_weights
    return vertices, weights


@njit(parallel=True)
def interpolate_fill_njit(
    values: np.ndarray,
    vtx: np.ndarray,
    wts: np.ndarray,
    fill_value: float = np.nan) -> np.ndarray:
    """
    Evaluate interpolation/extrapolation stencils, filling only nonfinite weights.
    """
    n = vtx.shape[0]
    ret = np.empty(n, dtype=values.dtype)
    
    for i in prange(n):
        has_invalid = False
        for j in range(wts.shape[1]):
            if not np.isfinite(wts[i, j]):
                has_invalid = True
                break
        
        if has_invalid:
            ret[i] = fill_value
        else:
            # Manual dot product
            val = 0.0
            for j in range(vtx.shape[1]):
                val += values[vtx[i, j]] * wts[i, j]
            ret[i] = val
    
    return ret


#@njit
def interpolate_fill(
    values: np.ndarray,
    vtx: np.ndarray,
    wts: np.ndarray,
    fill_value = np.nan) -> np.ndarray:
    """
    Interpolate based on previously computed vertices (vtx) and weights (wts) and fill.

    Args:
        values (NDArray): Array of values to interpolate.
        vtx (NDArray): Array of interpolation vertices.
        wts (NDArray): Array of interpolation weights.
        fill_value (float): Value used to fill.
    
    Returns:
        NDArray: Interpolated values with fill_value for invalid weights.
    """
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(np.array(wts) < 0, axis=1)] = fill_value
    return ret


#@njit(nopython = True)  #much faster using numba.njit but is giving an error
def index(array, item):
    """
    Finds the index of the first element equal to item.

    Args:
        array (NDArray):
        item (float):
    """
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.


def _unpack_grid_res(grid_res):
    """Unpack grid_res to per-axis (dx, dy, dz).

    Accepts either a scalar float (uniform resolution) or a length-3
    sequence ``(dx, dy, dz)`` giving the resolution for the x, y, and z
    directions respectively.  Scalar values are broadcast to all three axes
    for backward compatibility.
    """
    if isinstance(grid_res, (tuple, list, np.ndarray)):
        dx, dy, dz = float(grid_res[0]), float(grid_res[1]), float(grid_res[2])
    else:
        dx = dy = dz = float(grid_res)
    return dx, dy, dz


def get_grid_shape(limits: dict, grid_res, oracle_aligned=False) -> tuple[int, int, int]:
    """
    Compute the number of grid points along each axis for a uniform grid
    that fully covers the domain (extends half a cell beyond each limit).
    With oracle_aligned=True, use ceil(span/spacing)+1 nodes from the mesh minimum.

    Args:
        limits (dict): Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        grid_res: Grid resolution — either a scalar float (same for all axes)
                  or a (dx, dy, dz) tuple for per-direction resolution.

    Returns:
        tuple: (nx, ny, nz) number of grid points along x, y, z axes
    """
    dx, dy, dz = _unpack_grid_res(grid_res)

    if oracle_aligned:
        counts = []
        for axis, spacing in zip(('x', 'y', 'z'), (dx, dy, dz)):
            ratio = (limits[f'{axis}_max'] - limits[f'{axis}_min']) / spacing
            nearest = np.rint(ratio)
            if abs(ratio - nearest) <= 64 * np.finfo(float).eps * max(1., abs(ratio)):
                ratio = nearest
            counts.append(max(2, int(np.ceil(ratio)) + 1))
        return tuple(counts)

    x_start = limits['x_min'] - dx / 2
    x_end   = limits['x_max'] + dx / 2
    y_start = limits['y_min'] - dy / 2
    y_end   = limits['y_max'] + dy / 2
    z_start = limits['z_min'] - dz / 2
    z_end   = limits['z_max'] + dz / 2

    nx = int(round((x_end - x_start) / dx) + 1)
    ny = int(round((y_end - y_start) / dy) + 1)
    nz = int(round((z_end - z_start) / dz) + 1)
    return nx, ny, nz


def create_uniform_grid(limits: dict, grid_res, oracle_aligned=False):
    """
    Creates a uniform 3D grid that fully covers the domain
    (extends half a cell beyond each limit in every direction).
    With oracle_aligned=True, start at the mesh minimum and keep exact spacing.

    Args:
        limits (dict): Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        grid_res: Grid resolution — either a scalar float (same for all axes)
                  or a (dx, dy, dz) tuple for per-direction resolution.

    Returns:
        tuple: Three flattened arrays (XX0, YY0, ZZ0) representing the grid coordinates
    """
    dx, dy, dz = _unpack_grid_res(grid_res)
    nx, ny, nz = get_grid_shape(limits, grid_res, oracle_aligned=oracle_aligned)

    if oracle_aligned:
        axes = [limits[f'{axis}_min'] + np.arange(count) * spacing
                for axis, count, spacing in zip(('x', 'y', 'z'), (nx, ny, nz), (dx, dy, dz))]
        grid_x, grid_y, grid_z = np.meshgrid(*axes)
        return grid_x.ravel(), grid_y.ravel(), grid_z.ravel()

    X0 = np.linspace(limits['x_min'] - dx / 2, limits['x_max'] + dx / 2, num=nx)
    Y0 = np.linspace(limits['y_min'] - dy / 2, limits['y_max'] + dy / 2, num=ny)
    Z0 = np.linspace(limits['z_min'] - dz / 2, limits['z_max'] + dz / 2, num=nz)

    XX0, YY0, ZZ0 = np.meshgrid(X0, Y0, Z0)
    return XX0.flatten(), YY0.flatten(), ZZ0.flatten()


def unison_shuffled_copies(a, b):
    """Shuffle two arrays in unison."""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
    

def normalize_feature_data(input,
                            output,
                            standardization_method: str = "std",
                            normalization_factors_fn: str = "mean_std.npz",
                            load_existing: bool = False,
                            flatten_data: bool = False):
    """
    Normalize input and output data using different standardization methods.
    
    Args:
        input (ndarray): Input features
        output (ndarray): Output targets
        standardization_method (str): Method to use ('std', 'min_max', or 'max_abs')
        load_existing (bool): If True and the normalization file exists, load and reuse
                              the saved statistics instead of recomputing them.
        
    Returns:
        tuple: Normalized (x, y) arrays
    """
    if load_existing and os.path.exists(normalization_factors_fn):
        print(f'[normalize_feature_data] Loading existing normalization stats from {normalization_factors_fn}')
        data = np.load(normalization_factors_fn)
        try:
            if standardization_method == 'std':
                mean_in  = data['mean_in'];  std_in  = data['std_in']
                mean_out = data['mean_out']; std_out = data['std_out']
                x = (input  - mean_in)  / (std_in + 1e-8)
                y = (output - mean_out) / (std_out + 1e-8)
            elif standardization_method == 'min_max':
                min_in = data['min_in']; max_in = data['max_in']
                min_out = data['min_out']; max_out = data['max_out']
                x = (input  - min_in)  / (max_in  - min_in)
                y = (output - min_out) / (max_out - min_out)
            elif standardization_method == 'max_abs':
                max_abs_input_PCA = data['max_abs_input_PCA']
                max_abs_p_PCA     = data['max_abs_p_PCA']
                x = input  / max_abs_input_PCA
                y = output / max_abs_p_PCA
            return x, y
        except ValueError as e:
            # Channel layouts can change when add_* input flags are toggled.
            # In that case, stale stats are incompatible and must be recomputed.
            print(
                f"[normalize_feature_data] Existing normalization stats are incompatible with current data shape ({e}). "
                f"Recomputing and overwriting {normalization_factors_fn}."
            )

    if standardization_method == 'min_max':
        ## Option 2: Min-max scaling
        min_in = np.min(input, axis=0)
        max_in = np.max(input, axis=0)

        min_out = np.min(output, axis=0)
        max_out = np.max(output, axis=0)

        np.savez(normalization_factors_fn, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)

        # Perform min-max scaling
        x = (input - min_in) / (max_in - min_in)
        y = (output - min_out) / (max_out - min_out)
        
    elif standardization_method == 'std':
        if flatten_data:
            mean_in = input.mean(axis=0)
            std_in = input.std(axis=0)

            mean_out = output.mean(axis=0)
            std_out = output.std(axis=0)
        else:
            mean_in = input.mean(axis=(0,1,2,3), keepdims=True)
            std_in = input.std(axis=(0,1,2,3), keepdims=True)

            mean_out = output.mean(axis=(0,1,2,3), keepdims=True)
            std_out = output.std(axis=(0,1,2,3), keepdims=True)

        np.savez(normalization_factors_fn, mean_in=mean_in, std_in=std_in, mean_out=mean_out, std_out=std_out)

        x = (input - mean_in) / (std_in + 1e-8)
        y = (output - mean_out) / (std_out + 1e-8)

    elif standardization_method == 'max_abs':
        # Option 3 - Old method
        max_abs_input_PCA = np.max(np.abs(input))
        max_abs_p_PCA = np.max(np.abs(output))
        print( max_abs_input_PCA, max_abs_p_PCA)

        np.savez(normalization_factors_fn, max_abs_input_PCA=max_abs_input_PCA, max_abs_p_PCA=max_abs_p_PCA)

        x = input/max_abs_input_PCA
        y = output/max_abs_p_PCA

    return x, y
