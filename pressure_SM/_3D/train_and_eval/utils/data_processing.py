"""
Data interpolation, normalization, and grid operations.
"""

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


@njit(parallel=True)
def interpolate_fill_njit(
    values: np.ndarray,
    vtx: np.ndarray,
    wts: np.ndarray,
    fill_value: float = np.nan) -> np.ndarray:
    """
    Interpolate based on previously computed vertices (vtx) and weights (wts) and fill.
    """
    n = vtx.shape[0]
    ret = np.empty(n, dtype=values.dtype)
    
    for i in prange(n):
        # Check for negative weights
        has_negative = False
        for j in range(wts.shape[1]):
            if wts[i, j] < 0:
                has_negative = True
                break
        
        if has_negative:
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


def create_uniform_grid(limits, grid_res):
    """
    Creates an uniform 3D grid (should envolve every cell of the mesh).

    Args:
        limits (dict): Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        grid_res (float): Grid resolution
        
    Returns:
        tuple: Three flattened arrays (XX0, YY0, ZZ0) representing the grid coordinates
    """
    X0 = np.linspace(
        limits['x_min'] + grid_res/2,
        limits['x_max'] - grid_res/2,
        num = int(round( (limits['x_max'] - limits['x_min'])/grid_res )) 
    )
    Y0 = np.linspace(
        limits['y_min'] + grid_res/2,
        limits['y_max'] - grid_res/2,
        num = int(round( (limits['y_max'] - limits['y_min'])/grid_res )) 
    )
    Z0 = np.linspace(
        limits['z_min'] + grid_res/2,
        limits['z_max'] - grid_res/2,
        num = int(round( (limits['z_max'] - limits['z_min'])/grid_res )) 
    )

    XX0, YY0, ZZ0 = np.meshgrid(X0, Y0, Z0)
    return XX0.flatten(), YY0.flatten(), ZZ0.flatten()


def unison_shuffled_copies(a, b):
    """Shuffle two arrays in unison."""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
    

def normalize_feature_data(input, output, standardization_method: str = "std"):
    """
    Normalize input and output data using different standardization methods.
    
    Args:
        input (ndarray): Input features
        output (ndarray): Output targets
        standardization_method (str): Method to use ('std', 'min_max', or 'max_abs')
        
    Returns:
        tuple: Normalized (x, y) arrays
    """
    if standardization_method == 'min_max':
        ## Option 2: Min-max scaling
        min_in = np.min(input, axis=0)
        max_in = np.max(input, axis=0)

        min_out = np.min(output, axis=0)
        max_out = np.max(output, axis=0)

        np.savez('min_max_values.npz', min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)

        # Perform min-max scaling
        x = (input - min_in) / (max_in - min_in)
        y = (output - min_out) / (max_out - min_out)
        
    elif standardization_method == 'std':
        ## Option 1: Standardization
        mean_in = np.mean(input, axis=0)
        std_in = np.std(input, axis=0)

        mean_out = np.mean(output, axis=0)
        std_out = np.std(output, axis=0)

        np.savez('mean_std.npz', mean_in=mean_in, std_in=std_in, mean_out=mean_out, std_out=std_out)

        x = (input - mean_in) /std_in
        y = (output - mean_out) /std_out

    elif standardization_method == 'max_abs':
        # Option 3 - Old method
        max_abs_input_PCA = np.max(np.abs(input))
        max_abs_p_PCA = np.max(np.abs(output))
        print( max_abs_input_PCA, max_abs_p_PCA)

        np.savetxt('maxs_PCA', [max_abs_input_PCA, max_abs_p_PCA] )

        x = input/max_abs_input_PCA
        y = output/max_abs_p_PCA

    return x, y
