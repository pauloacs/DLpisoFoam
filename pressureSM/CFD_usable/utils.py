import time
import traceback
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.finalize = False
from mpi4py import MPI

from tensorflow.keras.models import load_model
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Activation, Input, Dense, Dropout

import scipy.spatial.qhull as qhull
from scipy.spatial import distance
import matplotlib.path as mpltPath
from shapely.geometry import MultiPoint
from scipy.spatial import cKDTree
import sklearn.neighbors


def create_uniform_grid(x_min, x_max, y_min, y_max, delta):
    """
    Creates a uniform quadrangular grid encompassing every cell of the mesh.

    Parameters:
    x_min (float): The minimum x-coordinate of the grid.
    x_max (float): The maximum x-coordinate of the grid.
    y_min (float): The minimum y-coordinate of the grid.
    y_max (float): The maximum y-coordinate of the grid.
    delta (float): The spacing between grid points.

    Returns:
    tuple: A tuple containing two arrays representing the x and y coordinates of the grid points.
    """

    X0 = np.linspace(x_min + delta/2, x_max - delta/2, num=int(round((x_max - x_min)/delta)))
    Y0 = np.linspace(y_min + delta/2, y_max - delta/2, num=int(round((y_max - y_min)/delta)))

    XX0, YY0 = np.meshgrid(X0, Y0)
    return XX0.flatten(), YY0.flatten()

# TODO: TRY with opt_einsum.contract
# check if it is faster: https://github.com/dgasmith/opt_einsum
# for small array seems a bit slower, but is essentially the same...
# from opt_einsum import contract 

def interpolate_fill(values, vtx, wts, fill_value=np.nan):
    """
    Interpolates values based on the given vertices and weights.

    Parameters:
    values (ndarray): The array of values to interpolate.
    vtx (ndarray): The array of vertices.
    wts (ndarray): The array of weights.
    fill_value (float, optional): The value to fill when weights are negative. Defaults to np.nan.

    Returns:
    ndarray: The interpolated values.
    """
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ## Filling not being used currently... 
    ## shouldn't be necessary since IDW is applied to extrapolate
    #ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

def domain_dist(top_boundary, obst_boundary, xy0, domain_limits):
    """
    Calculate the fluid-flow domain and the signal distance function (SDF).

    This function calculates a boolean array representing the fluid-flow domain and the signal distance function (SDF).
    The fluid-flow domain is determined by the top boundary and the obstacle boundary.
    The SDF represents the minimum distance from each point in `xy0` to the boundaries.

    Args:
        top_boundary (ndarray): The coordinates of the top boundary.
        obst_boundary (ndarray): The coordinates of the obstacle boundary.
        xy0 (ndarray): The coordinates of the points to calculate the SDF for.
        domain_limits

    Returns:
        ndarray: A boolean array representing the fluid-flow domain.
        ndarray: The signal distance function (SDF) for each point in `xy0`.
    """
    # Calculate the boundaries index
    top = top_boundary
    x_min, x_max, y_min, y_max = domain_limits
    max_x, max_y = np.max([(top[:,0]).max(), x_max]), np.min([(top[:,1]).max(), y_max])
    min_x, min_y = np.max([(top[:,0]).min(), x_min]), np.min([(top[:,1]).min(), y_min])

    is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y )

    obst = obst_boundary
    obst_points =  MultiPoint(obst)
    hull = obst_points.convex_hull
    hull_pts = hull.exterior.coords.xy
    hull_pts = np.c_[hull_pts[0], hull_pts[1]]

    path = mpltPath.Path(hull_pts)
    is_inside_obst = path.contains_points(xy0)
    domain_bool = is_inside_domain * ~is_inside_obst
    # If this causes an OOM error, increase the step
    step = 2
    top = top[0:top.shape[0]:step,:]
    obst = obst[0:obst.shape[0]:step,:]
    sdf = np.minimum( distance.cdist(xy0,obst).min(axis=1) , distance.cdist(xy0,top).min(axis=1) ) * domain_bool

    return domain_bool, sdf

def densePCA(input_shape, PC_p, n_layers, depth=512, dropout_rate=None, regularization=None):
    """
    Creates the MLP NN.
    """
    
    inputs = Input(int(input_shape))
    if len(depth) == 1:
        depth = [depth]*n_layers
    
    # Regularization parameter
    if regularization is not None:
        regularizer = regularizers.l2(regularization)
        print(f'\nUsing L2 regularization. Value: {regularization}\n')
    else:
        regularizer = None
    
    x = Dense(depth[0], activation='relu', kernel_regularizer=regularizer)(inputs)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    
    for i in range(n_layers - 1):
        x = Dense(depth[i+1], activation='relu', kernel_regularizer=regularizer)(x)
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
    
    outputs = Dense(PC_p)(x)

    model = Model(inputs, outputs, name="MLP")
    print(model.summary())

    return model


def memory():
    """
    Get node total memory and memory usage
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret


def interp_barycentric_weights(xyz, uvw, d=2):
    """
    Get interpolation weights and vertices using barycentric interpolation.

    This function calculates the interpolation weights and vertices for interpolating values from the original grid to the target grid.
    The interpolation is performed using Delaunay triangulation.

    Args:
        xyz (ndarray): Coordinates of the original grid.
        uvw (ndarray): Coordinates of the target grid.
        d (int, optional): Number of dimensions. Default is 2.

    Returns:
        ndarray: Vertices of the simplices that contain the target grid points.
        ndarray: Interpolation weights for each target grid point.
    """
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    valid = ~(simplex == -1)

    # Fill out-of-bounds points with Inverse-Distance Weighting
    if (~valid).any():
        tree = sklearn.neighbors.KDTree(xyz, leaf_size=40)
        nndist, nni = tree.query(np.array(uvw)[~valid], k=3)
        invalid = np.flatnonzero(~valid)
        vertices[invalid] = list(nni)
        wts[invalid] = list((1./np.maximum(nndist**2, 1e-6)) / (1./np.maximum(nndist**2, 1e-6)).sum(axis=-1)[:,None])
        
    return vertices, wts