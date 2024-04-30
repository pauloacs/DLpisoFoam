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
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Input, Dense

import scipy.spatial.qhull as qhull
from scipy.spatial import distance
import matplotlib.path as mpltPath
from shapely.geometry import MultiPoint


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

def interpolate(values, vtx, wts):
    """
    Interpolates values based on given vertices and weights.

    Parameters:
    values (ndarray): Array of values to interpolate.
    vtx (ndarray): Array of vertex indices.
    wts (ndarray): Array of weights.

    Returns:
    ndarray: Interpolated values.

    """
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

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
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

def domain_dist(top_boundary, obst_boundary, xy0):
    """
    Calculate the fluid-flow domain and the signal distance function (SDF).

    This function calculates a boolean array representing the fluid-flow domain and the signal distance function (SDF).
    The fluid-flow domain is determined by the top boundary and the obstacle boundary.
    The SDF represents the minimum distance from each point in `xy0` to the boundaries.

    Args:
        top_boundary (ndarray): The coordinates of the top boundary.
        obst_boundary (ndarray): The coordinates of the obstacle boundary.
        xy0 (ndarray): The coordinates of the points to calculate the SDF for.

    Returns:
        ndarray: A boolean array representing the fluid-flow domain.
        ndarray: The signal distance function (SDF) for each point in `xy0`.
    """
    # Calculate the boundaries index
    top = top_boundary
    max_x, max_y, min_x, min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])
    is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y )

    obst = obst_boundary
    obst_points =  MultiPoint(obst)
    hull = obst_points.convex_hull
    hull_pts = hull.exterior.coords.xy
    hull_pts = np.c_[hull_pts[0], hull_pts[1]]

    path = mpltPath.Path(hull_pts)
    is_inside_obst = path.contains_points(xy0)
    domain_bool = is_inside_domain * ~is_inside_obst

    top = top[0:top.shape[0]:10,:]
    obst = obst[0:obst.shape[0]:10,:]
    sdf = np.minimum( distance.cdist(xy0,obst).min(axis=1) , distance.cdist(xy0,top).min(axis=1) ) * domain_bool

    return domain_bool, sdf


def DENSE_PCA(input_shape, PC_p):
    """
    Initialized Neural Network.

    Args:
        input_shape (tuple): The shape of the input data.
        PC_p (int): The number of principal components for pressure.

    Returns:
        keras.models.Model: The initialized neural network model.
    """

    inputs = Input(input_shape)
    #
    x = Dense(512, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    #
    outputs = Dense(PC_p)(x)
    #
    model = Model(inputs, outputs, name="U-Net")
    #print(model.summary())

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


def interp_weights(xyz, uvw, d=2):
    """
    Get interpolation weights and vertices.

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
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
