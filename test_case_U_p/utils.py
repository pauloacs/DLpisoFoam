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


def create_uniform_grid(x_min, x_max,y_min,y_max, delta): #creates a uniform quadrangular grid envolving every cell of the mesh

  X0 = np.linspace(x_min + delta/2 , x_max - delta/2 , num = int(round( (x_max - x_min)/delta )) )
  Y0 = np.linspace(y_min + delta/2 , y_max - delta/2 , num = int(round( (y_max - y_min)/delta )) )

  XX0, YY0 = np.meshgrid(X0,Y0)
  return XX0.flatten(), YY0.flatten()


def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def interpolate_fill(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret

def domain_dist(top_boundary, obst_boundary, xy0):

  # boundaries index
  top = top_boundary
  max_x, max_y, min_x, min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])
  is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y ) #rhis is just for simplification

  obst = obst_boundary
  obst_points =  MultiPoint(obst)
  hull = obst_points.convex_hull       #only works for convex geometries
  hull_pts = hull.exterior.coords.xy    #have a code for any geometry . enven concave https://stackoverflow.com/questions/14263284/create-non-intersecting-polygon-passing-through-all-given-points/47410079
  hull_pts = np.c_[hull_pts[0], hull_pts[1]]

  path = mpltPath.Path(hull_pts)
  is_inside_obst = path.contains_points(xy0)
  domain_bool = is_inside_domain * ~is_inside_obst
  top = top[0:top.shape[0]:10,:]   #if this has too many values, using cdist can crash the memmory since it needs to evaluate the distance between ~1M points with thousands of points of top
  obst = obst[0:obst.shape[0]:10,:]
  sdf = np.minimum( distance.cdist(xy0,obst).min(axis=1) , distance.cdist(xy0,top).min(axis=1) ) * domain_bool

  return domain_bool, sdf


def DENSE_PCA(input_shape, PC_p):
    """
    Initialized Neural Network.

    Args:
        PC_input:
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


def interp_weights(xyz, uvw):
    d = 2 #2d interpolation
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))	