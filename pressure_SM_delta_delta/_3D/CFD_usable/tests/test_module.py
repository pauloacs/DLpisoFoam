import numpy as np
import matplotlib.pyplot as plt
from pressure_SM._3D.CFD_usable.main_mpi import load_tucker_and_NN, init_func, py_func
import os
from pressure_SM._3D.CFD_usable.test_data import (
    array, y_bot, y_top, z_bot, z_top, obst,
    tucker_factors_fn, maxs_fn, PCA_std_vals_fn, weights_fn
)

array = np.load(array)
y_bot = np.load(y_bot)
y_top = np.load(y_top)
z_bot = np.load(z_bot)
z_top = np.load(z_top)
obst_boundary = np.load(obst)
var = 0.95
model_arch = 'MLP_small'
apply_filter =  False
overlap_ratio = 0.25
filter_size = 3
array = np.concatenate([array, np.ones((array.shape[0], 2))], axis=-1)
block_size = 16
grid_res = 4e-3
dropout_rate = 0.1
regularization=None
ranks = 4

def test_whole_module():

    # plt.scatter(array[:,2],array[:,3], c = array[:,0])
    # plt.savefig('ux.png')

    # plt.scatter(array[:,2],array[:,3], c = array[:,1])
    # plt.savefig('uy.png')

    load_tucker_and_NN(
        tucker_factors_fn,
        maxs_fn,
        PCA_std_vals_fn,
        weights_fn,
        model_arch,
        apply_filter,
        overlap_ratio,
        (filter_size, filter_size, filter_size),
        block_size,
        grid_res,
        dropout_rate,
        regularization,
        ranks,
        verbose=True
    )
    init_func(array, z_top, z_bot, y_top, y_bot, obst_boundary)
    delta_p = py_func(array, 1.)
    assert delta_p.shape == (array.shape[0],)
    assert np.all(np.isfinite(delta_p))

if __name__=="__main__":
    test_whole_module()

# from numba import njit
# @njit
# def index(array, item):
#    for idx, val in np.ndenumerate(array):
#        if val == item:
#            return idx
#    # If no item was found return None, other return types might be a problem due to
#    # numbas type inference.

#path = '/home/paulo/dataset_unsteadyCil_fu_bound.hdf5' #adjust path

#frame = 40
#hdf5_file = h5py.File(path, "r")
##data = hdf5_file["sim_data"][:1, frame-1:frame, ...]
#top_boundary = hdf5_file["top_bound"][0, frame, ...]
#obst_boundary = hdf5_file["obst_bound"][0, frame, ...]
#hdf5_file.close()

#indice_top = index(top_boundary[:,0] , -100.0 )[0]
#top_boundary = top_boundary[:indice_top,:]

#indice_obst = index(obst_boundary[:,0] , -100.0 )[0]
#obst_boundary = obst_boundary[:indice_obst,:]

#indice = index(data[0,0,:,0] , -100.0 )[0]
#array = data[0,0,:indice,:4]
#array[:,2:4] = data[0,0,:indice,3:5]
