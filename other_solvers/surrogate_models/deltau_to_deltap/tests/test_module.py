import numpy as np
import matplotlib.pyplot as plt
from surrogate_models.deltau_to_deltap.main import load_pca_and_NN, init_func, py_func
from surrogate_models.deltau_to_deltap.test_data import array, top, obst, pca_input_fn, pca_output_fn, \
                                                        maxs_fn, PCA_std_vals_fn, weights_fn

array = np.load(array)
top_boundary = np.load(top)
obst_boundary = np.load(obst)
var = 0.95
model_arch = 'MLP_small'
apply_filter =  False
overlap_ratio = 0.25
filter_size = 3

def test_whole_module():

    plt.scatter(array[:,2],array[:,3], c = array[:,0])
    plt.savefig('ux.png')

    plt.scatter(array[:,2],array[:,3], c = array[:,1])
    plt.savefig('uy.png')

    load_pca_and_NN(pca_input_fn, pca_output_fn, maxs_fn, PCA_std_vals_fn, weights_fn, var, model_arch, apply_filter, overlap_ratio, filter_size)
    init_func(array, top_boundary, obst_boundary)
    p = py_func(array, 1.)

    plt.scatter(array[:,2], array[:,3], c=p, cmap = 'jet')
    plt.savefig('p.png')

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
