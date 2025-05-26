#f = open('python_log_file','w')
# f.write('Starting python module from OpenFOAM')
# f.close()

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

#from surrogate_models.deltau_to_deltap.main import load_pca_and_NN
from pressure_SM._3D.CFD_usable.main import load_tucker_and_NN

tucker_factors_fn = "tucker_factors.pkl"
maxs_fn = "maxs"
std_vals_fn = "mean_std.npz"
weights_fn = "weights.h5"
model_arch = "MLP_small"
apply_filter = True
overlap_ratio = 0.25
filter_tuple = (20,20)
verbose = True
block_size = 16
grid_res = 4e-3
dropout_rate = 0.1
regularization=None

# Load PCA and Neural Network models with specified parameters
load_tucker_and_NN(
    tucker_factors_fn,
    maxs_fn,
    std_vals_fn,
    weights_fn,
    model_arch,
    apply_filter,
    overlap_ratio,
    filter_tuple,
    block_size,
    grid_res,
    dropout_rate,
	regularization,
    verbose
)

#from surrogate_models.deltau_to_deltap.main import init_func, py_func
from pressure_SM._3D.CFD_usable.main import init_func, py_func

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFoam')
