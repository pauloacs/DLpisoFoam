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

#from pressure_SM._3D.CFD_usable.main import load_tucker_and_NN
from pressure_SM._3D.CFD_usable.main_mpi import load_tucker_and_NN

ML_data_folder    = 'ML_data'

tucker_factors_fn = f"{ML_data_folder}/tucker_factors.pkl"
maxs_fn           = f"{ML_data_folder}/maxs"
std_vals_fn       = f"{ML_data_folder}/mean_std.npz"
weights_fn        = f"{ML_data_folder}/weights.h5"
model_arch        = "MLP_small"
apply_filter      = True
overlap_ratio     = 0.1
filter_tuple      = (20,20)
verbose           = False
block_size        = 16
grid_res          = 1e-3
dropout_rate      = 0.1
ranks             = 3
regularization    = None

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
    ranks,
    verbose
)

#from pressure_SM._3D.CFD_usable.main import init_func, py_func
from pressure_SM._3D.CFD_usable.main_mpi import init_func, py_func

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFoam')
