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
from pressureSM.CFD_usable.main import load_pca_and_NN

pca_input_fn = "pca_in.pkl"
pca_output_fn = "pca_p.pkl"
maxs_fn = "maxs"
PCA_std_vals_fn = "mean_std.npz"
weights_fn = "weights.h5"
var = 0.95
model_arch = "MLP_huge"
apply_filter = True
overlap_ratio = 0.5
filter_tuple = (20,20)
verbose = True

load_pca_and_NN(pca_input_fn, pca_output_fn, maxs_fn, PCA_std_vals_fn, weights_fn, var, model_arch, apply_filter, overlap_ratio, filter_tuple, verbose)

#from surrogate_models.deltau_to_deltap.main import init_func, py_func
from pressureSM.CFD_usable.main import init_func, py_func

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFoam')
