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

from surrogate_model.main import load_pca_and_NN

ipca_input_fn = "ipca_input.pkl"
ipca_output_fn = "ipca_output.pkl"
maxs_fn = "maxs"
max_PCA_fn = "maxs_PCA"
weights_fn = "weights.h5"

load_pca_and_NN(ipca_input_fn, ipca_output_fn, maxs_fn, max_PCA_fn, weights_fn)

from surrogate_model.main import init_func, py_func

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFoam')