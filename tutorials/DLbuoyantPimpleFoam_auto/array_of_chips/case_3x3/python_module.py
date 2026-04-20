#f = open('python_log_file','w')
# f.write('Starting python module from OpenFOAM')
# f.close()

import time
import traceback
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

# === Shared configuration (imported by train_init.py and train_update.py) ===
ML_data_folder         = 'ML_data'
grid_res               = 1e-3       # Highly dependent on simulation length scales
block_size             = 16
ranks                  = 3
dropout_rate           = 0.1
regularization         = 1e-4
model_architecture     = 'MLP_small'
standardization_method = 'minmax'
n_samples_per_frame    = 2000
lr                     = 1e-3
batch_size             = 1024
beta                   = 0.5
num_epochs             = 100

tucker_factors_fn = f"{ML_data_folder}/tucker_factors.pkl"
maxs_fn           = f"{ML_data_folder}/maxs_list.npy"
std_vals_fn       = f"{ML_data_folder}/mean_std.npz"
weights_fn        = f"{ML_data_folder}/weights.h5"
apply_filter      = True
overlap_ratio     = 0.1
filter_tuple      = (20, 20, 20)
verbose           = False
feature_extraction_chunk_size = 5000 # chunk_size for FeatureExtractAndWrite during training
retrain_from_scratch          = False # If True, each retrain starts from a fresh model; if False, continues from current weights

# Only run OpenFOAM/MPI initialization when loaded by the solver, not by training scripts
if not os.environ.get('TRAIN_SCRIPT_MODE'):
    import mpi4py
    mpi4py.rc.initialize = False  # Don't call MPI_Init (OpenFOAM already did)
    mpi4py.rc.finalize = False
    from mpi4py import MPI

    # Manually attach to the already-initialized MPI environment
    if not MPI.Is_initialized():
        MPI.Init()

    #from pressure_SM._3D.CFD_usable.main import load_tucker_and_NN
    from pressure_SM._3D.CFD_usable.main_mpi import load_tucker_and_NN

    # Load PCA and Neural Network models with specified parameters
    load_tucker_and_NN(
        tucker_factors_fn,
        maxs_fn,
        std_vals_fn,
        weights_fn,
        model_architecture,
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
    from pressure_SM._3D.CFD_usable.main_mpi import reload_weights as _reload_weights_impl

    def reload_weights():
        _reload_weights_impl(weights_fn)

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFoam')
