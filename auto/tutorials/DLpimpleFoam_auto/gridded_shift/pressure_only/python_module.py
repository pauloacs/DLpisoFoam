import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

training_backend = 'lattice'
ML_data_folder = 'ML_data'
train = True
model_architecture = 'cnn_shifter_lightweight'
var_to_predict = 'ddp'
ddp_full_coarse_prediction = True
predict_dU_output = False
predict_ddUCorr_output = False
use_feature_decomposition = False
oracle_mode = False
verbose = False

grid_res = (0.01875, 0.01875, 0.03125)
block_size = 500
spatial_tucker_ranks = None
dropout_rate = 0.0
regularization = 0.0
standardization_method = 'std'
n_samples_per_frame = 1
lr = 5e-4
batch_size = 1
beta = 0.5
num_epochs = 8
feature_extraction_chunk_size = 1
n_representative_blocks = None
last_tucker_rank = 1
compression_clip_percentile = 99
use_s_roi_penalty = False

add_U_input = False
add_dUStar_input = False
add_ddUStar_input = False
add_ddUStarDiff_input = False
add_ddpPrev_input = True
add_dpPrev_input = True
add_gradDpPrev_input = True
include_rAU_input = True
add_pressureEqResidualp_input = True
enforce_zero_mean_pressure = True

output_weight_factor = 1.0
extrapolation_weight_factor = 1.0
apply_filter = False
overlap_ratio = 0.0
filter_tuple = (1, 1, 1)
inspect_results = False
inspect_output_dir = 'SM_inspect'
tucker_factors_fn = f'{ML_data_folder}/tucker_factors.pkl'
maxs_fn = f'{ML_data_folder}/maxs'
std_vals_fn = f'{ML_data_folder}/mean_std.npz'
weights_fn = f'{ML_data_folder}/weights.h5'

if not os.environ.get('TRAIN_SCRIPT_MODE'):
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    from pressure_SM_gridded_shift._3D.CFD_usable import main_mpi as _backend

    _backend.load_tucker_and_NN(
        tucker_factors_fn, maxs_fn, std_vals_fn, weights_fn,
        model_architecture, apply_filter, overlap_ratio, filter_tuple,
        block_size, grid_res, dropout_rate, regularization, spatial_tucker_ranks,
        verbose=verbose, use_feature_decomposition=False,
        add_ddUStar_input=False, add_ddUStarDiff_input=False,
        add_ddpPrev_input=add_ddpPrev_input, add_dpPrev_input=add_dpPrev_input,
        add_gradDpPrev_input=add_gradDpPrev_input, include_rAU_input=include_rAU_input,
        add_pressureEqResidualp_input=add_pressureEqResidualp_input,
        predict_ddUCorr=False, output_weight_factor=output_weight_factor,
        extrapolation_weight_factor=extrapolation_weight_factor,
        enforce_zero_mean_pressure=enforce_zero_mean_pressure,
        compression_clip_percentile=compression_clip_percentile,
        var_to_predict=var_to_predict, ddp_full_coarse_prediction=ddp_full_coarse_prediction,
        inspect_results=inspect_results, inspect_output_dir=inspect_output_dir)
    init_func = _backend.init_func
    py_func = _backend.py_func

    def reload_weights():
        _backend.reload_weights(weights_fn)