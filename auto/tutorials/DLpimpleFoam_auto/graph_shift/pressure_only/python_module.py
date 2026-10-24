import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')

training_backend = 'mesh_graph'
ML_data_folder = 'ML_data'
train = True
model_architecture = 'mesh_graph_shifter'
var_to_predict = 'ddp'
ddp_full_coarse_prediction = True
predict_dU_output = False
predict_ddUCorr_output = False
oracle_mode = False
use_feature_decomposition = False
verbose = False

add_ddpPrev_input = True
add_dpPrev_input = True
add_gradDpPrev_input = True
include_rAU_input = True
add_pressureEqResidualp_input = True

graph_levels = 2
graph_width = 16
graph_steps = 2
graph_patience = 5
graph_linear_prolongation = False
lr = 5e-4
num_epochs = 5
output_weight_factor = 1.0
extrapolation_weight_factor = 1.0
plot_training = False
inspect_results = False

if not os.environ.get('TRAIN_SCRIPT_MODE'):
    import mpi4py
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    from pressure_SM_mesh_graph._3D.inference import configure, init_func, py_func, reload_weights
    configure(globals())