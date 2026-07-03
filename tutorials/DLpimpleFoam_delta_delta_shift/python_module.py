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

grid_res               = (2.5e-2, 2.5e-2, 2.5e-2)  # (dx, dy, dz) — per-direction resolution; use a float for uniform resolution
block_size             = 500
# Specify the spatial ranks as a tuple (z, y, x)
spatial_tucker_ranks   = None
dropout_rate           = 0.1
regularization         = 1e-4
model_architecture     = 'cnn_shifter'
standardization_method = 'std'
n_samples_per_frame    = 1
lr                     = 5e-4
batch_size             = 1
beta                   = 0.5
num_epochs             = 100
n_representative_blocks= None
last_tucker_rank       = 11


#################################################### inputs ##################################################
# Input feature flags.
#
# Main pressure relation:
#     U ≈ HbyA - rAU ∇p
#
# Known pressure before ML correction:
#     pKnown = pPrev + dpPrev
#
# ML target:
#     ddpML
#
# Approximate equation for target:
#     ∇·(rAU ∇ddpML) ≈ ∇·HbyA - ∇·(rAU ∇pKnown)


add_U_input = False
# U: final/corrected velocity, 3 channels.

add_dUStar_input = False
# dUStar = UStar - UStarPrev, 3 channels.
# Disabled because dHbyA is the PISO-consistent replacement.

add_ddUStar_input = True
# ddUStar = dUStar - dUStarPrev, 3 channels.
# Temporal curvature of predictor velocity.
# Similar role to ddHbyA if HbyA ≈ UStar.

add_ddUStarDiff_input = True
# ddUStarDiff = ddUStar_n - ddUStar_{n-1}, 3 channels.

add_divUStar_input = False
# divUStar = ∇·UStar, 1 channel.
# Disabled because divHbyA is the pressure-equation source-like field.


add_dUCorrPrev_input = True
# dUCorrPrev = U_final_prev - UStar_prev, 3 channels.
# Previous pressure-correction velocity.
# Conceptually:
#     dUCorrPrev ≈ -rAU_prev ∇ddpPrev
# Not generally equal to:
#     -rAU ∇dpPrev

add_ddUCorrPrev_input = False
# ddUCorrPrev = dUCorrPrev_n - dUCorrPrev_{n-1}, 3 channels.


add_pPrev_input = True
# pPrev: absolute previous pressure, 1 channel.
# Disabled to avoid pressure-gauge/global-offset shortcuts.

add_dpPrev_input = True
# dpPrev = p_n - p_{n-1}, 1 channel.
# Previous pressure increment.

add_ddpPrev_input = True
# ddpPrev = dpPrev_n - dpPrev_{n-1}, 1 channel.
# Previous second pressure increment; closely related to current ddpML target.

add_gradDpPrev_input = True
# gradDpPrev = ∇dpPrev, 3 channels.
# Used by shifter formulation:
#     ddp ≈ a · ∇dpPrev

add_uDotGradDpPrev_input = False
# uDotGradDpPrev = U · ∇dpPrev, 1 channel.

add_gradDpPrevMag_input = True
# gradDpPrevMag = |∇dpPrev|, 1 channel.
# Useful feature-strength/ROI indicator for local pressure structures.


# PISO pressure-equation inputs

include_rAU_input = True
# rAU = A^{-1}, 1 channel.
# Pressure-equation coefficient:
#     ∇·(rAU ∇p) = ∇·HbyA

include_HbyA_input = True
# HbyA = A^{-1}H, 3 channels.
# Predictor velocity-like field:
#     U ≈ HbyA - rAU ∇p

include_divHbyA_input = True
# divHbyA = ∇·HbyA, 1 channel.
# RHS/source-like term of pressure equation.

include_dHbyA_input = True
# dHbyA = HbyA_n - HbyA_{n-1}, 3 channels.
# PISO-consistent replacement for dUStar.

include_dDivHbyA_input = True
# dDivHbyA = divHbyA_n - divHbyA_{n-1}, 1 channel.
# Temporal change of pressure-equation source.


# Additional pressure-equation inputs

add_rAUGradDpPrev_input = True
# rAUGradDpPrev = rAU ∇dpPrev, 3 channels.
# Velocity-correction-like effect of previous pressure increment:
#     -rAU ∇dpPrev

add_divRAUGradDpPrev_input = True
# divRAUGradDpPrev = ∇·(rAU ∇dpPrev), 1 channel.
# Prefer OpenFOAM-consistent:
#     fvc::laplacian(rAU, dpPrev)

add_laplacian_dpPrev_input = False
# laplacian_dpPrev = ∇²dpPrev, 1 channel.
# Disabled because similar to divRAUGradDpPrev.
# If rAU nearly constant:
#     ∇·(rAU ∇dpPrev) ≈ rAU ∇²dpPrev

add_pressureEqResidualp_input = True
# pressureEqResidualp, 1 channel.
# Known pressure-equation residual:
#     R_p = ∇·HbyA - ∇·(rAU ∇pKnown)
# where:
#     pKnown = pPrev + dpPrev
#
# This is what ddpML should correct:
#     ∇·(rAU ∇ddpML) ≈ R_p

add_rAUGradpPrev_input = False
# rAUGradpPrev = rAU ∇pPrev, 3 channels.
# Disabled; residualp already contains known-pressure operator effect.

add_divRAUGradpPrev_input = False
# divRAUGradpPrev = ∇·(rAU ∇pPrev), 1 channel.
# Disabled; redundant with pressureEqResidualp if pKnown residual is used.


add_UdotNwall_input = True
# UdotNwall = U · n_wall, 1 channel.
# n_wall = ∇sdf / |∇sdf|
# Wall-normal velocity feature.

clip_UdotNwall_to_inflow = True
# If True:
#     UdotNwall = max(-U · n_wall, 0)
# If False:
#     signed U · n_wall is used.

#################################################### \inputs\ ##################################################

predict_ddUCorr_output = False  # If True, model predicts [ddp, ddUx, ddUy, ddUz] (4 outputs)
output_weight_factor = 1

# Compression percentile used in asinh tail-compression (per sample/per call)
compression_clip_percentile = 99


oracle_mode = False  # Set to True to use ground-truth values when available
oracle_model_with_interp = False  # If True, apply the same interpolation used for ML inputs to the oracle data (for a more apples-to-apples comparison with ML predictions)
oracle_data_folder = 'oracle_data'  # Folder where oracle data (from training samples) is stored

# If False, skip training entirely and load an already-trained model from ML_data/.
# The files weights_fn, maxs_fn, and std_vals_fn must already exist.
# If they do not exist, the solver will exit with an error.
train = False

# If True (default), subtract the per-block domain mean from pressure fields
# (ddp output and p_prev/dpPrev/ddpPrev inputs) during feature extraction and inference.
# Set to False to disable mean removal (e.g. when the network should handle gauge implicitly).
enforce_zero_mean_pressure = True

# Extra static geometric input channels:
add_distance_to_outlet_input = True  # include distance to outlet (1 channel)
add_grad_sdf_input = False            # include grad(sdf) (3 channels: x, y, z)

# Feature decomposition flag:
#   True  (default) — Tucker decomposition is applied; NN operates on Tucker cores.
#   False — Tucker is skipped; the full 3D block is passed directly to the NN.
#           Default NN is CNN; set model_architecture to another 3D model to override.
use_feature_decomposition = False

# Shifter loss ROI penalty:
#   If True, apply ROI-based spatial regularization for the source term in ShifterLoss.
#   If False (default), only basic regularization terms are used.
use_s_roi_penalty = True

tucker_factors_fn = f"{ML_data_folder}/tucker_factors.pkl"
maxs_fn           = f"{ML_data_folder}/maxs"
std_vals_fn       = f"{ML_data_folder}/mean_std.npz"
weights_fn        = f"{ML_data_folder}/weights.h5"
apply_filter      = False
overlap_ratio     = 0.01
filter_tuple      = (2, 2, 2)
verbose           = True
feature_extraction_chunk_size = 5 # chunk_size for FeatureExtractAndWrite during training
retrain_from_scratch          = False # If True, each retrain starts from a fresh model; if False, continues from current weights
inspect_results   = False  # If True, mid-slice plots of the assembled SM result are saved after each py_func call
inspect_output_dir = 'SM_results'  # Directory where inspect plots are saved

# If train=False, verify all required model files exist before the solver starts
if not train and not os.environ.get('TRAIN_SCRIPT_MODE'):
    _missing = [f for f in [weights_fn, maxs_fn, std_vals_fn] if not os.path.exists(f)]
    if _missing:
        raise FileNotFoundError(
            f"[python_module] train=False but the following required model files are missing:\n"
            + "\n".join(f"  {f}" for f in _missing)
            + "\nSet train=True in python_module.py to run training first."
        )

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
    from pressure_SM_delta_delta_shift._3D.CFD_usable.main_mpi import load_tucker_and_NN

    # Load PCA and Neural Network models with specified parameters
    try:
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
            spatial_tucker_ranks,
            verbose,
            inspect_results=inspect_results,
            inspect_output_dir=inspect_output_dir,
            use_feature_decomposition=use_feature_decomposition,
            add_ddUStar_input=add_ddUStar_input,
            add_ddUStarDiff_input=add_ddUStarDiff_input,
            add_U_input=add_U_input,
            add_dpPrev_input=add_dpPrev_input,
            add_ddpPrev_input=add_ddpPrev_input,
            add_gradDpPrev_input=add_gradDpPrev_input,
            add_dUStar_input=add_dUStar_input,
            add_pPrev_input=add_pPrev_input,
            add_divUStar_input=add_divUStar_input,
            add_dUCorrPrev_input=add_dUCorrPrev_input,
            add_ddUCorrPrev_input=add_ddUCorrPrev_input,
            predict_ddUCorr=predict_ddUCorr_output,
            output_weight_factor=output_weight_factor,
            oracle_mode=oracle_mode,
            oracle_model_with_interp=oracle_model_with_interp,
            oracle_data_folder=oracle_data_folder,
            enforce_zero_mean_pressure=enforce_zero_mean_pressure,
            add_distance_to_outlet_input=add_distance_to_outlet_input,
            add_grad_sdf_input=add_grad_sdf_input,
            add_laplacian_dpPrev_input=add_laplacian_dpPrev_input,
            add_uDotGradDpPrev_input=add_uDotGradDpPrev_input,
            add_gradDpPrevMag_input=add_gradDpPrevMag_input,
            include_rAU_input=include_rAU_input,
            include_HbyA_input=include_HbyA_input,
            include_divHbyA_input=include_divHbyA_input,
            include_dHbyA_input=include_dHbyA_input,
            include_dDivHbyA_input=include_dDivHbyA_input,
            add_rAUGradDpPrev_input=add_rAUGradDpPrev_input,
            add_divRAUGradDpPrev_input=add_divRAUGradDpPrev_input,
            add_pressureEqResidualp_input=add_pressureEqResidualp_input,
            add_rAUGradpPrev_input=add_rAUGradpPrev_input,
            add_divRAUGradpPrev_input=add_divRAUGradpPrev_input,
            compression_clip_percentile=compression_clip_percentile,
            add_UdotNwall_input=add_UdotNwall_input,
            clip_UdotNwall_to_inflow=clip_UdotNwall_to_inflow,
        )
    except Exception as _e:
        import traceback as _tb
        print(f'[python_module] WARNING: load_tucker_and_NN failed: {_e}')
        _tb.print_exc()
        print('[python_module] Surrogate model will be uninitialised until init_func is called after training.')

    #from pressure_SM._3D.CFD_usable.main import init_func, py_func
    from pressure_SM_delta_delta_shift._3D.CFD_usable.main_mpi import init_func, py_func
    from pressure_SM_delta_delta_shift._3D.CFD_usable.main_mpi import reload_weights as _reload_weights_impl

    def reload_weights():
        """Reload NN weights from disk. Called by solver after each retrain."""
        _reload_weights_impl(weights_fn)

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFoam')
