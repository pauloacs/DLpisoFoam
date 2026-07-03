import numpy as np
import os
import h5py
import time
import shutil
from pressure_SM_delta_delta_shift._3D.train_and_eval.data_processor import CFDDataProcessor, FeatureExtractAndWrite
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import data_processing as utils_data
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import domain_geometry as utils_geo
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import io_operations as utils_io
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import sampling as utils_sampling
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import model_utils as utils_model
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils.data_processing import _unpack_grid_res
from pressure_SM_delta_delta_shift._3D.train_and_eval.train import Training
from pressure_SM_delta_delta_shift._3D.auto_CFD.hdf5_data_loader import load_hdf5_field_data, load_boundaries_dict
from pressure_SM_delta_delta_shift._3D.train_and_eval.neural_networks_shifter import SimpleCNN3D_ddp_shifter
from pressure_SM_delta_delta_shift._3D.train_and_eval.shifter_loss import ShifterLoss

# --- Feature Extraction and Training ---
def add_new_features_and_train():

    import argparse
    parser = argparse.ArgumentParser(description='Update ML model with new CFD samples.')
    parser.add_argument('--data_dir', type=str, default='ML_data', help='Directory where the ML data is stored (default: ML_data)')
    parser.add_argument('--window_frames', type=int, default=20, help='Sliding training window size in time frames (set via system/MLSamplingDict windowFrames)')
    args = parser.parse_args()
    data_dir = args.data_dir
    window_frames = args.window_frames

    # Import shared config from python_module in the case directory (CWD)
    # ALL THE IMPORTANT CONFIGURATION VARIABLES ARE DEFINED IN python_module.py
    # THIS python_module.py SHOULD BE IN YOUR CASE DIRECTORY
    import sys
    os.environ['TRAIN_SCRIPT_MODE'] = '1'
    sys.path.insert(0, os.getcwd())
    from python_module import (
        grid_res, block_size, spatial_tucker_ranks, dropout_rate, regularization,
        model_architecture, standardization_method, n_samples_per_frame,
        lr, batch_size, beta, num_epochs, feature_extraction_chunk_size,
        retrain_from_scratch, last_tucker_rank, use_feature_decomposition
    )

    try:
        from python_module import add_ddUStar_input
    except ImportError:
        add_ddUStar_input = True  # backward-compatible default

    try:
        from python_module import add_ddUStarDiff_input
    except ImportError:
        add_ddUStarDiff_input = True  # backward-compatible default

    try:
        from python_module import add_U_input
    except ImportError:
        add_U_input = False  # backward-compatible default

    try:
        from python_module import add_dpPrev_input
    except ImportError:
        add_dpPrev_input = False  # backward-compatible default

    try:
        from python_module import add_ddpPrev_input
    except ImportError:
        add_ddpPrev_input = False  # backward-compatible default

    try:
        from python_module import add_gradDpPrev_input
    except ImportError:
        add_gradDpPrev_input = False  # backward-compatible default

    try:
        from python_module import add_laplacian_dpPrev_input
    except ImportError:
        add_laplacian_dpPrev_input = False

    try:
        from python_module import add_uDotGradDpPrev_input
    except ImportError:
        add_uDotGradDpPrev_input = False

    try:
        from python_module import add_gradDpPrevMag_input
    except ImportError:
        add_gradDpPrevMag_input = False

    try:
        from python_module import include_rAU_input
    except ImportError:
        include_rAU_input = False

    try:
        from python_module import include_HbyA_input
    except ImportError:
        include_HbyA_input = False

    try:
        from python_module import include_divHbyA_input
    except ImportError:
        include_divHbyA_input = False

    try:
        from python_module import include_dHbyA_input
    except ImportError:
        include_dHbyA_input = False

    try:
        from python_module import include_dDivHbyA_input
    except ImportError:
        include_dDivHbyA_input = False

    try:
        from python_module import add_rAUGradDpPrev_input
    except ImportError:
        add_rAUGradDpPrev_input = False

    try:
        from python_module import add_divRAUGradDpPrev_input
    except ImportError:
        add_divRAUGradDpPrev_input = False

    try:
        from python_module import add_pressureEqResidualp_input
    except ImportError:
        add_pressureEqResidualp_input = False

    try:
        from python_module import add_rAUGradpPrev_input
    except ImportError:
        add_rAUGradpPrev_input = False

    try:
        from python_module import add_divRAUGradpPrev_input
    except ImportError:
        add_divRAUGradpPrev_input = False

    try:
        from python_module import add_dUStar_input
    except ImportError:
        add_dUStar_input = False  # backward-compatible default

    try:
        from python_module import add_pPrev_input
    except ImportError:
        add_pPrev_input = False  # backward-compatible default

    try:
        from python_module import add_divDDUStar_input
    except ImportError:
        add_divDDUStar_input = False  # backward-compatible default

    try:
        from python_module import add_divDUStar_input
    except ImportError:
        add_divDUStar_input = False  # backward-compatible default

    try:
        from python_module import add_divUStar_input
    except ImportError:
        add_divUStar_input = False  # backward-compatible default

    # Current C++ raw layout exports only divUFirstPred (mapped to add_divUStar_input).
    # Legacy divDDU/divDU channels are not exported separately.
    if add_divDDUStar_input or add_divDUStar_input:
        raise ValueError(
            "add_divDDUStar_input/add_divDUStar_input are unsupported with the current solver raw layout. "
            "Only divUFirstPred is exported (use add_divUStar_input)."
        )

    try:
        from python_module import predict_ddUCorr_output
    except ImportError:
        predict_ddUCorr_output = False  # default: predict only ddp

    try:
        from python_module import add_dUCorrPrev_input
    except ImportError:
        add_dUCorrPrev_input = False

    try:
        from python_module import add_ddUCorrPrev_input
    except ImportError:
        add_ddUCorrPrev_input = False

    try:
        from python_module import use_s_roi_penalty
    except ImportError:
        use_s_roi_penalty = False

    gridded_h5_fn = os.path.join(data_dir, 'gridded_data.h5')
    sample_idx_fn = os.path.join(data_dir, 'sample_idx_per_time.npy')
    maxs_list_fn = os.path.join(data_dir, 'maxs')
    tucker_factors_fn = os.path.join(data_dir, 'tucker_factors.pkl')
    core_data_fn = os.path.join(data_dir, 'core_data.h5')

    # --- Load old features from previous training ---
    import tables
    with tables.open_file(core_data_fn, mode='r') as f:
        old_input_cores = f.root.inputs[...]
        old_output_cores = f.root.outputs[...]

    # --- Load new data from HDF5 ---
    hdf5_file = os.path.join(data_dir, 'data.h5')
    hdf5_file_copy = os.path.join(data_dir, 'data_update_copy.h5')
    print(f"Copying {hdf5_file} to {hdf5_file_copy} for safe reading...")

    # Wait until the file exists and is non-empty
    max_wait = 30  # seconds
    waited = 0
    while not os.path.exists(hdf5_file) or os.path.getsize(hdf5_file) == 0:
        if waited >= max_wait:
            print(f"Timeout waiting for {hdf5_file} to be available.")
            exit(1)
        time.sleep(1)
        waited += 1

    shutil.copy2(hdf5_file, hdf5_file_copy)
    print(f"Copied to {hdf5_file_copy}.")

    try:
        ddUStar, ddUStarDiff, ddp, dpPrev, ddpPrev, gradDpPrev, laplaceDpPrev, uDotGradDpPrev, gradDpPrevMag, rAU, HbyA, divHbyA, dHbyA, dDivHbyA, rAUGradDpPrev, divRAUGradDpPrev, pressureEqResidualp, rAUGradpPrev, divRAUGradpPrev, dUStar, dUCorrPrev, ddUCorrPrev, p_prev, U, divDDUStar, divDUStar, divUStar, ddUCorr, timestamps, u_max_norm_arr = load_hdf5_field_data(hdf5_file_copy)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading HDF5 data: {e}")
        exit(1)
    finally:
        if os.path.exists(hdf5_file_copy):
            os.remove(hdf5_file_copy)
        # Delete original so C++ creates a fresh file for the next batch
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)
            print(f"Deleted {hdf5_file} — C++ will create a fresh file for the next batch.")

    n_sample_frames = len(timestamps)
    print(f"Loaded {n_sample_frames} new samples from HDF5 file")
    print(f"ddUStar shape: {ddUStar.shape}, ddUStarDiff shape: {ddUStarDiff.shape}, ddp shape: {ddp.shape}")

    # --- Sliding window: keep (window_frames - n_sample_frames) oldest frames of features ---
    frames_to_keep = max(0, window_frames - n_sample_frames)
    rows_to_keep = frames_to_keep * n_samples_per_frame
    print(f"Sliding window: {window_frames} frames total, keeping {frames_to_keep} old + {n_sample_frames} new.")
    if rows_to_keep > 0 and old_input_cores.shape[0] >= rows_to_keep:
        old_input_cores_to_keep = old_input_cores[-rows_to_keep:]
        old_output_cores_to_keep = old_output_cores[-rows_to_keep:]
    elif rows_to_keep > 0:
        # Not enough history yet — keep whatever we have
        old_input_cores_to_keep = old_input_cores
        old_output_cores_to_keep = old_output_cores
    else:
        old_input_cores_to_keep = old_input_cores[0:0]
        old_output_cores_to_keep = old_output_cores[0:0]

    # --- Load interpolation weights, vertices, and grid info ---
    weights = np.load(os.path.join(data_dir, 'interp_weights.npy'))
    vert = np.load(os.path.join(data_dir, 'interp_vertices.npy'))
    indices_data = np.load(os.path.join(data_dir, 'interpolation_indices.npz'))
    indices_i = indices_data['indices_i']
    indices_j = indices_data['indices_j']
    indices_k = indices_data['indices_k']
    sdf = np.load(os.path.join(data_dir, 'grid_sdf_flat.npy'))
    domain_bool = np.load(os.path.join(data_dir, 'grid_domain_mask_flat.npy'))
    print("Interpolation weights, vertices, and grid info loaded.")

    # --- Interpolate delta_U and dpML to grid (per sample, per component) ---
    n_samples = ddUStar.shape[0]
    n_grid_points = weights.shape[0]

    U_grid_flat = None
    dU_grid_flat = None
    ddUStar_grid_flat      = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    ddUStarDiff_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    ddp_grid_flat      = np.full((n_samples, n_grid_points),    np.nan, dtype=np.float64)
    p_prev_grid_flat             = None
    dpPrev_grid_flat       = None
    ddpPrev_grid_flat     = None
    divDDUStar_grid_flat  = None
    divDUStar_grid_flat             = None
    divUStar_grid_flat              = None

    if add_U_input:
        U_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    if add_dUStar_input:
        dUStar_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    
    dUCorrPrev_grid_flat = None
    ddUCorrPrev_grid_flat = None
    if add_dUCorrPrev_input:
        dUCorrPrev_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    if add_ddUCorrPrev_input:
        ddUCorrPrev_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    if add_pPrev_input:
        p_prev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)
    
    if add_dpPrev_input:
        dpPrev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_ddpPrev_input:
        ddpPrev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    gradDpPrev_grid_flat = None
    if add_gradDpPrev_input:
        gradDpPrev_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    laplaceDpPrev_grid_flat = None
    if add_laplacian_dpPrev_input:
        laplaceDpPrev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    uDotGradDpPrev_grid_flat = None
    if add_uDotGradDpPrev_input:
        uDotGradDpPrev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    gradDpPrevMag_grid_flat = None
    if add_gradDpPrevMag_input:
        gradDpPrevMag_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    rAU_grid_flat = None
    if include_rAU_input:
        rAU_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    HbyA_grid_flat = None
    if include_HbyA_input:
        HbyA_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    divHbyA_grid_flat = None
    if include_divHbyA_input:
        divHbyA_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    dHbyA_grid_flat = None
    if include_dHbyA_input:
        dHbyA_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    dDivHbyA_grid_flat = None
    if include_dDivHbyA_input:
        dDivHbyA_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    rAUGradDpPrev_grid_flat = None
    if add_rAUGradDpPrev_input:
        rAUGradDpPrev_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    divRAUGradDpPrev_grid_flat = None
    if add_divRAUGradDpPrev_input:
        divRAUGradDpPrev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    pressureEqResidualp_grid_flat = None
    if add_pressureEqResidualp_input:
        pressureEqResidualp_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    rAUGradpPrev_grid_flat = None
    if add_rAUGradpPrev_input:
        rAUGradpPrev_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    divRAUGradpPrev_grid_flat = None
    if add_divRAUGradpPrev_input:
        divRAUGradpPrev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_divDDUStar_input:
        divDDUStar_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_divDUStar_input:
        divDUStar_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_divUStar_input:
        divUStar_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    ddUCorr_grid_flat = None
    if predict_ddUCorr_output:
        ddUCorr_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    for sample_idx in range(n_samples):
        if add_U_input:
            for component in range(3):
                U_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    U[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        if add_dUStar_input:
            for component in range(3):
                dUStar_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    dUStar[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        if add_dUCorrPrev_input:
            for component in range(3):
                dUCorrPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    dUCorrPrev[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )
        if add_ddUCorrPrev_input:
            for component in range(3):
                ddUCorrPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    ddUCorrPrev[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        for component in range(3):
            ddUStar_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                ddUStar[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )
            ddUStarDiff_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                ddUStarDiff[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )
        ddp_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
            ddp[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
        )

        if add_pPrev_input:
            p_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                p_prev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )
        
        if add_dpPrev_input:
            dpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                dpPrev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )

        if add_ddpPrev_input:
            ddpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                ddpPrev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )

        if add_gradDpPrev_input:
            for component in range(3):
                gradDpPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    gradDpPrev[sample_idx, :, component] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
                )

        if add_laplacian_dpPrev_input:
            laplaceDpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                laplaceDpPrev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )

        if add_uDotGradDpPrev_input:
            uDotGradDpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                uDotGradDpPrev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 3), vert, weights, fill_value=np.nan
            )

        if add_gradDpPrevMag_input:
            gradDpPrevMag_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                gradDpPrevMag[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )

        # rAU = 1/A(U) [s] — not velocity-scaled (only max_abs normalized later)
        if include_rAU_input:
            rAU_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                rAU[sample_idx, :], vert, weights, fill_value=np.nan
            )

        # HbyA [m/s] — velocity-scaled
        if include_HbyA_input:
            for component in range(3):
                HbyA_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    HbyA[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        # divHbyA [1/s] — velocity-scaled (consistent with div(U))
        if include_divHbyA_input:
            divHbyA_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                divHbyA[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        # dHbyA [m/s] — velocity-scaled (temporal variation of HbyA)
        if include_dHbyA_input:
            for component in range(3):
                dHbyA_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    dHbyA[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        # dDivHbyA [1/s] — velocity-scaled (temporal variation of divHbyA)
        if include_dDivHbyA_input:
            dDivHbyA_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                dDivHbyA[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        if add_rAUGradDpPrev_input:
            for component in range(3):
                rAUGradDpPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    rAUGradDpPrev[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        if add_divRAUGradDpPrev_input:
            divRAUGradDpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                divRAUGradDpPrev[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        if add_pressureEqResidualp_input:
            pressureEqResidualp_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                pressureEqResidualp[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        if add_rAUGradpPrev_input:
            for component in range(3):
                rAUGradpPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    rAUGradpPrev[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        if add_divRAUGradpPrev_input:
            divRAUGradpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                divRAUGradpPrev[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        if add_divDDUStar_input:
            divDDUStar_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                divDDUStar[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        if add_divDUStar_input:
            divDUStar_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                divDUStar[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        if add_divUStar_input:
            divUStar_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                divUStar[sample_idx, :] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )

        if predict_ddUCorr_output:
            for component in range(3):
                ddUCorr_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    ddUCorr[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )
    print("Interpolation to grid complete.")

    # Stack dataset based on enabled flags
    # Channel order: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU [p_prev if add_p_prev] [dpPrev if add_dpPrev] [ddpPrev if add_ddpPrev] [div_ddu] [div_du] [div_u] ddp
    dataset_parts = []
    if add_U_input:
        dataset_parts.append(U_grid_flat)
    if add_dUStar_input:
        dataset_parts.append(dUStar_grid_flat)
    if add_ddUStar_input:
        dataset_parts.append(ddUStar_grid_flat)
    if add_ddUStarDiff_input:
        dataset_parts.append(ddUStarDiff_grid_flat)
    if add_dUCorrPrev_input:
        dataset_parts.append(dUCorrPrev_grid_flat)
    if add_ddUCorrPrev_input:
        dataset_parts.append(ddUCorrPrev_grid_flat)
    if add_pPrev_input:
        dataset_parts.append(p_prev_grid_flat[..., np.newaxis])
    if add_dpPrev_input:
        dataset_parts.append(dpPrev_grid_flat[..., np.newaxis])
    if add_ddpPrev_input:
        dataset_parts.append(ddpPrev_grid_flat[..., np.newaxis])
    if add_gradDpPrev_input:
        dataset_parts.append(gradDpPrev_grid_flat)
    if add_laplacian_dpPrev_input:
        dataset_parts.append(laplaceDpPrev_grid_flat[..., np.newaxis])
    if add_uDotGradDpPrev_input:
        dataset_parts.append(uDotGradDpPrev_grid_flat[..., np.newaxis])
    if add_gradDpPrevMag_input:
        dataset_parts.append(gradDpPrevMag_grid_flat[..., np.newaxis])
    if include_rAU_input:
        dataset_parts.append(rAU_grid_flat[..., np.newaxis])
    if include_HbyA_input:
        dataset_parts.append(HbyA_grid_flat)
    if include_divHbyA_input:
        dataset_parts.append(divHbyA_grid_flat[..., np.newaxis])
    if include_dHbyA_input:
        dataset_parts.append(dHbyA_grid_flat)
    if include_dDivHbyA_input:
        dataset_parts.append(dDivHbyA_grid_flat[..., np.newaxis])
    if add_rAUGradDpPrev_input:
        dataset_parts.append(rAUGradDpPrev_grid_flat)
    if add_divRAUGradDpPrev_input:
        dataset_parts.append(divRAUGradDpPrev_grid_flat[..., np.newaxis])
    if add_pressureEqResidualp_input:
        dataset_parts.append(pressureEqResidualp_grid_flat[..., np.newaxis])
    if add_rAUGradpPrev_input:
        dataset_parts.append(rAUGradpPrev_grid_flat)
    if add_divRAUGradpPrev_input:
        dataset_parts.append(divRAUGradpPrev_grid_flat[..., np.newaxis])
    if add_divDDUStar_input:
        dataset_parts.append(divDDUStar_grid_flat[..., np.newaxis])
    if add_divDUStar_input:
        dataset_parts.append(divDUStar_grid_flat[..., np.newaxis])
    if add_divUStar_input:
        dataset_parts.append(divUStar_grid_flat[..., np.newaxis])
    dataset_parts.append(ddp_grid_flat[..., np.newaxis])
    if predict_ddUCorr_output:
        dataset_parts.append(ddUCorr_grid_flat)
    dataset = np.concatenate(dataset_parts, axis=-1)
    
    # Calculate number of grid channels
    n_grid_channels = 2  # sdf + ddp
    if add_U_input:
        n_grid_channels += 3
    if add_dUStar_input:
        n_grid_channels += 3
    if add_ddUStar_input:
        n_grid_channels += 3
    if add_ddUStarDiff_input:
        n_grid_channels += 3
    if add_dUCorrPrev_input:
        n_grid_channels += 3
    if add_ddUCorrPrev_input:
        n_grid_channels += 3
    if add_pPrev_input:
        n_grid_channels += 1
    if add_dpPrev_input:
        n_grid_channels += 1
    if add_ddpPrev_input:
        n_grid_channels += 1
    if add_gradDpPrev_input:
        n_grid_channels += 3
    if add_laplacian_dpPrev_input:
        n_grid_channels += 1
    if add_uDotGradDpPrev_input:
        n_grid_channels += 1
    if add_gradDpPrevMag_input:
        n_grid_channels += 1
    if include_rAU_input:
        n_grid_channels += 1
    if include_HbyA_input:
        n_grid_channels += 3
    if include_divHbyA_input:
        n_grid_channels += 1
    if include_dHbyA_input:
        n_grid_channels += 3
    if include_dDivHbyA_input:
        n_grid_channels += 1
    if add_rAUGradDpPrev_input:
        n_grid_channels += 3
    if add_divRAUGradDpPrev_input:
        n_grid_channels += 1
    if add_pressureEqResidualp_input:
        n_grid_channels += 1
    if add_rAUGradpPrev_input:
        n_grid_channels += 3
    if add_divRAUGradpPrev_input:
        n_grid_channels += 1
    if add_divDDUStar_input:
        n_grid_channels += 1
    if add_divDUStar_input:
        n_grid_channels += 1
    if add_divUStar_input:
        n_grid_channels += 1
    if predict_ddUCorr_output:
        n_grid_channels += 3  # ddU_CFD_x, ddU_CFD_y, ddU_CFD_z

    # --- Reconstruct 3D gridded array using ch_idx (same as train_init) ---
    grid_shape_z = indices_i.max() + 1
    grid_shape_y = indices_j.max() + 1
    grid_shape_x = indices_k.max() + 1
    grid_shape = (n_samples, grid_shape_z, grid_shape_y, grid_shape_x, n_grid_channels)
    dataset_gridded = np.full(grid_shape, np.nan, dtype=np.float64)

    # Compute ch_idx mapping (same order as train_init buildstep)
    _ci = 0
    _u_idx = (_ci, _ci+3) if add_U_input else None; _ci += 3 if add_U_input else 0
    _dU_idx = (_ci, _ci+3) if add_dUStar_input else None; _ci += 3 if add_dUStar_input else 0
    _ddu_idx = (_ci, _ci+3) if add_ddUStar_input else None; _ci += 3 if add_ddUStar_input else 0
    _dddu_idx = (_ci, _ci+3) if add_ddUStarDiff_input else None; _ci += 3 if add_ddUStarDiff_input else 0
    _dUCorrPrev_idx = (_ci, _ci+3) if add_dUCorrPrev_input else None; _ci += 3 if add_dUCorrPrev_input else 0
    _ddUCorrPrev_idx = (_ci, _ci+3) if add_ddUCorrPrev_input else None; _ci += 3 if add_ddUCorrPrev_input else 0
    _sdf_idx = _ci; _ci += 1
    _p_prev_idx = _ci if add_pPrev_input else None; _ci += 1 if add_pPrev_input else 0
    _dpPrev_idx = _ci if add_dpPrev_input else None; _ci += 1 if add_dpPrev_input else 0
    _ddpPrev_idx = _ci if add_ddpPrev_input else None; _ci += 1 if add_ddpPrev_input else 0
    _gradDpPrev_idx = (_ci, _ci+3) if add_gradDpPrev_input else None; _ci += 3 if add_gradDpPrev_input else 0
    _laplaceDpPrev_idx = _ci if add_laplacian_dpPrev_input else None; _ci += 1 if add_laplacian_dpPrev_input else 0
    _uDotGradDpPrev_idx = _ci if add_uDotGradDpPrev_input else None; _ci += 1 if add_uDotGradDpPrev_input else 0
    _gradDpPrevMag_idx = _ci if add_gradDpPrevMag_input else None; _ci += 1 if add_gradDpPrevMag_input else 0
    _rAU_idx = _ci if include_rAU_input else None; _ci += 1 if include_rAU_input else 0
    _HbyA_idx = (_ci, _ci+3) if include_HbyA_input else None; _ci += 3 if include_HbyA_input else 0
    _divHbyA_idx = _ci if include_divHbyA_input else None; _ci += 1 if include_divHbyA_input else 0
    _dHbyA_idx = (_ci, _ci+3) if include_dHbyA_input else None; _ci += 3 if include_dHbyA_input else 0
    _dDivHbyA_idx = _ci if include_dDivHbyA_input else None; _ci += 1 if include_dDivHbyA_input else 0
    _rAUGradDpPrev_idx = (_ci, _ci+3) if add_rAUGradDpPrev_input else None; _ci += 3 if add_rAUGradDpPrev_input else 0
    _divRAUGradDpPrev_idx = _ci if add_divRAUGradDpPrev_input else None; _ci += 1 if add_divRAUGradDpPrev_input else 0
    _pressureEqResidualp_idx = _ci if add_pressureEqResidualp_input else None; _ci += 1 if add_pressureEqResidualp_input else 0
    _rAUGradpPrev_idx = (_ci, _ci+3) if add_rAUGradpPrev_input else None; _ci += 3 if add_rAUGradpPrev_input else 0
    _divRAUGradpPrev_idx = _ci if add_divRAUGradpPrev_input else None; _ci += 1 if add_divRAUGradpPrev_input else 0
    _div_ddu_idx = _ci if add_divDDUStar_input else None; _ci += 1 if add_divDDUStar_input else 0
    _div_du_idx  = _ci if add_divDUStar_input  else None; _ci += 1 if add_divDUStar_input  else 0
    _div_u_idx   = _ci if add_divUStar_input   else None; _ci += 1 if add_divUStar_input   else 0
    _ddp_idx = _ci
    _ddU_CFD_idx = (_ci + 1, _ci + 4) if predict_ddUCorr_output else None
    # Flat dataset indices (no sdf)
    _ds_base = (_ddUCorrPrev_idx[1] if add_ddUCorrPrev_input else (_dUCorrPrev_idx[1] if add_dUCorrPrev_input else (_dddu_idx[1] if add_ddUStarDiff_input else (_ddu_idx[1] if add_ddUStar_input else (_dU_idx[1] if add_dUStar_input else (_u_idx[1] if add_U_input else 0))))))
    _ds_p_prev_ch   = _ds_base
    _ds_dpPrev_ch  = _ds_base + (1 if add_pPrev_input else 0)
    _ds_ddpPrev_ch = _ds_dpPrev_ch + (1 if add_dpPrev_input else 0)
    _ds_gradDpPrev_ch = _ds_ddpPrev_ch + (1 if add_ddpPrev_input else 0)
    _ds_laplaceDpPrev_ch = _ds_gradDpPrev_ch + (3 if add_gradDpPrev_input else 0)
    _ds_uDotGradDpPrev_ch = _ds_laplaceDpPrev_ch + (1 if add_laplacian_dpPrev_input else 0)
    _ds_gradDpPrevMag_ch = _ds_uDotGradDpPrev_ch + (1 if add_uDotGradDpPrev_input else 0)
    _ds_rAU_ch      = _ds_gradDpPrevMag_ch + (1 if add_gradDpPrevMag_input else 0)
    _ds_HbyA_ch     = _ds_rAU_ch + (1 if include_rAU_input else 0)
    _ds_divHbyA_ch  = _ds_HbyA_ch + (3 if include_HbyA_input else 0)
    _ds_dHbyA_ch    = _ds_divHbyA_ch + (1 if include_divHbyA_input else 0)
    _ds_dDivHbyA_ch = _ds_dHbyA_ch + (3 if include_dHbyA_input else 0)
    _ds_rAUGradDpPrev_ch = _ds_dDivHbyA_ch + (1 if include_dDivHbyA_input else 0)
    _ds_divRAUGradDpPrev_ch = _ds_rAUGradDpPrev_ch + (3 if add_rAUGradDpPrev_input else 0)
    _ds_pressureEqResidualp_ch = _ds_divRAUGradDpPrev_ch + (1 if add_divRAUGradDpPrev_input else 0)
    _ds_rAUGradpPrev_ch = _ds_pressureEqResidualp_ch + (1 if add_pressureEqResidualp_input else 0)
    _ds_divRAUGradpPrev_ch = _ds_rAUGradpPrev_ch + (3 if add_rAUGradpPrev_input else 0)
    _ds_div_ddu_ch  = _ds_divRAUGradpPrev_ch + (1 if add_divRAUGradpPrev_input else 0)
    _ds_div_du_ch   = _ds_div_ddu_ch  + (1 if add_divDDUStar_input else 0)
    _ds_div_u_ch    = _ds_div_du_ch   + (1 if add_divDUStar_input  else 0)
    _ds_ddp_ch      = _ds_div_u_ch    + (1 if add_divUStar_input   else 0)

    for step in range(n_samples):
        if add_U_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _u_idx[0]:_u_idx[1]] = dataset[step, :, _u_idx[0]:_u_idx[1]]
        if add_dUStar_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dU_idx[0]:_dU_idx[1]] = dataset[step, :, _dU_idx[0]:_dU_idx[1]]
        if add_ddUStar_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _ddu_idx[0]:_ddu_idx[1]] = dataset[step, :, _ddu_idx[0]:_ddu_idx[1]]
        if add_ddUStarDiff_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dddu_idx[0]:_dddu_idx[1]] = dataset[step, :, _dddu_idx[0]:_dddu_idx[1]]
        if add_dUCorrPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dUCorrPrev_idx[0]:_dUCorrPrev_idx[1]] = dataset[step, :, _dUCorrPrev_idx[0]:_dUCorrPrev_idx[1]]
        if add_ddUCorrPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _ddUCorrPrev_idx[0]:_ddUCorrPrev_idx[1]] = dataset[step, :, _ddUCorrPrev_idx[0]:_ddUCorrPrev_idx[1]]
        dataset_gridded[step, indices_i, indices_j, indices_k, _sdf_idx] = sdf
        if add_pPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _p_prev_idx] = dataset[step, :, _ds_p_prev_ch]
        if add_dpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dpPrev_idx] = dataset[step, :, _ds_dpPrev_ch]
        if add_ddpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _ddpPrev_idx] = dataset[step, :, _ds_ddpPrev_ch]
        if add_gradDpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _gradDpPrev_idx[0]:_gradDpPrev_idx[1]] = dataset[step, :, _ds_gradDpPrev_ch:_ds_gradDpPrev_ch+3]
        if add_laplacian_dpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _laplaceDpPrev_idx] = dataset[step, :, _ds_laplaceDpPrev_ch]
        if add_uDotGradDpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _uDotGradDpPrev_idx] = dataset[step, :, _ds_uDotGradDpPrev_ch]
        if add_gradDpPrevMag_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _gradDpPrevMag_idx] = dataset[step, :, _ds_gradDpPrevMag_ch]
        if include_rAU_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _rAU_idx] = dataset[step, :, _ds_rAU_ch]
        if include_HbyA_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _HbyA_idx[0]:_HbyA_idx[1]] = dataset[step, :, _ds_HbyA_ch:_ds_HbyA_ch+3]
        if include_divHbyA_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _divHbyA_idx] = dataset[step, :, _ds_divHbyA_ch]
        if include_dHbyA_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dHbyA_idx[0]:_dHbyA_idx[1]] = dataset[step, :, _ds_dHbyA_ch:_ds_dHbyA_ch+3]
        if include_dDivHbyA_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dDivHbyA_idx] = dataset[step, :, _ds_dDivHbyA_ch]
        if add_rAUGradDpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _rAUGradDpPrev_idx[0]:_rAUGradDpPrev_idx[1]] = dataset[step, :, _ds_rAUGradDpPrev_ch:_ds_rAUGradDpPrev_ch+3]
        if add_divRAUGradDpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _divRAUGradDpPrev_idx] = dataset[step, :, _ds_divRAUGradDpPrev_ch]
        if add_pressureEqResidualp_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _pressureEqResidualp_idx] = dataset[step, :, _ds_pressureEqResidualp_ch]
        if add_rAUGradpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _rAUGradpPrev_idx[0]:_rAUGradpPrev_idx[1]] = dataset[step, :, _ds_rAUGradpPrev_ch:_ds_rAUGradpPrev_ch+3]
        if add_divRAUGradpPrev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _divRAUGradpPrev_idx] = dataset[step, :, _ds_divRAUGradpPrev_ch]
        if add_divDDUStar_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _div_ddu_idx] = dataset[step, :, _ds_div_ddu_ch]
        if add_divDUStar_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _div_du_idx] = dataset[step, :, _ds_div_du_ch]
        if add_divUStar_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _div_u_idx] = dataset[step, :, _ds_div_u_ch]
        dataset_gridded[step, indices_i, indices_j, indices_k, _ddp_idx] = dataset[step, :, _ds_ddp_ch]
        if predict_ddUCorr_output:
            dataset_gridded[step, indices_i, indices_j, indices_k, _ddU_CFD_idx[0]:_ddU_CFD_idx[1]] = dataset[step, :, _ds_ddp_ch+1:_ds_ddp_ch+4]

    if os.path.exists(gridded_h5_fn):
        os.remove(gridded_h5_fn)

    with h5py.File(gridded_h5_fn, 'w') as f:
        f.create_dataset('data', data=dataset_gridded)
    print(f"Stacked data (U, p, sdf) saved to {gridded_h5_fn}.")

    # --- Debug Plots: Save slices of gridded data like in train_init.py ---
    import matplotlib.pyplot as plt
    os.makedirs('plots_debug', exist_ok=True)
    var_names = []
    if add_U_input:
        var_names.extend(['u_x', 'u_y', 'u_z'])
    if add_dUStar_input:
        var_names.extend(['dU_x', 'dU_y', 'dU_z'])
    if add_ddUStar_input:
        var_names.extend(['ddU_x', 'ddU_y', 'ddU_z'])
    if add_ddUStarDiff_input:
        var_names.extend(['dddU_x', 'dddU_y', 'dddU_z'])
    if add_dUCorrPrev_input:
        var_names.extend(['dUCorrPrev_x', 'dUCorrPrev_y', 'dUCorrPrev_z'])
    if add_ddUCorrPrev_input:
        var_names.extend(['ddUCorrPrev_x', 'ddUCorrPrev_y', 'ddUCorrPrev_z'])
    var_names.extend(['sdf'])
    if add_pPrev_input:
        var_names.append('p_prev')
    if add_dpPrev_input:
        var_names.append('dpPrev')
    if add_ddpPrev_input:
        var_names.append('ddpPrev')
    if add_gradDpPrev_input:
        var_names.extend(['gradDpPrev_x', 'gradDpPrev_y', 'gradDpPrev_z'])
    if add_laplacian_dpPrev_input:
        var_names.append('laplaceDpPrev')
    if add_uDotGradDpPrev_input:
        var_names.append('uDotGradDpPrev')
    if add_gradDpPrevMag_input:
        var_names.append('gradDpPrevMag')
    if include_rAU_input:
        var_names.append('rAU')
    if include_HbyA_input:
        var_names.extend(['HbyA_x', 'HbyA_y', 'HbyA_z'])
    if include_divHbyA_input:
        var_names.append('divHbyA')
    if include_dHbyA_input:
        var_names.extend(['dHbyA_x', 'dHbyA_y', 'dHbyA_z'])
    if include_dDivHbyA_input:
        var_names.append('dDivHbyA')
    if add_rAUGradDpPrev_input:
        var_names.extend(['rAUGradDpPrev_x', 'rAUGradDpPrev_y', 'rAUGradDpPrev_z'])
    if add_divRAUGradDpPrev_input:
        var_names.append('divRAUGradDpPrev')
    if add_pressureEqResidualp_input:
        var_names.append('pressureEqResidualp')
    if add_rAUGradpPrev_input:
        var_names.extend(['rAUGradpPrev_x', 'rAUGradpPrev_y', 'rAUGradpPrev_z'])
    if add_divRAUGradpPrev_input:
        var_names.append('divRAUGradpPrev')
    if add_divDDUStar_input:
        var_names.append('divDDUStar')
    if add_divDUStar_input:
        var_names.append('divDUStar')
    if add_divUStar_input:
        var_names.append('divUStar')
    var_names.append('ddp')
    if predict_ddUCorr_output:
        var_names.extend(['ddUCorr_x', 'ddUCorr_y', 'ddUCorr_z'])
    # Use the first sample for plotting
    grid = dataset_gridded[0]
    for var_idx in range(n_grid_channels):
        # Plot slice through middle of grid (z-x plane at middle y)
        plt.figure(figsize=(10, 6))
        plt.imshow(grid[1:-1, int(grid.shape[1] / 2), 1:-1, var_idx], cmap='jet')
        plt.colorbar(label=var_names[var_idx])
        plt.title(f'{var_names[var_idx]} - Z-X slice (middle Y)')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.savefig(f'plots_debug/{var_names[var_idx]}_zx_slice_t0.png')
        plt.close()
        # Plot slice through middle of grid (y-x plane at middle z)
        plt.figure(figsize=(10, 6))
        plt.imshow(grid[int(grid.shape[0] / 2), :, :, var_idx], cmap='jet')
        plt.colorbar(label=var_names[var_idx])
        plt.title(f'{var_names[var_idx]} - Y-X slice (middle Z)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'plots_debug/{var_names[var_idx]}_yx_slice_t0.png')
        plt.close()

    # --- Define sampling indices ---
    boundaries_dict = load_boundaries_dict(data_dir)
    dx, dy, dz = _unpack_grid_res(grid_res)
    grid_limits = {
        'x_min': float(indices_k.min()) * dx,
        'x_max': float(indices_k.max()) * dx,
        'y_min': float(indices_j.min()) * dy,
        'y_max': float(indices_j.max()) * dy,
        'z_min': float(indices_i.min()) * dz,
        'z_max': float(indices_i.max()) * dz,
    }
    sampling_indices = utils_sampling.define_sample_indexes(
        n_samples_per_frame,
        block_size,
        grid_res,
        0, 0, 0,
        n_sample_frames,
        None,
        sample_idx_fn,
        grid_limits
    )

    # --- Use the fixed block maximums from initial training ---
    maxs_list = np.loadtxt(maxs_list_fn)
    print(f"[train_update] Using fixed maxs_list from initial training: {maxs_list_fn}")

    # --- Extract features from new data ---
    feature_writer = FeatureExtractAndWrite(
        grid_res=grid_res,
        block_size=block_size,
        original_dataset_path=None,
        n_samples_per_frame=n_samples_per_frame,
        first_sim=0,
        last_sim=0,
        first_t=0,
        last_t=n_sample_frames,
        standardization_method=standardization_method,
        chunk_size=feature_extraction_chunk_size,
        gridded_h5_fn=None,
        spatial_tucker_ranks=spatial_tucker_ranks,
        sample_indices_fn=sample_idx_fn,
        tucker_factors_fn=tucker_factors_fn,
        gridded_h5_filenames=[gridded_h5_fn],
        flatten_data=use_feature_decomposition,
        maxs_list=maxs_list,
        add_ddUStar_input=add_ddUStar_input,
        add_ddUStarDiff_input=add_ddUStarDiff_input,
        add_U_input=add_U_input,
        add_dUStar_input=add_dUStar_input,
        add_dpPrev_input=add_dpPrev_input,
        add_pPrev_input=add_pPrev_input,
        add_ddpPrev_input=add_ddpPrev_input,
        add_gradDpPrev_input=add_gradDpPrev_input,
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
        add_divDDUStar_input=add_divDDUStar_input,
        add_divDUStar_input=add_divDUStar_input,
        add_divUStar_input=add_divUStar_input,
        add_dUCorrPrev_input=add_dUCorrPrev_input,
        add_ddUCorrPrev_input=add_ddUCorrPrev_input,
        predict_ddUCorr_output=predict_ddUCorr_output,
    )
    feature_writer(core_data_fn, compute_tucker_factors=False)
    print("Feature extraction complete.")

    # --- Combine old and new features ---
    with tables.open_file(core_data_fn, mode='r') as f:
        new_input_cores = f.root.inputs[...]
        new_output_cores = f.root.outputs[...]

    all_input_cores = np.concatenate([old_input_cores_to_keep, new_input_cores], axis=0)
    all_output_cores = np.concatenate([old_output_cores_to_keep, new_output_cores], axis=0)

    with tables.open_file(core_data_fn, mode='w') as f:
        atom = tables.Atom.from_dtype(all_input_cores.dtype)
        input_array = f.create_carray(f.root, 'inputs', atom, all_input_cores.shape)
        output_array = f.create_carray(f.root, 'outputs', atom, all_output_cores.shape)
        input_array[:] = all_input_cores
        output_array[:] = all_output_cores
    print("Old and new features combined and saved to core_data.h5.")

    # --- Retrain model with combined data ---
    n_layers, width = utils_model.define_model_arch(model_architecture)
    model_name = f'{model_architecture}-{standardization_method}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'
    train_tfrecord_fn = os.path.join(data_dir, 'train_data.tfrecords')
    test_tfrecord_fn = os.path.join(data_dir, 'test_data.tfrecords')
    normalization_factors_fn = os.path.join(data_dir, 'mean_std.npz')

    Train = Training(standardization_method, train_tfrecord_fn, test_tfrecord_fn)

    # For the AUTO CFD solver:
    # Always regenerate TFRecords so they match the current sliding-window data.
    for fn in (train_tfrecord_fn, test_tfrecord_fn):
        if os.path.exists(fn):
            os.remove(fn)

    is_shifter_arch = model_architecture.lower() in [
        'cnn_shifter',
        'cnn_shifter_lightweight',
        'simplecnn3d_ddp_shifter',
        'simplecnn3d_ddp_shifter_lightweight',
        'cnn_shifter_velocity',
        'simplecnn3d_ddp_shifter_velocity',
    ]
    is_velocity_shifter_arch = model_architecture.lower() in [
        'cnn_shifter_velocity',
        'simplecnn3d_ddp_shifter_velocity',
    ]
    grad_ch_tuple = None
    u_ch_tuple = None
    if is_shifter_arch:
        grad_ch_tuple = (_ds_gradDpPrev_ch, _ds_gradDpPrev_ch + 1, _ds_gradDpPrev_ch + 2)
    if is_velocity_shifter_arch:
        u_ch_tuple = (0, 1, 2)

    Train.prepare_data_to_tf(
        core_data_fn,
        normalization_factors_fn,
        flatten_data=use_feature_decomposition,
        load_existing_normalization=False,
        include_dp_prev_in_y=not is_shifter_arch,
        include_gradDpPrev_in_y=is_shifter_arch and not is_velocity_shifter_arch,
        include_velocity_components_in_y=is_velocity_shifter_arch,
        include_uDotGradDpPrev_in_y=False,
        dp_prev_input_ch_idx=_ds_dpPrev_ch,
        gradDpPrev_input_ch_idxs=grad_ch_tuple,
        U_input_ch_idxs=u_ch_tuple,
        uDotGradDpPrev_input_ch_idx=None,
    )
    Train.load_data_and_train(
        lr=lr,
        batch_size=batch_size,
        model_name=model_name,
        beta_1=beta,
        num_epoch=num_epochs,
        n_layers=n_layers,
        width=width,
        dropout_rate=dropout_rate,
        regularization=regularization,
        model_architecture=model_architecture,
        new_model=retrain_from_scratch,
        spatial_tucker_ranks=spatial_tucker_ranks,
        flatten_data=use_feature_decomposition,
        weights_fn=os.path.join(data_dir, 'weights.h5'),
        model_h5_path=data_dir,
        last_tucker_rank=last_tucker_rank if use_feature_decomposition else (1 + 3 * (int(add_U_input) + int(add_dUStar_input) + int(add_ddUStar_input) + int(add_ddUStarDiff_input) + int(add_dUCorrPrev_input) + int(add_ddUCorrPrev_input)) + int(add_pPrev_input) + int(add_dpPrev_input) + int(add_ddpPrev_input) + 3 * int(add_gradDpPrev_input) + int(add_laplacian_dpPrev_input) + int(add_uDotGradDpPrev_input) + int(add_gradDpPrevMag_input) + int(include_rAU_input) + 3 * int(include_HbyA_input) + int(include_divHbyA_input) + 3 * int(include_dHbyA_input) + int(include_dDivHbyA_input) + 3 * int(add_rAUGradDpPrev_input) + int(add_divRAUGradDpPrev_input) + int(add_pressureEqResidualp_input) + 3 * int(add_rAUGradpPrev_input) + int(add_divRAUGradpPrev_input) + int(add_divDDUStar_input) + int(add_divDUStar_input) + int(add_divUStar_input)),
        use_feature_decomposition=use_feature_decomposition,
        block_size=block_size,
        predict_ddUCorr_output=predict_ddUCorr_output,
        dp_prev_input_ch_idx=_ds_dpPrev_ch,
        dp_prev_maxs_idx=_ds_dpPrev_ch + 1,  # +1: SDF at _vel_end shifts pressure channels by 1 in maxs file
        gradDpPrev_input_ch_idxs=grad_ch_tuple,
        U_input_ch_idxs=u_ch_tuple,
        uDotGradDpPrev_input_ch_idx=None,
            use_s_roi_penalty=use_s_roi_penalty,
    )
    print("Model training complete.")


if __name__ == '__main__':
    add_new_features_and_train()
