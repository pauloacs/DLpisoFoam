import numpy as np
import os
import glob
import shutil
import time
import pandas as pd

from pressure_SM_delta_delta_shift._3D.train_and_eval.data_processor import CFDDataProcessor
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import data_processing as utils_data
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import domain_geometry as utils_geo
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import io_operations as utils_io
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import sampling as utils_sampling
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import model_utils as utils_model
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils.data_processing import _unpack_grid_res

import h5py
from pressure_SM_delta_delta_shift._3D.train_and_eval.data_processor import FeatureExtractAndWrite
from pressure_SM_delta_delta_shift._3D.train_and_eval.train import Training
from pressure_SM_delta_delta_shift._3D.auto_CFD.hdf5_data_loader import load_hdf5_samples, save_cell_centers_and_boundaries, load_boundaries_dict
from pressure_SM_delta_delta_shift._3D.train_and_eval.neural_networks_shifter import SimpleCNN3D_ddp_shifter
from pressure_SM_delta_delta_shift._3D.train_and_eval.shifter_loss import ShifterLoss

# For debug plots
import matplotlib.pyplot as plt

# Interpolate field to grid using weights and vertices
# field_values: (N_cells,)
# vert: (N_grid, K), weights: (N_grid, K)
def interpolate_to_grid(field_values, vert, weights):
    # IDW interpolation: weighted sum over K nearest neighbors
    return np.sum(field_values[vert] * weights, axis=1)

def get_grid_limits(cell_centers, boundary_points):
    all_points = np.vstack([cell_centers, boundary_points])
    limits = {
        'x_min': np.min(all_points[:, 0]),
        'x_max': np.max(all_points[:, 0]),
        'y_min': np.min(all_points[:, 1]),
        'y_max': np.max(all_points[:, 1]),
        'z_min': np.min(all_points[:, 2]),
        'z_max': np.max(all_points[:, 2]),
    }
    return limits

def read_cell_centers():
    cell_centres = pd.read_csv('cell_centres.csv')  # DataFrame with columns x, y, z
    return cell_centres[['x', 'y', 'z']].values  # shape: (n_cells, 3)


def main():

        import argparse
        parser = argparse.ArgumentParser(description='Initialize interpolation weights and Tucker factors for ML training.')
        parser.add_argument('--data_dir', type=str, default='ML_data', help='Directory where the ML data is stored (default: ML_data)')
        args = parser.parse_args()
        data_dir = args.data_dir

        # Import shared config from python_module in the case directory (CWD)
        import sys
        import os
        os.environ['TRAIN_SCRIPT_MODE'] = '1'
        sys.path.insert(0, os.getcwd())

        from python_module import (
            grid_res, block_size, spatial_tucker_ranks, dropout_rate, regularization,
            model_architecture, standardization_method, n_samples_per_frame,
            lr, batch_size, beta, num_epochs, feature_extraction_chunk_size,
            n_representative_blocks, last_tucker_rank
        )

        try:
            from python_module import compression_clip_percentile
        except ImportError:
            compression_clip_percentile = 75.0  # backward-compatible default

        # use_feature_decomposition controls whether Tucker decomposition is applied.
        # When False, the full 3D block is passed directly to the NN (default: CNN).
        try:
            from python_module import use_feature_decomposition
        except ImportError:
            use_feature_decomposition = True  # backward-compatible default

        # add_ddUStar_input: include delta_delta_U as extra input alongside delta_delta_U_diff
        try:
            from python_module import add_ddUStar_input
        except ImportError:
            add_ddUStar_input = True  # backward-compatible default (current behaviour)

        # add_ddUStarDiff_input: include delta_delta_U_diff (dddU) as model input
        try:
            from python_module import add_ddUStarDiff_input
        except ImportError:
            add_ddUStarDiff_input = True  # backward-compatible default

        # add_U_input: include U (velocity) as extra input
        try:
            from python_module import add_U_input
        except ImportError:
            add_U_input = False  # default: don't include raw U

        # add_dpPrev_input: include dpPrev (previous pressure) as extra input
        try:
            from python_module import add_dpPrev_input
        except ImportError:
            add_dpPrev_input = False  # default: don't include previous pressure

        # add_ddpPrev_input: include ddpPrev (previous pressure double-increment) as extra input
        try:
            from python_module import add_ddpPrev_input
        except ImportError:
            add_ddpPrev_input = False  # default: don't include previous pressure double-increment

        # add_gradDpPrev_input: include grad(dpPrev) (3 channels) as extra input
        try:
            from python_module import add_gradDpPrev_input
        except ImportError:
            add_gradDpPrev_input = False  # default: don't include gradient of previous pressure increment

        # add_laplacian_dpPrev_input: include laplacian(dpPrev) (1 channel) as extra input
        try:
            from python_module import add_laplacian_dpPrev_input
        except ImportError:
            add_laplacian_dpPrev_input = False

        # add_uDotGradDpPrev_input: include U . grad(dpPrev) (1 channel) as extra input
        try:
            from python_module import add_uDotGradDpPrev_input
        except ImportError:
            add_uDotGradDpPrev_input = False

        # add_gradDpPrevMag_input: include |grad(dpPrev)| (1 channel) as extra input
        try:
            from python_module import add_gradDpPrevMag_input
        except ImportError:
            add_gradDpPrevMag_input = False

        # include_rAU_input: include rAU = 1/A(U) (1 channel) as extra input
        try:
            from python_module import include_rAU_input
        except ImportError:
            include_rAU_input = False

        # include_HbyA_input: include HbyA = rAU*H(U) (3 channels) as extra input
        try:
            from python_module import include_HbyA_input
        except ImportError:
            include_HbyA_input = False

        # include_divHbyA_input: include div(HbyA) (1 channel) as extra input
        try:
            from python_module import include_divHbyA_input
        except ImportError:
            include_divHbyA_input = False

        # include_dHbyA_input: include dHbyA = HbyA - HbyA_prev (3 channels) as extra input
        try:
            from python_module import include_dHbyA_input
        except ImportError:
            include_dHbyA_input = False

        # include_dDivHbyA_input: include dDivHbyA = divHbyA - divHbyA_prev (1 channel) as extra input
        try:
            from python_module import include_dDivHbyA_input
        except ImportError:
            include_dDivHbyA_input = False

        # add_rAUGradDpPrev_input: include rAU * grad(dpPrev) (3 channels) as extra input
        try:
            from python_module import add_rAUGradDpPrev_input
        except ImportError:
            add_rAUGradDpPrev_input = False

        # add_divRAUGradDpPrev_input: include div(rAU * grad(dpPrev)) (1 channel) as extra input
        try:
            from python_module import add_divRAUGradDpPrev_input
        except ImportError:
            add_divRAUGradDpPrev_input = False

        # add_pressureEqResidualp_input: include pressure equation residual div(HbyA)-div(rAU*grad(dpPrev)) (1 channel)
        try:
            from python_module import add_pressureEqResidualp_input
        except ImportError:
            add_pressureEqResidualp_input = False

        # add_rAUGradpPrev_input: include rAU * grad(pPrev) (3 channels) as extra input
        try:
            from python_module import add_rAUGradpPrev_input
        except ImportError:
            add_rAUGradpPrev_input = False

        # add_divRAUGradpPrev_input: include div(rAU * grad(pPrev)) (1 channel) as extra input
        try:
            from python_module import add_divRAUGradpPrev_input
        except ImportError:
            add_divRAUGradpPrev_input = False

        # add_dUStar_input: include delta_U (first velocity increment) as extra input
        try:
            from python_module import add_dUStar_input
        except ImportError:
            add_dUStar_input = False  # default: don't include first velocity increment

        # add_pPrev_input: include p_rgh_prev (absolute previous pressure) as extra input
        try:
            from python_module import add_pPrev_input
        except ImportError:
            add_pPrev_input = False  # default: don't include absolute previous pressure

        # add_divDDUStar_input: include div(delta_delta_U) (divergence of velocity double-increment) as extra input
        try:
            from python_module import add_divDDUStar_input
        except ImportError:
            add_divDDUStar_input = False  # default: don't include divergence

        # add_divDUStar_input: include div(delta_U) (divergence of velocity increment) as extra input
        try:
            from python_module import add_divDUStar_input
        except ImportError:
            add_divDUStar_input = False  # default: don't include divergence

        # add_divUStar_input: include div(U) (divergence of velocity) as extra input
        try:
            from python_module import add_divUStar_input
        except ImportError:
            add_divUStar_input = False  # default: don't include divergence

        # Current C++ raw layout exports only divUFirstPred (mapped to add_divUStar_input).
        # Legacy divDDU/divDU channels are not exported separately.
        if add_divDDUStar_input or add_divDUStar_input:
            raise ValueError(
                "add_divDDUStar_input/add_divDUStar_input are unsupported with the current solver raw layout. "
                "Only divUFirstPred is exported (use add_divUStar_input)."
            )

        # predict_ddUCorr_output: if True, model predicts [ddp, ddU_CFD_x, ddU_CFD_y, ddU_CFD_z] (4 outputs)
        try:
            from python_module import predict_ddUCorr_output
        except ImportError:
            predict_ddUCorr_output = False  # default: predict only ddp

        # add_dUCorrPrev_input: include dUCorrPrev (previous pressure-correction velocity increment) as extra input
        try:
            from python_module import add_dUCorrPrev_input
        except ImportError:
            add_dUCorrPrev_input = False  # default: don't include

        # add_ddUCorrPrev_input: include ddUCorrPrev (previous second pressure-correction velocity increment) as extra input
        try:
            from python_module import add_ddUCorrPrev_input
        except ImportError:
            add_ddUCorrPrev_input = False  # default: don't include

        # enforce_zero_mean_pressure: if True (default), subtract per-block/timestep domain mean
        # from pressure fields (ddp output and p_prev/dpPrev/ddpPrev inputs) before max_abs
        # normalization and during feature extraction. Set to False to disable mean removal.
        try:
            from python_module import enforce_zero_mean_pressure
        except ImportError:
            enforce_zero_mean_pressure = True  # default: keep current behaviour

        # save_plots_debug: if True, save debug PNG plots under plots_debug/
        try:
            from python_module import save_plots_debug
        except ImportError:
            save_plots_debug = False

        # generate_gif: if True, generate GIFs from debug variables under plots_debug/
        try:
            from python_module import generate_gif
        except ImportError:
            generate_gif = False

        # add_distance_to_outlet_input: include distance to outlet (1 channel) as model input
        try:
            from python_module import add_distance_to_outlet_input
        except ImportError:
            add_distance_to_outlet_input = False  # default: don't include

        # add_grad_sdf_input: include grad(sdf) (3 channels: x, y, z) as model input
        try:
            from python_module import add_grad_sdf_input
        except ImportError:
            add_grad_sdf_input = False  # default: don't include

        # add_UdotNwall_input: include U dot wall-normal (1 channel) as model input
        try:
            from python_module import add_UdotNwall_input
        except ImportError:
            add_UdotNwall_input = False  # default: don't include

        # clip_UdotNwall_to_inflow: if True with add_UdotNwall_input, clip to max(-Un*, 0)
        try:
            from python_module import clip_UdotNwall_to_inflow
        except ImportError:
            clip_UdotNwall_to_inflow = False  # default: keep signed values

        # use_s_roi_penalty: if True, enable ROI-based spatial regularization in Shifter loss
        try:
            from python_module import use_s_roi_penalty
        except ImportError:
            use_s_roi_penalty = True  # default: disabled

        # =========================================================
        # Configuration validation
        # predict_ddUCorr_output=True  → cnn_multi_out_divu + my_weighted_loss_split
        #   requires: use_feature_decomposition=False, add_divUStar_input=True
        # predict_ddUCorr_output=False → cnn_two_heads + my_mixed_weighted_mse_loss_masked
        # =========================================================
        if predict_ddUCorr_output:
            if use_feature_decomposition:
                raise ValueError(
                    "[predict_ddUCorr_output=True] 'cnn_multi_out_divu' requires raw spatial "
                    "blocks. Set 'use_feature_decomposition = False' in python_module."
                )
            if not add_divUStar_input:
                raise ValueError(
                    "[predict_ddUCorr_output=True] 'cnn_multi_out_divu' needs the divU channel "
                    "as input. Set 'add_divUStar_input = True' in python_module."
                )
        print(
            f"[Config] predict_ddUCorr_output={predict_ddUCorr_output} → "
            f"model='{'cnn_multi_out_divu' if predict_ddUCorr_output else 'cnn_two_heads'}', "
            f"use_feature_decomposition={use_feature_decomposition}"
        )

        # Allow block_size to be int or tuple, and always convert to tuple
        if isinstance(block_size, int):
            block_size_tuple = (block_size, block_size, block_size)
        else:
            block_size_tuple = tuple(block_size)

        # Use spatial_tucker_ranks everywhere instead of ranks
        if use_feature_decomposition:
            tucker_ranks_tuple = tuple(spatial_tucker_ranks)
        else:
            tucker_ranks_tuple = None

        gridded_h5_fn = os.path.join(data_dir, 'gridded_data.h5')
        sample_idx_fn = os.path.join(data_dir, 'sample_idx_per_time.npy')
        tucker_factors_fn = os.path.join(data_dir, 'tucker_factors.pkl')

        core_data_fn = os.path.join(data_dir, 'core_data.h5')

        # --- Load data from HDF5 ---
        hdf5_file = os.path.join(data_dir, 'data.h5')
        hdf5_file_copy = os.path.join(data_dir, 'data_init_copy.h5')
        print(f"Copying {hdf5_file} to {hdf5_file_copy} for safe reading...")

        # Wait until the file exists and is non-empty
        max_wait = 30
        waited = 0
        while not os.path.exists(hdf5_file) or os.path.getsize(hdf5_file) == 0:
            if waited >= max_wait:
                print(f"Timeout waiting for {hdf5_file} to be available.")
                exit(1)
            time.sleep(1)
            waited += 1

        shutil.copy2(hdf5_file, hdf5_file_copy)
        print(f"Copied to {hdf5_file_copy}.")


        # Load delta-delta fields (second differences) from HDF5
        try:
            cell_centers, boundary_coords, boundary_patches, patch_names, U, ddUStar, ddUStarDiff, dpPrev, ddpPrev, gradDpPrev, laplaceDpPrev, uDotGradDpPrev, gradDpPrevMag, rAU, HbyA, divHbyA, dHbyA, dDivHbyA, rAUGradDpPrev, divRAUGradDpPrev, pressureEqResidualp, rAUGradpPrev, divRAUGradpPrev, divDDUStar, divUStar, divDUStar, dUStar, dUCorrPrev, ddUCorrPrev, p_prev, ddp, ddUCorr, timestamps, U_MAX_NORM_arr = \
                load_hdf5_samples(hdf5_file_copy)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading HDF5 data: {e}")
            exit(1)
        finally:
            if os.path.exists(hdf5_file_copy):
                os.remove(hdf5_file_copy)
            # Delete original so C++ creates a fresh file for the next batch
            if os.path.exists(hdf5_file):
                os.remove(hdf5_file)
                print(f"Deleted {hdf5_file} after data extraction — C++ will create a fresh file for the next batch.")

        n_sample_frames = len(timestamps)
        print(f"Loaded {n_sample_frames} samples from HDF5 file")
        print(f"Cell centers shape: {cell_centers.shape}")
        print(f"Boundary coordinates shape: {boundary_coords.shape if boundary_coords is not None else 'None'}")
        print(f"U shape: {U.shape}, ddUStar shape: {ddUStar.shape}, ddp shape: {ddp.shape}, dpPrev shape: {dpPrev.shape}, ddpPrev shape: {ddpPrev.shape}, gradDpPrev shape: {gradDpPrev.shape}")
        print(f"laplaceDpPrev shape: {laplaceDpPrev.shape}, uDotGradDpPrev shape: {uDotGradDpPrev.shape}, gradDpPrevMag shape: {gradDpPrevMag.shape}")
        print(f"rAUGradDpPrev shape: {rAUGradDpPrev.shape}, divRAUGradDpPrev shape: {divRAUGradDpPrev.shape}, pressureEqResidualp shape: {pressureEqResidualp.shape}")
        print(f"rAUGradpPrev shape: {rAUGradpPrev.shape}, divRAUGradpPrev shape: {divRAUGradpPrev.shape}")
        print(f"dUStar shape: {dUStar.shape}, p_prev shape: {p_prev.shape}")
        print(f"dUCorrPrev shape: {dUCorrPrev.shape}, ddUCorrPrev shape: {ddUCorrPrev.shape}")
        print(f"U_MAX_NORM array shape: {U_MAX_NORM_arr.shape}")

        # Save coordinates and boundaries for compatibility
        save_cell_centers_and_boundaries(cell_centers, boundary_coords, boundary_patches, 
                                         patch_names, data_dir)

        # Use boundary coordinates (already concatenated) for domain_dist
        if boundary_coords is not None and len(boundary_coords) > 0:
            boundary_points = boundary_coords
        else:
            raise ValueError("No boundary points loaded")

        vert, weights = utils_data.interp_weights(cell_centers, boundary_points, interp_method='IDW')
        np.save(os.path.join(data_dir, 'interp_weights.npy'), weights)
        np.save(os.path.join(data_dir, 'interp_vertices.npy'), vert)
        print("Interpolation weights and vertices saved.")

        # Get grid limits using both cell centers and boundaries
        cfd_mesh_limits = get_grid_limits(cell_centers, boundary_points)
        print('CFD mesh limits:', cfd_mesh_limits)

        # Create the grid using the limits and grid_res
        X0, Y0, Z0 = utils_data.create_uniform_grid(cfd_mesh_limits, grid_res)
        grid_points = np.concatenate(
            (np.expand_dims(X0, axis=1),
             np.expand_dims(Y0, axis=1),
             np.expand_dims(Z0, axis=1)),
            axis=-1
        )
        grid_limits = {
            'x_min': X0.min(),
            'x_max': X0.max(),
            'y_min': Y0.min(),
            'y_max': Y0.max(),
            'z_min': Z0.min(),
            'z_max': Z0.max(),
        }

        boundaries_dict = load_boundaries_dict(data_dir)
        # Calculate SDF and domain mask on the grid using boundary dict
        grid_shape_x, grid_shape_y, grid_shape_z = utils_data.get_grid_shape(cfd_mesh_limits, grid_res)

        # Adjust block_size_tuple if any dimension exceeds grid shape
        block_size_z, block_size_y, block_size_x = block_size_tuple
        if block_size_z > grid_shape_z:
            block_size_z = grid_shape_z
        if block_size_y > grid_shape_y:
            block_size_y = grid_shape_y
        if block_size_x > grid_shape_x:
            block_size_x = grid_shape_x
        block_size_tuple = (block_size_z, block_size_y, block_size_x)
        print(f"Adjusted block_size_tuple to: {block_size_tuple} (grid_shape: z={grid_shape_z}, y={grid_shape_y}, x={grid_shape_x})")

        domain_bool, sdf = utils_geo.domain_dist(boundaries_dict, grid_points, grid_res)
        np.save(os.path.join(data_dir, 'grid_sdf_flat.npy'), sdf)
        np.save(os.path.join(data_dir, 'grid_domain_mask_flat.npy'), domain_bool)
        print('SDF and domain mask arrays saved.')

        # Compute interpolation weights and vertices from cell centers to grid points
        vert, weights = utils_data.interp_weights(cell_centers, grid_points, interp_method='IDW')
        np.save(os.path.join(data_dir, 'interp_weights.npy'), weights)
        np.save(os.path.join(data_dir, 'interp_vertices.npy'), vert)
        print("Interpolation weights and vertices (cell centers -> grid) saved.")


        # --- Interpolate delta_delta_U and ddp to grid ---
        # delta_delta_U has shape (n_samples, n_cells, 3) - need to interpolate each component
        # ddp has shape (n_samples, n_cells) - scalar field

        print("Interpolating delta-delta fields to grid and normalizing with U_MAX_NORM...")
        n_samples = ddUStar.shape[0]
        n_grid_points = grid_points.shape[0]

        # Pass block_size_tuple to all downstream processing and feature extraction
        # Example: when initializing CFDDataProcessor or FeatureExtractAndWrite, use block_size=block_size_tuple
        # ...existing code...

        # Initialize output arrays
        U_grid_flat = None
        dU_grid_flat = None
        ddUStar_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
        ddUStarDiff_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
        ddp_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)
        p_prev_grid_flat = None
        dpPrev_grid_flat = None
        ddpPrev_grid_flat = None
        divDDUStar_grid_flat = None
        divDUStar_grid_flat = None
        divUStar_grid_flat = None

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

        # Interpolate and normalize each sample
        for sample_idx in range(n_samples):
            norm = U_MAX_NORM_arr[sample_idx]

            # Interpolate U if requested
            if add_U_input and 'U' in locals():
                try:
                    for component in range(3):
                        U_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                            U[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                        )
                except Exception as e:
                    print(f"Warning: Could not interpolate U (add_U_input=True but U not available): {e}")
                    add_U_input = False

            # Interpolate dUStar if requested
            if add_dUStar_input:
                for component in range(3):
                    dUStar_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        dUStar[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )

            # Interpolate dUCorrPrev and ddUCorrPrev if requested
            if add_dUCorrPrev_input:
                for component in range(3):
                    dUCorrPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        dUCorrPrev[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )
            if add_ddUCorrPrev_input:
                for component in range(3):
                    ddUCorrPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        ddUCorrPrev[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )

            # Interpolate and normalize ddUStar components
            for component in range(3):
                ddUStar_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    ddUStar[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                )
                ddUStarDiff_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    ddUStarDiff[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                )

            # Interpolate and normalize previous pressure (if enabled)
            if add_pPrev_input:
                p_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    p_prev[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
                )

            # Interpolate and normalize previous pressure (if enabled)
            if add_dpPrev_input:
                dpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    dpPrev[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
                )

            # Interpolate and normalize previous pressure double-increment (if enabled)
            if add_ddpPrev_input:
                ddpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    ddpPrev[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
                )

            if add_gradDpPrev_input:
                for component in range(3):
                    gradDpPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        gradDpPrev[sample_idx, :, component] / (norm ** 2), vert, weights, fill_value=np.nan
                    )

            if add_laplacian_dpPrev_input:
                laplaceDpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    laplaceDpPrev[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
                )

            if add_uDotGradDpPrev_input:
                uDotGradDpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    uDotGradDpPrev[sample_idx, :] / (norm ** 3), vert, weights, fill_value=np.nan
                )

            if add_gradDpPrevMag_input:
                gradDpPrevMag_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    gradDpPrevMag[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
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
                        HbyA[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )

            # divHbyA [1/s] — velocity-scaled (consistent with div(U))
            if include_divHbyA_input:
                divHbyA_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    divHbyA[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # dHbyA [m/s] — velocity-scaled (temporal variation of HbyA)
            if include_dHbyA_input:
                for component in range(3):
                    dHbyA_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        dHbyA[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )

            # dDivHbyA [1/s] — velocity-scaled (temporal variation of divHbyA)
            if include_dDivHbyA_input:
                dDivHbyA_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    dDivHbyA[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # rAU*grad(dpPrev) [m/s] — velocity-scaled
            if add_rAUGradDpPrev_input:
                for component in range(3):
                    rAUGradDpPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        rAUGradDpPrev[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )

            # div(rAU*grad(dpPrev)) [1/s] — velocity-scaled
            if add_divRAUGradDpPrev_input:
                divRAUGradDpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    divRAUGradDpPrev[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # Pressure equation residual [1/s] — velocity-scaled
            if add_pressureEqResidualp_input:
                pressureEqResidualp_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    pressureEqResidualp[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # rAU*grad(pPrev) [m/s] — velocity-scaled
            if add_rAUGradpPrev_input:
                for component in range(3):
                    rAUGradpPrev_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        rAUGradpPrev[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )

            # div(rAU*grad(pPrev)) [1/s] — velocity-scaled
            if add_divRAUGradpPrev_input:
                divRAUGradpPrev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    divRAUGradpPrev[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # Interpolate and normalize divergence of ddUStar (if enabled)
            if add_divDDUStar_input:
                divDDUStar_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    divDDUStar[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # Interpolate and normalize divergence of dUStar (if enabled)
            if add_divDUStar_input:
                divDUStar_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    divDUStar[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # Interpolate and normalize divergence of UStar (if enabled)
            if add_divUStar_input:
                divUStar_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                    divUStar[sample_idx, :] / norm, vert, weights, fill_value=np.nan
                )

            # Interpolate and normalize ddUCorr (training output for predict_ddUCorr)
            if predict_ddUCorr_output:
                for component in range(3):
                    ddUCorr_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                        ddUCorr[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                    )

            # Interpolate and normalize delta-delta pressure
            ddp_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                ddp[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
            )


        print("Interpolation and normalization to grid complete.")

        # Stack: choose dataset channels based on flags
        # Channel order: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU [p_prev if add_p_prev] [dpPrev if add_dpPrev] [ddpPrev if add_ddpPrev] [div_ddu if add_div_ddu] [div_du if add_div_du] [div_u if add_div_u] ddp
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


        # Save indices for later reuse
        # Generate indices mapping grid points to (i, j, k) indices
        dx, dy, dz = _unpack_grid_res(grid_res)
        x0 = grid_points[:, 0].min()
        y0 = grid_points[:, 1].min()
        z0 = grid_points[:, 2].min()

        xyz0 = grid_points
        indices = np.full((xyz0.shape[0], 3), np.nan, dtype=float)
        sdfunct = np.full((grid_shape_z, grid_shape_y, grid_shape_x, 1), 0.0, dtype=float)
        obst_bool = np.zeros_like(sdfunct, dtype=int)

        # Example: using delta_ux_interp as delta_U_grid[..., 0]
        delta_ux_interp = ddUStar_grid_flat[0, :, 0]  # first sample, x-component

        for step, x_y_z in enumerate(xyz0):
            ii = int(round((x_y_z[2] - z0) / dz))
            jj = int(round((x_y_z[1] - y0) / dy))
            kk = int(round((x_y_z[0] - x0) / dx))
            indices[step, 0] = ii
            indices[step, 1] = jj
            indices[step, 2] = kk
            if domain_bool[step] and not np.isnan(delta_ux_interp[step]):
                sdfunct[ii, jj, kk, 0] = sdf[step]
                obst_bool[ii, jj, kk, 0] = 1

        indices = indices.astype(int)
        indices_i = indices[:, 0].astype(np.int32)
        indices_j = indices[:, 1].astype(np.int32)
        indices_k = indices[:, 2].astype(np.int32)

        # Compute static geometric input grids (shared across all time steps)
        dist_to_outlet_flat = None
        grad_sdf_flat_x = None
        grad_sdf_flat_y = None
        grad_sdf_flat_z = None
        UdotNwall_grid_flat = None

        if add_distance_to_outlet_input:
            if 'outlet_boundary' in boundaries_dict and boundaries_dict['outlet_boundary'] is not None and len(boundaries_dict['outlet_boundary']) > 0:
                from scipy.spatial import cKDTree as _cKDTree
                _outlet_pts = boundaries_dict['outlet_boundary']
                _tree = _cKDTree(_outlet_pts)
                dist_to_outlet_flat = _tree.query(grid_points)[0]
            else:
                # Fallback: distance to x_max face (typical outlet position)
                dist_to_outlet_flat = np.max(X0) - grid_points[:, 0]
            dist_to_outlet_flat = dist_to_outlet_flat * domain_bool.astype(float)
            print(f"[add_distance_to_outlet_input] computed distance_to_outlet: max={dist_to_outlet_flat.max():.4f}")

        if add_grad_sdf_input or add_UdotNwall_input:
            _dx_g, _dy_g, _dz_g = _unpack_grid_res(grid_res)
            _sdf_3d = sdfunct[:, :, :, 0]
            _grad_sdf_z_3d, _grad_sdf_y_3d, _grad_sdf_x_3d = np.gradient(_sdf_3d, _dz_g, _dy_g, _dx_g)
            # Map 3D gradients back to flat (grid-point order), components in physical x,y,z order
            grad_sdf_flat_x = _grad_sdf_x_3d[indices_i, indices_j, indices_k]
            grad_sdf_flat_y = _grad_sdf_y_3d[indices_i, indices_j, indices_k]
            grad_sdf_flat_z = _grad_sdf_z_3d[indices_i, indices_j, indices_k]
            print(f"[add_grad_sdf_input] computed grad_sdf: max_x={np.abs(grad_sdf_flat_x).max():.4f}, max_y={np.abs(grad_sdf_flat_y).max():.4f}, max_z={np.abs(grad_sdf_flat_z).max():.4f}")

        if add_UdotNwall_input:
            _grad_mag = np.sqrt(grad_sdf_flat_x**2 + grad_sdf_flat_y**2 + grad_sdf_flat_z**2) + 1e-12
            _nwall_x = grad_sdf_flat_x / _grad_mag
            _nwall_y = grad_sdf_flat_y / _grad_mag
            _nwall_z = grad_sdf_flat_z / _grad_mag
            UdotNwall_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)
            for sample_idx in range(n_samples):
                norm = U_MAX_NORM_arr[sample_idx]
                if add_U_input and U_grid_flat is not None:
                    _Ux = U_grid_flat[sample_idx, :, 0]
                    _Uy = U_grid_flat[sample_idx, :, 1]
                    _Uz = U_grid_flat[sample_idx, :, 2]
                else:
                    _Ux = utils_data.interpolate_fill_njit(U[sample_idx, :, 0] / norm, vert, weights, fill_value=np.nan)
                    _Uy = utils_data.interpolate_fill_njit(U[sample_idx, :, 1] / norm, vert, weights, fill_value=np.nan)
                    _Uz = utils_data.interpolate_fill_njit(U[sample_idx, :, 2] / norm, vert, weights, fill_value=np.nan)

                _Un = _Ux * _nwall_x + _Uy * _nwall_y + _Uz * _nwall_z
                if clip_UdotNwall_to_inflow:
                    _Un = np.maximum(-_Un, 0.0)
                UdotNwall_grid_flat[sample_idx, :] = _Un * domain_bool.astype(float)
            print(f"[add_UdotNwall_input] computed UdotNwall: max={np.nanmax(np.abs(UdotNwall_grid_flat)):.4f}")

        indices_save_path = os.path.join(data_dir, 'interpolation_indices.npz')
        np.savez(indices_save_path, indices=indices, indices_i=indices_i, indices_j=indices_j, indices_k=indices_k)
        print(f"Saved interpolation indices to {indices_save_path}.")
        indices_save_path = os.path.join(data_dir, 'interpolation_indices.npz')


        # Prepare gridded array for saving
        # Calculate total channels: sdf(1) + ddp(1) + optional flags
        n_grid_channels = 2  # sdf, dpML (+ optional flags)
        if add_U_input:
            n_grid_channels += 3  # U
        if add_dUStar_input:
            n_grid_channels += 3  # dU
        if add_ddUStar_input:
            n_grid_channels += 3  # ddU
        if add_ddUStarDiff_input:
            n_grid_channels += 3  # dddU
        if add_dUCorrPrev_input:
            n_grid_channels += 3  # dUCorrPrev
        if add_ddUCorrPrev_input:
            n_grid_channels += 3  # ddUCorrPrev
        if add_pPrev_input:
            n_grid_channels += 1  # p_prev
        if add_dpPrev_input:
            n_grid_channels += 1  # dpPrev
        if add_ddpPrev_input:
            n_grid_channels += 1  # ddpPrev
        if add_gradDpPrev_input:
            n_grid_channels += 3  # gradDpPrev
        if add_laplacian_dpPrev_input:
            n_grid_channels += 1  # laplace(dpPrev)
        if add_uDotGradDpPrev_input:
            n_grid_channels += 1  # U . grad(dpPrev)
        if add_gradDpPrevMag_input:
            n_grid_channels += 1  # |grad(dpPrev)|
        if include_rAU_input:
            n_grid_channels += 1  # rAU
        if include_HbyA_input:
            n_grid_channels += 3  # HbyA
        if include_divHbyA_input:
            n_grid_channels += 1  # div(HbyA)
        if include_dHbyA_input:
            n_grid_channels += 3  # dHbyA
        if include_dDivHbyA_input:
            n_grid_channels += 1  # dDivHbyA
        if add_rAUGradDpPrev_input:
            n_grid_channels += 3  # rAUGradDpPrev
        if add_divRAUGradDpPrev_input:
            n_grid_channels += 1  # divRAUGradDpPrev
        if add_pressureEqResidualp_input:
            n_grid_channels += 1  # pressureEqResidualp
        if add_rAUGradpPrev_input:
            n_grid_channels += 3  # rAUGradpPrev
        if add_divRAUGradpPrev_input:
            n_grid_channels += 1  # divRAUGradpPrev
        if add_divDDUStar_input:
            n_grid_channels += 1  # div_delta_delta_U
        if add_divDUStar_input:
            n_grid_channels += 1  # div_dU
        if add_divUStar_input:
            n_grid_channels += 1  # div_U
        if add_distance_to_outlet_input:
            n_grid_channels += 1  # distance_to_outlet
        if add_grad_sdf_input:
            n_grid_channels += 3  # grad_sdf_x, grad_sdf_y, grad_sdf_z
        if add_UdotNwall_input:
            n_grid_channels += 1  # U dot wall-normal
        if predict_ddUCorr_output:
            n_grid_channels += 3  # ddU_CFD_x, ddU_CFD_y, ddU_CFD_z

        grid_shape = (n_samples,) + sdfunct.shape[:3] + (n_grid_channels,)
        dataset_gridded = np.full(grid_shape, np.nan, dtype=np.float64)

        # Calculate channel indices based on what's included
        ch_idx = 0
        u_idx = (ch_idx, ch_idx+3) if add_U_input else None
        if add_U_input:
            ch_idx += 3
        dU_idx = (ch_idx, ch_idx+3) if add_dUStar_input else None
        if add_dUStar_input:
            ch_idx += 3
        ddu_idx = (ch_idx, ch_idx+3) if add_ddUStar_input else None
        if add_ddUStar_input:
            ch_idx += 3
        dddu_idx = (ch_idx, ch_idx+3) if add_ddUStarDiff_input else None
        if add_ddUStarDiff_input:
            ch_idx += 3
        dUCorrPrev_idx = (ch_idx, ch_idx+3) if add_dUCorrPrev_input else None
        if add_dUCorrPrev_input:
            ch_idx += 3
        ddUCorrPrev_idx = (ch_idx, ch_idx+3) if add_ddUCorrPrev_input else None
        if add_ddUCorrPrev_input:
            ch_idx += 3

        p_prev_idx = ch_idx if add_pPrev_input else None
        ch_idx += 1 if add_pPrev_input else 0
        dpPrev_idx = ch_idx if add_dpPrev_input else None
        ch_idx += 1 if add_dpPrev_input else 0
        ddpPrev_idx = ch_idx if add_ddpPrev_input else None
        ch_idx += 1 if add_ddpPrev_input else 0
        gradDpPrev_idx = (ch_idx, ch_idx + 3) if add_gradDpPrev_input else None
        ch_idx += 3 if add_gradDpPrev_input else 0
        laplaceDpPrev_idx = ch_idx if add_laplacian_dpPrev_input else None
        ch_idx += 1 if add_laplacian_dpPrev_input else 0
        uDotGradDpPrev_idx = ch_idx if add_uDotGradDpPrev_input else None
        ch_idx += 1 if add_uDotGradDpPrev_input else 0
        gradDpPrevMag_idx = ch_idx if add_gradDpPrevMag_input else None
        ch_idx += 1 if add_gradDpPrevMag_input else 0
        rAU_idx = ch_idx if include_rAU_input else None
        ch_idx += 1 if include_rAU_input else 0
        HbyA_idx = (ch_idx, ch_idx + 3) if include_HbyA_input else None
        ch_idx += 3 if include_HbyA_input else 0
        divHbyA_idx = ch_idx if include_divHbyA_input else None
        ch_idx += 1 if include_divHbyA_input else 0
        dHbyA_idx = (ch_idx, ch_idx + 3) if include_dHbyA_input else None
        ch_idx += 3 if include_dHbyA_input else 0
        dDivHbyA_idx = ch_idx if include_dDivHbyA_input else None
        ch_idx += 1 if include_dDivHbyA_input else 0
        rAUGradDpPrev_idx = (ch_idx, ch_idx + 3) if add_rAUGradDpPrev_input else None
        ch_idx += 3 if add_rAUGradDpPrev_input else 0
        divRAUGradDpPrev_idx = ch_idx if add_divRAUGradDpPrev_input else None
        ch_idx += 1 if add_divRAUGradDpPrev_input else 0
        pressureEqResidualp_idx = ch_idx if add_pressureEqResidualp_input else None
        ch_idx += 1 if add_pressureEqResidualp_input else 0
        rAUGradpPrev_idx = (ch_idx, ch_idx + 3) if add_rAUGradpPrev_input else None
        ch_idx += 3 if add_rAUGradpPrev_input else 0
        divRAUGradpPrev_idx = ch_idx if add_divRAUGradpPrev_input else None
        ch_idx += 1 if add_divRAUGradpPrev_input else 0
        div_ddu_idx = ch_idx if add_divDDUStar_input else None
        ch_idx += 1 if add_divDDUStar_input else 0
        div_du_idx = ch_idx if add_divDUStar_input else None
        ch_idx += 1 if add_divDUStar_input else 0
        div_u_idx = ch_idx if add_divUStar_input else None
        ch_idx += 1 if add_divUStar_input else 0
        sdf_idx = ch_idx  # sdf after all velocity/pressure/div inputs, before static geometric extras
        ch_idx += 1
        dist_to_outlet_idx = ch_idx if add_distance_to_outlet_input else None
        ch_idx += 1 if add_distance_to_outlet_input else 0
        grad_sdf_idx = (ch_idx, ch_idx + 3) if add_grad_sdf_input else None
        ch_idx += 3 if add_grad_sdf_input else 0
        UdotNwall_idx = ch_idx if add_UdotNwall_input else None
        ch_idx += 1 if add_UdotNwall_input else 0
        ddp_idx = ch_idx
        ch_idx += 1
        ddU_CFD_idx = (ch_idx, ch_idx + 3) if predict_ddUCorr_output else None
        ch_idx += 3 if predict_ddUCorr_output else 0

        # Compute explicit flat channel indices in 'dataset' (which has no sdf channel)
        _vel_end = (ddUCorrPrev_idx[1] if add_ddUCorrPrev_input else (dUCorrPrev_idx[1] if add_dUCorrPrev_input else (dddu_idx[1] if add_ddUStarDiff_input else (ddu_idx[1] if add_ddUStar_input else (dU_idx[1] if add_dUStar_input else (u_idx[1] if add_U_input else 0))))))
        _ds_base = _vel_end  # start of pressure channels in dataset
        dataset_p_prev_ch = _ds_base  # p_prev position in dataset (if add_pPrev_input)
        dataset_dpPrev_ch = _ds_base + (1 if add_pPrev_input else 0)  # dpPrev position
        dataset_ddpPrev_ch = dataset_dpPrev_ch + (1 if add_dpPrev_input else 0)  # ddpPrev position
        dataset_gradDpPrev_ch = dataset_ddpPrev_ch + (1 if add_ddpPrev_input else 0)  # gradDpPrev position
        dataset_laplaceDpPrev_ch = dataset_gradDpPrev_ch + (3 if add_gradDpPrev_input else 0)
        dataset_uDotGradDpPrev_ch = dataset_laplaceDpPrev_ch + (1 if add_laplacian_dpPrev_input else 0)
        dataset_gradDpPrevMag_ch = dataset_uDotGradDpPrev_ch + (1 if add_uDotGradDpPrev_input else 0)
        dataset_rAU_ch = dataset_gradDpPrevMag_ch + (1 if add_gradDpPrevMag_input else 0)  # rAU position
        dataset_HbyA_ch = dataset_rAU_ch + (1 if include_rAU_input else 0)  # HbyA position (3 ch)
        dataset_divHbyA_ch = dataset_HbyA_ch + (3 if include_HbyA_input else 0)  # divHbyA position
        dataset_dHbyA_ch = dataset_divHbyA_ch + (1 if include_divHbyA_input else 0)  # dHbyA position (3 ch)
        dataset_dDivHbyA_ch = dataset_dHbyA_ch + (3 if include_dHbyA_input else 0)  # dDivHbyA position
        dataset_rAUGradDpPrev_ch = dataset_dDivHbyA_ch + (1 if include_dDivHbyA_input else 0)  # rAUGradDpPrev position (3 ch)
        dataset_divRAUGradDpPrev_ch = dataset_rAUGradDpPrev_ch + (3 if add_rAUGradDpPrev_input else 0)
        dataset_pressureEqResidualp_ch = dataset_divRAUGradDpPrev_ch + (1 if add_divRAUGradDpPrev_input else 0)
        dataset_rAUGradpPrev_ch = dataset_pressureEqResidualp_ch + (1 if add_pressureEqResidualp_input else 0)  # rAUGradpPrev position (3 ch)
        dataset_divRAUGradpPrev_ch = dataset_rAUGradpPrev_ch + (3 if add_rAUGradpPrev_input else 0)
        dataset_div_ddu_ch = dataset_divRAUGradpPrev_ch + (1 if add_divRAUGradpPrev_input else 0)  # div_ddu position
        dataset_div_du_ch = dataset_div_ddu_ch + (1 if add_divDDUStar_input else 0)  # div_du position
        dataset_div_u_ch = dataset_div_du_ch + (1 if add_divDUStar_input else 0)  # div_u position
        dataset_ddp_ch = dataset_div_u_ch + (1 if add_divUStar_input else 0)  # ddp position



        # ============================================================
        # Setup: prepare compression function and settings
        # ============================================================
        clip_ddp_and_gradDpPrev = True
        clip_percentile = compression_clip_percentile

        fluid_mask_3d = obst_bool[:, :, :, 0] != 0  # True inside flow domain

        def compress_tail_asinh_with_threshold(arr3d, fluid_mask, threshold):
            """
            Apply asinh compression using a provided threshold (not computed from data).
            
            Parameters
            ----------
            arr3d : ndarray
                3D array to compress
            fluid_mask : ndarray (bool)
                Mask indicating fluid domain (True = fluid)
            threshold : float
                Pre-computed compression threshold
            
            Returns
            -------
            out : ndarray
                Compressed field
            """
            out = arr3d.copy()
            x = out[fluid_mask]
            ax = np.abs(x)
            mask = ax > threshold
            out_vals = x.copy()
            excess = ax[mask] - threshold
            compressed = threshold + threshold * np.arcsinh(excess / (threshold + 1e-30))
            out_vals[mask] = np.sign(x[mask]) * compressed
            out[fluid_mask] = out_vals
            return out

        def compress_tail_asinh(arr3d, fluid_mask, q=95.0):
            vals = np.abs(arr3d[fluid_mask])
            if vals.size == 0:
                return arr3d, np.nan

            thr = np.percentile(vals, q)
            out = arr3d.copy()

            x = out[fluid_mask]
            ax = np.abs(x)
            mask = ax > thr
            out_vals = x.copy()

            excess = ax[mask] - thr
            compressed = thr + thr * np.arcsinh(excess / (thr + 1e-30))
            out_vals[mask] = np.sign(x[mask]) * compressed
            out[fluid_mask] = out_vals
            return out, thr

        # First pass: fill dataset_gridded with data from all samples
        for step in range(n_samples):
            if add_U_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, u_idx[0]:u_idx[1]] = dataset[step, :, u_idx[0]:u_idx[1]]
            if add_dUStar_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, dU_idx[0]:dU_idx[1]] = dataset[step, :, dU_idx[0]:dU_idx[1]]
            if add_ddUStar_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, ddu_idx[0]:ddu_idx[1]] = dataset[step, :, ddu_idx[0]:ddu_idx[1]]
            if add_ddUStarDiff_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, dddu_idx[0]:dddu_idx[1]] = dataset[step, :, dddu_idx[0]:dddu_idx[1]]
            if add_dUCorrPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, dUCorrPrev_idx[0]:dUCorrPrev_idx[1]] = dataset[step, :, dUCorrPrev_idx[0]:dUCorrPrev_idx[1]]
            if add_ddUCorrPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, ddUCorrPrev_idx[0]:ddUCorrPrev_idx[1]] = dataset[step, :, ddUCorrPrev_idx[0]:ddUCorrPrev_idx[1]]
            if add_pPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, p_prev_idx] = dataset[step, :, dataset_p_prev_ch]
            if add_dpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, dpPrev_idx] = dataset[step, :, dataset_dpPrev_ch]
            if add_ddpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, ddpPrev_idx] = dataset[step, :, dataset_ddpPrev_ch]
            if add_gradDpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, gradDpPrev_idx[0]:gradDpPrev_idx[1]] = dataset[step, :, dataset_gradDpPrev_ch:dataset_gradDpPrev_ch+3]
            if add_laplacian_dpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, laplaceDpPrev_idx] = dataset[step, :, dataset_laplaceDpPrev_ch]
            if add_uDotGradDpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, uDotGradDpPrev_idx] = dataset[step, :, dataset_uDotGradDpPrev_ch]
            if add_gradDpPrevMag_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, gradDpPrevMag_idx] = dataset[step, :, dataset_gradDpPrevMag_ch]
            if include_rAU_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, rAU_idx] = dataset[step, :, dataset_rAU_ch]
            if include_HbyA_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, HbyA_idx[0]:HbyA_idx[1]] = dataset[step, :, dataset_HbyA_ch:dataset_HbyA_ch+3]
            if include_divHbyA_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, divHbyA_idx] = dataset[step, :, dataset_divHbyA_ch]
            if include_dHbyA_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, dHbyA_idx[0]:dHbyA_idx[1]] = dataset[step, :, dataset_dHbyA_ch:dataset_dHbyA_ch+3]
            if include_dDivHbyA_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, dDivHbyA_idx] = dataset[step, :, dataset_dDivHbyA_ch]
            if add_rAUGradDpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, rAUGradDpPrev_idx[0]:rAUGradDpPrev_idx[1]] = dataset[step, :, dataset_rAUGradDpPrev_ch:dataset_rAUGradDpPrev_ch+3]
            if add_divRAUGradDpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, divRAUGradDpPrev_idx] = dataset[step, :, dataset_divRAUGradDpPrev_ch]
            if add_pressureEqResidualp_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, pressureEqResidualp_idx] = dataset[step, :, dataset_pressureEqResidualp_ch]
            if add_rAUGradpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, rAUGradpPrev_idx[0]:rAUGradpPrev_idx[1]] = dataset[step, :, dataset_rAUGradpPrev_ch:dataset_rAUGradpPrev_ch+3]
            if add_divRAUGradpPrev_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, divRAUGradpPrev_idx] = dataset[step, :, dataset_divRAUGradpPrev_ch]
            if add_divDDUStar_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, div_ddu_idx] = dataset[step, :, dataset_div_ddu_ch]
            if add_divDUStar_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, div_du_idx] = dataset[step, :, dataset_div_du_ch]
            if add_divUStar_input:
                dataset_gridded[step, indices_i, indices_j, indices_k, div_u_idx] = dataset[step, :, dataset_div_u_ch]
            if add_distance_to_outlet_input and dist_to_outlet_flat is not None:
                dataset_gridded[step, indices_i, indices_j, indices_k, dist_to_outlet_idx] = dist_to_outlet_flat
            if add_grad_sdf_input and grad_sdf_flat_x is not None:
                dataset_gridded[step, indices_i, indices_j, indices_k, grad_sdf_idx[0]]     = grad_sdf_flat_x
                dataset_gridded[step, indices_i, indices_j, indices_k, grad_sdf_idx[0] + 1] = grad_sdf_flat_y
                dataset_gridded[step, indices_i, indices_j, indices_k, grad_sdf_idx[0] + 2] = grad_sdf_flat_z
            if add_UdotNwall_input and UdotNwall_grid_flat is not None:
                dataset_gridded[step, indices_i, indices_j, indices_k, UdotNwall_idx] = UdotNwall_grid_flat[step, :]
            dataset_gridded[step, indices_i, indices_j, indices_k, sdf_idx] = sdf  # sdf last among inputs

            # ddp to PREDICT
            dataset_gridded[step, indices_i, indices_j, indices_k, ddp_idx] = dataset[step, :, dataset_ddp_ch]
            if predict_ddUCorr_output:
                dataset_gridded[step, indices_i, indices_j, indices_k, ddU_CFD_idx[0]:ddU_CFD_idx[1]] = dataset[step, :, dataset_ddp_ch+1:dataset_ddp_ch+4]

            # Values inside non-flow regions are not important for training and can be set to zero or masked out.
            mask = obst_bool[:, :, :, 0] == 0
            dataset_gridded[step, mask] = 0.0

        # ============================================================
        # APPLY PER-SAMPLE COMPRESSION (threshold computed per sample)
        # ============================================================
        for step in range(n_samples):
            if clip_ddp_and_gradDpPrev:
                _ddp_arr = dataset_gridded[step, :, :, :, ddp_idx]
                ddp_soft, _thr_ddp = compress_tail_asinh(
                    _ddp_arr,
                    fluid_mask_3d,
                    q=clip_percentile
                )
                dataset_gridded[step, :, :, :, ddp_idx] = ddp_soft

                if add_ddpPrev_input:
                    ddpPrev_soft, _thr_ddpPrev = compress_tail_asinh(
                        dataset_gridded[step, :, :, :, ddpPrev_idx],
                        fluid_mask_3d,
                        q=clip_percentile
                    )
                    dataset_gridded[step, :, :, :, ddpPrev_idx] = ddpPrev_soft



        # Plot before filtering
        import matplotlib.pyplot as plt
        import os

        if save_plots_debug:
            os.makedirs('plots_debug', exist_ok=True)

            # Build variable names based on what's included
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
            var_names.extend(['sdf'])  # sdf after all velocity/pressure/div inputs
            if add_distance_to_outlet_input:
                var_names.append('dist_to_outlet')
            if add_grad_sdf_input:
                var_names.extend(['grad_sdf_x', 'grad_sdf_y', 'grad_sdf_z'])
            if add_UdotNwall_input:
                var_names.append('UdotNwall')
            var_names.append('ddp')
            if predict_ddUCorr_output:
                var_names.extend(['ddUCorr_x', 'ddUCorr_y', 'ddUCorr_z'])

            n_plot_vars = n_grid_channels
            n_samples_to_show = min(20, n_samples)
            sample_indices_to_plot = np.linspace(0, n_samples - 1, n_samples_to_show, dtype=int)

            mid_y = dataset_gridded.shape[2] // 2
            mid_z = dataset_gridded.shape[1] // 2

            for var_idx in range(n_plot_vars):
                var_dir = os.path.join('plots_debug', var_names[var_idx])
                os.makedirs(var_dir, exist_ok=True)

                for t_idx in sample_indices_to_plot:
                    grid_t = dataset_gridded[t_idx]

                    # Z-X slice at middle Y
                    fig, ax = plt.subplots(figsize=(20, 6))
                    masked_arr = np.ma.array(
                        grid_t[:, mid_y, :, var_idx],
                        mask=obst_bool[:, mid_y, :, 0] == 0,
                    )
                    im = ax.imshow(masked_arr, cmap='jet', aspect='auto')
                    fig.colorbar(im, ax=ax, label=var_names[var_idx])
                    ax.set_title(f'{var_names[var_idx]} | Z-X slice (mid Y={mid_y}) | t={t_idx}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Z')
                    fig.savefig(os.path.join(var_dir, f'zx_slice_t{t_idx:04d}.png'), dpi=100, bbox_inches='tight')
                    plt.close(fig)

                    # Y-X slice at middle Z
                    fig, ax = plt.subplots(figsize=(20, 6))
                    masked_arr = np.ma.array(
                        grid_t[mid_z, :, :, var_idx],
                        mask=obst_bool[mid_z, :, :, 0] == 0,
                    )
                    im = ax.imshow(masked_arr, cmap='jet', aspect='auto')
                    fig.colorbar(im, ax=ax, label=var_names[var_idx])
                    ax.set_title(f'{var_names[var_idx]} | Y-X slice (mid Z={mid_z}) | t={t_idx}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    fig.savefig(os.path.join(var_dir, f'yx_slice_t{t_idx:04d}.png'), dpi=100, bbox_inches='tight')
                    plt.close(fig)

            print(f"Debug plots saved to plots_debug/ ({n_samples_to_show} samples, {n_plot_vars} variables).")

            if generate_gif:
                # Generate GIFs for ALL variables (all time steps)
                _mask_zx = obst_bool[:, mid_y, :, 0] == 0
                _mask_yx = obst_bool[mid_z, :, :, 0] == 0

                for _gif_var_idx in range(n_plot_vars):
                    _gif_name = var_names[_gif_var_idx]
                    _gif_dir = os.path.join('plots_debug', _gif_name)
                    os.makedirs(_gif_dir, exist_ok=True)
                    for _slice_label, _mask_2d, _xlabel, _ylabel in [
                        ('zx', _mask_zx, 'X', 'Z'),
                        ('yx', _mask_yx, 'X', 'Y'),
                    ]:
                        _png_paths = []
                        for _t_idx in range(n_samples):
                            _grid_t = dataset_gridded[_t_idx]
                            _data_2d = (_grid_t[:, mid_y, :, _gif_var_idx]
                                        if _slice_label == 'zx' else
                                        _grid_t[mid_z, :, :, _gif_var_idx])
                            _masked = np.ma.array(_data_2d, mask=_mask_2d)
                            _fig, _ax = plt.subplots(figsize=(20, 6))
                            _im = _ax.imshow(_masked, cmap='jet', aspect='auto')
                            _fig.colorbar(_im, ax=_ax, label=_gif_name)
                            _ax.set_title(f'{_gif_name} | {_slice_label.upper()}-slice | t={_t_idx}')
                            _ax.set_xlabel(_xlabel)
                            _ax.set_ylabel(_ylabel)
                            _png_path = os.path.join(_gif_dir, f'{_slice_label}_t{_t_idx:04d}.png')
                            _fig.savefig(_png_path, dpi=80, bbox_inches='tight')
                            plt.close(_fig)
                            _png_paths.append(_png_path)
                        _gif_path = os.path.join('plots_debug', f'{_gif_name}_{_slice_label}.gif')
                        # Enforce total GIF duration ~30s using explicit ms timing (more reliable than backend defaults).
                        from PIL import Image as _PILImage
                        _frame_duration_ms = max(10, int(round(10000.0 / max(len(_png_paths), 1))))
                        _frames = [_PILImage.open(_p).convert('P', palette=_PILImage.ADAPTIVE) for _p in _png_paths]
                        _frames[0].save(
                            _gif_path,
                            save_all=True,
                            append_images=_frames[1:],
                            duration=_frame_duration_ms,
                            loop=0,
                            optimize=False,
                            disposal=2,
                        )
                        for _fr in _frames:
                            _fr.close()
                        print(f'GIF saved: {_gif_path} (frame_duration_ms={_frame_duration_ms}, frames={len(_png_paths)})')

        #dataset_gridded[..., :6] = gaussian_filter(dataset_gridded[..., :6], sigma=(0, 2, 2, 2, 0))
        #dataset_gridded[..., 7] = gaussian_filter(dataset_gridded[..., 7], sigma=(0, 2, 2, 2))

        

        with h5py.File(gridded_h5_fn, 'w') as f:
            f.create_dataset('data', data=dataset_gridded)
        print(f"Stacked data (U, p, sdf) saved to {gridded_h5_fn}.")

        # Plot after filtering
        #grid_after = dataset_gridded[0]
        #for var_idx in range(n_plot_vars):
        #    # Z-X slice at middle Y
        #    plt.figure(figsize=(20, 6))
        #    plt.imshow(grid_after[1:-1, int(grid_after.shape[1] / 2), 1:-1, var_idx], cmap='jet')
        #    plt.colorbar(label=var_names[var_idx])
        #    plt.title(f'{var_names[var_idx]} - Z-X slice (middle Y) AFTER filter')
        #    plt.xlabel('X')
        #    plt.ylabel('Z')
        #    plt.savefig(f'plots_debug/{var_names[var_idx]}_zx_slice_t0_after_filter.png')
        #    plt.close()
        #    # Y-X slice at middle Z
        #    plt.figure(figsize=(20, 6))
        #    plt.imshow(grid_after[int(grid_after.shape[0] / 2), :, :, var_idx], cmap='jet')
        #    plt.colorbar(label=var_names[var_idx])
        #    plt.title(f'{var_names[var_idx]} - Y-X slice (middle Z) AFTER filter')
        #    plt.xlabel('X')
        #    plt.ylabel('Y')
        #    plt.savefig(f'plots_debug/{var_names[var_idx]}_yx_slice_t0_after_filter.png')
        #    plt.close()

        maxs_fn = os.path.join(data_dir, 'maxs')

        # Single-block mode: the block covers the full domain — skip LHS sampling and block extraction
        single_block_mode = (block_size_z == grid_shape_z and block_size_y == grid_shape_y and block_size_x == grid_shape_x)

        if single_block_mode:
            print("Single-block mode: block covers the full domain. Computing sampling index and maxs directly.")
            import pickle as pk

            # One center point per time step: the geometric center of the grid
            center = np.array([[grid_shape_z // 2, grid_shape_y // 2, grid_shape_x // 2]])
            sampling_indices = [[center for _ in range(n_sample_frames)]]
            with open(sample_idx_fn, 'wb') as f:
                pk.dump(sampling_indices, f)
            print(f"Sampling indices (single center point) saved to {sample_idx_fn}.")

            # Compute absolute maxs directly from the full gridded dataset
            # Channel order: [U if add_U_input] [ddU if add_ddUStar_input] dddU sdf [dpPrev if add_dpPrev_input] ddp
            maxs_list = []
            ch = 0

            # U values
            if add_U_input:
                max_abs_u_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
                max_abs_u_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
                max_abs_u_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
                maxs_list.extend([max_abs_u_x, max_abs_u_y, max_abs_u_z])
                ch += 3

            # dU values
            if add_dUStar_input:
                max_abs_dU_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
                max_abs_dU_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
                max_abs_dU_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
                maxs_list.extend([max_abs_dU_x, max_abs_dU_y, max_abs_dU_z])
                ch += 3

            # ddU values
            if add_ddUStar_input:
                max_abs_ddU_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
                max_abs_ddU_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
                max_abs_ddU_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
                maxs_list.extend([max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z])
                ch += 3

            # dddU values (optional)
            if add_ddUStarDiff_input:
                max_abs_dddU_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
                max_abs_dddU_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
                max_abs_dddU_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
                maxs_list.extend([max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z])
                ch += 3

            # dUCorrPrev values (optional)
            if add_dUCorrPrev_input:
                max_abs_dUCorrPrev_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
                max_abs_dUCorrPrev_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
                max_abs_dUCorrPrev_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
                maxs_list.extend([max_abs_dUCorrPrev_x, max_abs_dUCorrPrev_y, max_abs_dUCorrPrev_z])
                ch += 3

            # ddUCorrPrev values (optional)
            if add_ddUCorrPrev_input:
                max_abs_ddUCorrPrev_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
                max_abs_ddUCorrPrev_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
                max_abs_ddUCorrPrev_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
                maxs_list.extend([max_abs_ddUCorrPrev_x, max_abs_ddUCorrPrev_y, max_abs_ddUCorrPrev_z])
                ch += 3

            # SDF — use sdf_idx directly (ch is only used for velocity groups above)
            max_abs_dist = float(np.nanmax(np.abs(dataset_gridded[..., sdf_idx])))
            maxs_list.append(max_abs_dist)

            # Pressure (ddp to predict): subtract per-timestep domain mean before computing max,
            # consistent with what sample_blocks does for each block.
            # Use ddp_idx explicitly (NOT -1 which becomes dpPrev when add_dpPrev_input=True)
            ddpML_data = dataset_gridded[..., ddp_idx].copy()
            obst_mask = dataset_gridded[0, ..., sdf_idx] != 0  # sdf != 0 => inside domain (same for all timesteps)
            if enforce_zero_mean_pressure:
                for t in range(n_samples):
                    ddpML_in_domain = ddpML_data[t][obst_mask]
                    if not np.all(np.isnan(ddpML_in_domain)):
                        ddpML_data[t][obst_mask] -= np.nanmean(ddpML_in_domain)

            # Absolute previous pressure (if enabled)
            if add_pPrev_input:
                p_prev_data = dataset_gridded[..., p_prev_idx].copy()
                if enforce_zero_mean_pressure:
                    for t in range(n_samples):
                        p_prev_in_domain = p_prev_data[t][obst_mask]
                        if not np.all(np.isnan(p_prev_in_domain)):
                            p_prev_data[t][obst_mask] -= np.nanmean(p_prev_in_domain)
                max_abs_p_prev = float(np.nanmax(np.abs(p_prev_data)))
                maxs_list.append(max_abs_p_prev)

            # Previous Pressure (if enabled): same mean-removal treatment; use dpPrev_idx explicitly
            if add_dpPrev_input:
                dpPrev_data = dataset_gridded[..., dpPrev_idx].copy()
                if enforce_zero_mean_pressure:
                    for t in range(n_samples):
                        dpPrev_in_domain = dpPrev_data[t][obst_mask]
                        if not np.all(np.isnan(dpPrev_in_domain)):
                            dpPrev_data[t][obst_mask] -= np.nanmean(dpPrev_in_domain)
                max_abs_dpPrev = float(np.nanmax(np.abs(dpPrev_data)))
                maxs_list.append(max_abs_dpPrev)

            # Previous pressure double-increment (if enabled): same mean-removal treatment
            if add_ddpPrev_input:
                ddpPrev_data = dataset_gridded[..., ddpPrev_idx].copy()
                if enforce_zero_mean_pressure:
                    for t in range(n_samples):
                        ddpPrev_in_domain = ddpPrev_data[t][obst_mask]
                        if not np.all(np.isnan(ddpPrev_in_domain)):
                            ddpPrev_data[t][obst_mask] -= np.nanmean(ddpPrev_in_domain)
                max_abs_ddpPrev = float(np.nanmax(np.abs(ddpPrev_data)))
                maxs_list.append(max_abs_ddpPrev)

            if add_gradDpPrev_input:
                for _gi in range(3):
                    _gdp = dataset_gridded[..., gradDpPrev_idx[0] + _gi].copy()
                    maxs_list.append(float(np.nanmax(np.abs(_gdp))))

            if add_laplacian_dpPrev_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., laplaceDpPrev_idx]))))

            if add_uDotGradDpPrev_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., uDotGradDpPrev_idx]))))

            if add_gradDpPrevMag_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., gradDpPrevMag_idx]))))

            if include_rAU_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., rAU_idx]))))

            if include_HbyA_input:
                for _hi in range(3):
                    maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., HbyA_idx[0] + _hi]))))

            if include_divHbyA_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., divHbyA_idx]))))

            if include_dHbyA_input:
                for _hi in range(3):
                    maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., dHbyA_idx[0] + _hi]))))

            if include_dDivHbyA_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., dDivHbyA_idx]))))

            if add_rAUGradDpPrev_input:
                for _hi in range(3):
                    maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., rAUGradDpPrev_idx[0] + _hi]))))

            if add_divRAUGradDpPrev_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., divRAUGradDpPrev_idx]))))

            if add_pressureEqResidualp_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., pressureEqResidualp_idx]))))

            if add_rAUGradpPrev_input:
                for _hi in range(3):
                    maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., rAUGradpPrev_idx[0] + _hi]))))

            if add_divRAUGradpPrev_input:
                maxs_list.append(float(np.nanmax(np.abs(dataset_gridded[..., divRAUGradpPrev_idx]))))

            # Divergence of velocity double-increment (if enabled)
            if add_divDDUStar_input:
                div_ddu_data = dataset_gridded[..., div_ddu_idx].copy()
                max_abs_div_ddu = float(np.nanmax(np.abs(div_ddu_data)))
                maxs_list.append(max_abs_div_ddu)

            # Divergence of velocity increment (if enabled)
            if add_divDUStar_input:
                div_du_data = dataset_gridded[..., div_du_idx].copy()
                max_abs_div_du = float(np.nanmax(np.abs(div_du_data)))
                maxs_list.append(max_abs_div_du)

            # Divergence of velocity (if enabled)
            if add_divUStar_input:
                div_u_data = dataset_gridded[..., div_u_idx].copy()
                max_abs_div_u = float(np.nanmax(np.abs(div_u_data)))
                maxs_list.append(max_abs_div_u)

            # Distance to outlet (if enabled)
            if add_distance_to_outlet_input:
                _dist_data = dataset_gridded[..., dist_to_outlet_idx].copy()
                max_abs_dist_to_outlet = float(np.nanmax(np.abs(_dist_data)))
                maxs_list.append(max_abs_dist_to_outlet)

            # Grad(sdf) components (if enabled): x, y, z order
            if add_grad_sdf_input:
                for _gi in range(3):
                    _gsd = dataset_gridded[..., grad_sdf_idx[0] + _gi].copy()
                    maxs_list.append(float(np.nanmax(np.abs(_gsd))))

            if add_UdotNwall_input:
                _udn = dataset_gridded[..., UdotNwall_idx].copy()
                maxs_list.append(float(np.nanmax(np.abs(_udn))))

            max_abs_ddp = float(np.nanmax(np.abs(ddpML_data)))
            maxs_list.append(max_abs_ddp)

            if predict_ddUCorr_output:
                max_abs_ddU_CFD_x = float(np.nanmax(np.abs(dataset_gridded[..., ddU_CFD_idx[0]])))
                max_abs_ddU_CFD_y = float(np.nanmax(np.abs(dataset_gridded[..., ddU_CFD_idx[0]+1])))
                max_abs_ddU_CFD_z = float(np.nanmax(np.abs(dataset_gridded[..., ddU_CFD_idx[0]+2])))
                maxs_list.extend([max_abs_ddU_CFD_x, max_abs_ddU_CFD_y, max_abs_ddU_CFD_z])

            np.savetxt(maxs_fn, maxs_list)
            print(f"Absolute maxs saved to {maxs_fn}: {maxs_list}")

        else:
            sampling_indices = utils_sampling.define_sample_indexes(
                n_samples_per_frame,
                block_size_tuple,
                grid_res,
                0,  # first_sim
                0,  # last_sim
                0,  # first_t
                n_sample_frames,  # last_t
                None,
                sample_idx_fn,
                grid_limits
            )

            maxs_list = utils_sampling.calculate_and_save_block_abs_max(
                0,
                0,
                0,
                n_sample_frames,
                sample_idx_fn,
                None,
                block_size_tuple,
                [gridded_h5_fn],
                for_auto_CFD=True,
                maxs_fn=maxs_fn
            )

        # Extract features and write them using FeatureExtractAndWrite
        # When use_feature_decomposition=False, raw normalized blocks are written instead of Tucker cores.
        # In that case, flatten_data must be False (CNN takes the full 3D block).
        flatten_data = True if use_feature_decomposition else False
        feature_writer = FeatureExtractAndWrite(
            grid_res=grid_res,
            block_size=block_size_tuple,
            original_dataset_path=None,
            n_samples_per_frame=n_samples_per_frame,
            first_sim=0,
            last_sim=0,
            first_t=0,
            last_t=n_sample_frames,
            standardization_method=standardization_method,
            chunk_size=feature_extraction_chunk_size,
            gridded_h5_fn=None,
            spatial_tucker_ranks=tucker_ranks_tuple,
            sample_indices_fn=sample_idx_fn,
            tucker_factors_fn=tucker_factors_fn,
            gridded_h5_filenames=[gridded_h5_fn],
            flatten_data=flatten_data,
            maxs_list=maxs_list,
            last_tucker_rank=last_tucker_rank if use_feature_decomposition else (1 + 3 * (int(add_U_input) + int(add_dUStar_input) + int(add_ddUStar_input) + int(add_ddUStarDiff_input) + int(add_dUCorrPrev_input) + int(add_ddUCorrPrev_input)) + int(add_pPrev_input) + int(add_dpPrev_input) + int(add_ddpPrev_input) + 3 * int(add_gradDpPrev_input) + int(add_laplacian_dpPrev_input) + int(add_uDotGradDpPrev_input) + int(add_gradDpPrevMag_input) + int(include_rAU_input) + 3 * int(include_HbyA_input) + int(include_divHbyA_input) + 3 * int(include_dHbyA_input) + int(include_dDivHbyA_input) + 3 * int(add_rAUGradDpPrev_input) + int(add_divRAUGradDpPrev_input) + int(add_pressureEqResidualp_input) + 3 * int(add_rAUGradpPrev_input) + int(add_divRAUGradpPrev_input) + int(add_divDDUStar_input) + int(add_divDUStar_input) + int(add_divUStar_input) + int(add_distance_to_outlet_input) + 3 * int(add_grad_sdf_input) + int(add_UdotNwall_input)),
            use_feature_decomposition=use_feature_decomposition,
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
            enforce_zero_mean_pressure=enforce_zero_mean_pressure,
            add_distance_to_outlet_input=add_distance_to_outlet_input,
            add_grad_sdf_input=add_grad_sdf_input,
            add_UdotNwall_input=add_UdotNwall_input,
            clip_UdotNwall_to_inflow=clip_UdotNwall_to_inflow,
        )
        feature_writer(core_data_fn, compute_tucker_factors=True, n_representative_blocks=n_representative_blocks)
        print("Feature extraction and writing complete.")
        if use_feature_decomposition:
            print(f"Tucker decomposition complete and factors saved to {tucker_factors_fn}.")
        print(f"Core data with features saved to {core_data_fn}.")

        # THE RESULT IS 
        # - interp_weights.npy
        # - interp_vertices.npy
        # - grid_X.npy, grid_Y.npy, grid_Z.npy
        # - grid_sdf.npy, grid_domain_mask.npy
        # - gridded_data.h5 (with stacked U, p, sdf)
        # - sample_idx_per_time.npy
        # - maxs (text file with max values for normalization)
        # - tucker_factors.pkl (with Tucker factors for the dataset)

        n_layers, width = utils_model.define_model_arch(model_architecture)
        model_name = f'{model_architecture}-{standardization_method}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'
        train_tfrecord_fn = os.path.join(data_dir, 'train_data.tfrecords')
        test_tfrecord_fn = os.path.join(data_dir, 'test_data.tfrecords')
        normalization_factors_fn = os.path.join(data_dir, 'mean_std.npz')

        # RUN the FIRST NN train
        Train = Training(standardization_method, train_tfrecord_fn, test_tfrecord_fn)
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
            grad_ch_tuple = (
                dataset_gradDpPrev_ch,
                dataset_gradDpPrev_ch + 1,
                dataset_gradDpPrev_ch + 2,
            )
        if is_velocity_shifter_arch:
            u_ch_tuple = (0, 1, 2)

        Train.prepare_data_to_tf(
            core_data_fn,
            normalization_factors_fn,
            flatten_data=flatten_data,
            include_dp_prev_in_y=not is_shifter_arch,
            include_gradDpPrev_in_y=is_shifter_arch and not is_velocity_shifter_arch,
            include_velocity_components_in_y=is_velocity_shifter_arch,
            include_uDotGradDpPrev_in_y=False,
            dp_prev_input_ch_idx=dataset_dpPrev_ch,
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
            new_model=True,
            spatial_tucker_ranks=tucker_ranks_tuple,
            flatten_data=flatten_data,
            weights_fn=os.path.join(data_dir, 'weights.h5'),
            model_h5_path=data_dir,
            last_tucker_rank=last_tucker_rank if use_feature_decomposition else (1 + 3 * (int(add_U_input) + int(add_dUStar_input) + int(add_ddUStar_input) + int(add_ddUStarDiff_input) + int(add_dUCorrPrev_input) + int(add_ddUCorrPrev_input)) + int(add_pPrev_input) + int(add_dpPrev_input) + int(add_ddpPrev_input) + 3 * int(add_gradDpPrev_input) + int(add_laplacian_dpPrev_input) + int(add_uDotGradDpPrev_input) + int(add_gradDpPrevMag_input) + int(include_rAU_input) + 3 * int(include_HbyA_input) + int(include_divHbyA_input) + 3 * int(include_dHbyA_input) + int(include_dDivHbyA_input) + 3 * int(add_rAUGradDpPrev_input) + int(add_divRAUGradDpPrev_input) + int(add_pressureEqResidualp_input) + 3 * int(add_rAUGradpPrev_input) + int(add_divRAUGradpPrev_input) + int(add_divDDUStar_input) + int(add_divDUStar_input) + int(add_divUStar_input) + int(add_distance_to_outlet_input) + 3 * int(add_grad_sdf_input) + int(add_UdotNwall_input)),
            use_feature_decomposition=use_feature_decomposition,
            block_size=block_size_tuple,
            obst_bool=obst_bool,
            predict_ddUCorr_output=predict_ddUCorr_output,
            div_u_ch_idx=div_u_idx,
            div_u_grid=np.nanmean(dataset_gridded[:, :, :, :, div_u_idx], axis=0) if div_u_idx is not None else None,
            grid_res=grid_res,
            dp_prev_input_ch_idx=dataset_dpPrev_ch,
            dp_prev_maxs_idx=dataset_dpPrev_ch + 1,  # +1: SDF at _vel_end shifts pressure channels by 1 in maxs file
            gradDpPrev_input_ch_idxs=grad_ch_tuple,
            use_s_roi_penalty=use_s_roi_penalty,
            U_input_ch_idxs=u_ch_tuple,
            uDotGradDpPrev_input_ch_idx=None,
        )




if __name__ == "__main__":
    main()
