import numpy as np
import os
import glob
import shutil
import time
import pandas as pd

from pressure_SM_delta_delta._3D.train_and_eval.data_processor import CFDDataProcessor
from pressure_SM_delta_delta._3D.train_and_eval.utils import data_processing as utils_data
from pressure_SM_delta_delta._3D.train_and_eval.utils import domain_geometry as utils_geo
from pressure_SM_delta_delta._3D.train_and_eval.utils import io_operations as utils_io
from pressure_SM_delta_delta._3D.train_and_eval.utils import sampling as utils_sampling
from pressure_SM_delta_delta._3D.train_and_eval.utils import model_utils as utils_model
from pressure_SM_delta_delta._3D.train_and_eval.utils.data_processing import _unpack_grid_res

import h5py
from pressure_SM_delta_delta._3D.train_and_eval.data_processor import FeatureExtractAndWrite
from pressure_SM_delta_delta._3D.train_and_eval.train import Training
from pressure_SM_delta_delta._3D.auto_CFD.hdf5_data_loader import load_hdf5_samples, save_cell_centers_and_boundaries, load_boundaries_dict

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


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Initialize interpolation weights and Tucker factors for ML training.')
    parser.add_argument('--data_dir', type=str, default='ML_data', help='Directory where the ML data is stored (default: ML_data)')
    args = parser.parse_args()
    data_dir = args.data_dir

    # Import shared config from python_module in the case directory (CWD)
    import sys
    os.environ['TRAIN_SCRIPT_MODE'] = '1'
    sys.path.insert(0, os.getcwd())

    from python_module import (
        grid_res, block_size, spatial_tucker_ranks, dropout_rate, regularization,
        model_architecture, standardization_method, n_samples_per_frame,
        lr, batch_size, beta, num_epochs, feature_extraction_chunk_size,
        n_representative_blocks, last_tucker_rank
    )

    # use_feature_decomposition controls whether Tucker decomposition is applied.
    # When False, the full 3D block is passed directly to the NN (default: CNN).
    try:
        from python_module import use_feature_decomposition
    except ImportError:
        use_feature_decomposition = True  # backward-compatible default

    # add_ddu_input: include delta_delta_U as extra input alongside delta_delta_U_diff
    try:
        from python_module import add_ddu_input
    except ImportError:
        add_ddu_input = True  # backward-compatible default (current behaviour)

    # add_U_input: include U (velocity) as extra input
    try:
        from python_module import add_U_input
    except ImportError:
        add_U_input = False  # default: don't include raw U

    # use_previous_dp_input: include delta_p_prev (previous pressure) as extra input
    try:
        from python_module import use_previous_dp_input
    except ImportError:
        use_previous_dp_input = False  # default: don't include previous pressure

    # add_ddp_prev_input: include delta_delta_p_prev (previous pressure double-increment) as extra input
    try:
        from python_module import add_ddp_prev_input
    except ImportError:
        add_ddp_prev_input = False  # default: don't include previous pressure double-increment

    # add_dU_input: include delta_U (first velocity increment) as extra input
    try:
        from python_module import add_dU_input
    except ImportError:
        add_dU_input = False  # default: don't include first velocity increment

    # add_p_prev_input: include p_rgh_prev (absolute previous pressure) as extra input
    try:
        from python_module import add_p_prev_input
    except ImportError:
        add_p_prev_input = False  # default: don't include absolute previous pressure

    # add_div_ddu_input: include div(delta_delta_U) (divergence of velocity double-increment) as extra input
    try:
        from python_module import add_div_ddu_input
    except ImportError:
        add_div_ddu_input = False  # default: don't include divergence

    # add_div_du_input: include div(delta_U) (divergence of velocity increment) as extra input
    try:
        from python_module import add_div_du_input
    except ImportError:
        add_div_du_input = False  # default: don't include divergence

    # add_div_u_input: include div(U) (divergence of velocity) as extra input
    try:
        from python_module import add_div_u_input
    except ImportError:
        add_div_u_input = False  # default: don't include divergence

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
        cell_centers, boundary_coords, boundary_patches, patch_names, U, delta_delta_U, delta_delta_U_diff, delta_p_prev, delta_delta_p_prev, div_delta_delta_U, div_U, div_dU, delta_U, p_prev, delta_delta_p, timestamps, U_MAX_NORM_arr = \
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
            print(f"Deleted {hdf5_file} — C++ will create a fresh file for the next batch.")

    n_sample_frames = len(timestamps)
    print(f"Loaded {n_sample_frames} samples from HDF5 file")
    print(f"Cell centers shape: {cell_centers.shape}")
    print(f"Boundary coordinates shape: {boundary_coords.shape if boundary_coords is not None else 'None'}")
    print(f"U shape: {U.shape}, delta_delta_U shape: {delta_delta_U.shape}, delta_delta_p shape: {delta_delta_p.shape}, delta_p_prev shape: {delta_p_prev.shape}, delta_delta_p_prev shape: {delta_delta_p_prev.shape}")
    print(f"delta_U shape: {delta_U.shape}, p_prev shape: {p_prev.shape}")
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


    # --- Interpolate delta_delta_U and delta_delta_p to grid ---
    # delta_delta_U has shape (n_samples, n_cells, 3) - need to interpolate each component
    # delta_delta_p has shape (n_samples, n_cells) - scalar field

    print("Interpolating delta-delta fields to grid and normalizing with U_MAX_NORM...")
    n_samples = delta_delta_U.shape[0]
    n_grid_points = grid_points.shape[0]

    # Pass block_size_tuple to all downstream processing and feature extraction
    # Example: when initializing CFDDataProcessor or FeatureExtractAndWrite, use block_size=block_size_tuple
    # ...existing code...

    # Initialize output arrays
    U_grid_flat = None
    dU_grid_flat = None
    delta_delta_U_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    delta_delta_U_diff_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    delta_delta_p_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)
    p_prev_grid_flat = None
    delta_p_prev_grid_flat = None
    delta_ddp_prev_grid_flat = None
    div_delta_delta_U_grid_flat = None
    div_dU_grid_flat = None
    div_U_grid_flat = None
    
    if add_U_input:
        U_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)

    if add_dU_input:
        dU_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    
    if add_p_prev_input:
        p_prev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if use_previous_dp_input:
        delta_p_prev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_ddp_prev_input:
        delta_ddp_prev_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_div_ddu_input:
        div_delta_delta_U_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_div_du_input:
        div_dU_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    if add_div_u_input:
        div_U_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

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

        # Interpolate dU if requested
        if add_dU_input:
            for component in range(3):
                dU_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    delta_U[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
                )
        
        # Interpolate and normalize delta-delta velocity components
        for component in range(3):
            delta_delta_U_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                delta_delta_U[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
            )
            delta_delta_U_diff_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                delta_delta_U_diff[sample_idx, :, component] / norm, vert, weights, fill_value=np.nan
            )
        
        # Interpolate and normalize previous pressure (if enabled)
        if add_p_prev_input:
            p_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                p_prev[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
            )

        # Interpolate and normalize previous pressure (if enabled)
        if use_previous_dp_input:
            delta_p_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                delta_p_prev[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
            )

        # Interpolate and normalize previous pressure double-increment (if enabled)
        if add_ddp_prev_input:
            delta_ddp_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                delta_delta_p_prev[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
            )

        # Interpolate and normalize divergence of delta-delta velocity (if enabled)
        if add_div_ddu_input:
            div_delta_delta_U_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                div_delta_delta_U[sample_idx, :] / norm, vert, weights, fill_value=np.nan
            )

        # Interpolate and normalize divergence of delta velocity (if enabled)
        if add_div_du_input:
            div_dU_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                div_dU[sample_idx, :] / norm, vert, weights, fill_value=np.nan
            )

        # Interpolate and normalize divergence of velocity (if enabled)
        if add_div_u_input:
            div_U_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                div_U[sample_idx, :] / norm, vert, weights, fill_value=np.nan
            )

        # Interpolate and normalize delta-delta pressure
        delta_delta_p_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
            delta_delta_p[sample_idx, :] / (norm ** 2), vert, weights, fill_value=np.nan
        )


    print("Interpolation and normalization to grid complete.")

    # Stack: choose dataset channels based on flags
    # Channel order: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU [p_prev if add_p_prev] [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] [div_ddu if add_div_ddu] [div_du if add_div_du] [div_u if add_div_u] ddp
    dataset_parts = []
    if add_U_input:
        dataset_parts.append(U_grid_flat)
    if add_dU_input:
        dataset_parts.append(dU_grid_flat)
    if add_ddu_input:
        dataset_parts.append(delta_delta_U_grid_flat)
    dataset_parts.append(delta_delta_U_diff_grid_flat)
    if add_p_prev_input:
        dataset_parts.append(p_prev_grid_flat[..., np.newaxis])
    if use_previous_dp_input:
        dataset_parts.append(delta_p_prev_grid_flat[..., np.newaxis])
    if add_ddp_prev_input:
        dataset_parts.append(delta_ddp_prev_grid_flat[..., np.newaxis])
    if add_div_ddu_input:
        dataset_parts.append(div_delta_delta_U_grid_flat[..., np.newaxis])
    if add_div_du_input:
        dataset_parts.append(div_dU_grid_flat[..., np.newaxis])
    if add_div_u_input:
        dataset_parts.append(div_U_grid_flat[..., np.newaxis])
    dataset_parts.append(delta_delta_p_grid_flat[..., np.newaxis])
    
    dataset = np.concatenate(dataset_parts, axis=-1)


    # Save indices for later reuse
    # Generate indices mapping grid points to (i, j, k) indices
    dx, dy, dz = _unpack_grid_res(grid_res)
    x0 = grid_points[:, 0].min()
    y0 = grid_points[:, 1].min()
    z0 = grid_points[:, 2].min()

    xyz0 = grid_points
    indices = np.full((xyz0.shape[0], 3), np.nan, dtype=float)
    sdfunct = np.full((grid_shape_z, grid_shape_y, grid_shape_x, 1), 0)
    obst_bool = np.zeros_like(sdfunct, dtype=int)

    # Example: using delta_ux_interp as delta_U_grid[..., 0]
    delta_ux_interp = delta_delta_U_grid_flat[0, :, 0]  # first sample, x-component

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

    indices_save_path = os.path.join(data_dir, 'interpolation_indices.npz')
    np.savez(indices_save_path, indices=indices, indices_i=indices_i, indices_j=indices_j, indices_k=indices_k)
    print(f"Saved interpolation indices to {indices_save_path}.")
    indices_save_path = os.path.join(data_dir, 'interpolation_indices.npz')


    # Prepare gridded array for saving
    # Calculate total channels: dddU(3) + sdf(1) + ddp(1) + optional flags
    n_grid_channels = 2 + 3  # sdf, delta_p, dddU (+ optional flags)
    if add_U_input:
        n_grid_channels += 3  # U
    if add_dU_input:
        n_grid_channels += 3  # dU
    if add_ddu_input:
        n_grid_channels += 3  # ddU
    if add_p_prev_input:
        n_grid_channels += 1  # p_prev
    if use_previous_dp_input:
        n_grid_channels += 1  # delta_p_prev
    if add_ddp_prev_input:
        n_grid_channels += 1  # delta_delta_p_prev
    if add_div_ddu_input:
        n_grid_channels += 1  # div_delta_delta_U
    if add_div_du_input:
        n_grid_channels += 1  # div_dU
    if add_div_u_input:
        n_grid_channels += 1  # div_U
    
    grid_shape = (n_samples,) + sdfunct.shape[:3] + (n_grid_channels,)
    dataset_gridded = np.full(grid_shape, np.nan, dtype=np.float64)

    # Calculate channel indices based on what's included
    ch_idx = 0
    u_idx = (ch_idx, ch_idx+3) if add_U_input else None
    if add_U_input:
        ch_idx += 3
    dU_idx = (ch_idx, ch_idx+3) if add_dU_input else None
    if add_dU_input:
        ch_idx += 3
    ddu_idx = (ch_idx, ch_idx+3) if add_ddu_input else None
    if add_ddu_input:
        ch_idx += 3
    dddu_idx = (ch_idx, ch_idx+3)
    ch_idx += 3
    sdf_idx = ch_idx
    ch_idx += 1
    p_prev_idx = ch_idx if add_p_prev_input else None
    ch_idx += 1 if add_p_prev_input else 0
    dp_prev_idx = ch_idx if use_previous_dp_input else None
    ch_idx += 1 if use_previous_dp_input else 0
    ddp_prev_idx = ch_idx if add_ddp_prev_input else None
    ch_idx += 1 if add_ddp_prev_input else 0
    div_ddu_idx = ch_idx if add_div_ddu_input else None
    ch_idx += 1 if add_div_ddu_input else 0
    div_du_idx = ch_idx if add_div_du_input else None
    ch_idx += 1 if add_div_du_input else 0
    div_u_idx = ch_idx if add_div_u_input else None
    ch_idx += 1 if add_div_u_input else 0
    ddp_idx = ch_idx
    ch_idx += 1

    # Compute explicit flat channel indices in 'dataset' (which has no sdf channel)
    _ds_base = dddu_idx[1]  # start of pressure channels in dataset
    dataset_p_prev_ch = _ds_base  # p_prev position in dataset (if add_p_prev_input)
    dataset_dp_prev_ch = _ds_base + (1 if add_p_prev_input else 0)  # dp_prev position
    dataset_ddp_prev_ch = dataset_dp_prev_ch + (1 if use_previous_dp_input else 0)  # ddp_prev position
    dataset_div_ddu_ch = dataset_ddp_prev_ch + (1 if add_ddp_prev_input else 0)  # div_ddu position
    dataset_div_du_ch = dataset_div_ddu_ch + (1 if add_div_ddu_input else 0)  # div_du position
    dataset_div_u_ch = dataset_div_du_ch + (1 if add_div_du_input else 0)  # div_u position
    dataset_ddp_ch = dataset_div_u_ch + (1 if add_div_u_input else 0)  # ddp position

    for step in range(n_samples):
        if add_U_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, u_idx[0]:u_idx[1]] = dataset[step, :, u_idx[0]:u_idx[1]]
        if add_dU_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, dU_idx[0]:dU_idx[1]] = dataset[step, :, dU_idx[0]:dU_idx[1]]
        if add_ddu_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, ddu_idx[0]:ddu_idx[1]] = dataset[step, :, ddu_idx[0]:ddu_idx[1]]
        dataset_gridded[step, indices_i, indices_j, indices_k, dddu_idx[0]:dddu_idx[1]] = dataset[step, :, dddu_idx[0]:dddu_idx[1]]
        dataset_gridded[step, indices_i, indices_j, indices_k, sdf_idx] = sdf
        if add_p_prev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, p_prev_idx] = dataset[step, :, dataset_p_prev_ch]
        if use_previous_dp_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, dp_prev_idx] = dataset[step, :, dataset_dp_prev_ch]
        if add_ddp_prev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, ddp_prev_idx] = dataset[step, :, dataset_ddp_prev_ch]
        if add_div_ddu_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, div_ddu_idx] = dataset[step, :, dataset_div_ddu_ch]
        if add_div_du_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, div_du_idx] = dataset[step, :, dataset_div_du_ch]
        if add_div_u_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, div_u_idx] = dataset[step, :, dataset_div_u_ch]
        
        # delta_delta_p to PREDICT
        dataset_gridded[step, indices_i, indices_j, indices_k, ddp_idx] = dataset[step, :, dataset_ddp_ch]
    
    # Plot before filtering
    import matplotlib.pyplot as plt
    import os
    os.makedirs('plots_debug', exist_ok=True)

    # Build variable names based on what's included
    var_names = []
    if add_U_input:
        var_names.extend(['u_x', 'u_y', 'u_z'])
    if add_dU_input:
        var_names.extend(['dU_x', 'dU_y', 'dU_z'])
    if add_ddu_input:
        var_names.extend(['ddU_x', 'ddU_y', 'ddU_z'])
    var_names.extend(['dddU_x', 'dddU_y', 'dddU_z', 'sdf'])
    if add_p_prev_input:
        var_names.append('p_prev')
    if use_previous_dp_input:
        var_names.append('delta_p_prev')
    if add_ddp_prev_input:
        var_names.append('delta_delta_p_prev')
    if add_div_ddu_input:
        var_names.append('div_delta_delta_U')
    if add_div_du_input:
        var_names.append('div_dU')
    if add_div_u_input:
        var_names.append('div_U')
    var_names.append('delta_delta_p')
    
    n_plot_vars = n_grid_channels
    grid_before = dataset_gridded[0].copy()
    for var_idx in range(n_plot_vars):
        # Z-X slice at middle Y
        plt.figure(figsize=(20, 6))
        plt.imshow(grid_before[1:-1, int(grid_before.shape[1] / 2), 1:-1, var_idx], cmap='jet')
        plt.colorbar(label=var_names[var_idx])
        plt.title(f'{var_names[var_idx]} - Z-X slice (middle Y) BEFORE filter')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.savefig(f'plots_debug/{var_names[var_idx]}_zx_slice_t0_before_filter.png')
        plt.close()
        # Y-X slice at middle Z
        plt.figure(figsize=(20, 6))
        plt.imshow(grid_before[int(grid_before.shape[0] / 2), :, :, var_idx], cmap='jet')
        plt.colorbar(label=var_names[var_idx])
        plt.title(f'{var_names[var_idx]} - Y-X slice (middle Z) BEFORE filter')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'plots_debug/{var_names[var_idx]}_yx_slice_t0_before_filter.png')
        plt.close()

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
        # Channel order: [U if add_U_input] [ddU if add_ddu_input] dddU sdf [delta_p_prev if use_previous_dp_input] delta_delta_p
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
        if add_dU_input:
            max_abs_dU_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
            max_abs_dU_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
            max_abs_dU_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
            maxs_list.extend([max_abs_dU_x, max_abs_dU_y, max_abs_dU_z])
            ch += 3
        
        # ddU values
        if add_ddu_input:
            max_abs_ddU_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
            max_abs_ddU_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
            max_abs_ddU_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
            maxs_list.extend([max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z])
            ch += 3
        
        # dddU values (always present)
        max_abs_dddU_x = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
        max_abs_dddU_y = float(np.nanmax(np.abs(dataset_gridded[..., ch+1])))
        max_abs_dddU_z = float(np.nanmax(np.abs(dataset_gridded[..., ch+2])))
        maxs_list.extend([max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z])
        ch += 3
        
        # SDF and pressure
        max_abs_dist = float(np.nanmax(np.abs(dataset_gridded[..., ch])))
        maxs_list.append(max_abs_dist)
        ch += 1

        # Pressure (delta_delta_p to predict): subtract per-timestep domain mean before computing max,
        # consistent with what sample_blocks does for each block.
        # Use ddp_idx explicitly (NOT -1 which becomes delta_p_prev when use_previous_dp_input=True)
        ddp_data = dataset_gridded[..., ddp_idx].copy()
        obst_mask = dataset_gridded[0, ..., sdf_idx] != 0  # sdf != 0 => inside domain (same for all timesteps)
        for t in range(n_samples):
            ddp_in_domain = ddp_data[t][obst_mask]
            if not np.all(np.isnan(ddp_in_domain)):
                ddp_data[t][obst_mask] -= np.nanmean(ddp_in_domain)

        # Absolute previous pressure (if enabled)
        if add_p_prev_input:
            p_prev_data = dataset_gridded[..., p_prev_idx].copy()
            for t in range(n_samples):
                p_prev_in_domain = p_prev_data[t][obst_mask]
                if not np.all(np.isnan(p_prev_in_domain)):
                    p_prev_data[t][obst_mask] -= np.nanmean(p_prev_in_domain)
            max_abs_p_prev = float(np.nanmax(np.abs(p_prev_data)))
            maxs_list.append(max_abs_p_prev)

        # Previous Pressure (if enabled): same mean-removal treatment; use dp_prev_idx explicitly
        if use_previous_dp_input:
            dp_prev_data = dataset_gridded[..., dp_prev_idx].copy()
            for t in range(n_samples):
                dp_prev_in_domain = dp_prev_data[t][obst_mask]
                if not np.all(np.isnan(dp_prev_in_domain)):
                    dp_prev_data[t][obst_mask] -= np.nanmean(dp_prev_in_domain)
            max_abs_delta_p_prev = float(np.nanmax(np.abs(dp_prev_data)))
            maxs_list.append(max_abs_delta_p_prev)

        # Previous pressure double-increment (if enabled): same mean-removal treatment
        if add_ddp_prev_input:
            ddp_prev_data = dataset_gridded[..., ddp_prev_idx].copy()
            for t in range(n_samples):
                ddp_prev_in_domain = ddp_prev_data[t][obst_mask]
                if not np.all(np.isnan(ddp_prev_in_domain)):
                    ddp_prev_data[t][obst_mask] -= np.nanmean(ddp_prev_in_domain)
            max_abs_ddp_prev = float(np.nanmax(np.abs(ddp_prev_data)))
            maxs_list.append(max_abs_ddp_prev)

        # Divergence of velocity double-increment (if enabled)
        if add_div_ddu_input:
            div_ddu_data = dataset_gridded[..., div_ddu_idx].copy()
            max_abs_div_ddu = float(np.nanmax(np.abs(div_ddu_data)))
            maxs_list.append(max_abs_div_ddu)

        # Divergence of velocity increment (if enabled)
        if add_div_du_input:
            div_du_data = dataset_gridded[..., div_du_idx].copy()
            max_abs_div_du = float(np.nanmax(np.abs(div_du_data)))
            maxs_list.append(max_abs_div_du)

        # Divergence of velocity (if enabled)
        if add_div_u_input:
            div_u_data = dataset_gridded[..., div_u_idx].copy()
            max_abs_div_u = float(np.nanmax(np.abs(div_u_data)))
            maxs_list.append(max_abs_div_u)

        max_abs_ddp = float(np.nanmax(np.abs(ddp_data)))
        maxs_list.append(max_abs_ddp)

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
        last_tucker_rank=last_tucker_rank if use_feature_decomposition else (1 + 3 * (1 + int(add_U_input) + int(add_dU_input) + int(add_ddu_input)) + int(add_p_prev_input) + int(use_previous_dp_input) + int(add_ddp_prev_input) + int(add_div_ddu_input) + int(add_div_du_input) + int(add_div_u_input)),
        use_feature_decomposition=use_feature_decomposition,
        add_ddu_input=add_ddu_input,
        add_U_input=add_U_input,
        add_dU_input=add_dU_input,
        use_previous_dp_input=use_previous_dp_input,
        add_p_prev_input=add_p_prev_input,
        add_ddp_prev_input=add_ddp_prev_input,
        add_div_ddu_input=add_div_ddu_input,
        add_div_du_input=add_div_du_input,
        add_div_u_input=add_div_u_input,
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
    Train.prepare_data_to_tf(core_data_fn, normalization_factors_fn, flatten_data=flatten_data)
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
        last_tucker_rank=last_tucker_rank if use_feature_decomposition else (1 + 3 * (1 + int(add_U_input) + int(add_dU_input) + int(add_ddu_input)) + int(add_p_prev_input) + int(use_previous_dp_input) + int(add_ddp_prev_input) + int(add_div_ddu_input) + int(add_div_du_input) + int(add_div_u_input)),
        use_feature_decomposition=use_feature_decomposition,
        block_size=block_size_tuple,
    )

    

