import numpy as np
import os
import h5py
import time
import shutil
from pressure_SM_delta_delta._3D.train_and_eval.data_processor import CFDDataProcessor, FeatureExtractAndWrite
from pressure_SM_delta_delta._3D.train_and_eval.utils import data_processing as utils_data
from pressure_SM_delta_delta._3D.train_and_eval.utils import domain_geometry as utils_geo
from pressure_SM_delta_delta._3D.train_and_eval.utils import io_operations as utils_io
from pressure_SM_delta_delta._3D.train_and_eval.utils import sampling as utils_sampling
from pressure_SM_delta_delta._3D.train_and_eval.utils import model_utils as utils_model
from pressure_SM_delta_delta._3D.train_and_eval.utils.data_processing import _unpack_grid_res
from pressure_SM_delta_delta._3D.train_and_eval.train import Training
from pressure_SM_delta_delta._3D.auto_CFD.hdf5_data_loader import load_hdf5_field_data, load_boundaries_dict

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
        from python_module import add_ddu_input
    except ImportError:
        add_ddu_input = True  # backward-compatible default

    try:
        from python_module import add_U_input
    except ImportError:
        add_U_input = False  # backward-compatible default

    try:
        from python_module import use_previous_dp_input
    except ImportError:
        use_previous_dp_input = False  # backward-compatible default

    try:
        from python_module import add_ddp_prev_input
    except ImportError:
        add_ddp_prev_input = False  # backward-compatible default

    try:
        from python_module import add_dU_input
    except ImportError:
        add_dU_input = False  # backward-compatible default

    try:
        from python_module import add_p_prev_input
    except ImportError:
        add_p_prev_input = False  # backward-compatible default

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
        delta_delta_U, delta_delta_U_diff, delta_delta_p, delta_p_prev, delta_delta_p_prev, delta_U, p_prev, U, timestamps, u_max_norm_arr = load_hdf5_field_data(hdf5_file_copy)
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
    print(f"delta_delta_U shape: {delta_delta_U.shape}, delta_delta_U_diff shape: {delta_delta_U_diff.shape}, delta_delta_p shape: {delta_delta_p.shape}")

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

    # --- Interpolate delta_U and delta_p to grid (per sample, per component) ---
    n_samples = delta_delta_U.shape[0]
    n_grid_points = weights.shape[0]

    U_grid_flat = None
    dU_grid_flat = None
    delta_delta_U_grid_flat      = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    delta_delta_U_diff_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    delta_delta_p_grid_flat      = np.full((n_samples, n_grid_points),    np.nan, dtype=np.float64)
    p_prev_grid_flat     = None
    delta_p_prev_grid_flat       = None
    delta_ddp_prev_grid_flat     = None
    
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

    for sample_idx in range(n_samples):
        if add_U_input:
            for component in range(3):
                U_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    U[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )

        if add_dU_input:
            for component in range(3):
                dU_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                    delta_U[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
                )
        
        for component in range(3):
            delta_delta_U_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                delta_delta_U[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )
            delta_delta_U_diff_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                delta_delta_U_diff[sample_idx, :, component] / u_max_norm_arr[sample_idx], vert, weights, fill_value=np.nan
            )
        delta_delta_p_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
            delta_delta_p[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
        )

        if add_p_prev_input:
            p_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                p_prev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )
        
        if use_previous_dp_input:
            delta_p_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                delta_p_prev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )

        if add_ddp_prev_input:
            delta_ddp_prev_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
                delta_delta_p_prev[sample_idx, :] / (u_max_norm_arr[sample_idx] ** 2), vert, weights, fill_value=np.nan
            )
    print("Interpolation to grid complete.")

    # Stack dataset based on enabled flags
    # Channel order: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU [p_prev if add_p_prev] [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] ddp
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
    dataset_parts.append(delta_delta_p_grid_flat[..., np.newaxis])
    dataset = np.concatenate(dataset_parts, axis=-1)
    
    # Calculate number of grid channels
    n_grid_channels = 2 + 3  # sdf + ddp + dddU
    if add_U_input:
        n_grid_channels += 3
    if add_dU_input:
        n_grid_channels += 3
    if add_ddu_input:
        n_grid_channels += 3
    if add_p_prev_input:
        n_grid_channels += 1
    if use_previous_dp_input:
        n_grid_channels += 1
    if add_ddp_prev_input:
        n_grid_channels += 1

    # --- Reconstruct 3D gridded array using ch_idx (same as train_init) ---
    grid_shape_z = indices_i.max() + 1
    grid_shape_y = indices_j.max() + 1
    grid_shape_x = indices_k.max() + 1
    grid_shape = (n_samples, grid_shape_z, grid_shape_y, grid_shape_x, n_grid_channels)
    dataset_gridded = np.full(grid_shape, np.nan, dtype=np.float64)

    # Compute ch_idx mapping (same order as train_init buildstep)
    _ci = 0
    _u_idx = (_ci, _ci+3) if add_U_input else None; _ci += 3 if add_U_input else 0
    _dU_idx = (_ci, _ci+3) if add_dU_input else None; _ci += 3 if add_dU_input else 0
    _ddu_idx = (_ci, _ci+3) if add_ddu_input else None; _ci += 3 if add_ddu_input else 0
    _dddu_idx = (_ci, _ci+3); _ci += 3
    _sdf_idx = _ci; _ci += 1
    _p_prev_idx = _ci if add_p_prev_input else None; _ci += 1 if add_p_prev_input else 0
    _dp_prev_idx = _ci if use_previous_dp_input else None; _ci += 1 if use_previous_dp_input else 0
    _ddp_prev_idx = _ci if add_ddp_prev_input else None; _ci += 1 if add_ddp_prev_input else 0
    _ddp_idx = _ci
    # Flat dataset indices (no sdf)
    _ds_base = _dddu_idx[1]
    _ds_p_prev_ch    = _ds_base
    _ds_dp_prev_ch   = _ds_base + (1 if add_p_prev_input else 0)
    _ds_ddp_prev_ch  = _ds_dp_prev_ch + (1 if use_previous_dp_input else 0)
    _ds_ddp_ch       = _ds_ddp_prev_ch + (1 if add_ddp_prev_input else 0)

    for step in range(n_samples):
        if add_U_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _u_idx[0]:_u_idx[1]] = dataset[step, :, _u_idx[0]:_u_idx[1]]
        if add_dU_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dU_idx[0]:_dU_idx[1]] = dataset[step, :, _dU_idx[0]:_dU_idx[1]]
        if add_ddu_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _ddu_idx[0]:_ddu_idx[1]] = dataset[step, :, _ddu_idx[0]:_ddu_idx[1]]
        dataset_gridded[step, indices_i, indices_j, indices_k, _dddu_idx[0]:_dddu_idx[1]] = dataset[step, :, _dddu_idx[0]:_dddu_idx[1]]
        dataset_gridded[step, indices_i, indices_j, indices_k, _sdf_idx] = sdf
        if add_p_prev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _p_prev_idx] = dataset[step, :, _ds_p_prev_ch]
        if use_previous_dp_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _dp_prev_idx] = dataset[step, :, _ds_dp_prev_ch]
        if add_ddp_prev_input:
            dataset_gridded[step, indices_i, indices_j, indices_k, _ddp_prev_idx] = dataset[step, :, _ds_ddp_prev_ch]
        dataset_gridded[step, indices_i, indices_j, indices_k, _ddp_idx] = dataset[step, :, _ds_ddp_ch]

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
    var_names.append('delta_delta_p')
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
        ranks=spatial_tucker_ranks,
        sample_indices_fn=sample_idx_fn,
        tucker_factors_fn=tucker_factors_fn,
        gridded_h5_filenames=[gridded_h5_fn],
        flatten_data=True,
        maxs_list=maxs_list,
        add_ddu_input=add_ddu_input,
        add_U_input=add_U_input,
        add_dU_input=add_dU_input,
        use_previous_dp_input=use_previous_dp_input,
        add_p_prev_input=add_p_prev_input,
        add_ddp_prev_input=add_ddp_prev_input,
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

    Train.prepare_data_to_tf(core_data_fn, normalization_factors_fn, flatten_data=True, load_existing_normalization=True)
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
        flatten_data=True,
        weights_fn=os.path.join(data_dir, 'weights.h5'),
        model_h5_path=data_dir,
        last_tucker_rank=last_tucker_rank if use_feature_decomposition else (1 + 3 * (1 + int(add_U_input) + int(add_dU_input) + int(add_ddu_input)) + int(add_p_prev_input) + int(use_previous_dp_input) + int(add_ddp_prev_input)),
        use_feature_decomposition=use_feature_decomposition,
        block_size=block_size,
        add_U_input=add_U_input,
        add_ddu_input=add_ddu_input,
        use_previous_dp_input=use_previous_dp_input,
    )
    print("Model training complete.")


if __name__ == '__main__':
    add_new_features_and_train()
