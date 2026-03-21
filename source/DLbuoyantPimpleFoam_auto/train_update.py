import numpy as np
import os
import h5py
import time
import shutil
from pressure_SM._3D.train_and_eval.data_processor import CFDDataProcessor, FeatureExtractAndWrite
from pressure_SM._3D.train_and_eval.utils import data_processing as utils_data
from pressure_SM._3D.train_and_eval.utils import domain_geometry as utils_geo
from pressure_SM._3D.train_and_eval.utils import io_operations as utils_io
from pressure_SM._3D.train_and_eval.utils import sampling as utils_sampling
from pressure_SM._3D.train_and_eval.utils import model_utils as utils_model
from pressure_SM._3D.train_and_eval.train import Training
from pressure_SM._3D.auto_CFD.hdf5_data_loader import load_hdf5_samples, save_cell_centers_and_boundaries, load_boundaries_dict

# --- Feature Extraction and Training ---
def add_new_features_and_train():

    import argparse
    parser = argparse.ArgumentParser(description='Update ML model with new CFD samples.')
    parser.add_argument('--data_dir', type=str, default='ML_data', help='Directory where the ML data is stored (default: ML_data)')
    args = parser.parse_args()
    data_dir = args.data_dir

    grid_res = 0.001 # This one is highly dependent on the simulation length scales (aim for at least 50 grid points across the domain in each direction)
    block_size = 16
    n_samples_per_frame = 1000
    standardization_method = 'minmax'
    gridded_h5_fn = os.path.join(data_dir, 'gridded_data.h5')
    sample_idx_fn = os.path.join(data_dir, 'sample_idx_per_time.npy')
    maxs_list_fn = os.path.join(data_dir, 'maxs_list.npy')
    tucker_factors_fn = os.path.join(data_dir, 'tucker_factors.pkl')
    ranks = 3
    chunk_size = 500
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
        cell_centers, boundary_coords, boundary_patches, patch_names, delta_U, delta_p, timestamps = \
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
    print(f"Loaded {n_sample_frames} new samples from HDF5 file")
    print(f"delta_U shape: {delta_U.shape}, delta_p shape: {delta_p.shape}")

    # Save coordinates for compatibility
    save_cell_centers_and_boundaries(cell_centers, boundary_coords, boundary_patches, 
                                     patch_names, data_dir)

    # --- Remove old features corresponding to the new samples ---
    # (Assuming we want to re-extract features for the newly collected samples)
    print(f"Removing {n_sample_frames} sample frames from the previous dataset.")
    old_input_cores_to_keep = old_input_cores[:-n_sample_frames * n_samples_per_frame]
    old_output_cores_to_keep = old_output_cores[:-n_sample_frames * n_samples_per_frame]

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
    n_samples = delta_U.shape[0]
    n_grid_points = weights.shape[0]

    delta_U_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    delta_p_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)

    for sample_idx in range(n_samples):
        for component in range(3):
            delta_U_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                delta_U[sample_idx, :, component], vert, weights, fill_value=np.nan
            )
        delta_p_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
            delta_p[sample_idx, :], vert, weights, fill_value=np.nan
        )
    print("Interpolation to grid complete.")

    dataset = np.concatenate([delta_U_grid_flat, delta_p_grid_flat[..., np.newaxis]], axis=-1)

    # --- Reconstruct 3D gridded array ---
    grid_shape_z = indices_i.max() + 1
    grid_shape_y = indices_j.max() + 1
    grid_shape_x = indices_k.max() + 1
    grid_shape = (n_samples, grid_shape_z, grid_shape_y, grid_shape_x, 5)
    dataset_gridded = np.full(grid_shape, np.nan, dtype=np.float64)

    for step in range(n_samples):
        dataset_gridded[step, indices_i, indices_j, indices_k, :3] = dataset[step, :, :3]
        dataset_gridded[step, indices_i, indices_j, indices_k, 3] = sdf
        dataset_gridded[step, indices_i, indices_j, indices_k, 4] = dataset[step, :, 3]

    if os.path.exists(gridded_h5_fn):
        os.remove(gridded_h5_fn)

    with h5py.File(gridded_h5_fn, 'w') as f:
        f.create_dataset('data', data=dataset_gridded)
    print(f"Stacked data (U, p, sdf) saved to {gridded_h5_fn}.")

    # --- Define sampling indices ---
    boundaries_dict = load_boundaries_dict(data_dir)
    grid_limits = {
        'x_min': float(indices_k.min()) * grid_res,
        'x_max': float(indices_k.max()) * grid_res,
        'y_min': float(indices_j.min()) * grid_res,
        'y_max': float(indices_j.max()) * grid_res,
        'z_min': float(indices_i.min()) * grid_res,
        'z_max': float(indices_i.max()) * grid_res,
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

    # --- Calculate block maximums ---
    current_maxs_list = np.load(maxs_list_fn, allow_pickle=True)

    maxs_list = utils_sampling.calculate_and_save_block_abs_max(
        0, 0, 0,
        n_sample_frames,
        sample_idx_fn,
        None,
        block_size,
        [gridded_h5_fn],
        for_auto_CFD=True
    )

    maxs_list = np.maximum(maxs_list, current_maxs_list)
    np.save(maxs_list_fn, maxs_list)

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
        gridded_h5_fn=None,
        ranks=ranks,
        sample_indices_fn=sample_idx_fn,
        tucker_factors_fn=tucker_factors_fn,
        gridded_h5_filenames=[gridded_h5_fn],
        flatten_data=True,
        maxs_list=maxs_list
    )
    feature_writer(core_data_fn, n_representative_blocks=1000, compute_tucker_factors=False)
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
    dropout_rate = 0.1
    lr = 1e-3
    batch_size = 1024
    beta = 0.5
    regularization = 1e-4
    model_architecture = 'MLP_small'
    n_layers, width = utils_model.define_model_arch(model_architecture)
    model_name = f'{model_architecture}-{standardization_method}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'
    train_tfrecord_fn = os.path.join(data_dir, 'train_data.tfrecords')
    test_tfrecord_fn = os.path.join(data_dir, 'test_data.tfrecords')
    normalization_factors_fn = os.path.join(data_dir, 'mean_std.npz')

    Train = Training(standardization_method, train_tfrecord_fn, test_tfrecord_fn)
    Train.prepare_data_to_tf(core_data_fn, normalization_factors_fn, flatten_data=True)
    Train.load_data_and_train(
        lr=lr,
        batch_size=batch_size,
        model_name=model_name,
        beta_1=beta,
        num_epoch=100,
        n_layers=n_layers,
        width=width,
        dropout_rate=dropout_rate,
        regularization=regularization,
        model_architecture=model_architecture,
        new_model=False,
        ranks=ranks,
        flatten_data=True,
        weights_fn=os.path.join(data_dir, 'weights.h5'),
        model_h5_path=data_dir
    )
    print("Model training complete.")


if __name__ == '__main__':
    add_new_features_and_train()
