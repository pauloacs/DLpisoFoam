import numpy as np
import os
import glob
import pandas as pd
from pressure_SM._3D.train_and_eval.data_processor import CFDDataProcessor
from pressure_SM._3D.train_and_eval.utils import data_processing as utils_data
from pressure_SM._3D.train_and_eval.utils import domain_geometry as utils_geo
from pressure_SM._3D.train_and_eval.utils import io_operations as utils_io
from pressure_SM._3D.train_and_eval.utils import sampling as utils_sampling
from pressure_SM._3D.train_and_eval.utils import model_utils as utils_model

import h5py
from pressure_SM._3D.train_and_eval.data_processor import FeatureExtractAndWrite
from pressure_SM._3D.train_and_eval.train import Training
from pressure_SM._3D.auto_CFD.hdf5_data_loader import load_hdf5_samples, save_cell_centers_and_boundaries, load_boundaries_dict

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

def read_boundary_points():
    boundary_points = pd.read_csv('boundary_points.csv')  # DataFrame with columns patch, x, y, z
    # Group boundary points by patch if needed:
    boundary_groups = boundary_points.groupby('patch')
    # Example: get all points for a patch
    #for patch_name, group in boundary_groups:
    #    points = group[['x', 'y', 'z']].values  # shape: (n_points, 3)

    return boundary_points, boundary_groups


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Initialize interpolation weights and Tucker factors for ML training.')
    parser.add_argument('--data_dir', type=str, default='ML_data', help='Directory where the ML data is stored (default: ML_data)')
    args = parser.parse_args()
    data_dir = args.data_dir

    grid_res = 0.001  # Example value
    block_size = 16   # Example value
    n_samples_per_frame = 2000
    standardization_method = 'minmax'  # or 'zscore', etc.
    gridded_h5_fn = os.path.join(data_dir, 'gridded_data.h5')
    sample_idx_fn = os.path.join(data_dir, 'sample_idx_per_time.npy')
    tucker_factors_fn = os.path.join(data_dir, 'tucker_factors.pkl')
    ranks = 3
    chunk_size = 500
    
    core_data_fn = os.path.join(data_dir, 'core_data.h5')

    # --- Load data from HDF5 ---
    hdf5_file = os.path.join(data_dir, 'data.h5')
    print(f"Loading data from {hdf5_file}...")
    
    try:
        cell_centers, boundary_coords, boundary_patches, patch_names, delta_U, delta_p, timestamps = \
            load_hdf5_samples(hdf5_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading HDF5 data: {e}")
        exit(1)
    
    n_sample_frames = len(timestamps)
    print(f"Loaded {n_sample_frames} samples from HDF5 file")
    print(f"Cell centers shape: {cell_centers.shape}")
    print(f"Boundary coordinates shape: {boundary_coords.shape if boundary_coords is not None else 'None'}")
    print(f"delta_U shape: {delta_U.shape}, delta_p shape: {delta_p.shape}")

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
    domain_bool, sdf = utils_geo.domain_dist(boundaries_dict, grid_points, grid_res)
    np.save(os.path.join(data_dir, 'grid_sdf_flat.npy'), sdf)
    np.save(os.path.join(data_dir, 'grid_domain_mask_flat.npy'), domain_bool)
    print('SDF and domain mask arrays saved.')

    # Compute interpolation weights and vertices from cell centers to grid points
    vert, weights = utils_data.interp_weights(cell_centers, grid_points, interp_method='IDW')
    np.save(os.path.join(data_dir, 'interp_weights.npy'), weights)
    np.save(os.path.join(data_dir, 'interp_vertices.npy'), vert)
    print("Interpolation weights and vertices (cell centers -> grid) saved.")

    # --- Interpolate delta_U and delta_p to grid ---
    # delta_U has shape (n_samples, n_cells, 3) - need to interpolate each component
    # delta_p has shape (n_samples, n_cells) - scalar field
    
    print("Interpolating fields to grid...")
    n_samples = delta_U.shape[0]
    n_grid_points = grid_points.shape[0]
    
    # Initialize output arrays
    delta_U_grid_flat = np.full((n_samples, n_grid_points, 3), np.nan, dtype=np.float64)
    delta_p_grid_flat = np.full((n_samples, n_grid_points), np.nan, dtype=np.float64)
    
    # Interpolate each sample
    for sample_idx in range(n_samples):
        
        # Interpolate velocity components (3 separate scalar fields)
        for component in range(3):
            delta_U_grid_flat[sample_idx, :, component] = utils_data.interpolate_fill_njit(
                delta_U[sample_idx, :, component], vert, weights, fill_value=np.nan
            )
        
        # Interpolate pressure (scalar field)
        delta_p_grid_flat[sample_idx, :] = utils_data.interpolate_fill_njit(
            delta_p[sample_idx, :], vert, weights, fill_value=np.nan
        )
    
    print("Interpolation to grid complete.")

    # Stack gridded_U, gridded_p, and sdf, then save to gridded_h5_fn
    dataset = np.concatenate([delta_U_grid_flat, delta_p_grid_flat[..., np.newaxis]], axis=-1)

    # Save indices for later reuse
    # Generate indices mapping grid points to (i, j, k) indices
    dx = grid_res
    dy = grid_res
    dz = grid_res
    x0 = grid_points[:, 0].min()
    y0 = grid_points[:, 1].min()
    z0 = grid_points[:, 2].min()

    xyz0 = grid_points
    indices = np.full((xyz0.shape[0], 3), np.nan, dtype=float)
    sdfunct = np.full((grid_shape_z, grid_shape_y, grid_shape_x, 1), 0)
    obst_bool = np.zeros_like(sdfunct, dtype=int)

    # Example: using delta_ux_interp as delta_U_grid[..., 0]
    delta_ux_interp = delta_U_grid_flat[0, :, 0]  # first sample, x-component

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


    # Prepare gridded array for saving: shape (Z, Y, X, 5)
    # [delta_U_x, delta_U_y, delta_U_z, sdf, delta_p]
    grid_shape = (n_samples,) + sdfunct.shape[:3] + (5,)
    dataset_gridded = np.full(grid_shape, np.nan, dtype=np.float64)

    for step in range(n_samples):
        dataset_gridded[step, indices_i, indices_j, indices_k, :3] = dataset[step, :, :3] # delta_U components
        dataset_gridded[step, indices_i, indices_j, indices_k, 3] = sdf
        dataset_gridded[step, indices_i, indices_j, indices_k, 4] = dataset[step, :, 3]  # delta_p

    with h5py.File(gridded_h5_fn, 'w') as f:
        f.create_dataset('data', data=dataset_gridded)
    print(f"Stacked data (U, p, sdf) saved to {gridded_h5_fn}.")

    sampling_indices = utils_sampling.define_sample_indexes(
        n_samples_per_frame,
        block_size,
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
        block_size,
        [gridded_h5_fn],
        for_auto_CFD= True
        )
    
    np.save(os.path.join(data_dir, 'maxs_list.npy'), np.array(maxs_list))

    # Extract features and write them using FeatureExtractAndWrite
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
    feature_writer(core_data_fn, n_representative_blocks=200, compute_tucker_factors=True)
    print("Feature extraction and writing complete.")
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

    dropout_rate = 0.1
    lr = 1e-3
    batch_size = 1024
    beta = 0.5
    regularization = 1e-4
    model_architecture='MLP_small'
    n_layers, width = utils_model.define_model_arch(model_architecture)
    model_name = f'{model_architecture}-{standardization_method}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'
    train_tfrecord_fn = os.path.join(data_dir, 'train_data.tfrecords')
    test_tfrecord_fn = os.path.join(data_dir, 'test_data.tfrecords')
    normalization_factors_fn = os.path.join(data_dir, 'mean_std.npz')
    
    # RUN the FIRST NN train
    Train = Training(standardization_method, train_tfrecord_fn, test_tfrecord_fn)
    Train.prepare_data_to_tf(core_data_fn, normalization_factors_fn, flatten_data=True)
    Train.load_data_and_train(
        lr=lr,
        batch_size=batch_size,
        model_name=model_name,
        beta_1=beta,
        num_epoch=20,
        n_layers=n_layers,
        width=width,
        dropout_rate=dropout_rate,
        regularization=regularization,
        model_architecture=model_architecture,
        new_model=True,
        ranks=ranks,
        flatten_data=True,
        weights_fn=os.path.join(data_dir, 'weights.h5'),
        model_h5_path=data_dir
    )

