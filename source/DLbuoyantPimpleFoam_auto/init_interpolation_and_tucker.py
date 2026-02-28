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
from pressure_SM._3D.train_and_eval.training import Training

from pressure_SM._3D.auto_CFD.utils import read_component_bin_files

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

    grid_res = 0.01  # Example value
    block_size = 8   # Example value
    n_samples_per_frame = 1000
    standardization_method = 'minmax'  # or 'zscore', etc.
    gridded_h5_fn = os.path.join(data_dir, 'gridded_data.h5')
    sample_idx_fn = os.path.join(data_dir, 'sample_idx_per_time.npy')
    tucker_factors_fn = os.path.join(data_dir, 'tucker_factors.pkl')
    ranks = 3
    chunk_size = 500
    
    core_data_fn = os.path.join(data_dir, 'core_data.h5')

    # Example: Compute interpolation weights directly (if needed)
    cell_centers = read_cell_centers()
    boundary_points, boundary_groups = read_boundary_points()
    vert, weights = utils_data.interp_weights(cell_centers, boundary_points, interp_method='IDW')
    np.save(os.path.join(data_dir, 'interp_weights.npy'), weights)
    np.save(os.path.join(data_dir, 'interp_vertices.npy'), vert)
    print("Interpolation weights and vertices saved.")

    # Get grid limits using both cell centers and boundaries
    limits = get_grid_limits(cell_centers, boundary_points)
    print('Grid limits:', limits)

    # Create the grid using the limits and grid_res
    X0, Y0, Z0 = utils_data.create_uniform_grid(limits, grid_res)
    # Stack grid points for interpolation
    grid_points = np.concatenate(
        (np.expand_dims(X0, axis=1),
         np.expand_dims(Y0, axis=1),
         np.expand_dims(Z0, axis=1)),
        axis=-1
    )

    # Calculate SDF and domain mask on the grid
    domain_bool, sdf = utils_geo.domain_dist(boundary_points, grid_points, grid_res)
    np.save(os.path.join(data_dir, 'grid_sdf.npy'), sdf)
    np.save(os.path.join(data_dir, 'grid_domain_mask.npy'), domain_bool)
    print('SDF and domain mask arrays saved.')

    # Compute interpolation weights and vertices from cell centers to grid points
    vert, weights = utils_data.interp_weights(cell_centers, grid_points, interp_method='IDW')
    np.save(os.path.join(data_dir, 'interp_weights.npy'), weights)
    np.save(os.path.join(data_dir, 'interp_vertices.npy'), vert)
    print("Interpolation weights and vertices (cell centers -> grid) saved.")


    delta_p_files = sorted(glob.glob(os.path.join(data_dir, 'delta_p_*.bin')))
    n_sample_frames = len(delta_p_files)

    with open(os.path.join(data_dir, 'n_cells.txt'), 'r') as f:
        n_cells = int(f.read().strip())

    if n_sample_frames > 0:
        delta_U, delta_p = read_component_bin_files(n_sample_frames, n_cells)
        print(f"Loaded {n_sample_frames} samples from binary component files.")
    else:
        print("No binary component field files found.")

    delta_U_grid = utils_data.interpolate_fill_njit(delta_U, vert, weights, fill_value=np.nan)
    delta_p_grid = utils_data.interpolate_fill_njit(delta_p, vert, weights, fill_value=np.nan)
    print("Interpolation to grid complete.")

    # Stack gridded_U, gridded_p, and sdf, then save to gridded_h5_fn_sim
    sdf_stack = np.broadcast_to(sdf, delta_U_grid.shape)
    dataset = np.stack([delta_U_grid, delta_p_grid, sdf_stack], axis=-1)  # shape: (N_samples, z, y, x, 3 + 1 + 1)
        
    with h5py.File(gridded_h5_fn, 'w') as f:
        f.create_dataset('data', data=dataset)
    print(f"Stacked data (U, p, sdf) saved to {gridded_h5_fn}.")

    sampling_indices = utils_sampling.define_sample_indexes(
        n_samples_per_frame,
        block_size,
        grid_res,
        0,  # first_sim
        0,  # last_sim
        0,  # first_t
        n_sample_frames,  # last_t
        gridded_h5_fn,
        sample_idx_fn
    )

    maxs_list = utils_sampling.calculate_and_save_block_abs_max(
        0,
        0,
        0,
        n_sample_frames,
        sample_idx_fn,
        gridded_h5_fn,
        block_size
        )
    
    maxs_list.save(os.path.join(data_dir, 'maxs_list.npy'))

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
        gridded_h5_fn=gridded_h5_fn,
        sample_indices_fn=sample_idx_fn,
        tucker_factors_fn=tucker_factors_fn,
        flatten_data=False,
        maxs_list=maxs_list
    )
    feature_writer(core_data_fn, n_representative_blocks=7500, compute_tucker_factors=True)
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


    # DO THE FIRST NN train
    Train = Training(standardization_method)
    Train.prepare_data_to_tf(core_data_fn, flatten_data=True)
    Train.load_data_and_train(
        lr=lr,
        batch_size=batch_size,
        model_name=model_name,
        beta=beta,
        num_epoch=100,
        n_layers=n_layers,
        width=width,
        dropout_rate=dropout_rate,
        regularization=regularization,
        model_architecture=model_architecture,
        new_model=True,
        ranks=ranks,
        flatten_data=True
    )

