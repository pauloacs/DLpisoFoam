import numpy as np
import glob
import os
import h5py

from pressure_SM._3D.train_and_eval.utils import data_processing as utils_data
from pressure_SM._3D.train_and_eval.utils import domain_geometry as utils_geo
from pressure_SM._3D.train_and_eval.utils import io_operations as utils_io
from pressure_SM._3D.train_and_eval.utils import sampling as utils_sampling
from pressure_SM._3D.train_and_eval.utils import model_utils as utils_model

from pressure_SM._3D.train_and_eval.data_processor import FeatureExtractAndWrite
from pressure_SM._3D.train_and_eval.training import Training

from pressure_SM._3D.auto_CFD.utils import read_component_bin_files


# --- Load interpolation weights and Tucker factors ---
interp_weights = np.load('interp_weights.npy')
tucker_U = np.load('tucker_U.npy')
tucker_S = np.load('tucker_S.npy')
tucker_V = np.load('tucker_V.npy')

# --- Utility to read new delta_U and delta_p files ---
def read_new_fields(existing_files, pattern):
    all_files = sorted(glob.glob(pattern))
    new_files = [f for f in all_files if f not in existing_files]
    return new_files

# --- Load previous dataset file lists ---
if os.path.exists('used_delta_U_files.txt'):
    with open('used_delta_U_files.txt', 'r') as f:
        used_u_files = set(line.strip() for line in f)
else:
    used_u_files = set()
if os.path.exists('used_delta_p_files.txt'):
    with open('used_delta_p_files.txt', 'r') as f:
        used_p_files = set(line.strip() for line in f)
else:
    used_p_files = set()



# --- Feature Extraction and Training ---
from pressure_SM._3D.train_and_eval.data_processor import FeatureExtractAndWrite
import tables

def add_new_features_and_train():

    import argparse
    parser = argparse.ArgumentParser(description='Initialize interpolation weights and Tucker factors for ML training.')
    parser.add_argument('--data_dir', type=str, default='ML_data', help='Directory where the ML data is stored (default: ML_data)')
    args = parser.parse_args()
    data_dir = args.data_dir

    grid_res = 0.001 # This one is highly dependent on the simulation length scales (aim for at least 50 grid points across the domain in each direction)
    block_size = 16
    n_samples_per_frame = 1000
    standardization_method = 'minmax'  # or 'zscore', etc.
    gridded_h5_fn = os.path.join(data_dir, 'gridded_data.h5')
    sample_idx_fn = os.path.join(data_dir, 'sample_idx_per_time.npy')
    maxs_list_fn = os.path.join(data_dir, 'maxs_list.npy')
    tucker_factors_fn = os.path.join(data_dir, 'tucker_factors.pkl')
    ranks = 3
    chunk_size = 500
    core_data_fn = os.path.join(data_dir, 'core_data.h5')

    # Before writing new features, grab some from the last run
    with tables.open_file(core_data_fn, mode='r') as f:
      old_input_cores = f.root.inputs[...] 
      old_output_cores = f.root.outputs[...] 

    # --- Determine which samples are available ---
    delta_p_files = sorted(glob.glob(os.path.join(data_dir, 'delta_p_*.bin')))
    n_sample_frames = len(delta_p_files)

    print(f"Found {n_sample_frames} delta_p files. Removing {n_sample_frames} from the previous dataset.")
    print("Next it will be updated with the new features extracted from these files.")
    old_input_cores_to_keep = old_input_cores[:-n_sample_frames * n_samples_per_frame]
    old_output_cores_to_keep = old_output_cores[:-n_sample_frames * n_samples_per_frame]

    with open(os.path.join(data_dir, 'n_cells.txt'), 'r') as f:
        n_cells = int(f.read().strip())

    if n_sample_frames > 0:
        delta_U, delta_p = read_component_bin_files(n_sample_frames, n_cells)
        print(f"Loaded {n_sample_frames} samples from binary component files.")
    else:
        print("No binary component field files found.")

    weights =np.load(os.path.join(data_dir, 'interp_weights.npy'))
    vert = np.load(os.path.join(data_dir, 'interp_vertices.npy'))
    print("Interpolation weights and vertices loaded.")

    delta_U_grid = utils_data.interpolate_fill_njit(delta_U, vert, weights, fill_value=np.nan)
    delta_p_grid = utils_data.interpolate_fill_njit(delta_p, vert, weights, fill_value=np.nan)
    sdf = np.load(os.path.join(data_dir, 'grid_sdf.npy'))

    # Stack gridded_U, gridded_p, and sdf, then save to gridded_h5_fn_sim
    sdf_stack = np.broadcast_to(sdf, delta_U_grid.shape)
    dataset = np.stack([delta_U_grid, delta_p_grid, sdf_stack], axis=-1)  # shape: (N_samples, z, y, x, 3 + 1 + 1)

    # Clean old gridded data file if it exists
    if os.path.exists(gridded_h5_fn):
        os.remove(gridded_h5_fn)

    with h5py.File(gridded_h5_fn, 'w') as f:
        f.create_dataset('data', data=dataset)
    print(f"Stacked data (U, p, sdf) saved to {gridded_h5_fn}.")

    # Using sample indices defined in init_interpolation_and_tucker.py
    current_maxs_list = np.load(maxs_list_fn, allow_pickle=True)

    maxs_list = utils_sampling.calculate_and_save_block_abs_max(
        0,
        0,
        0,
        n_sample_frames,
        sample_idx_fn,
        gridded_h5_fn,
        block_size
        )

    # Update maxs_list with new maximums
    maxs_list = np.maximum(maxs_list, current_maxs_list)
    np.save(maxs_list_fn, maxs_list)

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
    # Run with precalculated tucker factors
    feature_writer(core_data_fn, compute_tucker_factors=False)
    print("Feature extraction and writing complete.")
    print(f"New core data with features saved to {core_data_fn}.")

    # --- Add old_input_cores_to_keep and old_output_cores_to_keep to core_data_fn ---

    # Read new features from core_data_fn
    with tables.open_file(core_data_fn, mode='r') as f:
        new_input_cores = f.root.inputs[...]
        new_output_cores = f.root.outputs[...]

    # Concatenate old and new features
    all_input_cores = np.concatenate([old_input_cores_to_keep, new_input_cores], axis=0)
    all_output_cores = np.concatenate([old_output_cores_to_keep, new_output_cores], axis=0)

    # Overwrite core_data_fn with the combined data
    with tables.open_file(core_data_fn, mode='w') as f:
        atom = tables.Atom.from_dtype(all_input_cores.dtype)
        input_array = f.create_carray(f.root, 'inputs', atom, all_input_cores.shape)
        output_array = f.create_carray(f.root, 'outputs', atom, all_output_cores.shape)
        input_array[:] = all_input_cores
        output_array[:] = all_output_cores
    print("Old and new features combined and saved to core_data.h5.")

    dropout_rate = 0.1
    lr = 1e-3
    batch_size = 1024
    beta = 0.5
    regularization = 1e-4
    model_architecture='MLP_small'
    n_layers, width = utils_model.define_model_arch(model_architecture)
    model_name = f'{model_architecture}-{standardization_method}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'

    # Continue the training with new data
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
        new_model=False,  # Continue training the existing model
        ranks=ranks,
        flatten_data=True
    )


if __name__ == '__main__':
    add_new_features_and_train()
