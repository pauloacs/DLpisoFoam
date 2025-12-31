"""
Block sampling strategies and error computation.
"""

import numpy as np
import pickle as pk
import tables
from pyDOE import lhs
from . import io_operations as utils_io

def compute_in_block_error(
    pred: np.ndarray,
    true: np.ndarray,
    flow_bool: np.ndarray):
    """
    Compute normalized error metrics within blocks.
    
    Args:
        pred: Predicted values
        true: True values
        flow_bool: Boolean mask for flow region
        
    Returns:
        tuple: (pred_minus_true_block, pred_minus_true_squared_block)
    """
    true_masked = true[flow_bool]
    pred_masked = pred[flow_bool]

    # Calculate norm based on reference data a predicted data
    norm_true = np.max(true_masked) - np.min(true_masked)
    norm_pred = np.max(pred_masked) - np.min(pred_masked)

    norm = norm_true

    mask_nan = ~np.isnan( pred_masked  - true_masked )

    BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
    RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
    STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )
    
    print(f"""
            norm_true = {norm_true};
            norm_pred = {norm_pred};

    ** Error in delta_p (blocks) **

        normVal  = {norm} Pa
        biasNorm = {BIAS_norm:.3f}%
        stdeNorm = {STDE_norm:.3f}%
        rmseNorm = {RMSE_norm:.3f}%
    """, flush = True)

    pred_minus_true_block = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm
    pred_minus_true_squared_block = np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2
    return pred_minus_true_block, pred_minus_true_squared_block


def define_sample_indexes(
    n_samples_per_frame: int,
    block_size: int,
    grid_res: float,
    first_sim: int,
    last_sim: int,
    first_t: int,
    last_t: int,
    original_dataset_path: str,
    output_pkl_path: str = None
):
    """
    Define sampling indexes using Latin Hypercube Sampling.
    
    Args:
        n_samples_per_frame: Number of samples per frame
        block_size: Size of blocks to sample
        grid_res: Grid resolution
        first_sim: First simulation index
        last_sim: Last simulation index (exclusive)
        first_t: First time step
        last_t: Last time step
        dataset_path: Path to HDF5 dataset
        output_pkl_path: Optional path to save indices
        
    Returns:
        list: Indices per simulation per time
    """

    indices_per_sim_per_time = []
    for sim_i in range(first_sim, last_sim + 1):

        _, limits = utils_io.read_cells_and_limits(original_dataset_path, sim_i, first_t, last_t)

        indices_per_time = []
        for time_i in range(last_t - first_t):
            lower_bound = np.array([
                0 + block_size * grid_res / 2,
                0 + block_size * grid_res / 2,
                0 + block_size * grid_res / 2
            ])

            upper_bound = np.array([
                (limits['z_max'] - limits['z_min']) - block_size * grid_res / 2,
                (limits['y_max'] - limits['y_min']) - block_size * grid_res / 2,
                (limits['x_max'] - limits['x_min']) - block_size * grid_res / 2
            ])
            ZYX = lower_bound + (upper_bound - lower_bound) * lhs(3, n_samples_per_frame)
            ZYX_indices = (np.round(ZYX / grid_res)).astype(int)
            ZYX_indices = np.unique([tuple(row) for row in ZYX_indices], axis=0)
            indices_per_time.append(ZYX_indices)

        indices_per_sim_per_time.append(indices_per_time)

    # Save to file if output_pkl_path is provided
    if output_pkl_path is not None:
        with open(output_pkl_path, 'wb') as f:
            pk.dump(indices_per_sim_per_time, f)

    return indices_per_sim_per_time


def sample_blocks(
    block_size: int,
    sim_i: int,
    t_start: int,
    t_end: int,
    calculate_maxs: bool = False,
    sample_indices = None,
    gridded_h5_fn: str = None,
):
    """
    Sample N blocks from each time step based on LHS.
    
    Args:
        block_size: Size of blocks
        sim_i: Current simulation index
        t_start: Start time for sampling
        t_end: End time for sampling
        calculate_maxs: Whether to calculate maximum values
        sample_indices: Pre-computed sample indices
        gridded_h5_fn: Path to gridded HDF5 file
        
    Returns:
        tuple: (inputs_u, inputs_obst, outputs, and updated max values)
    """
    inputs_u_list = []
    inputs_obst_list = []
    outputs_list = []

    count = 0

    for time in range(t_start, t_end):

        with tables.open_file(gridded_h5_fn, mode='r') as f:
            grid = f.root.data[time, :, :, :, :]

        ZYX_indices = sample_indices[sim_i][time]

        for [ii, jj, kk] in ZYX_indices:

            i_idx_first = int(ii - block_size / 2)
            i_idx_last = int(ii + block_size / 2)

            j_idx_first = int(jj - block_size / 2)
            j_idx_last = int(jj + block_size / 2)

            k_idx_first = int(kk - block_size / 2)
            k_idx_last = int(kk + block_size / 2)

            inputs_u_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 0:3]
            inputs_obst_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 3:4]
            outputs_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 4:5]

            # Remove all the blocks with delta_U = 0 and delta_p = 0
            if not ((inputs_u_sample == 0).all() and (outputs_sample == 0).all()):
                inputs_u_list.append(inputs_u_sample)
                inputs_obst_list.append(inputs_obst_sample)
                outputs_list.append(outputs_sample)
            else:
                count += 1

    inputs_u = np.array(inputs_u_list)
    inputs_obst = np.array(inputs_obst_list)
    outputs = np.array(outputs_list)

    # Remove mean from each output block
    for step in range(outputs.shape[0]):
        outputs[step, ...][inputs_obst[step, ...] != 0] -= np.mean(outputs[step, ...][inputs_obst[step, ...] != 0])

    print('Removing duplicate blocks ...', flush=True)
    array = np.c_[inputs_u, inputs_obst, outputs]
    reshaped_array = array.reshape(array.shape[0], -1)
    # Find unique rows
    unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
    unique_array = array[unique_indices]
    inputs_u, inputs_obst, outputs = unique_array[..., 0:3], unique_array[..., 3:4], unique_array[..., 4:5]

    if count > 0:
        print(f'    {count} blocks discarded')
        
    maxs_dict = {}

    if calculate_maxs:
        maxs_dict['max_abs_delta_Ux'] = np.abs(inputs_u[..., 0]).max()
        maxs_dict['max_abs_delta_Uy'] = np.abs(inputs_u[..., 1]).max()
        maxs_dict['max_abs_delta_Uz'] = np.abs(inputs_u[..., 2]).max()
        maxs_dict['max_abs_dist']     = np.abs(inputs_obst).max()
        maxs_dict['max_abs_delta_p']  = np.abs(outputs).max()

    return inputs_u, inputs_obst, outputs, maxs_dict


def calculate_and_save_block_abs_max(
    first_sim: int,
    last_sim: int,
    first_t: int,
    last_t: int,
    sample_indices_fn: str,
    gridded_h5_fn: str,
    block_size: int
):
    """
    Calculate and save absolute maximum values for normalization.
    
    Args:
        first_sim: First simulation index
        last_sim: Last simulation index
        first_t: First time step
        last_t: Last time step
        sample_indices_fn: Path to sample indices pickle file
        gridded_h5_fn: Path to gridded HDF5 file
        block_size: Size of blocks
    """
    max_abs_delta_Ux = 0
    max_abs_delta_Uy = 0
    max_abs_delta_Uz = 0
    max_abs_dist = 0
    max_abs_delta_p = 0

    with open(sample_indices_fn, 'rb') as f:
        sample_indices_per_sim_per_time = pk.load(f)

    print('Calculating absolute maxs to normalize data...')

    gridded_h5_filenames = utils_io.get_gridded_h5_filenames(
      gridded_h5_fn,
      first_sim,
      last_sim
      )
    
    for sim_i in range(first_sim, last_sim + 1):
        for time in range(last_t - first_t):
            _, _, _, maxs_dict = sample_blocks(
                block_size,
                sim_i - first_sim,
                t_start=time,
                t_end=time + 1,
                calculate_maxs=True,
                sample_indices=sample_indices_per_sim_per_time,
                gridded_h5_fn=gridded_h5_filenames[sim_i - first_sim],
            )
            max_abs_delta_Ux = max(max_abs_delta_Ux, maxs_dict['max_abs_delta_Ux'])
            max_abs_delta_Uy = max(max_abs_delta_Uy, maxs_dict['max_abs_delta_Uy'])
            max_abs_delta_Uz = max(max_abs_delta_Uz, maxs_dict['max_abs_delta_Uz'])
            max_abs_dist     = max(max_abs_dist, maxs_dict['max_abs_dist'])    
            max_abs_delta_p  = max(max_abs_delta_p, maxs_dict['max_abs_delta_p'])

    np.savetxt('maxs', [
        max_abs_delta_Ux,
        max_abs_delta_Uy,
        max_abs_delta_Uz,
        max_abs_dist,
        max_abs_delta_p
    ])
