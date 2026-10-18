"""
Block sampling strategies and error computation.
"""

import numpy as np
import pickle as pk
import tables
import h5py
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
    block_size: tuple,
    grid_res: float,
    first_sim: int,
    last_sim: int,
    first_t: int,
    last_t: int,
    original_dataset_path: str,
    output_pkl_path: str = None,
    limits: dict = None
):
    """
    Define sampling indexes using Latin Hypercube Sampling.
    
    Args:
        n_samples_per_frame: Number of samples per frame
        block_size: Tuple (z, y, x) size of blocks to sample
        grid_res: Grid resolution
        first_sim: First simulation index
        last_sim: Last simulation index (exclusive)
        first_t: First time step
        last_t: Last time step
        dataset_path: Path to HDF5 dataset
        output_pkl_path: Optional path to save indices
        limits: Optional dictionary of limits
        
    Returns:
        list: Indices per simulation per time
    """

    indices_per_sim_per_time = []
    for sim_i in range(first_sim, last_sim + 1):
        if limits is None:
            _, limits = utils_io.read_cells_and_limits(original_dataset_path, sim_i, first_t, last_t, grid_res)

        indices_per_time = []
        for time_i in range(last_t - first_t):
            sampling_indices = get_sampling_indices(block_size, grid_res, limits, n_samples_per_frame)
            indices_per_time.append(sampling_indices)

        indices_per_sim_per_time.append(indices_per_time)

    # Save to file if output_pkl_path is provided
    if output_pkl_path is not None:
        with open(output_pkl_path, 'wb') as f:
            pk.dump(indices_per_sim_per_time, f)

    return indices_per_sim_per_time

def get_sampling_indices(block_size, grid_res, limits, n_samples_per_frame):
    """Get sampling indices for a single time step based on LHS."""

    from .data_processing import get_grid_shape, _unpack_grid_res
    dx, dy, dz = _unpack_grid_res(grid_res)
    block_size_z, block_size_y, block_size_x = block_size

    # Use get_grid_shape for robust grid size calculation
    grid_shape_x, grid_shape_y, grid_shape_z = get_grid_shape(limits, grid_res)

    # The center index must be in [block_size//2, grid_shape - block_size//2 - 1]
    min_z = block_size_z // 2
    max_z = grid_shape_z - (block_size_z - block_size_z // 2)
    min_y = block_size_y // 2
    max_y = grid_shape_y - (block_size_y - block_size_y // 2)
    min_x = block_size_x // 2
    max_x = grid_shape_x - (block_size_x - block_size_x // 2)

    res_zyx = np.array([dz, dy, dx])
    lower_bound = np.array([min_z, min_y, min_x]) * res_zyx
    upper_bound = np.array([max_z, max_y, max_x]) * res_zyx

    ZYX = lower_bound + (upper_bound - lower_bound) * lhs(3, n_samples_per_frame)
    ZYX_indices = (np.round(ZYX / res_zyx)).astype(int)
    # Clamp indices to valid range
    ZYX_indices[:, 0] = np.clip(ZYX_indices[:, 0], min_z, max_z - 1)
    ZYX_indices[:, 1] = np.clip(ZYX_indices[:, 1], min_y, max_y - 1)
    ZYX_indices[:, 2] = np.clip(ZYX_indices[:, 2], min_x, max_x - 1)
    ZYX_indices = np.unique([tuple(row) for row in ZYX_indices], axis=0)

    return ZYX_indices

def sample_blocks(
    block_size: tuple,
    sim_i: int,
    t_start: int,
    t_end: int,
    calculate_maxs: bool = False,
    sample_indices = None,
    gridded_h5_fn: str = None,
    for_auto_CFD: bool = False
):
    """
    Sample N blocks from each time step based on LHS.
    
    Args:
        block_size: Tuple (z, y, x) size of blocks
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

        if for_auto_CFD:
            with h5py.File(gridded_h5_fn, mode='r') as f:
                grid = f['data'][time, :, :, :, :]
        else:
            with tables.open_file(gridded_h5_fn, mode='r') as f:
                grid = f.root.data[time, :, :, :, :]

        ZYX_indices = sample_indices[sim_i][time]

        block_size_z, block_size_y, block_size_x = block_size

        for [ii, jj, kk] in ZYX_indices:
            i_idx_first = ii - block_size_z // 2
            i_idx_last = i_idx_first + block_size_z

            j_idx_first = jj - block_size_y // 2
            j_idx_last = j_idx_first + block_size_y

            k_idx_first = kk - block_size_x // 2
            k_idx_last = k_idx_first + block_size_x

            # Skip blocks that extend outside the grid
            if (i_idx_first < 0 or i_idx_last > grid.shape[0] or
                    j_idx_first < 0 or j_idx_last > grid.shape[1] or
                    k_idx_first < 0 or k_idx_last > grid.shape[2]):
                count += 1
                continue

            # Auto-detect channel layout from grid shape:
            #   8 channels → add_ddu=True:  [ddU_x, ddU_y, ddU_z, dddU_x, dddU_y, dddU_z, sdf, delta_p]
            #   5 channels → add_ddu=False: [dddU_x, dddU_y, dddU_z, sdf, delta_p]
            _n_ch = grid.shape[-1]
            if _n_ch == 8:
                inputs_u_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 0:6]   # ddU + dddU
                inputs_obst_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 6:7] # sdf
                outputs_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 7:8]    # delta_p
            else:  # 5 channels
                inputs_u_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 0:3]   # dddU
                inputs_obst_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 3:4] # sdf
                outputs_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 4:5]    # delta_p

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
    # Recover channel slices — same split as used when sampling
    _nu = inputs_u.shape[-1]   # 6 (add_ddu) or 3 (no ddu)
    inputs_u, inputs_obst, outputs = unique_array[..., :_nu], unique_array[..., _nu:_nu+1], unique_array[..., _nu+1:_nu+2]

    if count > 0:
        print(f'    {count} blocks discarded')
        
    maxs_dict = {}

    if calculate_maxs:
        if _nu == 6:  # add_ddu mode
            maxs_dict['max_abs_ddU_x']   = np.abs(inputs_u[..., 0]).max()
            maxs_dict['max_abs_ddU_y']   = np.abs(inputs_u[..., 1]).max()
            maxs_dict['max_abs_ddU_z']   = np.abs(inputs_u[..., 2]).max()
            maxs_dict['max_abs_dddU_x']  = np.abs(inputs_u[..., 3]).max()
            maxs_dict['max_abs_dddU_y']  = np.abs(inputs_u[..., 4]).max()
            maxs_dict['max_abs_dddU_z']  = np.abs(inputs_u[..., 5]).max()
        else:  # no-ddu mode
            maxs_dict['max_abs_dddU_x']  = np.abs(inputs_u[..., 0]).max()
            maxs_dict['max_abs_dddU_y']  = np.abs(inputs_u[..., 1]).max()
            maxs_dict['max_abs_dddU_z']  = np.abs(inputs_u[..., 2]).max()
        maxs_dict['max_abs_dist']    = np.abs(inputs_obst).max()
        maxs_dict['max_abs_delta_p'] = np.abs(outputs).max()

    return inputs_u, inputs_obst, outputs, maxs_dict


def calculate_and_save_block_abs_max(
    first_sim: int,
    last_sim: int,
    first_t: int,
    last_t: int,
    sample_indices_fn: str,
    base_gridded_h5_fn: str,
    block_size: tuple,
    gridded_h5_filenames: list = None,
    for_auto_CFD: bool = False,
    maxs_fn: str = 'maxs'
) -> list:
    """
    Calculate and save absolute maximum values for normalization.
    
    Args:
        first_sim: First simulation index
        last_sim: Last simulation index
        first_t: First time step
        last_t: Last time step
        sample_indices_fn: Path to sample indices pickle file
        base_gridded_h5_fn: Path to base gridded HDF5 file
        block_size: Tuple (z, y, x) size of blocks
    """
    # Tracking vars — ddU group only used when add_ddu_input=True (detected from data)
    max_abs_ddU_x  = 0
    max_abs_ddU_y  = 0
    max_abs_ddU_z  = 0
    max_abs_dddU_x = 0
    max_abs_dddU_y = 0
    max_abs_dddU_z = 0
    max_abs_dist   = 0
    max_abs_delta_p = 0
    _add_ddu = None  # detected on first iteration

    with open(sample_indices_fn, 'rb') as f:
        sample_indices_per_sim_per_time = pk.load(f)

    print('Calculating absolute maxs to normalize data...')

    if gridded_h5_filenames is None:
        gridded_h5_filenames = utils_io.get_gridded_h5_filenames(
        base_gridded_h5_fn,
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
                for_auto_CFD=for_auto_CFD
            )
            if _add_ddu is None:
                _add_ddu = 'max_abs_ddU_x' in maxs_dict  # detect once from first sample
            if _add_ddu:
                max_abs_ddU_x  = max(max_abs_ddU_x,  maxs_dict['max_abs_ddU_x'])
                max_abs_ddU_y  = max(max_abs_ddU_y,  maxs_dict['max_abs_ddU_y'])
                max_abs_ddU_z  = max(max_abs_ddU_z,  maxs_dict['max_abs_ddU_z'])
            max_abs_dddU_x = max(max_abs_dddU_x, maxs_dict['max_abs_dddU_x'])
            max_abs_dddU_y = max(max_abs_dddU_y, maxs_dict['max_abs_dddU_y'])
            max_abs_dddU_z = max(max_abs_dddU_z, maxs_dict['max_abs_dddU_z'])
            max_abs_dist     = max(max_abs_dist,     maxs_dict['max_abs_dist'])
            max_abs_delta_p  = max(max_abs_delta_p,  maxs_dict['max_abs_delta_p'])

            if _add_ddu:
                print(f"""    Absolute maxs calculated:
                max_abs_ddU_x  = {max_abs_ddU_x:.6f}
                max_abs_ddU_y  = {max_abs_ddU_y:.6f}
                max_abs_ddU_z  = {max_abs_ddU_z:.6f}
                max_abs_dddU_x = {max_abs_dddU_x:.6f}
                max_abs_dddU_y = {max_abs_dddU_y:.6f}
                max_abs_dddU_z = {max_abs_dddU_z:.6f}
                max_abs_dist   = {max_abs_dist:.6f}
                max_abs_delta_p= {max_abs_delta_p:.6f}
            """, flush=True)
            else:
                print(f"""    Absolute maxs calculated:
                max_abs_dddU_x = {max_abs_dddU_x:.6f}
                max_abs_dddU_y = {max_abs_dddU_y:.6f}
                max_abs_dddU_z = {max_abs_dddU_z:.6f}
                max_abs_dist   = {max_abs_dist:.6f}
                max_abs_delta_p= {max_abs_delta_p:.6f}
            """, flush=True)
    
    if _add_ddu:
        maxs_list = [
            max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z,
            max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z,
            max_abs_dist,
            max_abs_delta_p
        ]
    else:
        maxs_list = [
            max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z,
            max_abs_dist,
            max_abs_delta_p
        ]



    np.savetxt(maxs_fn, maxs_list)
    print(f'Absolute maxs saved to "{maxs_fn}" file.')

    return maxs_list
