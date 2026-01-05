import numpy as np
from scipy import ndimage
from numba import njit

#### ASSEMBLY ALGORITHM

@njit(cache=True, fastmath=True)
def fast_masked_mean_njit(arr_flat, mask_flat):
    """Numba-optimized masked mean."""
    total = 0.0
    count = 0
    for i in range(len(arr_flat)):
        if mask_flat[i]:
            total += arr_flat[i]
            count += 1
    return total / count if count > 0 else 0.0

def correct_pred(
        field_block,
        bool_block,
        i, j, k,
        p_i, p_j, p_k,
        shape,
        overlap,
        n_x, n_z,
        BC_col,
        BC_rows,
        BC_depths,
        Ref_BC):
    """
    Optimized version with pre-computed means and cached slicing.
    """
    
    # Pre-compute mask once
    mask = bool_block != 0
    
    # Pre-compute commonly used slices to avoid recomputation
    overlap_slice_start = slice(None, overlap)
    overlap_slice_end = slice(-overlap, None)
    last_col_slice = slice(None, None, None), slice(None, None, None), slice(-1, None)
    second_last_col_slice = slice(None, None, None), slice(None, None, None), slice(-2, -1)
    
    # Helper function for fast masked mean - use views to avoid copies
    def fast_masked_mean(arr_slice, mask_slice):
        """Compute mean using numba with contiguous arrays."""
        if arr_slice.size == 0:
            return 0.0
        # Use reshape(-1) instead of ravel() - it returns a view when possible
        arr_flat = np.ascontiguousarray(arr_slice).reshape(-1)
        mask_flat = np.ascontiguousarray(mask_slice).reshape(-1)
        return fast_masked_mean_njit(arr_flat, mask_flat)
    
    # Case 1 - 1st correction - based on the outlet fixed pressure boundary condition (Ref_BC)
    if i == 0 and j == 0 and k == n_x - 1:
        if not (bool_block[:, :, -1] == 0).all():
            BC_corr = fast_masked_mean(field_block[:, :, -1], mask[:, :, -1]) - Ref_BC
        else:
            BC_corr = fast_masked_mean(field_block[:, :, -2], mask[:, :, -2]) - Ref_BC
        
        field_block -= BC_corr
        BC_col = fast_masked_mean(field_block[:, :, :overlap], mask[:, :, :overlap])
        BC_rows[k] = fast_masked_mean(field_block[:, -overlap:, :], mask[:, -overlap:, :])
        BC_depths[j, k] = fast_masked_mean(field_block[-overlap:, :, :], mask[-overlap:, :, :])
    
    # Case 2
    elif i == 0 and j == 0:
        if k > 0:
            BC_corr = fast_masked_mean(field_block[:, :, -overlap:], mask[:, :, -overlap:]) - BC_col
            field_block -= BC_corr
            BC_col = fast_masked_mean(field_block[:, :, :overlap], mask[:, :, :overlap])
        else:
            intersect_zone_limit_k = overlap - p_k
            BC_corr = fast_masked_mean(field_block[:, :, -intersect_zone_limit_k:], 
                                      mask[:, :, -intersect_zone_limit_k:]) - BC_col
            field_block -= BC_corr
        
        BC_rows[k] = fast_masked_mean(field_block[:, -overlap:, :], mask[:, -overlap:, :])
        BC_depths[j, k] = fast_masked_mean(field_block[-overlap:, :, :], mask[-overlap:, :, :])
    
    # Case 3
    elif i == 0:
        down_most_j = BC_depths.shape[0] - 1
        if j < down_most_j:
            BC_corr = fast_masked_mean(field_block[:, :overlap, :], mask[:, :overlap, :]) - BC_rows[k]
            field_block -= BC_corr
            
            if j == down_most_j - 1:
                BC_rows[k] = fast_masked_mean(field_block[:, -(shape-p_j):, :], mask[:, -(shape-p_j):, :])
            else:
                BC_rows[k] = fast_masked_mean(field_block[:, -overlap:, :], mask[:, -overlap:, :])
        else:
            BC_corr = fast_masked_mean(field_block[:, :-p_j, :], mask[:, :-p_j, :]) - BC_rows[k]
            field_block -= BC_corr
        
        BC_depths[j, k] = fast_masked_mean(field_block[-overlap:, :, :], mask[-overlap:, :, :])
    
    # Case 4
    elif i < n_z - 1:
        BC_corr = fast_masked_mean(field_block[:overlap, :, :], mask[:overlap, :, :]) - BC_depths[j, k]
        field_block -= BC_corr
        BC_depths[j, k] = fast_masked_mean(field_block[-overlap:, :, :], mask[-overlap:, :, :])
    
    # Case 5
    else:
        i_0 = -p_i - overlap
        i_f = -p_i
        BC_corr = fast_masked_mean(field_block[i_0:i_f, :, :], mask[i_0:i_f, :, :]) - BC_depths[j, k]
        field_block -= BC_corr
    
    return field_block


def assemble_prediction(
    array,
    indices_list,
    n_x,
    n_y,
    n_z,
    overlap,
    shape,
    Ref_BC,
    x_array,
    apply_filter,
    shape_x,
    shape_y,
    shape_z,
    deltaU_change_grid,
    deltaP_prev_grid,
    apply_deltaU_change_wgt,
):
    """
    Reconstructs the flow domain based on squared blocks.
    """

    # Pre-allocate result array
    result = np.empty((shape_z, shape_y, shape_x), dtype=np.float32)

    # Arrays to store average pressure in overlap regions
    BC_col = 0.0
    BC_rows = np.zeros(n_x, dtype=np.float32)
    BC_depths = np.zeros((n_y, n_x), dtype=np.float32)

    # i index where the lower blocks are located
    p_i = shape_z - ((shape - overlap) * (n_z - 2) + shape)
    # j index where the left-most blocks are located
    p_j = shape_y - ((shape - overlap) * (n_y - 2) + shape)
    # k index where the left-most blocks are located
    p_k = shape_x - ((shape - overlap) * (n_x - 1) + shape)

    # Vectorized range computation
    z_idx = np.arange(n_z - 1, dtype=np.int32)
    z_ranges = np.empty((n_z, 2), dtype=np.int32)
    z_ranges[:-1, 0] = (shape - overlap) * z_idx
    z_ranges[:-1, 1] = (shape - overlap) * z_idx + shape
    z_ranges[-1] = [shape_z - p_i, shape_z]
    
    y_idx = np.arange(n_y - 1, dtype=np.int32)
    y_ranges = np.empty((n_y, 2), dtype=np.int32)
    y_ranges[:-1, 0] = (shape - overlap) * y_idx
    y_ranges[:-1, 1] = (shape - overlap) * y_idx + shape
    y_ranges[-1] = [shape_y - p_j, shape_y]
    
    x_idx = np.arange(n_x, dtype=np.int32)
    x_idx_rev = n_x - x_idx - 1
    x_ranges = np.empty((n_x, 2), dtype=np.int32)
    x_ranges[:, 0] = shape_x - shape - x_idx_rev * (shape - overlap)
    x_ranges[:, 1] = shape_x - x_idx_rev * (shape - overlap)
    x_ranges[0, 0] = 0
    x_ranges[0, 1] = shape
    
    # Pre-convert indices_list to numpy array for faster indexing
    if not isinstance(indices_list, np.ndarray):
        indices_list = np.array(indices_list, dtype=np.int32)

    # Group blocks by their type to minimize branching
    # Type encoding: i_type * 100 + j_type
    # i_type: 0=first depth, 1=middle depth, 2=last depth
    # j_type: 0=first row, 1=middle row, 2=last row
    block_types = np.zeros(len(indices_list), dtype=np.int32)
    for idx, (i, j, k) in enumerate(indices_list):
        i_type = 0 if i == 0 else (2 if i == n_z - 1 else 1)
        j_type = 0 if j == 0 else (2 if j == n_y - 1 else 1)
        block_types[idx] = i_type * 100 + j_type
    
    # Sort blocks by type for better cache locality
    sorted_indices = np.argsort(block_types)
    
    n_blocks = x_array.shape[0]
    for i_block in sorted_indices:
        i, j, k = indices_list[i_block]
        flow_bool = x_array[i_block, :, :, :, 3]
        pred_field = array[i_block]

        # Applying the correction
        pred_field = correct_pred(
            pred_field, flow_bool, i, j, k, p_i, p_j, p_k,
            shape, overlap, n_x, n_z,
            BC_col, BC_rows, BC_depths, Ref_BC
        )

        # Get pre-computed ranges for this block
        z0, z1 = z_ranges[i]
        y0, y1 = y_ranges[j]
        x0, x1 = x_ranges[k]

        # Simplified assignment logic - use block type to reduce branching
        block_type = block_types[i_block]
        
        # Decode block type
        is_last_depth = (block_type // 100) == 2
        is_last_row = (block_type % 100) == 2
        
        if is_last_depth:
            if is_last_row:
                result[z0:z1, y0:y1, x0:x1] = pred_field[-p_i:, -p_j:, :]
            else:
                result[z0:z1, y0:y1, x0:x1] = pred_field[-p_i:, :, :]
        else:
            if is_last_row:
                result[z0:z1, y0:y1, x0:x1] = pred_field[:, -p_j:, :]
            else:
                result[z0:z1, y0:y1, x0:x1] = pred_field

    # Correction based on BC
    if ~(flow_bool[:, :, -1] == 0).all():
        # Vectorized operation
        correction = np.mean(3.0 * result[:, :, -1] - result[:, :, -2]) / 3.0
    else:
        correction = np.mean(3.0 * result[:, :, -2] - result[:, :, -3]) / 3.0
    result -= correction

    # Apply Gaussian filter
    if apply_filter:
        result = ndimage.gaussian_filter(result, sigma=(10, 10, 10), order=0, mode='constant', truncate=3.0)

    change_in_deltap = None
    if apply_deltaU_change_wgt:
        deltaU_change_grid = ndimage.gaussian_filter(deltaU_change_grid, sigma=(5, 5, 5), order=0, mode='constant', truncate=3.0)
        # Fused operation
        change_in_deltap = ndimage.gaussian_filter((result - deltaP_prev_grid) * deltaU_change_grid, 
                                                    sigma=(10, 10, 10), order=0, mode='constant', truncate=3.0)

    return result, change_in_deltap