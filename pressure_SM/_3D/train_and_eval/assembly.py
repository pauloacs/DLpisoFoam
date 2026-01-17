import numpy as np
from scipy import ndimage
from numba import njit

#### ASSEMBLY ALGORITHM

@njit(cache=True, fastmath=True, nogil=True, inline="always")  # CHANGED: remove parallel=True
def masked_mean_3d_bounds(arr, mask, z0, z1, y0, y1, x0, x1):
    """Mean of arr over bounds where mask != 0 (Numba, no temps)."""
    total = 0.0
    count = 0
    for zz in range(z0, z1):  # CHANGED: range, not prange
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if mask[zz, yy, xx] != 0:
                    total += arr[zz, yy, xx]
                    count += 1
    return total / count if count > 0 else 0.0

@njit(cache=True, fastmath=True, nogil=True, inline="always")
def any_nonzero_outlet_plane(mask):
    """Return True if any cell on the outlet x=-1 plane is non-zero (Numba)."""
    sz, sy, sx = mask.shape
    x = sx - 1
    for z in range(sz):
        for y in range(sy):
            if mask[z, y, x] != 0:
                return True
    return False

@njit(cache=True, fastmath=True, nogil=True, inline="always")  # CHANGED: inline to cut call overhead
def correct_pred_jit(
        field_block,
        bool_block,
        i, j, k,
        p_i, p_j, p_k,
        shape,
        overlap,
        n_x, n_z,
        BC_col_arr,
        BC_rows,
        BC_depths,
        Ref_BC):
    sz = field_block.shape[0]
    sy = field_block.shape[1]
    sx = field_block.shape[2]

    # Case 1
    if i == 0 and j == 0 and k == n_x - 1:
        if any_nonzero_outlet_plane(bool_block):
            out_mean = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, sy, sx - 1, sx)
        else:
            out_mean = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, sy, sx - 2, sx - 1)

        BC_corr = out_mean - Ref_BC
        sub_scalar_inplace(field_block, BC_corr)

        BC_col_arr[0] = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, sy, 0, overlap)
        BC_rows[k] = masked_mean_3d_bounds(field_block, bool_block, 0, sz, sy - overlap, sy, 0, sx)
        BC_depths[j, k] = masked_mean_3d_bounds(field_block, bool_block, sz - overlap, sz, 0, sy, 0, sx)

    # Case 2
    elif i == 0 and j == 0:
        if k > 0:
            m = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, sy, sx - overlap, sx)
            BC_corr = m - BC_col_arr[0]
            sub_scalar_inplace(field_block, BC_corr)
            BC_col_arr[0] = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, sy, 0, overlap)
        else:
            intersect_zone_limit_k = overlap - p_k
            x0 = sx - intersect_zone_limit_k
            m = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, sy, x0, sx)
            BC_corr = m - BC_col_arr[0]
            sub_scalar_inplace(field_block, BC_corr)

        BC_rows[k] = masked_mean_3d_bounds(field_block, bool_block, 0, sz, sy - overlap, sy, 0, sx)
        BC_depths[j, k] = masked_mean_3d_bounds(field_block, bool_block, sz - overlap, sz, 0, sy, 0, sx)

    # Case 3
    elif i == 0:
        down_most_j = BC_depths.shape[0] - 1
        if j < down_most_j:
            m = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, overlap, 0, sx)
            BC_corr = m - BC_rows[k]
            sub_scalar_inplace(field_block, BC_corr)

            if j == down_most_j - 1:
                y0 = sy - (shape - p_j)
                BC_rows[k] = masked_mean_3d_bounds(field_block, bool_block, 0, sz, y0, sy, 0, sx)
            else:
                BC_rows[k] = masked_mean_3d_bounds(field_block, bool_block, 0, sz, sy - overlap, sy, 0, sx)
        else:
            y1 = sy - p_j
            m = masked_mean_3d_bounds(field_block, bool_block, 0, sz, 0, y1, 0, sx)
            BC_corr = m - BC_rows[k]
            sub_scalar_inplace(field_block, BC_corr)

        BC_depths[j, k] = masked_mean_3d_bounds(field_block, bool_block, sz - overlap, sz, 0, sy, 0, sx)

    # Case 4
    elif i < n_z - 1:
        m = masked_mean_3d_bounds(field_block, bool_block, 0, overlap, 0, sy, 0, sx)
        BC_corr = m - BC_depths[j, k]
        sub_scalar_inplace(field_block, BC_corr)
        BC_depths[j, k] = masked_mean_3d_bounds(field_block, bool_block, sz - overlap, sz, 0, sy, 0, sx)

    # Case 5
    else:
        i_0 = sz - p_i - overlap
        i_f = sz - p_i
        m = masked_mean_3d_bounds(field_block, bool_block, i_0, i_f, 0, sy, 0, sx)
        BC_corr = m - BC_depths[j, k]
        sub_scalar_inplace(field_block, BC_corr)

    return field_block


@njit(cache=True, fastmath=True, nogil=True, inline="always")  # CHANGED: remove parallel=True
def plane_combo_mean(result, a, x1, b, x2, denom):
    """mean((a*result[:,:,x1] + b*result[:,:,x2]) / denom) without temporaries."""
    sz, sy, _sx = result.shape
    total = 0.0
    count = sz * sy
    for z in range(sz):  # CHANGED: range, not prange
        for y in range(sy):
            total += (a * result[z, y, x1] + b * result[z, y, x2]) / denom
    return total / count if count > 0 else 0.0

@njit(cache=True, fastmath=True, nogil=True, inline="always")  # CHANGED: remove parallel=True
def sub_scalar_inplace(arr, val):
    sz, sy, sx = arr.shape
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                arr[z, y, x] -= val

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

    if apply_filter or apply_deltaU_change_wgt:
        tmp0 = np.empty_like(result, dtype=np.float32)
        tmp1 = np.empty_like(result, dtype=np.float32)

    # Arrays to store average pressure in overlap regions
    BC_col_arr = np.zeros(1, dtype=np.float32)
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
    
    if not isinstance(indices_list, np.ndarray):
        indices_list = np.array(indices_list, dtype=np.int32)

    # CHANGED: revert to simple Python loop (process_blocks_jit wrapper added overhead)
    n_blocks = indices_list.shape[0]
    for b in range(n_blocks):
        i, j, k = indices_list[b]
        flow_bool = x_array[b, :, :, :, 3]
        pred_field = array[b]

        pred_field = correct_pred_jit(
            pred_field, flow_bool, i, j, k, p_i, p_j, p_k,
            shape, overlap, n_x, n_z,
            BC_col_arr, BC_rows, BC_depths, Ref_BC
        )

        z0, z1 = z_ranges[i]; y0, y1 = y_ranges[j]; x0, x1 = x_ranges[k]

        if i == n_z - 1:
            if j == n_y - 1:
                result[z0:z1, y0:y1, x0:x1] = pred_field[-p_i:, -p_j:, :]
            else:
                result[z0:z1, y0:y1, x0:x1] = pred_field[-p_i:, :, :]
        else:
            if j == n_y - 1:
                result[z0:z1, y0:y1, x0:x1] = pred_field[:, -p_j:, :]
            else:
                result[z0:z1, y0:y1, x0:x1] = pred_field

    outlet_has_fluid = np.any(x_array[:, :, :, -1, 3] != 0)

    if outlet_has_fluid:
        correction = plane_combo_mean(result, 3.0, -1, -1.0, -2, 3.0)
    else:
        correction = plane_combo_mean(result, 3.0, -2, -1.0, -3, 3.0)
    sub_scalar_inplace(result, correction)

    if apply_filter:
        ndimage.gaussian_filter1d(result, sigma=10, axis=0, order=0, mode="constant", truncate=3.0, output=tmp0)
        ndimage.gaussian_filter1d(tmp0, sigma=10, axis=1, order=0, mode="constant", truncate=3.0, output=tmp1)
        ndimage.gaussian_filter1d(tmp1, sigma=10, axis=2, order=0, mode="constant", truncate=3.0, output=result)

    change_in_deltap = None
    if apply_deltaU_change_wgt:
        ndimage.gaussian_filter1d(deltaU_change_grid, sigma=5, axis=0, order=0, mode="constant", truncate=3.0, output=tmp0)
        ndimage.gaussian_filter1d(tmp0, sigma=5, axis=1, order=0, mode="constant", truncate=3.0, output=tmp1)
        ndimage.gaussian_filter1d(tmp1, sigma=5, axis=2, order=0, mode="constant", truncate=3.0, output=deltaU_change_grid)

        np.subtract(result, deltaP_prev_grid, out=tmp0)
        np.multiply(tmp0, deltaU_change_grid, out=tmp0)

        ndimage.gaussian_filter1d(tmp0, sigma=10, axis=0, order=0, mode="constant", truncate=3.0, output=tmp1)
        ndimage.gaussian_filter1d(tmp1, sigma=10, axis=1, order=0, mode="constant", truncate=3.0, output=tmp0)
        ndimage.gaussian_filter1d(tmp0, sigma=10, axis=2, order=0, mode="constant", truncate=3.0, output=tmp1)
        change_in_deltap = tmp1

    return result, change_in_deltap