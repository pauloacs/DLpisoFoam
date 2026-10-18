import numpy as np
from scipy import ndimage
from numba import njit
import os

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

def _gaussian_filter3d(arr, buf0, buf1, sigmas, radii, output):
    """Apply separable 3D Gaussian filter (axis 0→1→2) using pre-allocated buffers."""
    if arr is buf0:
        # Input overlaps with buf0: use buf1 first to avoid overwriting input.
        ndimage.gaussian_filter1d(arr,  sigma=sigmas[0], axis=0, order=0, mode="constant", radius=radii[0], output=buf1)
        ndimage.gaussian_filter1d(buf1, sigma=sigmas[1], axis=1, order=0, mode="constant", radius=radii[1], output=buf0)
        ndimage.gaussian_filter1d(buf0, sigma=sigmas[2], axis=2, order=0, mode="constant", radius=radii[2], output=output)
    else:
        ndimage.gaussian_filter1d(arr,  sigma=sigmas[0], axis=0, order=0, mode="constant", radius=radii[0], output=buf0)
        ndimage.gaussian_filter1d(buf0, sigma=sigmas[1], axis=1, order=0, mode="constant", radius=radii[1], output=buf1)
        ndimage.gaussian_filter1d(buf1, sigma=sigmas[2], axis=2, order=0, mode="constant", radius=radii[2], output=output)


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
    delta_U_change_grid,
    delta_delta_p_prev_grid,
    apply_deltaU_change_wgt,
    filter_tuple=(10, 10, 10),
    filter_tuple_deltaU=(1, 1, 1),
    plot_results=False,
    plot_call_count=1,
    plot_output_dir="assembly_plots",
):
    """
    Reconstructs the flow domain based on squared blocks (delta-delta fields).
    """

    # Convert kernel sizes to (sigma, radius) for gaussian_filter1d.
    # radius = (k-1)//2  →  actual kernel = 2*radius+1 (rounded down to odd).
    # sigma = radius/3   →  Gaussian falls to ~1% at the kernel edge.
    _f_radii = tuple((max(int(k), 1) - 1) // 2 for k in filter_tuple)
    _f_sigmas = tuple(max(r / 3.0, 0.1) for r in _f_radii)

    # Separate filter for delta_U_change_grid
    if filter_tuple_deltaU is None:
        filter_tuple_deltaU = filter_tuple
    _f_radii_deltaU = tuple((max(int(k), 1) - 1) // 2 for k in filter_tuple_deltaU)
    _f_sigmas_deltaU = tuple(max(r / 3.0, 0.1) for r in _f_radii_deltaU)

    # Pre-allocate result array and flow boolean (obstacle) grid
    result = np.empty((shape_z, shape_y, shape_x), dtype=np.float32)
    flow_bool_grid = np.zeros((shape_z, shape_y, shape_x), dtype=np.float32)

    # Arrays to store average pressure in overlap regions
    BC_col_arr = np.zeros(1, dtype=np.float32)
    BC_rows = np.zeros(n_x, dtype=np.float32)
    BC_depths = np.zeros((n_y, n_x), dtype=np.float32)


    # --- Support for non-cubic blocks when n_z==1 and n_y==1 ---
    if n_z == 1 and n_y == 1:
        block_size_z = x_array.shape[1]
        block_size_y = x_array.shape[2]
        block_size_x = x_array.shape[3]
        shape_z, shape_y, shape_x = result.shape
        # For this case, the block covers the full z and y, only x is split
        p_i = 0
        p_j = 0
        p_k = shape_x - ((block_size_x - overlap) * (n_x - 1) + block_size_x)

        z_ranges = np.array([[0, block_size_z]])
        y_ranges = np.array([[0, block_size_y]])
        x_idx = np.arange(n_x, dtype=np.int32)
        x_idx_rev = n_x - x_idx - 1
        x_ranges = np.empty((n_x, 2), dtype=np.int32)
        x_ranges[:, 0] = shape_x - block_size_x - x_idx_rev * (block_size_x - overlap)
        x_ranges[:, 1] = shape_x - x_idx_rev * (block_size_x - overlap)
        x_ranges[0, 0] = 0
        x_ranges[0, 1] = block_size_x
    else:
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

        if n_z == 1 and n_y == 1:
            # No BC correction needed for 1-block z/y case
            z0, z1 = z_ranges[0]
            y0, y1 = y_ranges[0]
            x0, x1 = x_ranges[k]
            result[z0:z1, y0:y1, x0:x1] = pred_field
            flow_bool_grid[z0:z1, y0:y1, x0:x1] = flow_bool
        else:
            pred_field = correct_pred_jit(
                pred_field, flow_bool, i, j, k, p_i, p_j, p_k,
                shape, overlap, n_x, n_z,
                BC_col_arr, BC_rows, BC_depths, Ref_BC
            )

            z0, z1 = z_ranges[i]; y0, y1 = y_ranges[j]; x0, x1 = x_ranges[k]

            if i == n_z - 1:
                if j == n_y - 1:
                    result[z0:z1, y0:y1, x0:x1] = pred_field[-p_i:, -p_j:, :]
                    flow_bool_grid[z0:z1, y0:y1, x0:x1] = flow_bool[-p_i:, -p_j:, :]
                else:
                    result[z0:z1, y0:y1, x0:x1] = pred_field[-p_i:, :, :]
                    flow_bool_grid[z0:z1, y0:y1, x0:x1] = flow_bool[-p_i:, :, :]
            else:
                if j == n_y - 1:
                    result[z0:z1, y0:y1, x0:x1] = pred_field[:, -p_j:, :]
                    flow_bool_grid[z0:z1, y0:y1, x0:x1] = flow_bool[:, -p_j:, :]
                else:
                    result[z0:z1, y0:y1, x0:x1] = pred_field
                    flow_bool_grid[z0:z1, y0:y1, x0:x1] = flow_bool


    outlet_has_fluid = np.any(x_array[:, :, :, -1, 3] != 0)

    if outlet_has_fluid:
        correction = plane_combo_mean(result, 3.0, -1, -1.0, -2, 3.0)
    else:
        correction = plane_combo_mean(result, 3.0, -2, -1.0, -3, 3.0)
    sub_scalar_inplace(result, correction)


    if apply_filter:
        result = ndimage.gaussian_filter(result, sigma=filter_tuple)
        #_gaussian_filter3d(result, tmp0, tmp1, _f_sigmas, _f_radii, output=result)

    result_unweighted = result

    if apply_deltaU_change_wgt:
        print(f'delta_U_change_grid.max(): {delta_U_change_grid.max()}, delta_U_change_grid.min(): {delta_U_change_grid.min()}, delta_U_change_grid.mean(): {delta_U_change_grid.mean()}')
        # Use separate filter for delta_U_change_grid
        delta_U_change_grid = ndimage.gaussian_filter(delta_U_change_grid, sigma=filter_tuple_deltaU)
        #_gaussian_filter3d(delta_U_change_grid, tmp0, tmp1, _f_sigmas_deltaU, _f_radii_deltaU, output=delta_U_change_grid)
        print(f'After filter: delta_U_change_grid.max(): {delta_U_change_grid.max()}, delta_U_change_grid.min(): {delta_U_change_grid.min()}, delta_U_change_grid.mean(): {delta_U_change_grid.mean()}')
        delta_U_change_grid = delta_U_change_grid / (np.max(np.abs(delta_U_change_grid)) + 1e-8)  # Normalize to [-1, 1]
        print(f'After normalization: delta_U_change_grid.max(): {delta_U_change_grid.max()}, delta_U_change_grid.min(): {delta_U_change_grid.min()}, delta_U_change_grid.mean(): {delta_U_change_grid.mean()}')

    #    np.subtract(result, delta_delta_p_prev_grid, out=tmp0)
    #    pressure_delta_before_weight = tmp0.copy()
        result = result_unweighted * delta_U_change_grid

    #    if apply_filter:
    #        _gaussian_filter3d(tmp0, tmp0, tmp1, _f_sigmas, _f_radii, output=tmp1)
    #        change_in_deltap = tmp1
    #    else:
    #        change_in_deltap = tmp0

    # Plot all intermediate and final results (optional, disabled by default)
    if plot_results:
        _plot_sm_result(
            assembled_change_in_delta_p=result_unweighted,
            previous_change_in_delta_p=delta_delta_p_prev_grid,
            delta_u_change_weight=delta_U_change_grid,
            change_in_delta_p_before_weight=result_unweighted,
            weighted_change_in_delta_p=result,
            final_pressure_prediction=None,
            obstacle_mask=(flow_bool_grid == 0),
            call_count=plot_call_count,
            output_dir=plot_output_dir,
        )
    
    return result


def _plot_sm_result(
    assembled_change_in_delta_p=None,
    previous_change_in_delta_p=None,
    delta_u_change_weight=None,
    change_in_delta_p_before_weight=None,
    weighted_change_in_delta_p=None,
    final_pressure_prediction=None,
    obstacle_mask=None,
    call_count=1,
    output_dir='SM_inspect',
    requested_slices=[0, 20, 40, 60, 80, 95],
    axis=0,
    save_png=True
):
    """
    Unified plotting function for all assembly fields (change in delta p version).

    Parameters:
    - assembled_change_in_delta_p: Assembled change in delta p field
    - previous_change_in_delta_p: Previous change in delta p field (delta_delta_p_prev_grid)
    - delta_u_change_weight: Weight grid based on delta-U change
    - change_in_delta_p_before_weight: result - delta_delta_p_prev_grid
    - weighted_change_in_delta_p: Final weighted change in delta p
    - final_pressure_prediction: weighted_change_in_delta_p + previous_change_in_delta_p
    - call_count: Call counter for filename
    - output_dir: Output directory for saving
    - n_slices: Number of slices for additional delta_u_change_weight analysis plot (default 5)
    - axis: Axis to slice along (0=z, 1=y, 2=x; default 0)
    - save_png: Whether to save as PNG (True) or display (False)
    """
    # Import matplotlib only when plotting is needed
    import matplotlib.pyplot as plt


    def _apply_mask(field, mask):
        """Apply boolean mask: set field to NaN where mask is True (obstacle)."""
        if field is None or mask is None:
            return field
        masked = np.array(field, copy=True)
        masked[mask] = np.nan
        return masked

    def _safe_vmax(arr, fallback=1e-8):
        """Return nanmax(|arr|), falling back to `fallback` when result is zero, NaN, or Inf."""
        v = float(np.nanmax(np.abs(arr)))
        return v if np.isfinite(v) and v > 0.0 else fallback

    def _slice_indices_for_axis(axis_len, requested_slices):
        # If a list/tuple/array is provided, use it directly (as percent of axis_len if all are numbers and <=100)
        if isinstance(requested_slices, (list, tuple, np.ndarray)):
            # Only treat as percent if all elements are numbers
            if all(isinstance(x, (int, float)) for x in requested_slices):
                if all(x <= 100 for x in requested_slices):
                    return [int((x / 100) * axis_len) for x in requested_slices]
                else:
                    return [int(x) for x in requested_slices]
            else:
                raise ValueError("All elements in slices_indices must be int or float.")
        # Otherwise, fall back to old behavior
        n_req = max(1, int(requested_slices))
        if axis_len <= 1 or n_req == 1:
            return [axis_len // 2]
        idx = np.linspace(0, axis_len - 1, num=n_req, dtype=np.int32)
        return list(np.unique(idx))

    fields_available = sum([
        assembled_change_in_delta_p is not None,
        previous_change_in_delta_p is not None,
        delta_u_change_weight is not None,
        change_in_delta_p_before_weight is not None,
        weighted_change_in_delta_p is not None,
        final_pressure_prediction is not None,
    ])
    if fields_available == 0:
        print("[plot_sm_result] No fields to plot")
        return

    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(60, 12))
    axes = axes.flatten()

    ref_field = None
    for candidate in (
        assembled_change_in_delta_p,
        previous_change_in_delta_p,
        delta_u_change_weight,
        change_in_delta_p_before_weight,
        weighted_change_in_delta_p,
        final_pressure_prediction,
    ):
        if candidate is not None:
            ref_field = candidate
            break

    shape_along_axis = [ref_field.shape[0], ref_field.shape[1], ref_field.shape[2]][axis]

    # Use the center slice along the selected axis.
    slice_idx = 3

    axis_names = ["z", "y", "x"]
    axis_name = axis_names[axis]
    axis_labels = [["x", "y"], ["x", "z"], ["y", "z"]][axis]

    # Share one color scale only across assembled, previous, and final pressure fields.
    shared_pressure_slices = []
    for candidate in (assembled_change_in_delta_p, previous_change_in_delta_p, final_pressure_prediction):
        if candidate is not None:
            shared_pressure_slices.append(_extract_slice(candidate, axis, slice_idx))

    shared_pressure_vmax = None
    if shared_pressure_slices:
        shared_pressure_vmax = max(_safe_vmax(slice_data) for slice_data in shared_pressure_slices)
    shared_pressure_cmap = "RdBu_r"

    plot_idx = 0

    # Prepare masked fields for plotting (like plot_delta_p_comparison_slices)
    masked_assembled_change_in_delta_p = _apply_mask(assembled_change_in_delta_p, obstacle_mask)
    masked_previous_change_in_delta_p = _apply_mask(previous_change_in_delta_p, obstacle_mask)
    masked_delta_u_change_weight = _apply_mask(delta_u_change_weight, obstacle_mask)
    masked_change_in_delta_p_before_weight = _apply_mask(change_in_delta_p_before_weight, obstacle_mask)
    masked_weighted_change_in_delta_p = _apply_mask(weighted_change_in_delta_p, obstacle_mask)
    masked_final_pressure_prediction = _apply_mask(final_pressure_prediction, obstacle_mask)

    if masked_assembled_change_in_delta_p is not None:
        slice_data = _extract_slice(masked_assembled_change_in_delta_p, axis, slice_idx)
        vmax = shared_pressure_vmax if shared_pressure_vmax is not None else np.nanmax(np.abs(slice_data))
        im = axes[plot_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
        axes[plot_idx].set_title(f"Assembled change in delta p - {axis_name}-slice {slice_idx}")
        axes[plot_idx].set_xlabel(axis_labels[0])
        axes[plot_idx].set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    if masked_previous_change_in_delta_p is not None:
        slice_data = _extract_slice(masked_previous_change_in_delta_p, axis, slice_idx)
        vmax = shared_pressure_vmax if shared_pressure_vmax is not None else np.nanmax(np.abs(slice_data))
        im = axes[plot_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
        axes[plot_idx].set_title(f"Previous change in delta p - {axis_name}-slice {slice_idx}")
        axes[plot_idx].set_xlabel(axis_labels[0])
        axes[plot_idx].set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    if masked_delta_u_change_weight is not None:
        slice_data = _extract_slice(masked_delta_u_change_weight, axis, slice_idx)
        im = axes[plot_idx].imshow(slice_data, origin="lower", aspect="auto", cmap="plasma")
        axes[plot_idx].set_title(f"Delta-U change weight - {axis_name}-slice {slice_idx}")
        axes[plot_idx].set_xlabel(axis_labels[0])
        axes[plot_idx].set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    if masked_change_in_delta_p_before_weight is not None:
        slice_data = _extract_slice(masked_change_in_delta_p_before_weight, axis, slice_idx)
        vmax = _safe_vmax(slice_data)
        im = axes[plot_idx].imshow(slice_data, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[plot_idx].set_title(f"Change in delta p before weight - {axis_name}-slice {slice_idx}")
        axes[plot_idx].set_xlabel(axis_labels[0])
        axes[plot_idx].set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    if masked_weighted_change_in_delta_p is not None:
        slice_data = _extract_slice(masked_weighted_change_in_delta_p, axis, slice_idx)
        vmax = _safe_vmax(slice_data)
        im = axes[plot_idx].imshow(slice_data, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[plot_idx].set_title(f"Weighted change in delta p - {axis_name}-slice {slice_idx}")
        axes[plot_idx].set_xlabel(axis_labels[0])
        axes[plot_idx].set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    if masked_final_pressure_prediction is not None:
        slice_data = _extract_slice(masked_final_pressure_prediction, axis, slice_idx)
        vmax = shared_pressure_vmax if shared_pressure_vmax is not None else np.nanmax(np.abs(slice_data))
        im = axes[plot_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
        axes[plot_idx].set_title(f"Final pressure prediction (change in delta p + prev) - {axis_name}-slice {slice_idx}")
        axes[plot_idx].set_xlabel(axis_labels[0])
        axes[plot_idx].set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Assembly Results - Call {call_count}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_png:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"assembly_result_{call_count:05d}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot_sm_result] Saved: {fname}")
    else:
        plt.show()

    # Additional diagnostic figure: multiple slices for delta_u_change_weight.
    if delta_u_change_weight is not None and len(requested_slices) > 0:
        axis_len = [
            delta_u_change_weight.shape[0],
            delta_u_change_weight.shape[1],
            delta_u_change_weight.shape[2],
        ][axis]
        slice_indices = _slice_indices_for_axis(axis_len, requested_slices)
        n_panels = len(slice_indices)
        fig_height = 2.5 * n_panels
        fig_width = 12
        fig2, axes2 = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height), squeeze=False)
        axes2 = axes2.flatten()

        for panel_idx, s_idx in enumerate(slice_indices):
            slice_data = _extract_slice(delta_u_change_weight, axis, s_idx)
            im = axes2[panel_idx].imshow(slice_data, origin="lower", aspect="auto", cmap="plasma")
            axes2[panel_idx].set_title(f"Delta-U change weight - {axis_name}-slice {s_idx}", fontsize=18, fontweight='bold')
            axes2[panel_idx].set_xlabel(axis_labels[0])
            axes2[panel_idx].set_ylabel(axis_labels[1])
            plt.colorbar(im, ax=axes2[panel_idx])
            axes2[panel_idx].axis("off")

        plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
        fig2.suptitle(f"Delta-U Change Weight Multi-Slice Analysis - Call {call_count}", fontsize=22, fontweight="bold")

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            fname2 = os.path.join(output_dir, f"delta_u_change_weight_slices_{call_count:05d}.png")
            plt.savefig(fname2, dpi=100, bbox_inches="tight")
            plt.close(fig2)
            print(f"[plot_sm_result] Saved: {fname2}")
        else:
            plt.show()

    # Additional diagnostic figure: multiple z-slices for assembled pressure, masked by obstacle_mask
    if assembled_change_in_delta_p is not None and len(requested_slices) > 0:
        z_len = assembled_change_in_delta_p.shape[0]
        z_slice_indices = _slice_indices_for_axis(z_len, requested_slices)
        n_panels = len(z_slice_indices)
        fig_height = 2.5 * n_panels
        fig_width = 12
        fig3, axes3 = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height), squeeze=False)
        axes3 = axes3.flatten()

        # Prepare mask for z-slices
        mask = obstacle_mask if obstacle_mask is not None else None

        for panel_idx, z_idx in enumerate(z_slice_indices):
            slice_data = assembled_change_in_delta_p[z_idx, :, :]
            if mask is not None:
                slice_mask = mask[z_idx, :, :]
                slice_data = np.array(slice_data, copy=True)
                slice_data[slice_mask] = np.nan

            vmax = _safe_vmax(slice_data)
            im = axes3[panel_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
            axes3[panel_idx].set_title(f"Assembled change in delta p - z-slice {z_idx}", fontsize=18, fontweight='bold')
            axes3[panel_idx].set_xlabel("x")
            axes3[panel_idx].set_ylabel("y")
            plt.colorbar(im, ax=axes3[panel_idx])
            axes3[panel_idx].axis("off")

        plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
        fig3.suptitle(f"Assembled Change in Delta p Multi-Z-Slice Analysis - Call {call_count}", fontsize=22, fontweight="bold")

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            fname3 = os.path.join(output_dir, f"assembled_change_in_delta_p_slices_{call_count:05d}.png")
            plt.savefig(fname3, dpi=100, bbox_inches="tight")
            plt.close(fig3)
            print(f"[plot_sm_result] Saved: {fname3}")
        else:
            plt.show()

    # Additional diagnostic figure: multiple z-slices for previous change in delta p
    if previous_change_in_delta_p is not None and len(requested_slices) > 0:
        z_len = previous_change_in_delta_p.shape[0]
        z_slice_indices = _slice_indices_for_axis(z_len, requested_slices)
        n_panels = len(z_slice_indices)
        fig_height = 2.5 * n_panels
        fig_width = 12
        fig4, axes4 = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height), squeeze=False)
        axes4 = axes4.flatten()

        mask = obstacle_mask if obstacle_mask is not None else None

        for panel_idx, z_idx in enumerate(z_slice_indices):
            slice_data = previous_change_in_delta_p[z_idx, :, :]
            if mask is not None:
                slice_mask = mask[z_idx, :, :]
                slice_data = np.array(slice_data, copy=True)
                slice_data[slice_mask] = np.nan

            vmax = _safe_vmax(slice_data)
            im = axes4[panel_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
            axes4[panel_idx].set_title(f"Previous change in delta p - z-slice {z_idx}", fontsize=18, fontweight='bold')
            axes4[panel_idx].set_xlabel("x")
            axes4[panel_idx].set_ylabel("y")
            plt.colorbar(im, ax=axes4[panel_idx])
            axes4[panel_idx].axis("off")

        plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
        fig4.suptitle(f"Previous Change in Delta p Multi-Z-Slice Analysis - Call {call_count}", fontsize=22, fontweight="bold")

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            fname4 = os.path.join(output_dir, f"previous_change_in_delta_p_slices_{call_count:05d}.png")
            plt.savefig(fname4, dpi=100, bbox_inches="tight")
            plt.close(fig4)
            print(f"[plot_sm_result] Saved: {fname4}")
        else:
            plt.show()

    # Additional diagnostic figure: multiple z-slices for change in delta p before weight
    if change_in_delta_p_before_weight is not None and len(requested_slices) > 0:
        z_len = change_in_delta_p_before_weight.shape[0]
        z_slice_indices = _slice_indices_for_axis(z_len, requested_slices)
        n_panels = len(z_slice_indices)
        fig_height = 2.5 * n_panels
        fig_width = 12
        fig5, axes5 = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height), squeeze=False)
        axes5 = axes5.flatten()

        mask = obstacle_mask if obstacle_mask is not None else None

        for panel_idx, z_idx in enumerate(z_slice_indices):
            slice_data = change_in_delta_p_before_weight[z_idx, :, :]
            if mask is not None:
                slice_mask = mask[z_idx, :, :]
                slice_data = np.array(slice_data, copy=True)
                slice_data[slice_mask] = np.nan

            vmax = _safe_vmax(slice_data)
            im = axes5[panel_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
            axes5[panel_idx].set_title(f"Change in delta p before weight - z-slice {z_idx}", fontsize=18, fontweight='bold')
            axes5[panel_idx].set_xlabel("x")
            axes5[panel_idx].set_ylabel("y")
            plt.colorbar(im, ax=axes5[panel_idx])
            axes5[panel_idx].axis("off")

        plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
        fig5.suptitle(f"Change in Delta p Before Weight Multi-Z-Slice Analysis - Call {call_count}", fontsize=22, fontweight="bold")

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            fname5 = os.path.join(output_dir, f"change_in_delta_p_before_weight_slices_{call_count:05d}.png")
            plt.savefig(fname5, dpi=100, bbox_inches="tight")
            plt.close(fig5)
            print(f"[plot_sm_result] Saved: {fname5}")
        else:
            plt.show()

    # Additional diagnostic figure: multiple z-slices for weighted change in delta p
    if weighted_change_in_delta_p is not None and len(requested_slices) > 0:
        z_len = weighted_change_in_delta_p.shape[0]
        z_slice_indices = _slice_indices_for_axis(z_len, requested_slices)
        n_panels = len(z_slice_indices)
        fig_height = 2.5 * n_panels
        fig_width = 12
        fig6, axes6 = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height), squeeze=False)
        axes6 = axes6.flatten()

        mask = obstacle_mask if obstacle_mask is not None else None

        for panel_idx, z_idx in enumerate(z_slice_indices):
            slice_data = weighted_change_in_delta_p[z_idx, :, :]
            if mask is not None:
                slice_mask = mask[z_idx, :, :]
                slice_data = np.array(slice_data, copy=True)
                slice_data[slice_mask] = np.nan

            vmax = _safe_vmax(slice_data)
            im = axes6[panel_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
            axes6[panel_idx].set_title(f"Weighted change in delta p - z-slice {z_idx}", fontsize=18, fontweight='bold')
            axes6[panel_idx].set_xlabel("x")
            axes6[panel_idx].set_ylabel("y")
            plt.colorbar(im, ax=axes6[panel_idx])
            axes6[panel_idx].axis("off")

        plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
        fig6.suptitle(f"Weighted Change in Delta p Multi-Z-Slice Analysis - Call {call_count}", fontsize=22, fontweight="bold")

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            fname6 = os.path.join(output_dir, f"weighted_change_in_delta_p_slices_{call_count:05d}.png")
            plt.savefig(fname6, dpi=100, bbox_inches="tight")
            plt.close(fig6)
            print(f"[plot_sm_result] Saved: {fname6}")
        else:
            plt.show()

    # Additional diagnostic figure: multiple z-slices for final pressure prediction
    if final_pressure_prediction is not None and len(requested_slices) > 0:
        z_len = final_pressure_prediction.shape[0]
        z_slice_indices = _slice_indices_for_axis(z_len, requested_slices)
        n_panels = len(z_slice_indices)
        fig_height = 2.5 * n_panels
        fig_width = 12
        fig7, axes7 = plt.subplots(n_panels, 1, figsize=(fig_width, fig_height), squeeze=False)
        axes7 = axes7.flatten()

        mask = obstacle_mask if obstacle_mask is not None else None

        for panel_idx, z_idx in enumerate(z_slice_indices):
            slice_data = final_pressure_prediction[z_idx, :, :]
            if mask is not None:
                slice_mask = mask[z_idx, :, :]
                slice_data = np.array(slice_data, copy=True)
                slice_data[slice_mask] = np.nan

            vmax = _safe_vmax(slice_data)
            im = axes7[panel_idx].imshow(slice_data, origin="lower", aspect="auto", cmap=shared_pressure_cmap, vmin=-vmax, vmax=vmax)
            axes7[panel_idx].set_title(f"Final pressure prediction - z-slice {z_idx}", fontsize=18, fontweight='bold')
            axes7[panel_idx].set_xlabel("x")
            axes7[panel_idx].set_ylabel("y")
            plt.colorbar(im, ax=axes7[panel_idx])
            axes7[panel_idx].axis("off")

        plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.15, hspace=0.25)
        fig7.suptitle(f"Final Pressure Prediction Multi-Z-Slice Analysis - Call {call_count}", fontsize=22, fontweight="bold")

        if save_png:
            os.makedirs(output_dir, exist_ok=True)
            fname7 = os.path.join(output_dir, f"final_pressure_prediction_slices_{call_count:05d}.png")
            plt.savefig(fname7, dpi=100, bbox_inches="tight")
            plt.close(fig7)
            print(f"[plot_sm_result] Saved: {fname7}")
        else:
            plt.show()


def _extract_slice(data_3d, axis, slice_idx):
    """Extract 2D slice from 3D data along specified axis."""
    if axis == 0:
        return data_3d[slice_idx, :, :]
    if axis == 1:
        return data_3d[:, slice_idx, :]
    return data_3d[:, :, slice_idx]
