import numpy as np
from scipy import ndimage

#### ASSEMBLE ALGORITH

def correct_pred(field_block, bool_block, i, j, k, p_i, p_j, p_k, shape, overlap, n_x, n_z, BC_col, BC_rows, BC_depths, Ref_BC):
    """
    Standalone version of _correct_pred for block correction.

    Args:
        field_block (ndarray): Block of field values.
        bool_block (ndarray): Boolean mask for the block.
        i, j, k (int): Block indices (depth, row, column).
        p_i, p_j, p_k (int): Index offsets for block placement.
        shape (int): Block shape.
        overlap (int): Overlap size.
        n_x (int): Number of blocks in x-direction.
        n_z (int): Number of blocks in z-direction.
        BC_col, BC_rows, BC_depths: Boundary condition arrays (can be None for stateless use).
        Ref_BC: Reference boundary condition (can be None for stateless use).

    Returns:
        ndarray: Corrected field block.
    """

    # i - depth index
    # j - row index
    # k - column index

    intersect_zone_limit_i = (-p_i-overlap, -p_i)
    intersect_zone_limit_j = (-p_j-overlap, -p_j)
    intersect_zone_limit_k = overlap - p_k

    # left_most_k = len(BC_rows) - 1
    down_most_j = BC_depths.shape[0] - 1

    # Case 1 - 1st correction - based on the outlet fixed pressure boundary condition (Ref_BC)
    if (i, j, k) == (0, 0, n_x-1):
        # check value at outlet BC
        if ~(bool_block[:, :, -1] == 0).all():
            BC_corr = np.mean(field_block[:, :, -1][bool_block[:, :, -1] != 0]) - Ref_BC
        else:
            BC_corr = np.mean(field_block[:, :, -2][bool_block[:, :, -2] != 0]) - Ref_BC

        field_block -= BC_corr
        BC_col = np.mean(field_block[:, :, :overlap][bool_block[:, :, :overlap] != 0])
        BC_rows[k] = np.mean(field_block[:, -overlap:, :][bool_block[:, -overlap:, :] != 0])
        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 2 - 1st depth and 1st row - correct from the left
    elif (i, j) == (0, 0):
        # Case 2 a)
        if k > 0:
            BC_corr = np.mean(field_block[:, :, -overlap:][bool_block[:, :, -overlap:] != 0]) - BC_col
            field_block -= BC_corr
            # left-most column
            if k == 0:
                BC_col = np.mean(field_block[:, :, :intersect_zone_limit_k][bool_block[:, :, :intersect_zone_limit_k] != 0])
            else:
                BC_col = np.mean(field_block[:, :, :overlap][bool_block[:, :, :overlap] != 0])
        # Case 2 b) - Left-most block
        else:
            BC_corr = np.mean(field_block[:, :, -intersect_zone_limit_k:][bool_block[:, :, -intersect_zone_limit_k:] != 0]) - BC_col
            field_block -= BC_corr

        BC_rows[k] = np.mean(field_block[:, -overlap:, :][bool_block[:, -overlap:, :] != 0])
        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 3 - 1st depth (non 1st row and column)
    # Correction based on the
    elif i == 0:
        # Case 3 a)
        if j < down_most_j:
            BC_corr = np.mean(field_block[:, :overlap, :][bool_block[:, :overlap, :] != 0]) - BC_rows[k]
            field_block -= BC_corr

            # Value stored to be used in the last row depends on p_i
            if j == down_most_j - 1:
                BC_rows[k] = np.mean(field_block[:, -(shape-p_j):, :][bool_block[:, -(shape-p_j):, :] != 0])
            else:
                BC_rows[k] = np.mean(field_block[:, -overlap:, :][bool_block[:, -overlap:, :] != 0])
        # Case 3 b) - Last Row
        else:
            # j_0 = #intersect_zone_limit_j[0]
            # j_f = #intersect_zone_limit_j[1]
            BC_corr = np.mean(field_block[:, :-p_j, :][bool_block[:, :-p_j, :] != 0]) - BC_rows[k]
            field_block -= BC_corr

        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 4 - non 1st depth (any row and column)
    # Correcting based on depth overlap -> correcting (i, j, k) = (i, 0, 0) for the pressure BC could improve respecting the outlet BC
    elif i < n_z - 1:
        BC_corr = np.mean(field_block[:overlap, :, :][bool_block[:overlap, :, :] != 0]) - BC_depths[j, k]
        field_block -= BC_corr
        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 5 - last depth
    else:
        i_0 = intersect_zone_limit_j[0]
        i_f = intersect_zone_limit_j[1]
        BC_corr = np.mean(field_block[i_0:i_f, :, :][bool_block[i_0:i_f, :, :] != 0]) - BC_depths[j, k]
        field_block -= BC_corr

    # # DEBUGGING print
    # print(f"(i,j,k): {(i,j,k)}")
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
    In the first row the correction is based on the outlet fixed value BC.

    In the following rows the correction is based on the overlap region at the top of each new block.
    This correction from the top ensures better agreement between different rows, leading to overall better results.

    Args:
        array (ndarray): The array containing the predicted flow fields for each block.
        indices_list (list): The list of indices representing the position of each block in the flow domain.
        n_x (int): The number of blocks in the x-direction.
        n_y (int): The number of blocks in the y-direction.
        n_z (int): The number of blocks in the z-direction.
        overlap (int): Overlap size between blocks.
        shape (int): Size of each block.
        Ref_BC: Reference boundary condition (not used directly here).
        x_array (ndarray): Array with block information.
        apply_filter (bool): Whether to apply a Gaussian filter.
        shape_x (int): Domain size in x.
        shape_y (int): Domain size in y.
        shape_z (int): Domain size in z.
        deltaU_change_grid (ndarray): Grid for deltaU change weighting.
        deltaP_prev_grid (ndarray): Previous deltaP grid.
        apply_deltaU_change_wgt (bool): Whether to apply deltaU change weighting.

    Returns:
        tuple: (reconstructed domain, change_in_deltap if requested)
    """

    result_array = np.empty(shape=(shape_z, shape_y, shape_x))

    # Arrays to store average pressure in overlap regions
    # BC_col - correction between side by side blocks
    # BC_rows - correction between top and down blocks
    # BC_depth - correction between blocks in the depth direction
    BC_col = 0.0
    BC_rows = np.zeros(n_x)
    BC_depths = np.zeros((n_y, n_x))

    print(f'Shape: {(shape_z, shape_y, shape_x)}')

    # i index where the lower blocks are located
    p_i = shape_z - ((shape - overlap) * (n_z - 2) + shape)
    # j index where the left-most blocks are located
    p_j = shape_y - ((shape - overlap) * (n_y - 2) + shape)
    # k index where the left-most blocks are located
    p_k = shape_x - ((shape - overlap) * (n_x - 1) + shape)

    result = result_array

    # Loop over all the blocks and apply corrections to ensure consistency between overlapping blocks
    for i_block in range(x_array.shape[0]):
        i, j, k = indices_list[i_block]
        flow_bool = x_array[i_block, :, :, :, 3]
        pred_field = array[i_block, ...]

        # Applying the correction
        pred_field = correct_pred(
            pred_field, flow_bool, i, j, k, p_i, p_j, p_k,
            shape=shape, overlap=overlap, n_x=n_x, n_z=n_z,
            BC_col=BC_col, BC_rows=BC_rows, BC_depths=BC_depths, Ref_BC=Ref_BC
        )

        # Last reassembly step:
        # Assigning the block to the right location in the flow domain

        # # DEBUGGING print
        # print(f"(i,j,k): {(i,j,k)}")

        # Non last depth
        if i < n_z - 1:
            # Last row, first column (right-most)
            if (j, k) == (n_y - 1, 0):
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape, shape_y - p_j, shape_y, 0, shape)
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       -p_j:shape_y, :shape] = pred_field[:, -p_j:, :]
            # Last column (left-most)
            elif k == 0:
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape,
                #       (shape - overlap) * j, (shape - overlap) * j + shape,
                #       0, shape)
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       0:shape] = pred_field
            # Last row
            elif j == (n_y - 1):
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape,
                #       shape_y - p_j, shape_y, shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       -p_j:,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field[:, -p_j:, :]
            else:
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape,
                #       (shape - overlap) * j, (shape - overlap) * j + shape,
                #       shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field
        # Last depth
        else:
            # Last row, first column (right-most)
            if (j, k) == (n_y - 1, 0):
                # # DEBUGGING print
                # print((shape_z - p_i, shape_z), (shape_y - p_j, shape_y), 0, shape)
                result[-p_i:,
                       -p_j:, :shape] = pred_field[-p_i:, -p_j:, :]
            # Last column (left-most)
            elif k == 0:
                # # DEBUGGING print
                # print(shape_z - p_i, shape_z, shape_y - p_j, shape_y, 0, shape)
                result[-p_i:,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       0:shape] = pred_field[-p_i:, :, :]
            # Last row
            elif j == (n_y - 1):
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print(shape_z - p_i, shape_z, shape_y - p_j, shape_y, shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[-p_i:,
                       -p_j:,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field[-p_i:, -p_j:, :]
            else:
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print(shape_z - p_i, shape_z, (shape - overlap) * j, (shape - overlap) * j + shape, shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[-p_i:,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field[-p_i:, :, :]

        # # DEBUGGING: plot slices
        # if i==0: # and j==3:
        #     fig, axs = plt.subplots(7, 1, figsize=(15, 5))
        #     axs[0].imshow(result[(shape-overlap) * i + 1, :, :])
        #     axs[1].imshow(result[(shape-overlap) * i + 3, :,:])
        #     axs[2].imshow(result[(shape-overlap) * i + 5, :,:])
        #     axs[3].imshow(result[(shape-overlap) * i + 7, :,:])
        #     axs[4].imshow(result[(shape-overlap) * i + 9,:,:])
        #     axs[5].imshow(result[(shape-overlap) * i + 11,:,:])
        #     axs[6].imshow(result[(shape-overlap) * i + 13,:,:])
        #     for ax in axs:
        #         plt.colorbar(ax.images[0], ax=ax)
        #     plt.savefig(f"reconstruct/reconstructed_{i_block}.png")
        #     plt.close(fig)

    # Correction based on the fact the BC is applied at the last cell center and not the cell face...
    if ~(flow_bool[:, :, -1] == 0).all():
        result -= np.mean(3 * result[:, :, -1] - result[:, :, -2]) / 3
    else:
        result -= np.mean(3 * result[:, :, -2] - result[:, :, -3]) / 3

    ################### this applies a gaussian filter to remove boundary artifacts #################
    filter_tuple = (10, 10, 10)
    if apply_filter:
        result = ndimage.gaussian_filter(result, sigma=filter_tuple, order=0)

    change_in_deltap = None
    if apply_deltaU_change_wgt:
        deltaU_change_grid = ndimage.gaussian_filter(deltaU_change_grid, sigma=(5, 5, 5), order=0)
        change_in_deltap = result - deltaP_prev_grid
        change_in_deltap = change_in_deltap * deltaU_change_grid
        change_in_deltap = ndimage.gaussian_filter(change_in_deltap, sigma=filter_tuple, order=0)

    return result, change_in_deltap