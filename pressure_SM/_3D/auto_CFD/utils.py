import numpy as np
import os

# --- Find and read new field files ---
def read_and_delete_bin_file(filename, n_cells, dtype=np.float32):
    """Read a binary field and delete it
    return as numpy array"""

    data = np.fromfile(filename, dtype=dtype, count=n_cells)
    os.remove(filename)
    return data
    
def read_component_bin_files(n_sample_frames, n_cells):
    """
    Reads delta_Ux_*, delta_Uy_*, delta_Uz_*, delta_p_rgh_* for the given sample indices.
    Returns stacked arrays: delta_U_grid (N, n_cells, 3), delta_p_grid (N, n_cells)
    """
    delta_U_list = []
    delta_p_list = []
    for idx in range(n_sample_frames):
        ux = read_and_delete_bin_file(f'delta_Ux_{idx}.bin', n_cells)
        uy = read_and_delete_bin_file(f'delta_Uy_{idx}.bin', n_cells)
        uz = read_and_delete_bin_file(f'delta_Uz_{idx}.bin', n_cells)
        p  = read_and_delete_bin_file(f'delta_p_rgh_{idx}.bin', n_cells)
        delta_U = np.stack([ux, uy, uz], axis=-1)  # shape: (n_cells, 3)
        delta_U_list.append(delta_U)
        delta_p_list.append(p)
    delta_U_grid = np.stack(delta_U_list)  # (N, n_cells, 3)
    delta_p_grid = np.stack(delta_p_list)  # (N, n_cells)
    return delta_U_grid, delta_p_grid