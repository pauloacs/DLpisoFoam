"""
Data loader for HDF5 format output from DLbuoyantPimpleFoam solver.
Reads coordinates, boundary data, and field samples.
"""

import h5py
import numpy as np
import os
from pathlib import Path


def load_hdf5_samples(data_file='ML_data/data.h5'):
    """
    Load all samples from master HDF5 file.
    
    Returns:
        coordinates: (n_cells, 3) array of cell center coordinates
        boundary_coords: (n_boundary_faces, 3) array of boundary face centers (concatenated)
        boundary_patches: (n_boundary_faces,) array of patch indices
        patch_names: dict mapping patch index to patch name
        U: (n_samples, n_cells, 3) array of velocities
        delta_delta_U: (n_samples, n_cells, 3) array of velocity double-increments
        delta_delta_U_diff: (n_samples, n_cells, 3) array of velocity time increments
        delta_delta_p: (n_samples, n_cells) array of pressure double-increments
        delta_p_prev: (n_samples, n_cells) array of previous pressure increments
        timestamps: (n_samples,) array of timesteps
        u_max_norm_arr: (n_samples,) array of velocity normalizations
    """
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"HDF5 data file not found: {data_file}")
    
    coordinates = None
    boundary_coords = None
    boundary_patches = None
    patch_names = {}
    delta_delta_U_list = []
    delta_delta_U_diff_list = []
    delta_delta_p_list = []
    delta_p_prev_list = []
    U_list = []
    timestamps = []
    u_max_norm_list = []
    
    with h5py.File(data_file, 'r') as f:
        # Load cell center coordinates
        if '/coordinates' in f:
            coordinates = f['/coordinates'][:]  # (n_cells, 3)
        else:
            raise ValueError("Coordinates dataset '/coordinates' not found in HDF5 file")

        # Load boundary face coordinates and patch indices
        if '/boundary_coordinates' in f:
            boundary_coords = f['/boundary_coordinates'][:]  # (n_boundary_faces, 3)
        if '/boundary_patches' in f:
            boundary_patches = f['/boundary_patches'][:]  # (n_boundary_faces,)

        # Load patch names from attributes
        for attr_name in f.attrs.keys():
            if attr_name.startswith('patch_'):
                patch_idx = int(attr_name.split('_')[1])
                patch_names[patch_idx] = f.attrs[attr_name]
                if isinstance(patch_names[patch_idx], bytes):
                    patch_names[patch_idx] = patch_names[patch_idx].decode('utf-8')

        # Filter out boundary faces with patch name 'outlet' or 'inlet'
        if boundary_coords is not None and boundary_patches is not None and patch_names:
            keep_mask = np.array([
                patch_names[idx] != 'outlet' and patch_names[idx] != 'inlet'
                for idx in boundary_patches
            ])
            boundary_coords = boundary_coords[keep_mask]
            boundary_patches = boundary_patches[keep_mask]

        n_cells = coordinates.shape[0]

        # Iterate over all sample groups
        sample_keys = sorted([key for key in f.keys() if key.startswith('sample_')])

        if len(sample_keys) == 0:
            raise ValueError("No sample groups (sample_*) found in HDF5 file")

        for sample_key in sample_keys:
            group = f[sample_key]

            # Load delta-delta velocity and pressure increments
            if 'delta_delta_U' not in group or 'pressure_increment' not in group:
                print(f"Warning: {sample_key} missing delta_delta_U or pressure_increment dataset, skipping")
                continue

            delta_delta_u = group['delta_delta_U'][:]     # (n_cells, 3)
            delta_delta_u_diff = group['delta_delta_U_diff'][:]  # (n_cells, 3)
            delta_delta_pressure = group['pressure_increment'][:]     # (n_cells,)
            
            # Load pressure_increment_prev if present
            if 'pressure_increment_prev' in group:
                delta_p_prev = group['pressure_increment_prev'][:]
            else:
                delta_p_prev = np.zeros(n_cells)  # placeholder if not available

            # Load U (velocity) if present
            if 'U' in group:
                U_list.append(group['U'][:])
            else:
                U_list.append(np.zeros((n_cells, 3)))  # placeholder if not available

            # Load timestep metadata
            timestep = group.attrs.get('timestep', -1)

            # Load U_MAX_NORM if present
            if 'U_MAX_NORM' in group:
                u_max = group['U_MAX_NORM'][()]
                if isinstance(u_max, np.ndarray):
                    u_max = float(u_max)
                u_max_norm_list.append(u_max)
            else:
                u_max_norm_list.append(1.0)

            delta_delta_U_list.append(delta_delta_u)
            delta_delta_U_diff_list.append(delta_delta_u_diff)
            delta_delta_p_list.append(delta_delta_pressure)
            delta_p_prev_list.append(delta_p_prev)
            timestamps.append(timestep)
    
    # Stack into arrays
    delta_delta_U = np.array(delta_delta_U_list)  # (n_samples, n_cells, 3)
    delta_delta_U_diff = np.array(delta_delta_U_diff_list)  # (n_samples, n_cells, 3)
    delta_delta_p = np.array(delta_delta_p_list)  # (n_samples, n_cells)
    delta_p_prev = np.array(delta_p_prev_list)  # (n_samples, n_cells)
    U = np.array(U_list)  # (n_samples, n_cells, 3)
    timestamps = np.array(timestamps)
    
    u_max_norm_arr = np.array(u_max_norm_list)
    return coordinates, boundary_coords, boundary_patches, patch_names, U, delta_delta_U, delta_delta_U_diff, delta_p_prev, delta_delta_p, timestamps, u_max_norm_arr


def load_hdf5_field_data(data_file='ML_data/data.h5'):
    """
    Load only field samples from master HDF5 file.
    Cell centers and boundary coordinates are NOT loaded — they were already
    saved to disk during train_init.py and do not need to be re-read on updates.

    Returns:
        delta_delta_U: (n_samples, n_cells, 3) array of velocity double-increments
        delta_delta_U_diff: (n_samples, n_cells, 3) array of velocity time increments
        delta_delta_p: (n_samples, n_cells) array of pressure double-increments
        delta_p_prev: (n_samples, n_cells) array of previous pressure increments
        U: (n_samples, n_cells, 3) array of velocities
        timestamps: (n_samples,) array of timesteps
        u_max_norm_arr: (n_samples,) array of velocity normalizations
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"HDF5 data file not found: {data_file}")

    delta_delta_U_list = []
    delta_delta_U_diff_list = []
    delta_delta_p_list = []
    delta_p_prev_list = []
    U_list = []
    timestamps = []
    u_max_norm_list = []

    with h5py.File(data_file, 'r') as f:
        sample_keys = sorted([key for key in f.keys() if key.startswith('sample_')])

        if len(sample_keys) == 0:
            raise ValueError("No sample groups (sample_*) found in HDF5 file")

        for sample_key in sample_keys:
            group = f[sample_key]
            if 'delta_delta_U' not in group or 'pressure_increment' not in group:
                print(f"Warning: {sample_key} missing delta_delta_U or pressure_increment dataset, skipping")
                continue
            delta_delta_U_list.append(group['delta_delta_U'][:])
            if 'delta_delta_U_diff' in group:
                delta_delta_U_diff_list.append(group['delta_delta_U_diff'][:])
            else:
                delta_delta_U_diff_list.append(np.zeros_like(group['delta_delta_U'][:]))
            delta_delta_p_list.append(group['pressure_increment'][:])
            
            # Load delta_p_prev if present
            if 'pressure_increment_prev' in group:
                delta_p_prev_list.append(group['pressure_increment_prev'][:])
            else:
                delta_p_prev_list.append(np.zeros_like(group['pressure_increment'][:]))
            
            if 'U' in group:
                U_list.append(group['U'][:])
            else:
                U_list.append(np.zeros_like(group['delta_delta_U'][:]))
            timestamps.append(group.attrs.get('timestep', -1))
            if 'U_MAX_NORM' in group:
                u_max = group['U_MAX_NORM'][()]
                u_max_norm_list.append(float(u_max) if not isinstance(u_max, np.ndarray) else float(u_max))
            else:
                u_max_norm_list.append(1.0)

    delta_delta_U = np.array(delta_delta_U_list)
    delta_delta_U_diff = np.array(delta_delta_U_diff_list)
    delta_delta_p = np.array(delta_delta_p_list)
    delta_p_prev = np.array(delta_p_prev_list)
    U = np.array(U_list)
    timestamps = np.array(timestamps)
    u_max_norm_arr = np.array(u_max_norm_list)

    return delta_delta_U, delta_delta_U_diff, delta_delta_p, delta_p_prev, U, timestamps, u_max_norm_arr


def load_boundaries_dict(data_dir='ML_data'):
    """
    Load boundaries as a dictionary from NPZ file.
    
    Returns:
        boundaries: dict with keys like 'z_top_boundary', 'y_bot_boundary', etc.
                   Each value is (N_points, 3) array
    """
    boundaries_file = os.path.join(data_dir, 'boundaries.npz')
    if not os.path.exists(boundaries_file):
        print(f"Warning: boundaries file not found at {boundaries_file}")
        return {}
    
    data = np.load(boundaries_file)
    boundaries = {key: data[key] for key in data.files}
    return boundaries


def save_cell_centers_and_boundaries(coordinates, boundary_coords, boundary_patches, 
                                      patch_names, data_dir='ML_data'):
    """
    Save coordinates as CSV and boundaries as both NPZ dictionary and CSV.
    
    Args:
        coordinates: (n_cells, 3) array of cell center coordinates
        boundary_coords: (n_boundary_faces, 3) array of boundary face centers
        boundary_patches: (n_boundary_faces,) array of patch indices
        patch_names: dict mapping patch index to patch name
        data_dir: directory to save files
    """
    import pandas as pd
    
    # Save cell centers
    df_cells = pd.DataFrame({
        'x': coordinates[:, 0],
        'y': coordinates[:, 1],
        'z': coordinates[:, 2]
    })
    df_cells.to_csv(os.path.join(data_dir, 'cell_centres.csv'), index=False)
    
    # Save number of cells to file
    with open(os.path.join(data_dir, 'n_cells.txt'), 'w') as f:
        f.write(str(coordinates.shape[0]))
    
    print(f"Saved {coordinates.shape[0]} cell centers to cell_centres.csv")
    
    # Save boundary points as dictionary in NPZ format
    if boundary_coords is not None and boundary_patches is not None:
        # Create mapping from patch name to boundary coordinates
        boundaries = {}
        
        for patch_idx, patch_name in patch_names.items():
            # Get mask for this patch
            mask = boundary_patches == patch_idx
            patch_coords = boundary_coords[mask]  # (n_faces_in_patch, 3)
            
            # Map standard OpenFOAM boundary names to dictionary keys
            boundary_key = None
            if 'top' in patch_name.lower():
                boundary_key = 'z_top_boundary'
            elif 'bot' in patch_name.lower():
                boundary_key = 'z_bot_boundary'
            elif 'front' in patch_name.lower():
                boundary_key = 'y_bot_boundary'
            elif 'back' in patch_name.lower():
                boundary_key = 'y_top_boundary'
            elif 'obstacle' in patch_name.lower():
                boundary_key = 'obst_boundary'
            else:
                # Generic fallback: use patch name as key
                boundary_key = f'{patch_name}_boundary'
            
            boundaries[boundary_key] = patch_coords
            print(f"Saved {len(patch_coords)} face centers for boundary '{boundary_key}' (patch: {patch_name})")
        
        # Save boundaries dictionary as NPZ file
        boundaries_file = os.path.join(data_dir, 'boundaries.npz')
        np.savez(boundaries_file, **boundaries)
        print(f"Saved boundaries dictionary to {boundaries_file}")
    else:
        print("Warning: No boundary data to save")


if __name__ == "__main__":
    # Example usage
    coordinates, boundary_coords, boundary_patches, patch_names, delta_U, delta_p, timestamps = \
        load_hdf5_samples('ML_data/data.h5')
    
    print(f"Loaded {len(timestamps)} samples")
    print(f"Coordinates shape: {coordinates.shape}")
    print(f"Boundary coordinates shape: {boundary_coords.shape if boundary_coords is not None else 'None'}")
    print(f"delta_U shape: {delta_U.shape}")
    print(f"delta_p shape: {delta_p.shape}")
    print(f"Patch names: {patch_names}")
    
    # Save coordinates for compatibility
    save_cell_centers_and_boundaries(coordinates, boundary_coords, boundary_patches, 
                                     patch_names, 'ML_data')
