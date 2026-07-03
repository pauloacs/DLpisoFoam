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
        ddUStar: (n_samples, n_cells, 3) array of non-conservative velocity double-increments
        ddUStarDiff: (n_samples, n_cells, 3) array of ddUStar time differences
        dpPrev: (n_samples, n_cells) array of previous pressure increments
        ddpPrev: (n_samples, n_cells) array of previous pressure double-increments
        gradDpPrev: (n_samples, n_cells, 3) array of gradient of previous pressure increments
        laplaceDpPrev: (n_samples, n_cells) array of laplacian of previous pressure increments
        uDotGradDpPrev: (n_samples, n_cells) array of U dot grad(dpPrev)
        gradDpPrevMag: (n_samples, n_cells) array of |grad(dpPrev)|
        divDDUStar: (n_samples, n_cells) array of divergence of ddUStar
        divUStar: (n_samples, n_cells) array of divergence of UStar
        divDUStar: (n_samples, n_cells) array of divergence of dUStar
        dUStar: (n_samples, n_cells, 3) array of non-conservative velocity increments
        p_prev: (n_samples, n_cells) array of absolute previous pressure
        ddp: (n_samples, n_cells) array of pressure double-increments
        ddUCorr: (n_samples, n_cells, 3) array of pressure-correction second velocity increments (training target)
        timestamps: (n_samples,) array of timesteps
        u_max_norm_arr: (n_samples,) array of velocity normalizations
    """
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"HDF5 data file not found: {data_file}")
    
    coordinates = None
    boundary_coords = None
    boundary_patches = None
    patch_names = {}
    ddUStar_list = []
    ddUStarDiff_list = []
    ddUCorr_list = []
    ddp_list = []
    dpPrev_list = []
    ddpPrev_list = []
    gradDpPrev_list = []
    laplaceDpPrev_list = []
    uDotGradDpPrev_list = []
    gradDpPrevMag_list = []
    rAU_list = []
    HbyA_list = []
    divHbyA_list = []
    dHbyA_list = []
    dDivHbyA_list = []
    rAUGradDpPrev_list = []
    divRAUGradDpPrev_list = []
    pressureEqResidualp_list = []
    rAUGradpPrev_list = []
    divRAUGradpPrev_list = []
    dUStar_list = []
    dUCorrPrev_list = []
    ddUCorrPrev_list = []
    p_prev_list = []
    U_list = []
    divDDUStar_list = []
    divUStar_list = []
    divDUStar_list = []
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
            # Support new key 'ddp', old key 'ddpML', and oldest 'pressure_increment'
            ddp_key = 'ddp' if 'ddp' in group else ('ddpML' if 'ddpML' in group else 'pressure_increment')
            if 'ddUStar' not in group or ddp_key not in group:
                print(f"Warning: {sample_key} missing ddUStar or pressure dataset, skipping")
                continue

            ddUStar_arr = group['ddUStar'][:]     # (n_cells, 3)
            ddUStarDiff_arr = group['ddUStarDiff'][:]  # (n_cells, 3)
            ddp = group[ddp_key][:]     # (n_cells,)
            
            # Load dpPrev if present (new key 'dpPrev', old key 'dpML_prev', oldest 'pressure_increment_prev')
            dpPrev_key = 'dpPrev' if 'dpPrev' in group else ('dpML_prev' if 'dpML_prev' in group else 'pressure_increment_prev')
            if dpPrev_key in group:
                dpPrev = group[dpPrev_key][:]
            else:
                dpPrev = np.zeros(n_cells)  # placeholder if not available

            # Load ddpPrev if present (new key 'ddpPrev', old key 'ddpML_prev' for backward compat)
            if 'ddpPrev' in group or 'ddpML_prev' in group:
                ddpPrev = group['ddpPrev' if 'ddpPrev' in group else 'ddpML_prev'][:]
            else:
                ddpPrev = np.zeros(n_cells)  # placeholder if not available

            if 'gradDpPrev' in group:
                gradDpPrev = group['gradDpPrev'][:]
            else:
                gradDpPrev = np.zeros((n_cells, 3))

            if 'laplaceDpPrev' in group:
                laplaceDpPrev = group['laplaceDpPrev'][:]
            else:
                laplaceDpPrev = np.zeros(n_cells)

            if 'uDotGradDpPrev' in group:
                uDotGradDpPrev = group['uDotGradDpPrev'][:]
            else:
                uDotGradDpPrev = np.zeros(n_cells)

            if 'gradDpPrevMag' in group:
                gradDpPrevMag = group['gradDpPrevMag'][:]
            else:
                gradDpPrevMag = np.zeros(n_cells)

            if 'rAU' in group:
                rAU = group['rAU'][:]
            else:
                rAU = np.zeros(n_cells)

            if 'HbyA' in group:
                HbyA = group['HbyA'][:]
            else:
                HbyA = np.zeros((n_cells, 3))

            if 'divHbyA' in group:
                divHbyA = group['divHbyA'][:]
            else:
                divHbyA = np.zeros(n_cells)

            if 'dHbyA' in group:
                dHbyA = group['dHbyA'][:]
            else:
                dHbyA = np.zeros((n_cells, 3))

            if 'dDivHbyA' in group:
                dDivHbyA = group['dDivHbyA'][:]
            else:
                dDivHbyA = np.zeros(n_cells)

            if 'rAUGradDpPrev' in group:
                rAUGradDpPrev = group['rAUGradDpPrev'][:]
            else:
                rAUGradDpPrev = np.zeros((n_cells, 3))

            if 'divRAUGradDpPrev' in group:
                divRAUGradDpPrev = group['divRAUGradDpPrev'][:]
            else:
                divRAUGradDpPrev = np.zeros(n_cells)

            if 'pressureEqResidualp' in group:
                pressureEqResidualp = group['pressureEqResidualp'][:]
            else:
                pressureEqResidualp = np.zeros(n_cells)

            if 'rAUGradpPrev' in group:
                rAUGradpPrev = group['rAUGradpPrev'][:]
            else:
                rAUGradpPrev = np.zeros((n_cells, 3))

            if 'divRAUGradpPrev' in group:
                divRAUGradpPrev = group['divRAUGradpPrev'][:]
            else:
                divRAUGradpPrev = np.zeros(n_cells)

            # Load divDDUStar if present
            if 'divDDUStar' in group:
                divDDUStar_arr = group['divDDUStar'][:]  # (n_cells,)
            else:
                divDDUStar_arr = np.zeros(n_cells)

            # Load divUFirstPred (written as 'divUFirstPred'; fall back to 'divUStar' for old data)
            if 'divUFirstPred' in group:
                divUStar_arr = group['divUFirstPred'][:]  # (n_cells,)
            elif 'divUStar' in group:
                divUStar_arr = group['divUStar'][:]  # (n_cells,)
            else:
                divUStar_arr = np.zeros(n_cells)

            # Load divDUStar if present
            if 'divDUStar' in group:
                divDUStar_arr = group['divDUStar'][:]  # (n_cells,)
            else:
                divDUStar_arr = np.zeros(n_cells)

            # Load ddUCorr if present (pressure-correction second velocity increment = training target)
            if 'ddUCorr' in group:
                ddUCorr_list.append(group['ddUCorr'][:])
            else:
                ddUCorr_list.append(np.zeros((n_cells, 3)))

            # Load U (velocity) if present
            if 'U' in group:
                U_list.append(group['U'][:])
            else:
                U_list.append(np.zeros((n_cells, 3)))  # placeholder if not available

            # Load dUStar (non-conservative velocity increment) if present
            if 'dUStar' in group:
                dUStar_list.append(group['dUStar'][:])
            else:
                dUStar_list.append(np.zeros((n_cells, 3)))

            # Load dUCorrPrev and ddUCorrPrev if present
            if 'dUCorrPrev' in group:
                dUCorrPrev_list.append(group['dUCorrPrev'][:])
            else:
                dUCorrPrev_list.append(np.zeros((n_cells, 3)))

            if 'ddUCorrPrev' in group:
                ddUCorrPrev_list.append(group['ddUCorrPrev'][:])
            else:
                ddUCorrPrev_list.append(np.zeros((n_cells, 3)))

            # Load p_prev (absolute previous pressure) if present
            if 'p_prev' in group:
                p_prev_list.append(group['p_prev'][:])
            else:
                p_prev_list.append(np.zeros(n_cells))  # placeholder if not available

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

            ddUStar_list.append(ddUStar_arr)
            ddUStarDiff_list.append(ddUStarDiff_arr)
            ddp_list.append(ddp)
            dpPrev_list.append(dpPrev)
            ddpPrev_list.append(ddpPrev)
            gradDpPrev_list.append(gradDpPrev)
            laplaceDpPrev_list.append(laplaceDpPrev)
            uDotGradDpPrev_list.append(uDotGradDpPrev)
            gradDpPrevMag_list.append(gradDpPrevMag)
            rAU_list.append(rAU)
            HbyA_list.append(HbyA)
            divHbyA_list.append(divHbyA)
            dHbyA_list.append(dHbyA)
            dDivHbyA_list.append(dDivHbyA)
            rAUGradDpPrev_list.append(rAUGradDpPrev)
            divRAUGradDpPrev_list.append(divRAUGradDpPrev)
            pressureEqResidualp_list.append(pressureEqResidualp)
            rAUGradpPrev_list.append(rAUGradpPrev)
            divRAUGradpPrev_list.append(divRAUGradpPrev)
            divDDUStar_list.append(divDDUStar_arr)
            divUStar_list.append(divUStar_arr)
            divDUStar_list.append(divDUStar_arr)
            timestamps.append(timestep)
    
    # Stack into arrays
    ddUStar = np.array(ddUStar_list)           # (n_samples, n_cells, 3)
    ddUStarDiff = np.array(ddUStarDiff_list)   # (n_samples, n_cells, 3)
    ddUCorr = np.array(ddUCorr_list)           # (n_samples, n_cells, 3)
    ddp = np.array(ddp_list)  # (n_samples, n_cells)
    dpPrev = np.array(dpPrev_list)  # (n_samples, n_cells)
    ddpPrev = np.array(ddpPrev_list)  # (n_samples, n_cells)
    gradDpPrev = np.array(gradDpPrev_list)  # (n_samples, n_cells, 3)
    laplaceDpPrev = np.array(laplaceDpPrev_list)  # (n_samples, n_cells)
    uDotGradDpPrev = np.array(uDotGradDpPrev_list)  # (n_samples, n_cells)
    gradDpPrevMag = np.array(gradDpPrevMag_list)  # (n_samples, n_cells)
    rAU = np.array(rAU_list)  # (n_samples, n_cells)
    HbyA = np.array(HbyA_list)  # (n_samples, n_cells, 3)
    divHbyA = np.array(divHbyA_list)  # (n_samples, n_cells)
    dHbyA = np.array(dHbyA_list)  # (n_samples, n_cells, 3)
    dDivHbyA = np.array(dDivHbyA_list)  # (n_samples, n_cells)
    rAUGradDpPrev = np.array(rAUGradDpPrev_list)  # (n_samples, n_cells, 3)
    divRAUGradDpPrev = np.array(divRAUGradDpPrev_list)  # (n_samples, n_cells)
    pressureEqResidualp = np.array(pressureEqResidualp_list)  # (n_samples, n_cells)
    rAUGradpPrev = np.array(rAUGradpPrev_list)  # (n_samples, n_cells, 3)
    divRAUGradpPrev = np.array(divRAUGradpPrev_list)  # (n_samples, n_cells)
    divDDUStar = np.array(divDDUStar_list)     # (n_samples, n_cells)
    divUStar = np.array(divUStar_list)         # (n_samples, n_cells)
    divDUStar = np.array(divDUStar_list)       # (n_samples, n_cells)
    dUStar = np.array(dUStar_list)             # (n_samples, n_cells, 3)
    dUCorrPrev = np.array(dUCorrPrev_list)     # (n_samples, n_cells, 3)
    ddUCorrPrev = np.array(ddUCorrPrev_list)   # (n_samples, n_cells, 3)
    p_prev = np.array(p_prev_list)             # (n_samples, n_cells)
    U = np.array(U_list)                       # (n_samples, n_cells, 3)
    timestamps = np.array(timestamps)
    
    u_max_norm_arr = np.array(u_max_norm_list)
    return coordinates, boundary_coords, boundary_patches, patch_names, U, ddUStar, ddUStarDiff, dpPrev, ddpPrev, gradDpPrev, laplaceDpPrev, uDotGradDpPrev, gradDpPrevMag, rAU, HbyA, divHbyA, dHbyA, dDivHbyA, rAUGradDpPrev, divRAUGradDpPrev, pressureEqResidualp, rAUGradpPrev, divRAUGradpPrev, divDDUStar, divUStar, divDUStar, dUStar, dUCorrPrev, ddUCorrPrev, p_prev, ddp, ddUCorr, timestamps, u_max_norm_arr


def load_hdf5_field_data(data_file='ML_data/data.h5'):
    """
    Load only field samples from master HDF5 file.
    Cell centers and boundary coordinates are NOT loaded — they were already
    saved to disk during train_init.py and do not need to be re-read on updates.

    Returns:
        ddUStar: (n_samples, n_cells, 3) array of non-conservative velocity double-increments
        ddUStarDiff: (n_samples, n_cells, 3) array of ddUStar time differences
        ddp: (n_samples, n_cells) array of pressure double-increments
        dpPrev: (n_samples, n_cells) array of previous pressure increments
        ddpPrev: (n_samples, n_cells) array of previous pressure double-increments
        gradDpPrev: (n_samples, n_cells, 3) array of gradient of previous pressure increments
        laplaceDpPrev: (n_samples, n_cells) array of laplacian of previous pressure increments
        uDotGradDpPrev: (n_samples, n_cells) array of U dot grad(dpPrev)
        gradDpPrevMag: (n_samples, n_cells) array of |grad(dpPrev)|
        dUStar: (n_samples, n_cells, 3) array of non-conservative velocity increments
        p_prev: (n_samples, n_cells) array of absolute previous pressure
        U: (n_samples, n_cells, 3) array of velocities
        divDDUStar: (n_samples, n_cells) array of divergence of ddUStar
        divDUStar: (n_samples, n_cells) array of divergence of dUStar
        divUStar: (n_samples, n_cells) array of divergence of UStar
        ddUCorr: (n_samples, n_cells, 3) array of pressure-correction second velocity increments
        timestamps: (n_samples,) array of timesteps
        u_max_norm_arr: (n_samples,) array of velocity normalizations
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"HDF5 data file not found: {data_file}")

    ddUStar_list = []
    ddUStarDiff_list = []
    ddUCorr_list = []
    ddp_list = []
    dpPrev_list = []
    ddpPrev_list = []
    gradDpPrev_list = []
    laplaceDpPrev_list = []
    uDotGradDpPrev_list = []
    gradDpPrevMag_list = []
    rAU_list = []
    HbyA_list = []
    divHbyA_list = []
    dHbyA_list = []
    dDivHbyA_list = []
    rAUGradDpPrev_list = []
    divRAUGradDpPrev_list = []
    pressureEqResidualp_list = []
    rAUGradpPrev_list = []
    divRAUGradpPrev_list = []
    dUStar_list = []
    dUCorrPrev_list = []
    ddUCorrPrev_list = []
    p_prev_list = []
    U_list = []
    divDDUStar_list = []
    divDUStar_list = []
    divUStar_list = []
    timestamps = []
    u_max_norm_list = []

    with h5py.File(data_file, 'r') as f:
        sample_keys = sorted([key for key in f.keys() if key.startswith('sample_')])

        if len(sample_keys) == 0:
            raise ValueError("No sample groups (sample_*) found in HDF5 file")

        for sample_key in sample_keys:
            group = f[sample_key]
            ddp_key = 'ddp' if 'ddp' in group else ('ddpML' if 'ddpML' in group else 'pressure_increment')
            if 'ddUStar' not in group or ddp_key not in group:
                print(f"Warning: {sample_key} missing ddUStar or pressure dataset, skipping")
                continue
            ddUStar_list.append(group['ddUStar'][:])
            if 'ddUStarDiff' in group:
                ddUStarDiff_list.append(group['ddUStarDiff'][:])
            else:
                ddUStarDiff_list.append(np.zeros_like(group['ddUStar'][:]))
            ddp_list.append(group[ddp_key][:])
            
            # Load dpPrev (new key 'dpPrev', old key 'dpML_prev', oldest 'pressure_increment_prev')
            dpPrev_key = 'dpPrev' if 'dpPrev' in group else ('dpML_prev' if 'dpML_prev' in group else 'pressure_increment_prev')
            if dpPrev_key in group:
                dpPrev_list.append(group[dpPrev_key][:])
            else:
                dpPrev_list.append(np.zeros_like(group[ddp_key][:]))

            # Load ddpPrev (new key 'ddpPrev', old key 'ddpML_prev' for backward compat)
            if 'ddpPrev' in group or 'ddpML_prev' in group:
                ddpPrev_list.append(group['ddpPrev' if 'ddpPrev' in group else 'ddpML_prev'][:])
            else:
                ddpPrev_list.append(np.zeros_like(group[ddp_key][:]))

            if 'gradDpPrev' in group:
                gradDpPrev_list.append(group['gradDpPrev'][:])
            else:
                gradDpPrev_list.append(np.zeros((group['ddUStar'].shape[0], 3)))

            if 'laplaceDpPrev' in group:
                laplaceDpPrev_list.append(group['laplaceDpPrev'][:])
            else:
                laplaceDpPrev_list.append(np.zeros_like(group[ddp_key][:]))

            if 'uDotGradDpPrev' in group:
                uDotGradDpPrev_list.append(group['uDotGradDpPrev'][:])
            else:
                uDotGradDpPrev_list.append(np.zeros_like(group[ddp_key][:]))

            if 'gradDpPrevMag' in group:
                gradDpPrevMag_list.append(group['gradDpPrevMag'][:])
            else:
                gradDpPrevMag_list.append(np.zeros_like(group[ddp_key][:]))

            if 'rAU' in group:
                rAU_list.append(group['rAU'][:])
            else:
                rAU_list.append(np.zeros_like(group[ddp_key][:]))

            if 'HbyA' in group:
                HbyA_list.append(group['HbyA'][:])
            else:
                HbyA_list.append(np.zeros((group['ddUStar'].shape[0], 3)))

            if 'divHbyA' in group:
                divHbyA_list.append(group['divHbyA'][:])
            else:
                divHbyA_list.append(np.zeros_like(group[ddp_key][:]))

            if 'dHbyA' in group:
                dHbyA_list.append(group['dHbyA'][:])
            else:
                dHbyA_list.append(np.zeros((group['ddUStar'].shape[0], 3)))

            if 'dDivHbyA' in group:
                dDivHbyA_list.append(group['dDivHbyA'][:])
            else:
                dDivHbyA_list.append(np.zeros_like(group[ddp_key][:]))

            if 'rAUGradDpPrev' in group:
                rAUGradDpPrev_list.append(group['rAUGradDpPrev'][:])
            else:
                rAUGradDpPrev_list.append(np.zeros((group['ddUStar'].shape[0], 3)))

            if 'divRAUGradDpPrev' in group:
                divRAUGradDpPrev_list.append(group['divRAUGradDpPrev'][:])
            else:
                divRAUGradDpPrev_list.append(np.zeros_like(group[ddp_key][:]))

            if 'pressureEqResidualp' in group:
                pressureEqResidualp_list.append(group['pressureEqResidualp'][:])
            else:
                pressureEqResidualp_list.append(np.zeros_like(group[ddp_key][:]))

            if 'rAUGradpPrev' in group:
                rAUGradpPrev_list.append(group['rAUGradpPrev'][:])
            else:
                rAUGradpPrev_list.append(np.zeros((group['ddUStar'].shape[0], 3)))

            if 'divRAUGradpPrev' in group:
                divRAUGradpPrev_list.append(group['divRAUGradpPrev'][:])
            else:
                divRAUGradpPrev_list.append(np.zeros_like(group[ddp_key][:]))
            if 'U' in group:
                U_list.append(group['U'][:])
            else:
                U_list.append(np.zeros_like(group['ddUStar'][:]))

            # Load dUStar (non-conservative velocity increment) if present
            if 'dUStar' in group:
                dUStar_list.append(group['dUStar'][:])
            else:
                dUStar_list.append(np.zeros_like(group['ddUStar'][:]))

            # Load dUCorrPrev and ddUCorrPrev if present
            if 'dUCorrPrev' in group:
                dUCorrPrev_list.append(group['dUCorrPrev'][:])
            else:
                dUCorrPrev_list.append(np.zeros_like(group['ddUStar'][:]))

            if 'ddUCorrPrev' in group:
                ddUCorrPrev_list.append(group['ddUCorrPrev'][:])
            else:
                ddUCorrPrev_list.append(np.zeros_like(group['ddUStar'][:]))

            # Load p_prev (absolute previous pressure) if present
            if 'p_prev' in group:
                p_prev_list.append(group['p_prev'][:])
            else:
                p_prev_list.append(np.zeros_like(group[ddp_key][:]))

            # Load divergence fields if present
            if 'divDDUStar' in group:
                divDDUStar_list.append(group['divDDUStar'][:])
            else:
                divDDUStar_list.append(np.zeros_like(group[ddp_key][:]))

            if 'divDUStar' in group:
                divDUStar_list.append(group['divDUStar'][:])
            else:
                divDUStar_list.append(np.zeros_like(group[ddp_key][:]))

            # Load divUFirstPred (fall back to 'divUStar' for old data)
            if 'divUFirstPred' in group:
                divUStar_list.append(group['divUFirstPred'][:])
            elif 'divUStar' in group:
                divUStar_list.append(group['divUStar'][:])
            else:
                divUStar_list.append(np.zeros_like(group[ddp_key][:]))

            # Load ddUCorr if present (pressure-correction second velocity increment)
            if 'ddUCorr' in group:
                ddUCorr_list.append(group['ddUCorr'][:])
            else:
                ddUCorr_list.append(np.zeros_like(group['ddUStar'][:]))

            timestamps.append(group.attrs.get('timestep', -1))
            if 'U_MAX_NORM' in group:
                u_max = group['U_MAX_NORM'][()]
                u_max_norm_list.append(float(u_max) if not isinstance(u_max, np.ndarray) else float(u_max))
            else:
                u_max_norm_list.append(1.0)

    ddUStar = np.array(ddUStar_list)
    ddUStarDiff = np.array(ddUStarDiff_list)
    ddUCorr = np.array(ddUCorr_list)
    ddp = np.array(ddp_list)
    dpPrev = np.array(dpPrev_list)
    ddpPrev = np.array(ddpPrev_list)
    gradDpPrev = np.array(gradDpPrev_list)
    laplaceDpPrev = np.array(laplaceDpPrev_list)
    uDotGradDpPrev = np.array(uDotGradDpPrev_list)
    gradDpPrevMag = np.array(gradDpPrevMag_list)
    rAU = np.array(rAU_list)
    HbyA = np.array(HbyA_list)
    divHbyA = np.array(divHbyA_list)
    dHbyA = np.array(dHbyA_list)
    dDivHbyA = np.array(dDivHbyA_list)
    rAUGradDpPrev = np.array(rAUGradDpPrev_list)
    divRAUGradDpPrev = np.array(divRAUGradDpPrev_list)
    pressureEqResidualp = np.array(pressureEqResidualp_list)
    rAUGradpPrev = np.array(rAUGradpPrev_list)
    divRAUGradpPrev = np.array(divRAUGradpPrev_list)
    dUStar = np.array(dUStar_list)
    dUCorrPrev = np.array(dUCorrPrev_list)
    ddUCorrPrev = np.array(ddUCorrPrev_list)
    p_prev = np.array(p_prev_list)
    U = np.array(U_list)
    divDDUStar = np.array(divDDUStar_list)
    divDUStar = np.array(divDUStar_list)
    divUStar = np.array(divUStar_list)
    timestamps = np.array(timestamps)
    u_max_norm_arr = np.array(u_max_norm_list)

    return ddUStar, ddUStarDiff, ddp, dpPrev, ddpPrev, gradDpPrev, laplaceDpPrev, uDotGradDpPrev, gradDpPrevMag, rAU, HbyA, divHbyA, dHbyA, dDivHbyA, rAUGradDpPrev, divRAUGradDpPrev, pressureEqResidualp, rAUGradpPrev, divRAUGradpPrev, dUStar, dUCorrPrev, ddUCorrPrev, p_prev, U, divDDUStar, divDUStar, divUStar, ddUCorr, timestamps, u_max_norm_arr


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
    coordinates, boundary_coords, boundary_patches, patch_names, delta_U, dpML, timestamps = \
        load_hdf5_samples('ML_data/data.h5')
    
    print(f"Loaded {len(timestamps)} samples")
    print(f"Coordinates shape: {coordinates.shape}")
    print(f"Boundary coordinates shape: {boundary_coords.shape if boundary_coords is not None else 'None'}")
    print(f"delta_U shape: {delta_U.shape}")
    print(f"dpML shape: {dpML.shape}")
    print(f"Patch names: {patch_names}")
    
    # Save coordinates for compatibility
    save_cell_centers_and_boundaries(coordinates, boundary_coords, boundary_patches, 
                                     patch_names, 'ML_data')
