import h5py
import os
import numpy as np
from tqdm import tqdm
import pyvista as pv

def padding(array, max):
    if array.shape[0] > max:
        raise ValueError(f"Number of cells ({array.shape[0]}) exceeds the maximum allowed ({max})")    
    # Vectorized padding
    out = np.full((max,), -100.0, dtype=array.dtype)
    n = min(array.shape[0], max)
    out[:n] = array[:n]
    return out

def extract_simulation_data_bound_single(serial, patch, path_to_sims, t_target, max):
    data = { 'Cx': None, 'Cy': None, 'Cz': None }
    vtk_dir = f"{path_to_sims}/{serial}/VTK/{patch}"
    if not os.path.exists(vtk_dir):
        print(f"Directory {vtk_dir} does not exist")
        return {k: np.full((max,), -100.0) for k in data}
    files = os.listdir(vtk_dir)
    file_map = {}
    for fname in files:
        if fname.startswith(f"{patch}_") and fname.endswith(".vtk"):
            try:
                t_str = fname[len(f"{patch}_"):-4]
                t_val = float(t_str)
                file_map[t_val] = fname
            except Exception:
                continue
    closest = min(file_map.keys(), key=lambda t: abs(t - t_target)) if file_map else None
    if closest is not None and abs(closest - t_target) < 0.1 * t_target:
        path = os.path.join(vtk_dir, file_map[closest])
        mesh = pv.read(path)
        cell_centers = mesh.cell_centers().points
        data['Cx'] = padding(cell_centers[:, 0], max)
        data['Cy'] = padding(cell_centers[:, 1], max)
        data['Cz'] = padding(cell_centers[:, 2], max)
    else:
        print(f"No matching VTK file for {patch} at t={t_target}")
        data = {k: np.full((max,), -100.0) for k in data}
    return data

def extract_simulation_data_single(serial, path_to_sims, t_target, max):
    data = {'Ux': None, 'Uy': None, 'Uz': None,
            'Cx': None, 'Cy': None, 'Cz': None,
            'delta_Ux': None, 'delta_Uy': None, 'delta_Uz': None,
            'p_rgh': None, 'delta_p_rgh': None,
            }
    vtk_dir = f"{path_to_sims}/{serial}/VTK"
    if not os.path.exists(vtk_dir):
        print(f"Directory {vtk_dir} does not exist")
        return {k: np.full((max,), -100.0) for k in data}
    files = os.listdir(vtk_dir)
    file_map = {}
    for fname in files:
        if fname.startswith(f"{serial}_") and fname.endswith(".vtk"):
            try:
                t_str = fname[len(f"{serial}_"):-4]
                t_val = float(t_str)
                file_map[t_val] = fname
            except Exception:
                continue
    closest = min(file_map.keys(), key=lambda t: abs(t - t_target)) if file_map else None
    if closest is not None and abs(closest - t_target) < 0.1 * t_target:
        path = os.path.join(vtk_dir, file_map[closest])
        mesh = pv.read(path)
        cell_centers = mesh.cell_centers().points
        mesh_cell_data = mesh.cell_data
        data['Cx'] = padding(cell_centers[:,0], max)
        data['Cy'] = padding(cell_centers[:,1], max)
        data['Cz'] = padding(cell_centers[:,2], max)
        # U_non_cons
        if 'U_non_cons' in mesh_cell_data:
            U = mesh_cell_data['U_non_cons']
            data['Ux'] = padding(U[:,0], max)
            data['Uy'] = padding(U[:,1], max)
            data['Uz'] = padding(U[:,2], max)
        else:
            data['Ux'] = np.full((max,), -100.0)
            data['Uy'] = np.full((max,), -100.0)
            data['Uz'] = np.full((max,), -100.0)
        # delta_U
        if 'delta_U' in mesh_cell_data:
            dU = mesh_cell_data['delta_U']
            data['delta_Ux'] = padding(dU[:,0], max)
            data['delta_Uy'] = padding(dU[:,1], max)
            data['delta_Uz'] = padding(dU[:,2], max)
        else:
            data['delta_Ux'] = np.full((max,), -100.0)
            data['delta_Uy'] = np.full((max,), -100.0)
            data['delta_Uz'] = np.full((max,), -100.0)
        # p_rgh
        if 'p_rgh' in mesh_cell_data:
            data['p_rgh'] = padding(mesh_cell_data['p_rgh'], max)
        else:
            data['p_rgh'] = np.full((max,), -100.0)
        # delta_p_rgh
        if 'delta_p_rgh' in mesh_cell_data:
            data['delta_p_rgh'] = padding(mesh_cell_data['delta_p_rgh'], max)
        else:
            data['delta_p_rgh'] = np.full((max,), -100.0)
    else:
        print(f"No matching VTK file for {serial} at t={t_target}")
        data = {k: np.full((max,), -100.0) for k in data}
    return data

def create_hdf5_file(hdf5_path, num_sims_actual, num_time_steps, max_n_cells_sim, max_n_cells_patch):
    train_shape = (num_sims_actual, num_time_steps, max_n_cells_sim, 11)
    #train_shape = (num_sims_actual, int(total_times), 600000, 12)
    top_shape = (num_sims_actual, num_time_steps, max_n_cells_patch, 3)
    obst_shape = (num_sims_actual, num_time_steps, max_n_cells_patch, 3)

    if os.path.exists(hdf5_path):
        hdf5_file = h5py.File(hdf5_path, mode='a')
        return hdf5_file
    
    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset('sim_data', train_shape, np.float32)
    hdf5_file.create_dataset('y_bot_bound', top_shape, np.float32)
    hdf5_file.create_dataset('z_bot_bound', top_shape, np.float32)
    hdf5_file.create_dataset('y_top_bound', top_shape, np.float32)
    hdf5_file.create_dataset('z_top_bound', top_shape, np.float32)
    hdf5_file.create_dataset('obst_bound', obst_shape, np.float32)

    return hdf5_file

def process_simulation(i_sim, sim, path_to_sims, avance_list, hdf5_file, first_t, last_t, max_sim, max_patch):
    # Read the deltaT_write
    control_dict_path = f"{path_to_sims}/{sim}/system/controlDict"
    with open(control_dict_path, 'r') as file:
        for line in file:
            if 'writeInterval' in line:
                deltat = float(line.split()[1].strip(';'))
                break       

    avance = avance_list[sim]
    time_steps = range(first_t, last_t + 1)
    for t_idx, time in enumerate(tqdm(time_steps, desc=f"Sim {sim} time steps", leave=False)):
        t_target = deltat * (time * avance + 1)
        data = extract_simulation_data_single(sim, path_to_sims, t_target, max_sim)
        data_obst = extract_simulation_data_bound_single(sim, 'cylinder', path_to_sims, t_target, max_patch)
        data_y_top = extract_simulation_data_bound_single(sim, 'back', path_to_sims, t_target, max_patch)
        data_z_bot = extract_simulation_data_bound_single(sim, 'bot', path_to_sims, t_target, max_patch)
        data_y_bot = extract_simulation_data_bound_single(sim, 'front', path_to_sims, t_target, max_patch)
        data_z_top = extract_simulation_data_bound_single(sim, 'top', path_to_sims, t_target, max_patch)

        hdf5_file['sim_data'][i_sim, t_idx, :, 0] = data['Ux']
        hdf5_file['sim_data'][i_sim, t_idx, :, 1] = data['Uy']
        hdf5_file['sim_data'][i_sim, t_idx, :, 2] = data['Uz']
        hdf5_file['sim_data'][i_sim, t_idx, :, 3] = data['p_rgh']
        hdf5_file['sim_data'][i_sim, t_idx, :, 4] = data['Cx']
        hdf5_file['sim_data'][i_sim, t_idx, :, 5] = data['Cy']
        hdf5_file['sim_data'][i_sim, t_idx, :, 6] = data['Cz']
        hdf5_file['sim_data'][i_sim, t_idx, :, 7] = data['delta_Ux']
        hdf5_file['sim_data'][i_sim, t_idx, :, 8] = data['delta_Uy']
        hdf5_file['sim_data'][i_sim, t_idx, :, 9] = data['delta_Uz']
        hdf5_file['sim_data'][i_sim, t_idx, :, 10] = data['delta_p_rgh']

        hdf5_file['y_bot_bound'][i_sim, t_idx, :, 0] = data_y_bot['Cx']
        hdf5_file['y_bot_bound'][i_sim, t_idx, :, 1] = data_y_bot['Cy']
        hdf5_file['y_bot_bound'][i_sim, t_idx, :, 2] = data_y_bot['Cz']

        hdf5_file['z_bot_bound'][i_sim, t_idx, :, 0] = data_z_bot['Cx']
        hdf5_file['z_bot_bound'][i_sim, t_idx, :, 1] = data_z_bot['Cy']
        hdf5_file['z_bot_bound'][i_sim, t_idx, :, 2] = data_z_bot['Cz']

        hdf5_file['y_top_bound'][i_sim, t_idx, :, 0] = data_y_top['Cx']
        hdf5_file['y_top_bound'][i_sim, t_idx, :, 1] = data_y_top['Cy']
        hdf5_file['y_top_bound'][i_sim, t_idx, :, 2] = data_y_top['Cz']

        hdf5_file['z_top_bound'][i_sim, t_idx, :, 0] = data_z_top['Cx']
        hdf5_file['z_top_bound'][i_sim, t_idx, :, 1] = data_z_top['Cy']
        hdf5_file['z_top_bound'][i_sim, t_idx, :, 2] = data_z_top['Cz']

        hdf5_file['obst_bound'][i_sim, t_idx, :, 0] = data_obst['Cx']
        hdf5_file['obst_bound'][i_sim, t_idx, :, 1] = data_obst['Cy']
        hdf5_file['obst_bound'][i_sim, t_idx, :, 2] = data_obst['Cz']

    hdf5_file.flush()

def main():
    first_sim = 5
    last_sim = 7
    first_t = 0
    last_t = 19
    num_time_steps = last_t - first_t + 1
    hdf5_path = 'dataset_heat_cyl_larger_3sim20t_test.hdf5'
    path_to_sims = 'simulations_larger/'
    max_n_cells_sim = int(8.5e5)
    max_n_cells_patch = int(150000)

    avance_list = [1] * 20

    if os.path.exists(hdf5_path):
        raise FileExistsError(f"HDF5 file {hdf5_path} already exists. Please remove it or change the target file name before running the script.")

    hdf5_file = create_hdf5_file(hdf5_path, int(last_sim-first_sim +1), num_time_steps, max_n_cells_sim, max_n_cells_patch)

    for i_sim, sim in tqdm(enumerate(range(first_sim, last_sim + 1)), total=last_sim-first_sim+1, desc="Simulations"):
        process_simulation(i_sim, sim, path_to_sims, avance_list, hdf5_file, first_t, last_t, max_n_cells_sim, max_n_cells_patch)

    hdf5_file.close()

if __name__ == "__main__":
    main()
