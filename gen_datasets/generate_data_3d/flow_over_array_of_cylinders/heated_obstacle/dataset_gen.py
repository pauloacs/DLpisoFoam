import h5py
import os
import numpy as np

import numpy as np
import h5py
import os
from tqdm import tqdm
import pyvista as pv

def padding(array, max):
        x=[]
        a =np.pad(array, [(0, max - array.shape[0] ) ], mode = 'constant', constant_values = -100.0)
        x.append(a)
        x= np.array(x).reshape((max,))
        return x

def extract_simulation_data_bound(serial, patch, path_to_sims, first_t, last_t, deltat, avance, max):
    data = { 'Cx': [], 'Cy': [], 'Cz': [] }
    for time in range(first_t, last_t + 1):

        # Try multiple roundings if the file is not found
        found = False
        # List all files in the directory
        vtk_dir = f"{path_to_sims}/{serial}/VTK/{patch}"
        if not os.path.exists(vtk_dir):
            print(f"Directory {vtk_dir} does not exist")
            continue
        files = os.listdir(vtk_dir)
        t_target = deltat * (time * avance + 1)
        path = f"{vtk_dir}/{patch}_{t_target}.vtk"
        closest_diff = float('inf')
        matched_file = None
        for fname in files:
            if fname.startswith(f"{patch}_") and fname.endswith(".vtk"):
                try:
                    t_str = fname[len(f"{patch}_"):-4]
                    t_val = float(t_str)
                    diff = abs(t_val - t_target)
                    if diff < 0.1 * t_target and diff < closest_diff:
                        closest_diff = diff
                        matched_file = fname
                except Exception:
                    continue
        if matched_file:
            path = os.path.join(vtk_dir, matched_file)
            print(path)
            found = True
            
        if not found:
            print(f"Path {path} does not exist after trying multiple roundings")
            print('Probably it corresponds to a steady state simulation - not saving more data for this simulation')
            continue

        mesh = pv.read(path)
        cell_centers = mesh.cell_centers().points

        data['Cx'].append(padding(cell_centers[:, 0], max))
        data['Cy'].append(padding(cell_centers[:, 1], max))
        data['Cz'].append(padding(cell_centers[:, 2], max))

    return data

def extract_simulation_data(serial, path_to_sims, first_t, last_t, deltat, avance, max):
    data = {'Ux': [], 'Uy': [], 'Uz': [],
            'Cx': [], 'Cy': [], 'Cz': [],
            'delta_Ux': [], 'delta_Uy': [], 'delta_Uz': [],
            'p_rgh': [], 'delta_p_rgh': [],
            }

    for entity in ['U_non_cons', 'Cx', 'Cy', 'Cz', 'delta_U', 'delta_p_rgh', 'p_rgh']:
      for time in range(first_t, last_t + 1):

        found = False
        # List all files in the directory
        vtk_dir = f"{path_to_sims}/{serial}/VTK"
        if not os.path.exists(vtk_dir):
            print(f"Directory {vtk_dir} does not exist")
            continue
        files = os.listdir(vtk_dir)
        t_target = deltat * (time * avance + 1)
        path = f"{vtk_dir}/{serial}_{t_target}.vtk"
        closest_diff = float('inf')
        matched_file = None
        for fname in files:
            if fname.startswith(f"{serial}_") and fname.endswith(".vtk"):
                try:
                    t_str = fname[len(f"{serial}_"):-4]
                    t_val = float(t_str)
                    diff = abs(t_val - t_target)
                    if diff < 0.1 * t_target and diff < closest_diff:
                        closest_diff = diff
                        matched_file = fname
                except Exception:
                    continue
        if matched_file:
            path = os.path.join(vtk_dir, matched_file)
            print(path)
            found = True
        if not found:
            print(f"Path {path} does not exist")
            print('Probably it corresponds to a steady state simulation - not saving more data for this simulation')
            continue

        mesh = pv.read(path)

        if entity in ['Cx', 'Cy', 'Cz']:
            cell_centers = mesh.cell_centers().points
            if entity == 'Cx':
                data['Cx'].append(padding(cell_centers[:,0],max))
            elif entity == 'Cy':
                data['Cy'].append(padding(cell_centers[:,1],max))
            elif entity == 'Cz':
                data['Cz'].append(padding(cell_centers[:,2],max))
        else:
            mesh_cell_data = mesh.cell_data

            if entity == 'U_non_cons':
                data['Ux'].append(padding(mesh_cell_data[entity][:,0],max))
                data['Uy'].append(padding(mesh_cell_data[entity][:,1],max))
                data['Uz'].append(padding(mesh_cell_data[entity][:,2],max))
            elif entity == 'delta_U':
                data['delta_Ux'].append(padding(mesh_cell_data[entity][:,0],max))
                data['delta_Uy'].append(padding(mesh_cell_data[entity][:,1],max))
                data['delta_Uz'].append(padding(mesh_cell_data[entity][:,2],max))
            else:
                extent = 0, 3, 0, 1
                data[entity].append(padding(mesh_cell_data[entity], max))
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

def process_simulation(sim, path_to_sims, avance_list, hdf5_file, first_t, last_t, max_sim, max_patch):

    # Read the deltaT_write
    control_dict_path = f"{path_to_sims}/{sim}/system/controlDict"
    with open(control_dict_path, 'r') as file:
        for line in file:
            if 'writeInterval' in line:
                deltat = float(line.split()[1].strip(';'))
                break       

    avance = avance_list[sim]
    data = extract_simulation_data(sim, path_to_sims, first_t, last_t, deltat, avance, max_sim)
    data_obst = extract_simulation_data_bound(sim, 'cylinder', path_to_sims, first_t, last_t, deltat, avance, max_patch)
    data_y_top = extract_simulation_data_bound(sim, 'back', path_to_sims, first_t, last_t, deltat, avance, max_patch)
    data_z_bot = extract_simulation_data_bound(sim, 'bot', path_to_sims, first_t, last_t, deltat, avance, max_patch)
    data_y_bot = extract_simulation_data_bound(sim, 'front', path_to_sims, first_t, last_t, deltat, avance, max_patch)
    data_z_top = extract_simulation_data_bound(sim, 'top', path_to_sims, first_t, last_t, deltat, avance, max_patch)

    hdf5_file['sim_data'][sim, ..., 0] = data['Ux']
    hdf5_file['sim_data'][sim, ..., 1] = data['Uy']
    hdf5_file['sim_data'][sim, ..., 2] = data['Uz']
    hdf5_file['sim_data'][sim, ..., 3] = data['p_rgh']
    hdf5_file['sim_data'][sim, ..., 4] = data['Cx']
    hdf5_file['sim_data'][sim, ..., 5] = data['Cy']
    hdf5_file['sim_data'][sim, ..., 6] = data['Cz']
    hdf5_file['sim_data'][sim, ..., 7] = data['delta_Ux']
    hdf5_file['sim_data'][sim, ..., 8] = data['delta_Uy']
    hdf5_file['sim_data'][sim, ..., 9] = data['delta_Uz']
    hdf5_file['sim_data'][sim, ..., 10] = data['delta_p_rgh']

    hdf5_file['y_bot_bound'][sim, ..., 0] = data_y_bot['Cx']
    hdf5_file['y_bot_bound'][sim, ..., 1] = data_y_bot['Cy']
    hdf5_file['y_bot_bound'][sim, ..., 2] = data_y_bot['Cz']

    hdf5_file['z_bot_bound'][sim, ..., 0] = data_z_bot['Cx']
    hdf5_file['z_bot_bound'][sim, ..., 1] = data_z_bot['Cy']
    hdf5_file['z_bot_bound'][sim, ..., 2] = data_z_bot['Cz']

    hdf5_file['y_top_bound'][sim, ..., 0] = data_y_top['Cx']
    hdf5_file['y_top_bound'][sim, ..., 1] = data_y_top['Cy']
    hdf5_file['y_top_bound'][sim, ..., 2] = data_y_top['Cz']

    hdf5_file['z_top_bound'][sim, ..., 0] = data_z_top['Cx']
    hdf5_file['z_top_bound'][sim, ..., 1] = data_z_top['Cy']
    hdf5_file['z_top_bound'][sim, ..., 2] = data_z_top['Cz']
    
    hdf5_file['obst_bound'][sim, ..., 0] = data_obst['Cx']
    hdf5_file['obst_bound'][sim, ..., 1] = data_obst['Cy']
    hdf5_file['obst_bound'][sim, ..., 2] = data_obst['Cz']

    hdf5_file.flush()

def main():
    first_sim = 0
    last_sim = 4
    first_t = 0
    last_t = 19
    num_time_steps = last_t - first_t + 1
    hdf5_path = 'dataset_heat_5sim20t.hdf5'
    path_to_sims = 'simulations/'
    max_n_cells_sim = int(6e5)
    max_n_cells_patch = 150000

    avance_list = [1] * 20

    hdf5_file = create_hdf5_file(hdf5_path, int(last_sim-first_sim +1), num_time_steps, max_n_cells_sim, max_n_cells_patch)

    for sim in tqdm(range(first_sim, last_sim + 1)):
        process_simulation(sim, path_to_sims, avance_list, hdf5_file, first_t, last_t, max_n_cells_sim, max_n_cells_patch)

    hdf5_file.close()

if __name__ == "__main__":
    main()
