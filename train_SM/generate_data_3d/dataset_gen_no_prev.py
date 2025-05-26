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

def extract_simulation_data_bound(serial, patch, path_to_sims, total_times, deltat, avance, max):
    data = { 'Cx': [], 'Cy': [], 'Cz': [] }
    for entity in ['Cx', 'Cy', 'Cz']:
      for time in range(int(total_times)):

        #t = deltat*(time*avance+1)
        t = round(deltat*(time*avance+1)*1000000)/1000000
        if t % 1 == 0 :
          t = round(t)

        #boundary patch:
        path = f"{path_to_sims}/{serial}/VTK/{patch}/{patch}_{t}.vtk"
        print(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            print('Probably it corresponds to a steady state simulation - not saving more data for this simulation')
            continue
        mesh_cell_data = pv.read(path).cell_data

        data[entity].append(padding(mesh_cell_data[entity] ,max))
    return data

def extract_simulation_data(serial, path_to_sims, total_times, deltat, avance, max):
    data = {'Ux': [], 'Uy': [], 'Uz': [],
            'Cx': [], 'Cy': [], 'Cz': [],
            'delta_Ux': [], 'delta_Uy': [], 'delta_Uz': [],
            'p': [], 'delta_p': [],
            }

    for entity in ['U_non_cons', 'Cx', 'Cy', 'Cz', 'delta_U', 'delta_p', 'p']:
      for time in range(int(total_times)):

        #t = round(deltat*(time*avance+1)*100)/100
        t = round(deltat*(time*avance+1)*1000000)/1000000
        if t % 1 == 0 :
          t = round(t)
        path = f"{path_to_sims}/{serial}/VTK/{serial}_{t}.vtk"
        print(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            print('Probably it corresponds to a steady state simulation - not saving more data for this simulation')
            continue
        mesh_cell_data = pv.read(path).cell_data

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

def create_hdf5_file(hdf5_path, num_sims_actual, total_times, max_n_cells_sim, max_n_cells_patch):
    train_shape = (num_sims_actual, int(total_times), max_n_cells_sim, 17)
    #train_shape = (num_sims_actual, int(total_times), 600000, 12)
    top_shape = (num_sims_actual, int(total_times), max_n_cells_patch, 3)
    obst_shape = (num_sims_actual, int(total_times), max_n_cells_patch, 3)

    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset('sim_data', train_shape, np.float32)
    hdf5_file.create_dataset('y_bot_bound', top_shape, np.float32)
    hdf5_file.create_dataset('z_bot_bound', top_shape, np.float32)
    hdf5_file.create_dataset('y_top_bound', top_shape, np.float32)
    hdf5_file.create_dataset('z_top_bound', top_shape, np.float32)
    hdf5_file.create_dataset('obst_bound', obst_shape, np.float32)

    return hdf5_file

def process_simulation(sim, path_to_sims, avance_list, hdf5_file, total_times, max_sim, max_patch):

    # Read the deltaT_write
    control_dict_path = f"{path_to_sims}/{sim}/system/controlDict"
    with open(control_dict_path, 'r') as file:
        for line in file:
            if 'writeInterval' in line:
                deltat = float(line.split()[1].strip(';'))
                break       

    avance = avance_list[sim]
    data = extract_simulation_data(sim, path_to_sims, total_times, deltat, avance, max_sim)
    data_obst = extract_simulation_data_bound(sim, 'geometry_obstacle', path_to_sims, total_times, deltat, avance, max_patch)
    data_z_bot = extract_simulation_data_bound(sim, 'geometry_z_bot', path_to_sims, total_times, deltat, avance, max_patch)
    data_y_bot = extract_simulation_data_bound(sim, 'geometry_y_bot', path_to_sims, total_times, deltat, avance, max_patch)
    data_z_top = extract_simulation_data_bound(sim, 'geometry_z_top', path_to_sims, total_times, deltat, avance, max_patch)
    data_y_top = extract_simulation_data_bound(sim, 'geometry_y_top', path_to_sims, total_times, deltat, avance, max_patch)

    hdf5_file['sim_data'][sim, ..., 0] = data['Ux']
    hdf5_file['sim_data'][sim, ..., 1] = data['Uy']
    hdf5_file['sim_data'][sim, ..., 2] = data['Uz']
    hdf5_file['sim_data'][sim, ..., 3] = data['p']
    hdf5_file['sim_data'][sim, ..., 4] = data['Cx']
    hdf5_file['sim_data'][sim, ..., 5] = data['Cy']
    hdf5_file['sim_data'][sim, ..., 6] = data['Cz']
    hdf5_file['sim_data'][sim, ..., 7] = data['delta_Ux']
    hdf5_file['sim_data'][sim, ..., 8] = data['delta_Uy']
    hdf5_file['sim_data'][sim, ..., 9] = data['delta_Uz']
    hdf5_file['sim_data'][sim, ..., 10] = data['delta_p']

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
    num_sims_actual = 5
    total_times = 20
    hdf5_path = 'dataset_plate_piso_16May_20t_5sim.hdf5'
    path_to_sims = 'simulation_data'
    max_n_cells_sim = 750000
    max_n_cells_patch = 100000

    avance_list = [1] * 10 #* 50

    hdf5_file = create_hdf5_file(hdf5_path, num_sims_actual, total_times, max_n_cells_sim, max_n_cells_patch)

    for sim in tqdm(range(num_sims_actual)):
        process_simulation(sim, path_to_sims, avance_list, hdf5_file, total_times, max_n_cells_sim, max_n_cells_patch)

    hdf5_file.close()

if __name__ == "__main__":
    main()
