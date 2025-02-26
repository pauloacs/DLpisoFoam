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
    data = { 'Cx': [], 'Cy': [] }
    for entity in ['Cx', 'Cy']:
      for time in range(int(total_times)):
        vtk_list = []
        t = round(deltat*(time*avance+1)*100)/100
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
        vtk_list.append(mesh_cell_data[entity])

        data[entity].append(padding(np.concatenate(vtk_list) ,max))
    return data

def extract_simulation_data(serial, path_to_sims, total_times, deltat, avance, max):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [], 'delta_Ux': [], 'delta_Uy': [], 'delta_p': [], 'delta_Ux_prev': [], 'delta_Uy_prev': [], 'delta_p_prev': []}
    for entity in ['p', 'U_non_cons', 'Cx', 'Cy', 'delta_U', 'delta_p', 'delta_U_prev', 'delta_p_prev']:
      for time in range(int(total_times)):
        vtk_list = []
        t = round(deltat*(time*avance+1)*100)/100
        if t % 1 == 0 :
          t = round(t)
        path = f"{path_to_sims}/{serial}/VTK/{serial}_{t}.vtk"
        print(path)
        if not os.path.exists(path):
            print(f"Path {path} does not exist")
            print('Probably it corresponds to a steady state simulation - not saving more data for this simulation')
            continue
        mesh_cell_data = pv.read(path).cell_data
        vtk_list.append(mesh_cell_data[entity])

        if entity == 'U_non_cons':
            data['Ux'].append(padding(np.concatenate(vtk_list)[:,0],max))
            data['Uy'].append(padding(np.concatenate(vtk_list)[:,1],max))
        elif entity == 'delta_U':
            data['delta_Ux'].append(padding(np.concatenate(vtk_list)[:,0],max))
            data['delta_Uy'].append(padding(np.concatenate(vtk_list)[:,1],max))
        elif entity == 'delta_U_prev':
            data['delta_Ux_prev'].append(padding(np.concatenate(vtk_list)[:,0],max))
            data['delta_Uy_prev'].append(padding(np.concatenate(vtk_list)[:,1],max))
        else:
            extent = 0, 3, 0, 1
            if entity == 'delta_p_prev':
              print(vtk_list)
            data[entity].append(padding(np.concatenate(vtk_list), max))
    return data

def create_hdf5_file(hdf5_path, num_sims_actual, total_times):
    train_shape = (num_sims_actual, int(total_times), 600000, 11)
    top_shape = (num_sims_actual, int(total_times), 50000, 2)
    obst_shape = (num_sims_actual, int(total_times), 50000, 2)

    hdf5_file = h5py.File(hdf5_path, mode='w')
    hdf5_file.create_dataset('sim_data', train_shape, np.float32)
    hdf5_file.create_dataset('top_bound', top_shape, np.float32)
    hdf5_file.create_dataset('obst_bound', obst_shape, np.float32)

    return hdf5_file

def process_simulation(sim, path_to_sims, deltat_write, avance_list, hdf5_file, total_times, max_sim, max_patch):
    deltat = deltat_write[sim]
    avance = avance_list[sim]

    data = extract_simulation_data(sim, path_to_sims, total_times, deltat, avance, max_sim)
    data_top = extract_simulation_data_bound(sim, 'geometry_top', path_to_sims, total_times, deltat, avance, max_patch)
    data_obst = extract_simulation_data_bound(sim, 'geometry_obstacle', path_to_sims, total_times, deltat, avance, max_patch)

    hdf5_file['sim_data'][sim, ..., 0] = data['Ux']
    hdf5_file['sim_data'][sim, ..., 1] = data['Uy']
    hdf5_file['sim_data'][sim, ..., 2] = data['p']
    hdf5_file['sim_data'][sim, ..., 3] = data['Cx']
    hdf5_file['sim_data'][sim, ..., 4] = data['Cy']
    hdf5_file['sim_data'][sim, ..., 5] = data['delta_Ux']
    hdf5_file['sim_data'][sim, ..., 6] = data['delta_Uy']
    hdf5_file['sim_data'][sim, ..., 7] = data['delta_p']
    hdf5_file['sim_data'][sim, ..., 5] = data['delta_Ux']
    hdf5_file['sim_data'][sim, ..., 6] = data['delta_Uy']
    hdf5_file['sim_data'][sim, ..., 7] = data['delta_p']
    hdf5_file['sim_data'][sim, ..., 8] = data['delta_Ux_prev']
    hdf5_file['sim_data'][sim, ..., 9] = data['delta_Uy_prev']
    hdf5_file['sim_data'][sim, ..., 10] = data['delta_p_prev']

    hdf5_file['top_bound'][sim, ..., 0] = data_top['Cx']
    hdf5_file['top_bound'][sim, ..., 1] = data_top['Cy']
    hdf5_file['obst_bound'][sim, ..., 0] = data_obst['Cx']
    hdf5_file['obst_bound'][sim, ..., 1] = data_obst['Cy']

    hdf5_file.flush()

def main():
    num_sims_actual = 50
    total_times = 5
    hdf5_path = 'datasets/dataset_plate_big.hdf5'
    path_to_sims = '/homes/up201605045/cetftstg/deltaU_to_deltaP/plate_sims/simulation_data_additional/'
    max_sim = 600000
    max_patch = 50000

    deltat_write = [0.5] * 10 + [0.75] * 10 + [1] * 10 + [1.25] * 10 + [1.5] * 10
    avance_list = [1] * 50

    hdf5_file = create_hdf5_file(hdf5_path, num_sims_actual, total_times)

    for sim in tqdm(range(num_sims_actual)):
        process_simulation(sim, path_to_sims, deltat_write, avance_list, hdf5_file, total_times, max_sim, max_patch)

    hdf5_file.close()

if __name__ == "__main__":
    main()