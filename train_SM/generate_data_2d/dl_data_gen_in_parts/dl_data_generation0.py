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

def extract_simulation_data_bound(serial, patch, path_to_sims):
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
        mesh_cell_data = pv.read(path).cell_arrays
        vtk_list.append(mesh_cell_data[entity])

        data[entity].append(padding(np.concatenate(vtk_list) ,max))
    return data

def extract_simulation_data(serial, path_to_sims):
    data = {'Ux': [], 'Uy': [], 'p': [], 'Cx': [], 'Cy': [], 'delta_Ux': [], 'delta_Uy': [], 'delta_p': []}
    for entity in ['p', 'U_non_cons', 'Cx', 'Cy', 'delta_U', 'delta_p']:
      for time in range(int(total_times)):
        vtk_list = []
        t = round(deltat*(time*avance+1)*100)/100
        if t % 1 == 0 :
          t = round(t)
        path = f"{path_to_sims}/{serial}/VTK/{serial}_{t}.vtk"
        print(path)
        mesh_cell_data = pv.read(path).cell_arrays
        vtk_list.append(mesh_cell_data[entity])

        if entity == 'U_non_cons':
            data['Ux'].append(padding(np.concatenate(vtk_list)[:,0],max))
            data['Uy'].append(padding(np.concatenate(vtk_list)[:,1],max))
        elif entity == 'delta_U':
            data['delta_Ux'].append(padding(np.concatenate(vtk_list)[:,0],max))
            data['delta_Uy'].append(padding(np.concatenate(vtk_list)[:,1],max))
        else:
            extent = 0, 3, 0, 1
            data[entity].append(padding(np.concatenate(vtk_list), max))
    return data

num_sims_actual = 10

total_times = 100
train_shape = (num_sims_actual, int(total_times) , 600000 , 8)

hdf5_path = 'datasets/dataset_plate_deltas_0-10sim100t.hdf5'
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset('sim_data', train_shape, np.float32)

top_shape = (num_sims_actual, int(total_times) , 50000 , 2)
hdf5_file.create_dataset('top_bound', top_shape, np.float32)

obst_shape = (num_sims_actual, int(total_times) , 50000 , 2)
hdf5_file.create_dataset('obst_bound', obst_shape, np.float32)


count = 0 
deltat_write = [0.5] * 10 + [0.75]*10 + [1]*10  + [1.25]*10 + [1.5]*10
avance_list = [1]*50

path_to_sims = '/homes/up201605045/cetftstg/deltaU_to_deltaP/plate_sims/simulation_data/'

for sim in tqdm(range(num_sims_actual)):
  sim+=0
  deltat = deltat_write[sim]
  init_time = deltat
  avance = avance_list[sim]
  max = 600000
  data = extract_simulation_data(sim, path_to_sims)
  max = 50000
  data_top = extract_simulation_data_bound(sim,'geometry_top', path_to_sims)
  data_obst = extract_simulation_data_bound(sim,'geometry_obstacle', path_to_sims)

  hdf5_file['sim_data'][count, ..., 0] = data['Ux']
  hdf5_file['sim_data'][count, ..., 1] = data['Uy']
  hdf5_file['sim_data'][count, ..., 2] = data['p']
  hdf5_file['sim_data'][count, ..., 3] = data['Cx']
  hdf5_file['sim_data'][count, ..., 4] = data['Cy']
  hdf5_file['sim_data'][count, ..., 5] = data['delta_Ux']
  hdf5_file['sim_data'][count, ..., 6] = data['delta_Uy']
  hdf5_file['sim_data'][count, ..., 7] = data['delta_p']

  hdf5_file['top_bound'][count, ..., 0] = data_top['Cx']
  hdf5_file['top_bound'][count, ..., 1] = data_top['Cy']
  hdf5_file['obst_bound'][count, ..., 0] = data_obst['Cx']
  hdf5_file['obst_bound'][count, ..., 1] = data_obst['Cy']

  count = count + 1

hdf5_file.close()
