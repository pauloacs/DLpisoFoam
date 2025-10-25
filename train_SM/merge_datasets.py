import h5py
import numpy as np

dataset1_path = 'dataset_plate_buoyant_4Jun_20t_6sim.hdf5'
dataset2_path = 'dataset_plate_buoyant_4Jun_20t_6sim.hdf5'
merged_dataset_path = 'dataset_plate_buoyant_6Jun_20t_9sim.h5'

num_sims_1 = 5
num_sims_2 = 4

with h5py.File(dataset1_path, 'r') as f1, \
    h5py.File(dataset2_path, 'r') as f2:

    # Get shapes and dtypes
    sim_data_shape = f1['sim_data'].shape
    bound_shape = f1['y_bot_bound'].shape

    total_times = sim_data_shape[1]
    max_n_cells_sim = sim_data_shape[2]
    max_n_cells_patch = bound_shape[2]

    # Create merged file
    with h5py.File(merged_dataset_path, 'w') as f_out:
       f_out.create_dataset('sim_data', (num_sims_1 + num_sims_2, total_times, max_n_cells_sim, 11), dtype='float32')
       for name in ['y_bot_bound', 'z_bot_bound', 'y_top_bound', 'z_top_bound', 'obst_bound']:
          f_out.create_dataset(name, (num_sims_1 + num_sims_2, total_times, max_n_cells_patch, 3), dtype='float32')

       # Copy from dataset1
       f_out['sim_data'][:num_sims_1] = f1['sim_data'][:num_sims_1]
       for name in ['y_bot_bound', 'z_bot_bound', 'y_top_bound', 'z_top_bound', 'obst_bound']:
          f_out[name][:num_sims_1] = f1[name][:num_sims_1]

       # Copy from dataset2
       f_out['sim_data'][num_sims_1:num_sims_1+num_sims_2] = f2['sim_data'][:num_sims_2]
       for name in ['y_bot_bound', 'z_bot_bound', 'y_top_bound', 'z_top_bound', 'obst_bound']:
          f_out[name][num_sims_1:num_sims_1+num_sims_2] = f2[name][:num_sims_2]

print(f"Merged {num_sims_1} sims from {dataset1_path} and {num_sims_2} sims from {dataset2_path} into {merged_dataset_path}")