import numpy as np
import tables
import pickle as pk
from math import ceil
import tensorly as tl
from tensorly.decomposition import tucker

from .utils import io_operations as utils_io
from .utils import domain_geometry as utils_geo
from .utils import sampling as utils_sampling
from .utils import data_processing as utils_data

import os

import dask.distributed

class CFDDataProcessor:
  """
  """
  def __init__(
        self,
        grid_res: float,
        block_size: int,
        original_dataset_path: str,
        n_samples_per_frame: int,
        first_sim: int,
        last_sim: int,
        first_t: int,
        last_t: int,
        standardization_method: str,
        chunk_size: int,
        gridded_h5_fn: str
    ):

    self.grid_res = grid_res
    self.block_size = block_size
    self.original_dataset_path = original_dataset_path
    self.n_samples_per_frame = n_samples_per_frame
    self.first_sim = first_sim
    self.last_sim = last_sim
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.chunk_size = chunk_size
    self.gridded_h5_filenames = utils_io.get_gridded_h5_filenames(
      gridded_h5_fn,
      first_sim,
      last_sim
      )

  def write_gridded_simulation_data(self) -> None:
    """
    Write CFD mesh data to a regular grid and save to HDF5 file.
    """

    for sim_i in range(self.first_sim, self.last_sim + 1):

      gridded_h5_fn_sim = self.gridded_h5_filenames[sim_i - self.first_sim]
      print(f'########## Writting CFD mesh data to a grid -> {gridded_h5_fn_sim} ############')
      NUM_COLUMNS = 5

      with tables.open_file(gridded_h5_fn_sim, mode='w') as file:

        atom = tables.Float32Atom()

        _, limits = utils_io.read_cells_and_limits(self.original_dataset_path, sim_i, self.first_t, self.last_t)

        self.grid_shape_z = int(round((limits['z_max'] - limits['z_min']) / self.grid_res))
        self.grid_shape_y = int(round((limits['y_max'] - limits['y_min']) / self.grid_res))
        self.grid_shape_x = int(round((limits['x_max'] - limits['x_min']) / self.grid_res))

        file.create_earray(file.root, 'data', atom, (0, self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, NUM_COLUMNS))

      print(f"\nProcessing sim {sim_i+1}/{self.last_sim - self.first_sim + 1}\n", flush=True)
      self.write_sim_fields(sim_i, gridded_h5_fn_sim)

  def write_sim_fields(self, sim_i, gridded_h5_fn_sim):
    """
    Write simulation fields for a single simulation index to the grid file.
    """

    sim_data_ts, limits = utils_io.read_cells_and_limits(
      self.original_dataset_path,
      sim_i,
      self.first_t,
      self.last_t
      )

    sim_data_t0 = sim_data_ts[0, :, :]
    boundaries = utils_io.read_boundaries(sim_i, self.original_dataset_path)

    X0, Y0, Z0 = utils_data.create_uniform_grid(limits, self.grid_res)
    xyz0 = np.concatenate(
      (np.expand_dims(X0, axis=1),
       np.expand_dims(Y0, axis=1),
       np.expand_dims(Z0, axis=1)),
        axis=-1
      )
    
    points = sim_data_t0[..., 4:7]

    vert, weights = utils_data.interp_weights(points, xyz0, interp_method='IDW')
    domain_bool, sdf = utils_geo.domain_dist(boundaries, xyz0, self.grid_res)

    x0 = np.min(X0)
    y0 = np.min(Y0)
    z0 = np.min(Z0)
    dx = self.grid_res
    dy = self.grid_res
    dz = self.grid_res

    indices = np.zeros((X0.shape[0], 3))
    obst_bool = np.zeros((self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 1))
    sdfunct = obst_bool.copy()

    delta_p = sim_data_t0[..., 10:11]
    p_interp = utils_data.interpolate_fill(delta_p, vert, weights)

    for (step, x_y_z) in enumerate(xyz0):
      if domain_bool[step] * (~np.isnan(p_interp[step])):
        ii = int(round((x_y_z[..., 2] - z0) / dz))
        jj = int(round((x_y_z[..., 1] - y0) / dy))
        kk = int(round((x_y_z[..., 0] - x0) / dx))
        indices[step, 0] = ii
        indices[step, 1] = jj
        indices[step, 2] = kk
        sdfunct[ii, jj, kk, :] = sdf[step]
        obst_bool[ii, jj, kk, :] = int(1)

    indices = indices.astype(int)
    self.stationary_ts = 0
    for j in range(sim_data_ts.shape[0]):
      data = sim_data_ts[j, :, :]
      self.write_time_step_fields(j, data, vert, weights, indices, sdfunct, gridded_h5_fn_sim)
      if self.stationary_ts > 5:
        print('This simulation is stationary, ignoring it...')
        break

  def write_time_step_fields(self, j, data_limited, vert, weights, indices, sdfunct, gridded_h5_fn_sim):
    """
    Write a single time step's fields to the grid file.
    """
    Ux = data_limited[..., 0:1]
    Uy = data_limited[..., 1:2]
    Uz = data_limited[..., 2:3]
    delta_p = data_limited[..., 10:11]
    delta_Ux = data_limited[..., 7:8]
    delta_Uy = data_limited[..., 8:9]
    delta_Uz = data_limited[..., 9:10]

    U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy) + np.square(Uz)))
    deltaU_max_norm = np.max(np.sqrt(np.square(delta_Ux) + np.square(delta_Uy) + np.square(delta_Uz)))

    threshold = 1e-4
    print(f"deltaU_max_norm = {deltaU_max_norm}")
    print(f"U_max_norm      = {U_max_norm}")
    irrelevant_ts = (deltaU_max_norm / U_max_norm) < threshold or deltaU_max_norm < 1e-6 or U_max_norm < 1e-6

    if irrelevant_ts:
      print(f"\n\n Irrelevant time step, skipping it...")
      self.stationary_ts += 1
      return 0

    delta_p_adim = delta_p / pow(U_max_norm, 2.0)
    delta_Ux_adim = delta_Ux / U_max_norm
    delta_Uy_adim = delta_Uy / U_max_norm
    delta_Uz_adim = delta_Uz / U_max_norm

    delta_p_interp = utils_data.interpolate_fill(delta_p_adim, vert, weights)
    delta_Ux_interp = utils_data.interpolate_fill(delta_Ux_adim, vert, weights)
    delta_Uy_interp = utils_data.interpolate_fill(delta_Uy_adim, vert, weights)
    delta_Uz_interp = utils_data.interpolate_fill(delta_Uz_adim, vert, weights)

    filter_tuple = (2, 2, 2)
    grid = np.zeros(shape=(self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 5))
    grid[:, :, :, 0:1][tuple(indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0], 1)
    grid[:, :, :, 1:2][tuple(indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0], 1)
    grid[:, :, :, 2:3][tuple(indices.T)] = delta_Uz_interp.reshape(delta_Uz_interp.shape[0], 1)
    grid[:, :, :, 3:4] = sdfunct
    grid[:, :, :, 4:5][tuple(indices.T)] = delta_p_interp.reshape(delta_p_interp.shape[0], 1)

    grid[np.isnan(grid)] = 0

    import matplotlib.pyplot as plt
    # Save plots for all variables
    os.makedirs('plots_debug', exist_ok=True)
    
    var_names = ['delta_Ux', 'delta_Uy', 'delta_Uz', 'sdf', 'delta_p']
    
    for var_idx in range(5):
      # Plot slice through middle of grid (z-x plane at middle y)
      plt.figure(figsize=(10, 6))
      plt.imshow(grid[1:-1, int(grid.shape[1] / 2), 1:-1, var_idx], cmap='jet')
      plt.colorbar(label=var_names[var_idx])
      plt.title(f'{var_names[var_idx]} - Z-X slice (middle Y)')
      plt.xlabel('X')
      plt.ylabel('Z')
      plt.savefig(f'plots_debug/{var_names[var_idx]}_zx_slice_t{j + self.first_t}.png')
      plt.close()
      
      # Plot slice through middle of grid (y-x plane at middle z)
      plt.figure(figsize=(10, 6))
      plt.imshow(grid[int(grid.shape[0] / 2), :, :, var_idx], cmap='jet')
      plt.colorbar(label=var_names[var_idx])
      plt.title(f'{var_names[var_idx]} - Y-X slice (middle Z)')
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.savefig(f'plots_debug/{var_names[var_idx]}_yx_slice_t{j + self.first_t}.png')
      plt.close()

    print(f"Writting t{j + self.first_t} to {gridded_h5_fn_sim}", flush=True)
    with tables.open_file(gridded_h5_fn_sim, mode='a') as file:
      file.root.data.append(np.array(np.expand_dims(grid, axis=0), dtype='float32'))


class FeatureExtractAndWrite:
  
  def __init__(
        self,
        grid_res: float,
        block_size: int,
        original_dataset_path: str,
        n_samples_per_frame: int,
        first_sim: int,
        last_sim: int,
        first_t: int,
        last_t: int,
        standardization_method: str,
        chunk_size: int,
        gridded_h5_fn: str,
        flatten_data: bool
    ):

    self.grid_res = grid_res
    self.block_size = block_size
    self.n_samples_per_frame = n_samples_per_frame
    self.first_sim = first_sim
    self.last_sim = last_sim
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.chunk_size = chunk_size
    self.original_dataset_path = original_dataset_path
    self.gridded_h5_filenames = utils_io.get_gridded_h5_filenames(
      gridded_h5_fn,
      first_sim,
      last_sim
      )
    self.flatten_data = flatten_data

  def write_features_to_h5(self, filename_flat: str, chunk_size: int = 500, ranks: int = 4) -> None:
    with open('sample_indices_per_sim_per_time.pkl', 'rb') as f:
      sample_indices_per_sim_per_time = pk.load(f)

    total_times = self.last_t - self.first_t
    if chunk_size > self.n_samples_per_frame:
      n_times_per_chunk = chunk_size // self.n_samples_per_frame
      n_chunks_per_sim = ceil(total_times / n_times_per_chunk)
      n_sub_chunks = 1
    else:
      n_sub_chunks = ceil(self.n_samples_per_frame / chunk_size)
      n_times_per_chunk = 1
      n_chunks_per_sim = total_times

    self.transform_and_write_blocks_to_features(
      filename_flat, n_chunks_per_sim, n_times_per_chunk, n_sub_chunks, ranks, sample_indices_per_sim_per_time
    )

  def sample_blocks_chunked(
    self,
    sim: int,
    t_start: int,
    t_end: int,
    i_chunk=None,
    n_chunks=False,
    sample_indices=None):
    """
    """
    inputs_u_list = []
    inputs_obst_list = []
    outputs_list = []
    use_subchunks = n_chunks > 1
    count = 0

    sim = sim - self.first_sim
    
    for time in range(t_start, t_end):
      with tables.open_file(self.gridded_h5_filenames[sim], mode='r') as f:
        grid = f.root.data[time, :, :, :, :]

      ZYX_indices = sample_indices[sim][time]

      if use_subchunks:
        elements_per_sub_chunk = ceil(ZYX_indices.shape[0] / n_chunks)
        i_element_start = i_chunk * elements_per_sub_chunk
        i_element_end = (i_chunk + 1) * elements_per_sub_chunk
        ZYX_indices = ZYX_indices[i_element_start:i_element_end]

      for [ii, jj, kk] in ZYX_indices:
        i_idx_fist = int(ii - self.block_size / 2)
        i_idx_last = int(ii + self.block_size / 2)
        j_idx_first = int(jj - self.block_size / 2)
        j_idx_last = int(jj + self.block_size / 2)
        k_idx_first = int(kk - self.block_size / 2)
        k_idx_last = int(kk + self.block_size / 2)

        inputs_u_sample = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 0:3]
        inputs_obst_sample = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 3:4]
        outputs_sample = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 4:5]

        if not ((inputs_u_sample == 0).all() and (outputs_sample == 0).all()):
          inputs_u_list.append(inputs_u_sample)
          inputs_obst_list.append(inputs_obst_sample)
          outputs_list.append(outputs_sample)
        else:
          count += 1

    inputs_u = np.array(inputs_u_list)
    inputs_obst = np.array(inputs_obst_list)
    outputs = np.array(outputs_list)

    for step in range(outputs.shape[0]):
      outputs[step, ...][inputs_obst[step, ...] != 0] -= np.mean(outputs[step, ...][inputs_obst[step, ...] != 0])

    array = np.c_[inputs_u, inputs_obst, outputs]
    reshaped_array = array.reshape(array.shape[0], -1)
    unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
    unique_array = array[unique_indices]
    inputs_u, inputs_obst, outputs = unique_array[..., 0:3], unique_array[..., 3:4], unique_array[..., 4:5]

    if count > 0:
      print(f'    {count} blocks discarded')

    return inputs_u, inputs_obst, outputs

  def get_representative_factors(self, blocks_data: np.ndarray, ranks):
    spatial_ranks = (ranks, ranks, ranks)
    inputs_u, inputs_obst, outputs = blocks_data
    chunk_size = inputs_u.shape[0]
    print(f'ACTUAL Chunk size: {chunk_size}')

    velocity = inputs_u / [self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz]
    obstacle = inputs_obst / self.max_abs_dist
    pressure = outputs[..., 0] / self.max_abs_delta_p

    print(f"Calculating representative Tucker factors ({chunk_size} samples) ...")
    input_tensor = np.concatenate([velocity, obstacle], axis=-1)
    last_rank = 4
    _, self.input_factors = tucker(input_tensor, rank=(chunk_size,) + spatial_ranks + (last_rank,))
    _, self.output_factors = tucker(pressure, rank=(chunk_size,) + spatial_ranks)

    with open('tucker_factors.pkl', 'wb') as f:
      pk.dump({'input_factors': self.input_factors, 'output_factors': self.output_factors}, f)

  def transform_data_with_tucker(self, blocks_data: np.ndarray, client) -> np.ndarray:
    inputs_u, inputs_obst, outputs = blocks_data
    chunk_size = inputs_u.shape[0]
    print(f'ACTUAL Chunk size: {chunk_size}')

    velocity = inputs_u / [self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz]
    obstacle = inputs_obst / self.max_abs_dist
    pressure = outputs[..., 0] / self.max_abs_delta_p

    print("Transforming data using precomputed Tucker factors...")
    input_tensor = np.concatenate([velocity, obstacle], axis=-1)
    input_core = tl.tenalg.multi_mode_dot(input_tensor, self.input_factors[1:], modes=[1, 2, 3, 4], transpose=True)
    output_core = tl.tenalg.multi_mode_dot(pressure, self.output_factors[1:], modes=[1, 2, 3], transpose=True)

    if self.flatten_data:
        input_core = input_core.reshape(chunk_size, -1)
        output_core = output_core.reshape(chunk_size, -1)

    return input_core, output_core

  def transform_and_write_blocks_to_features(
    self,
    filename_flat: str,
    chunks_per_sim: int,
    n_times_per_chunk: int,
    n_sub_chunks: int,
    ranks: int,
    sample_indices_per_sim_per_time
  ) -> None:
    with tables.open_file(filename_flat, mode='w') as file:
        atom = tables.Float32Atom()
        if self.flatten_data:
          input_shape = (0, ranks * ranks * ranks * 4)
          output_shape = (0, ranks * ranks * ranks)
        else:
          input_shape = (0, ranks, ranks, ranks, 4)
          output_shape = (0, ranks, ranks, ranks)

        file.create_earray(file.root, 'inputs', atom, input_shape)
        file.create_earray(file.root, 'outputs', atom, output_shape)

    client = dask.distributed.Client(processes=False)

    # Load maxs
    maxs = np.loadtxt('maxs')
    self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz, self.max_abs_dist, self.max_abs_delta_p = maxs

    # Compute representative factors once for all sims
    N_representative = 7500 #2500
    N_representative_per_sim = int(N_representative / (self.last_sim - self.first_sim + 1) / (self.last_t - self.first_t + 1))
    sample_indices_per_sim_per_time_representative = utils_sampling.define_sample_indexes(
      N_representative_per_sim,
      self.block_size,
      self.grid_res,
      self.first_sim,
      self.last_sim,
      self.first_t,
      self.last_t,
      self.original_dataset_path
    )

    all_inputs_u = []
    all_inputs_obst = []
    all_outputs = []

    for sim in range(self.first_sim, self.last_sim + 1):
        inputs_u, inputs_obst, outputs = self.sample_blocks_chunked(
            sim,
            t_start=self.first_t,
            t_end=self.last_t,
            i_chunk=None,
            n_chunks=False,
            sample_indices=sample_indices_per_sim_per_time_representative
            )
        all_inputs_u.append(inputs_u)
        all_inputs_obst.append(inputs_obst)
        all_outputs.append(outputs)

    all_inputs_u = np.concatenate(all_inputs_u)
    all_inputs_obst = np.concatenate(all_inputs_obst)
    all_outputs = np.concatenate(all_outputs)

    representative_blocks = (all_inputs_u, all_inputs_obst, all_outputs)
    self.get_representative_factors(representative_blocks, ranks=ranks)

    for sim in range(self.first_sim, self.last_sim + 1):
        print(f'Transforming data from sim {sim + 1}/[{self.first_sim + 1}, {self.last_sim + 1}]...')
        for i_chunk in range(chunks_per_sim):
            for sub_chunk in range(n_sub_chunks):
                print(f' -Sampling block data for chunk {i_chunk + 1}/{chunks_per_sim} - subchunk {sub_chunk + 1}/{n_sub_chunks}', flush=True)
                blocks_data = self.sample_blocks_chunked(
                  sim,
                  t_start=i_chunk * n_times_per_chunk,
                  t_end=min((i_chunk + 1) * n_times_per_chunk, self.last_t),
                  i_chunk=sub_chunk,
                  n_chunks=n_sub_chunks,
                  sample_indices=sample_indices_per_sim_per_time
                )

                print(f' -Transforming grid data to tensor cores for chunk {i_chunk + 1}/{chunks_per_sim} - subchunk {sub_chunk + 1}/{n_sub_chunks}', flush=True)
                in_features, out_features = self.transform_data_with_tucker(blocks_data, client)

                with tables.open_file(filename_flat, mode='a') as f:
                    f.root.inputs.append(np.array(in_features))
                    f.root.outputs.append(np.array(out_features))

    client.close()
