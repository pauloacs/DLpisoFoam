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

        _, limits = utils_io.read_cells_and_limits(self.original_dataset_path, sim_i, self.first_t, self.last_t, self.grid_res)

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
      self.last_t,
      self.grid_res
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
        original_dataset_path: str = None,
        n_samples_per_frame: int = 1000,
        first_sim: int = 0,
        last_sim: int = 0,
        first_t: int = 0,
        last_t: int = 0,
        standardization_method: str = 'minmax',
        chunk_size: int = 500,
        ranks: int = 4,
        gridded_h5_fn: str = 'gridded_data.h5',
        sample_indices_fn: str = 'sample_indices_per_sim_per_time.pkl',
        tucker_factors_fn: str = 'tucker_factors.pkl',
        flatten_data: bool = False,
        maxs_list: list = None
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
    self.ranks = ranks
    self.original_dataset_path = original_dataset_path
    self.gridded_h5_filenames = utils_io.get_gridded_h5_filenames(
      gridded_h5_fn,
      first_sim,
      last_sim
      )
    self.sample_indices_fn = sample_indices_fn
    self.tucker_factors_fn = tucker_factors_fn
    self.flatten_data = flatten_data
    self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz, self.max_abs_dist, self.max_abs_delta_p = maxs_list

    with open(self.sample_indices_fn, 'rb') as f:
      self.sample_indices_per_sim_per_time = pk.load(f)

  def __call__(self, core_data_fn: str,  n_representative_blocks=7500, compute_tucker_factors=True) -> None:

    """
    Extract features using Tucker decomposition and write to core data file.
    If tucker_factors are provided, use them; otherwise compute representative factors from the data.
    """

    client = dask.distributed.Client(processes=False)
    
    if compute_tucker_factors:
        input_factors, output_factors = self.compute_representative_factors(
                                          self.sample_indices_per_sim_per_time,
                                          self.ranks,
                                          n_representative_blocks
                                        )
    elif os.path.exists(self.tucker_factors_fn):
      with open(self.tucker_factors_fn, 'rb') as f:
          tucker_factors = pk.load(f)
      input_factors = tucker_factors['input_factors']
      output_factors = tucker_factors['output_factors']
    else:
        raise FileNotFoundError(f"Tucker factors file {self.tucker_factors_fn} not found. "
                                f"Set compute_tucker_factors=True to compute them from the data.")
        
    self.transform_and_write_blocks_to_core_data(
      core_data_fn,
      input_factors,
      output_factors,
      self.chunk_size,
      self.sample_indices_per_sim_per_time
    )
    client.close()

    print("Feature extraction and writing to core data complete.")


  def compute_representative_factors(self, 
                                     sample_indices_per_sim_per_time: list = None,
                                     ranks: int = 4,
                                     n_representative_blocks: int = 5000,
                                    ):
    
    # Compute representative factors once for all sims
    n_representative_blocks_per_frame = int(n_representative_blocks / (self.last_sim - self.first_sim + 1) / (self.last_t - self.first_t + 1))
    
    # if original dataset is available, use it to define the representative sample indices per sim and time step
    # otherwise randomly select them from the already sampled indices
    if self.original_dataset_path is None:
      sample_indices_per_sim_per_time_representative = []
      for sim_indices in sample_indices_per_sim_per_time:
        sim_representative = []
        for time_indices in sim_indices:
          # Randomly select n_representative_blocks_per_frame samples or all if fewer available
          if len(time_indices) > n_representative_blocks_per_frame:
            idx = np.random.choice(len(time_indices), n_representative_blocks_per_frame, replace=False)
            sim_representative.append(time_indices[idx])
          else:
            sim_representative.append(time_indices)
        sample_indices_per_sim_per_time_representative.append(sim_representative)
    else:
      sample_indices_per_sim_per_time_representative = utils_sampling.define_sample_indexes(
        n_representative_blocks_per_frame,
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
            self.gridded_h5_filenames[sim],
            sim,
            t_start=self.first_t,
            t_end=self.last_t,
            block_size=self.block_size,
            first_sim=self.first_sim,
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
    input_factors, output_factors = self.get_representative_factors(representative_blocks, ranks=ranks)
    
    return input_factors, output_factors



  def transform_and_write_blocks_to_core_data(self,
      core_data_fn,
      input_factors,
      output_factors,
      chunk_size,
      sample_indices_per_sim_per_time,
      ):

    total_times = self.last_t - self.first_t
    if chunk_size > self.n_samples_per_frame:
      n_times_per_chunk = chunk_size // self.n_samples_per_frame
      n_chunks_per_sim = ceil(total_times / n_times_per_chunk)
      n_sub_chunks = 1
    else:
      n_sub_chunks = ceil(self.n_samples_per_frame / chunk_size)
      n_times_per_chunk = 1
      n_chunks_per_sim = total_times

    for sim in range(self.first_sim, self.last_sim + 1):
        print(f'Transforming data from sim {sim + 1}/[{self.first_sim + 1}, {self.last_sim + 1}]...')
        for i_chunk in range(n_chunks_per_sim):
            for sub_chunk in range(n_sub_chunks):
                print(f' -Sampling block data for chunk {i_chunk + 1}/{n_chunks_per_sim} - subchunk {sub_chunk + 1}/{n_sub_chunks}', flush=True)
                blocks_data = self.sample_blocks_chunked(
                  sim,
                  t_start=i_chunk * n_times_per_chunk,
                  t_end=min((i_chunk + 1) * n_times_per_chunk, self.last_t),
                  i_chunk=sub_chunk,
                  n_chunks=n_sub_chunks,
                  sample_indices=sample_indices_per_sim_per_time
                )

                print(f' -Transforming grid data to tensor cores for chunk {i_chunk + 1}/{n_chunks_per_sim} - subchunk {sub_chunk + 1}/{n_sub_chunks}', flush=True)
                in_features, out_features = self.transform_data_with_tucker(blocks_data, input_factors, output_factors)

                with tables.open_file(core_data_fn, mode='a') as f:
                    f.root.inputs.append(np.array(in_features))
                    f.root.outputs.append(np.array(out_features))

    print(f"All data transformed and written to {core_data_fn}.")

  @staticmethod
  def sample_blocks_chunked(
    gridded_h5_fn_sim: str,
    sim: int,
    t_start: int,
    t_end: int,
    block_size: int,
    first_sim: int,
    n_chunks=False,
    i_chunk=None,
    sample_indices=None):
    """
    Static method to sample blocks chunked from gridded HDF5 simulation data.
    """


    inputs_u_list = []
    inputs_obst_list = []
    outputs_list = []
    use_subchunks = n_chunks > 1
    count = 0

    sim_idx = sim - first_sim

    for time in range(t_start, t_end):
      with tables.open_file(gridded_h5_fn_sim, mode='r') as f:
        grid = f.root.data[time, :, :, :, :]

      ZYX_indices = sample_indices[sim_idx][time]

      if use_subchunks:
        elements_per_sub_chunk = ceil(ZYX_indices.shape[0] / n_chunks)
        i_element_start = i_chunk * elements_per_sub_chunk
        i_element_end = (i_chunk + 1) * elements_per_sub_chunk
        ZYX_indices = ZYX_indices[i_element_start:i_element_end]

      for [ii, jj, kk] in ZYX_indices:
        i_idx_fist = int(ii - block_size / 2)
        i_idx_last = int(ii + block_size / 2)
        j_idx_first = int(jj - block_size / 2)
        j_idx_last = int(jj + block_size / 2)
        k_idx_first = int(kk - block_size / 2)
        k_idx_last = int(kk + block_size / 2)

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
    _, input_factors = tucker(input_tensor, rank=(chunk_size,) + spatial_ranks + (last_rank,))
    _, output_factors = tucker(pressure, rank=(chunk_size,) + spatial_ranks)

    with open(self.tucker_factors_fn, 'wb') as f:
      pk.dump({'input_factors': input_factors, 'output_factors': output_factors}, f)

    return input_factors, output_factors

  def transform_data_with_tucker(self, blocks_data: np.ndarray, input_factors, output_factors, n_representative_blocks) -> np.ndarray:
    inputs_u, inputs_obst, outputs = blocks_data
    chunk_size = inputs_u.shape[0]
    print(f'ACTUAL Chunk size: {chunk_size}')

    velocity = inputs_u / [self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz]
    obstacle = inputs_obst / self.max_abs_dist
    pressure = outputs[..., 0] / self.max_abs_delta_p

    print("Transforming data using precomputed Tucker factors...")
    input_tensor = np.concatenate([velocity, obstacle], axis=-1)
    input_core = tl.tenalg.multi_mode_dot(input_tensor, input_factors[1:], modes=[1, 2, 3, 4], transpose=True)
    output_core = tl.tenalg.multi_mode_dot(pressure, output_factors[1:], modes=[1, 2, 3], transpose=True)

    if self.flatten_data:
        input_core = input_core.reshape(chunk_size, -1)
        output_core = output_core.reshape(chunk_size, -1)

    return input_core, output_core