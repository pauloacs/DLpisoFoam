import numpy as np
import h5py
import tables
import pickle as pk
from math import ceil
import tensorly as tl
from tensorly.decomposition import tucker
from . import utils

import dask.distributed

class CFDDataProcessor:
  """
  """
  def __init__(
        self,
        grid_res: float,
        block_size: int,
        var_p: int,
        var_in: int,
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
    self.var_in = var_in
    self.var_p = var_p
    self.original_dataset_path = original_dataset_path
    self.n_samples_per_frame = n_samples_per_frame
    self.first_sim = first_sim
    self.last_sim = last_sim
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.chunk_size = chunk_size
    self.gridded_h5_fn = gridded_h5_fn

  def write_gridded_simulation_data(self) -> None:
    """
    Write CFD mesh data to a regular grid and save to HDF5 file.
    """
    print(f'########## Writting CFD mesh data to a grid -> {self.gridded_h5_fn} ############')
    NUM_COLUMNS = 5

    file = tables.open_file(self.gridded_h5_fn, mode='w')
    atom = tables.Float32Atom()

    # For now, assume grid shape is fixed as in Training
    self.grid_shape_z = 0.1 / self.grid_res
    self.grid_shape_y = 0.1 / self.grid_res
    self.grid_shape_x = 1 / self.grid_res

    file.create_earray(file.root, 'data', atom, (0, int(self.grid_shape_z), int(self.grid_shape_y), int(self.grid_shape_x), NUM_COLUMNS))
    file.close()

    for i in range(self.first_sim, self.last_sim):
      print(f"\nProcessing sim {i+1}/{self.last_sim - self.first_sim}\n", flush=True)
      self.write_sim_fields(i)

  def write_sim_fields(self, i):
    """
    Write simulation fields for a single simulation index to the grid file.
    """
    with h5py.File(self.original_dataset_path, "r") as f:
      data = np.array(f["sim_data"][i:i+1, self.first_t:(self.first_t + self.last_t), ...], dtype='float32')
      obst_boundary = np.array(f["obst_bound"][i, 0, ...], dtype='float32')
      y_bot_boundary = np.array(f["y_bot_bound"][i, 0, ...], dtype='float32')
      z_bot_boundary = np.array(f["z_bot_bound"][i, 0, ...], dtype='float32')
      y_top_boundary = np.array(f["y_top_bound"][i, 0, ...], dtype='float32')
      z_top_boundary = np.array(f["z_top_bound"][i, 0, ...], dtype='float32')

    indice = utils.index(data[0, 0, :, 0], -100.0)[0]
    data_limited = data[0, 0, :indice, :]

    self.x_min = round(np.min(data_limited[..., 4]), 2)
    self.x_max = round(np.max(data_limited[..., 4]), 2)
    self.y_min = round(np.min(data_limited[..., 5]), 2)
    self.y_max = round(np.max(data_limited[..., 5]), 2)
    self.z_min = round(np.min(data_limited[..., 6]), 2)
    self.z_max = round(np.max(data_limited[..., 6]), 2)

    X0, Y0, Z0 = utils.create_uniform_grid(self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max, self.grid_res)
    xyz0 = np.concatenate((np.expand_dims(X0, axis=1), np.expand_dims(Y0, axis=1), np.expand_dims(Z0, axis=1)), axis=-1)
    points = data_limited[..., 4:7]

    vert, weights = utils.interp_weights(points, xyz0)
    boundaries_list = [obst_boundary, y_bot_boundary, z_bot_boundary, y_top_boundary, z_top_boundary]
    domain_bool, sdf = utils.domain_dist(boundaries_list, xyz0, self.grid_res)

    div = 1
    self.grid_shape_z = int(round((self.z_max - self.z_min) / self.grid_res))
    self.grid_shape_y = int(round((self.y_max - self.y_min) / self.grid_res))
    self.grid_shape_x = int(round((self.x_max - self.x_min) / self.grid_res))

    x0 = np.min(X0)
    y0 = np.min(Y0)
    z0 = np.min(Z0)
    dx = self.grid_res
    dy = self.grid_res
    dz = self.grid_res

    indices = np.zeros((X0.shape[0], 3))
    obst_bool = np.zeros((self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 1))
    sdfunct = obst_bool.copy()

    delta_p = data_limited[..., 10:11]
    p_interp = utils.interpolate_fill(delta_p, vert, weights)

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
    for j in range(data.shape[1]):
      data_limited = data[0, j, :indice, :]
      self.write_time_step_fields(j, data_limited, vert, weights, indices, sdfunct)
      if self.stationary_ts > 5:
        print('This simulation is stationary, ignoring it...')
        break

  def write_time_step_fields(self, j, data_limited, vert, weights, indices, sdfunct):
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

    delta_p_interp = utils.interpolate_fill(delta_p_adim, vert, weights)
    delta_Ux_interp = utils.interpolate_fill(delta_Ux_adim, vert, weights)
    delta_Uy_interp = utils.interpolate_fill(delta_Uy_adim, vert, weights)
    delta_Uz_interp = utils.interpolate_fill(delta_Uz_adim, vert, weights)

    filter_tuple = (2, 2, 2)
    grid = np.zeros(shape=(self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 5))
    grid[:, :, :, 0:1][tuple(indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0], 1)
    grid[:, :, :, 1:2][tuple(indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0], 1)
    grid[:, :, :, 2:3][tuple(indices.T)] = delta_Uz_interp.reshape(delta_Uz_interp.shape[0], 1)
    grid[:, :, :, 3:4] = sdfunct
    grid[:, :, :, 4:5][tuple(indices.T)] = delta_p_interp.reshape(delta_p_interp.shape[0], 1)

    grid[np.isnan(grid)] = 0

    print(f"Writting t{j + self.first_t} to {self.gridded_h5_fn}", flush=True)
    with tables.open_file(self.gridded_h5_fn, mode='a') as file:
      file.root.data.append(np.array(np.expand_dims(grid, axis=0), dtype='float32'))


class FeatureExtractAndWrite:
  
  def __init__(
        self,
        grid_res: float,
        block_size: int,
        var_p: int,
        var_in: int,
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
    self.var_in = var_in
    self.var_p = var_p
    self.n_samples_per_frame = n_samples_per_frame
    self.first_sim = first_sim
    self.last_sim = last_sim
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.chunk_size = chunk_size
    self.original_dataset_path = original_dataset_path
    self.gridded_h5_fn = gridded_h5_fn
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

  def sample_blocks_chunked(self, sim, t_start, t_end, i_chunk=None, n_chunks=False, sample_indices=None):
    inputs_u_list = []
    inputs_obst_list = []
    outputs_list = []
    use_subchunks = n_chunks > 1
    count = 0

    sim = sim - self.first_sim
    for time in range(t_start, t_end):
      with tables.open_file(self.gridded_h5_fn, mode='r') as f:
        grid = f.root.data[sim * (self.last_t - self.first_t) + time, :, :, :, :]

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

  def transform_data_with_tucker(self, blocks_data: np.ndarray, client, ranks) -> np.ndarray:
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
          input_shape = (0, ranks * ranks * ranks * ranks)
          output_shape = (0, ranks * ranks * ranks)
        else:
          input_shape = (0, ranks, ranks, ranks, ranks)
          output_shape = (0, ranks, ranks, ranks)

        file.create_earray(file.root, 'inputs', atom, input_shape)
        file.create_earray(file.root, 'outputs', atom, output_shape)

    client = dask.distributed.Client(processes=False)

    # Load maxs
    maxs = np.loadtxt('maxs')
    self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz, self.max_abs_dist, self.max_abs_delta_p = maxs

    # Compute representative factors once for all sims
    N_representative = 2500 #2500
    N_representative_per_sim = int(N_representative / (self.last_sim - self.first_sim) / (self.last_t - self.first_t))
    sample_indices_per_sim_per_time_representative = utils.define_sample_indexes(
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

    for sim in range(self.first_sim, self.last_sim):
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

    for sim in range(self.first_sim, self.last_sim):
        print(f'Transforming data from sim {sim + 1}/[{self.first_sim}, {self.last_sim}]...')
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
                in_features, out_features = self.transform_data_with_tucker(blocks_data, client, ranks)

                with tables.open_file(filename_flat, mode='a') as f:
                    f.root.inputs.append(np.array(in_features))
                    f.root.outputs.append(np.array(out_features))

    client.close()
