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
        gridded_h5_fn: str,
    ):

    self.grid_res = grid_res
    # Allow block_size to be int or tuple
    if isinstance(block_size, int):
      self.block_size_z = self.block_size_y = self.block_size_x = block_size
    else:
      self.block_size_z, self.block_size_y, self.block_size_x = block_size
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

        self.grid_shape_x, self.grid_shape_y, self.grid_shape_z = utils_data.get_grid_shape(limits, self.grid_res)

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
    dx, dy, dz = utils_data._unpack_grid_res(self.grid_res)

    indices = np.zeros((X0.shape[0], 3))
    obst_bool = np.zeros((self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 1))
    sdfunct = obst_bool.copy()

    # Use delta_delta_p as the pressure field (from new solver output)
    delta_delta_p = sim_data_t0[..., 10:11]
    p_interp = utils_data.interpolate_fill(delta_delta_p, vert, weights)
    # delta_p_prev (from column 11 if available)
    if self.use_previous_dp_input and sim_data_t0.shape[1] > 11:
      delta_delta_p_prev = sim_data_t0[..., 11:12]
    else:
      delta_delta_p_prev = None

    for (step, x_y_z) in enumerate(xyz0):
      ii = int(round((x_y_z[..., 2] - z0) / dz))
      jj = int(round((x_y_z[..., 1] - y0) / dy))
      kk = int(round((x_y_z[..., 0] - x0) / dx))
      indices[step, 0] = ii
      indices[step, 1] = jj
      indices[step, 2] = kk      
      if domain_bool[step] * (~np.isnan(p_interp[step])):
        sdfunct[ii, jj, kk, :] = sdf[step]
        obst_bool[ii, jj, kk, :] = int(1)

    indices = indices.astype(int)
    self.stationary_ts = 0
    for j in range(sim_data_ts.shape[0]):
      data = sim_data_ts[j, :, :]
      p_prev_data = sim_data_ts[j, :, 11:12] if self.use_previous_dp_input and sim_data_ts.shape[2] > 11 else None
      self.write_time_step_fields(j, data, p_prev_data, vert, weights, indices, sdfunct, gridded_h5_fn_sim)
      if self.stationary_ts > 5:
        print('This simulation is stationary, ignoring it...')
        break

  def write_time_step_fields(self, j, data_limited, p_prev_data, vert, weights, indices, sdfunct, gridded_h5_fn_sim):
    """
    Write a single time step's fields to the grid file.
    """

    # Use delta_delta fields for ML surrogate modeling
    Ux = data_limited[..., 0:1]
    Uy = data_limited[..., 1:2]
    Uz = data_limited[..., 2:3]
    delta_delta_p = data_limited[..., 10:11]
    delta_delta_Ux = data_limited[..., 7:8]
    delta_delta_Uy = data_limited[..., 8:9]
    delta_delta_Uz = data_limited[..., 9:10]

    U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy) + np.square(Uz)))
    delta_delta_U_max_norm = np.max(np.sqrt(np.square(delta_delta_Ux) + np.square(delta_delta_Uy) + np.square(delta_delta_Uz)))

    threshold = 1e-4
    print(f"delta_delta_U_max_norm = {delta_delta_U_max_norm}")
    print(f"U_max_norm      = {U_max_norm}")
    irrelevant_ts = (delta_delta_U_max_norm / U_max_norm) < threshold or delta_delta_U_max_norm < 1e-6 or U_max_norm < 1e-6

    if irrelevant_ts:
      print(f"\n\n Irrelevant time step, skipping it...")
      self.stationary_ts += 1
      return 0

    delta_delta_p_adim = delta_delta_p / pow(U_max_norm, 2.0)
    delta_delta_Ux_adim = delta_delta_Ux / U_max_norm
    delta_delta_Uy_adim = delta_delta_Uy / U_max_norm
    delta_delta_Uz_adim = delta_delta_Uz / U_max_norm

    delta_delta_p_interp = utils_data.interpolate_fill(delta_delta_p_adim, vert, weights)
    delta_delta_Ux_interp = utils_data.interpolate_fill(delta_delta_Ux_adim, vert, weights)
    delta_delta_Uy_interp = utils_data.interpolate_fill(delta_delta_Uy_adim, vert, weights)
    delta_delta_Uz_interp = utils_data.interpolate_fill(delta_delta_Uz_adim, vert, weights)

    # Interpolate delta_p_prev if available and enabled
    if self.use_previous_dp_input and p_prev_data is not None:
      delta_delta_p_prev_adim = p_prev_data / pow(U_max_norm, 2.0)
      delta_delta_p_prev_interp = utils_data.interpolate_fill(delta_delta_p_prev_adim, vert, weights)
      n_cols = 6
    else:
      delta_delta_p_prev_interp = None
      n_cols = 5

    filter_tuple = (2, 2, 2)
    grid = np.zeros(shape=(self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, n_cols))
    grid[:, :, :, 0:1][tuple(indices.T)] = delta_delta_Ux_interp.reshape(delta_delta_Ux_interp.shape[0], 1)
    grid[:, :, :, 1:2][tuple(indices.T)] = delta_delta_Uy_interp.reshape(delta_delta_Uy_interp.shape[0], 1)
    grid[:, :, :, 2:3][tuple(indices.T)] = delta_delta_Uz_interp.reshape(delta_delta_Uz_interp.shape[0], 1)
    grid[:, :, :, 3:4] = sdfunct
    grid[:, :, :, 4:5][tuple(indices.T)] = delta_delta_p_interp.reshape(delta_delta_p_interp.shape[0], 1)
    if self.use_previous_dp_input and delta_delta_p_prev_interp is not None:
      grid[:, :, :, 5:6][tuple(indices.T)] = delta_delta_p_prev_interp.reshape(delta_delta_p_prev_interp.shape[0], 1)

    grid[np.isnan(grid)] = 0

    import matplotlib.pyplot as plt
    # Save plots for all variables
    os.makedirs('plots_debug', exist_ok=True)

    var_names = ['delta_delta_Ux', 'delta_delta_Uy', 'delta_delta_Uz', 'sdf', 'delta_delta_p']
    if self.use_previous_dp_input and n_cols == 6:
      var_names.append('delta_delta_p_prev')

    # Compute the domain mask for the current grid
    domain_mask = np.zeros(grid.shape[:3], dtype=bool)
    for idx, (ii, jj, kk) in enumerate(indices):
      if not np.isnan(ii) and not np.isnan(jj) and not np.isnan(kk):
        domain_mask[int(ii), int(jj), int(kk)] = True

    for var_idx in range(n_cols):
      # Plot slice through middle of grid (z-x plane at middle y)
      plt.figure(figsize=(10, 6))
      slice_zx = grid[1:-1, int(grid.shape[1] / 2), 1:-1, var_idx]
      mask_zx = domain_mask[1:-1, int(grid.shape[1] / 2), 1:-1]
      masked_zx = np.ma.masked_where(~mask_zx, slice_zx)
      plt.imshow(masked_zx, cmap='jet')
      plt.colorbar(label=var_names[var_idx])
      plt.title(f'{var_names[var_idx]} - Z-X slice (middle Y)')
      plt.xlabel('X')
      plt.ylabel('Z')
      plt.savefig(f'plots_debug/{var_names[var_idx]}_zx_slice_t{j + self.first_t}.png')
      plt.close()

      # Plot slice through middle of grid (y-x plane at middle z)
      plt.figure(figsize=(10, 6))
      slice_yx = grid[int(grid.shape[0] / 2), :, :, var_idx]
      mask_yx = domain_mask[int(grid.shape[0] / 2), :, :]
      masked_yx = np.ma.masked_where(~mask_yx, slice_yx)
      plt.imshow(masked_yx, cmap='jet')
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
        standardization_method: str = 'std',
        chunk_size: int = 500,
        spatial_tucker_ranks: tuple = (4, 4, 4),
        gridded_h5_fn: str = 'gridded_data.h5',
        sample_indices_fn: str = 'sample_indices_per_sim_per_time.pkl',
        tucker_factors_fn: str = 'tucker_factors.pkl',
        gridded_h5_filenames: list = None,
        flatten_data: bool = False,
        maxs_list: list = None,
        last_tucker_rank: int = 4,
        use_feature_decomposition: bool = True,
        add_ddu_input: bool = True,
        add_U_input: bool = False,
        add_dU_input: bool = False,
        use_previous_dp_input: bool = False,
        add_p_prev_input: bool = False,
        add_ddp_prev_input: bool = False,
        add_div_ddu_input: bool = False,
        add_div_du_input: bool = False,
        add_div_u_input: bool = False,
    ):

    self.grid_res = grid_res
    # Allow block_size to be int or tuple
    if isinstance(block_size, int):
      self.block_size_z = self.block_size_y = self.block_size_x = block_size
    else:
      self.block_size_z, self.block_size_y, self.block_size_x = block_size
    self.block_size = block_size
    self.n_samples_per_frame = n_samples_per_frame
    self.first_sim = first_sim
    self.last_sim = last_sim
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.chunk_size = chunk_size
    self.spatial_tucker_ranks = tuple(spatial_tucker_ranks) if use_feature_decomposition else None
    self.original_dataset_path = original_dataset_path
    if gridded_h5_fn is not None:
      self.gridded_h5_filenames = utils_io.get_gridded_h5_filenames(
        gridded_h5_fn,
        first_sim,
        last_sim
        )
    else:
      self.gridded_h5_filenames = gridded_h5_filenames

    self.sample_indices_fn = sample_indices_fn
    self.tucker_factors_fn = tucker_factors_fn
    self.flatten_data = flatten_data
    self.add_U_input = add_U_input
    self.add_dU_input = add_dU_input
    self.add_ddu_input = add_ddu_input
    self.use_previous_dp_input = use_previous_dp_input
    self.add_p_prev_input = add_p_prev_input
    self.add_ddp_prev_input = add_ddp_prev_input
    self.add_div_ddu_input = add_div_ddu_input
    self.add_div_du_input = add_div_du_input
    self.add_div_u_input = add_div_u_input
    # maxs_list layout: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU sdf [p_prev if add_p_prev] [delta_p_prev if use_prev_dp] [ddp_prev if add_ddp_prev] [div_ddu if add_div_ddu] [div_du if add_div_du] [div_u if add_div_u] delta_p
    ch_idx = 0
    if add_U_input:
      self.max_abs_U_x, self.max_abs_U_y, self.max_abs_U_z = maxs_list[ch_idx:ch_idx+3]
      ch_idx += 3
    else:
      self.max_abs_U_x = self.max_abs_U_y = self.max_abs_U_z = 1.0  # unused

    if add_dU_input:
      self.max_abs_dU_x, self.max_abs_dU_y, self.max_abs_dU_z = maxs_list[ch_idx:ch_idx+3]
      ch_idx += 3
    else:
      self.max_abs_dU_x = self.max_abs_dU_y = self.max_abs_dU_z = 1.0  # unused

    if add_ddu_input:
      self.max_abs_ddU_x, self.max_abs_ddU_y, self.max_abs_ddU_z = maxs_list[ch_idx:ch_idx+3]
      ch_idx += 3
    else:
      self.max_abs_ddU_x = self.max_abs_ddU_y = self.max_abs_ddU_z = 1.0  # unused
    
    self.max_abs_dddU_x, self.max_abs_dddU_y, self.max_abs_dddU_z = maxs_list[ch_idx:ch_idx+3]
    ch_idx += 3
    
    # SDF maximum is set to default (will be computed from actual data if needed)
    self.max_abs_dist = 1.0  # Default normalization for sdf
    if add_p_prev_input:
      self.max_abs_p_prev = maxs_list[ch_idx]
      ch_idx += 1
    else:
      self.max_abs_p_prev = 1.0  # unused
    if use_previous_dp_input:
      self.max_abs_delta_p_prev = maxs_list[ch_idx]
      ch_idx += 1
    else:
      self.max_abs_delta_p_prev = 1.0  # unused
    if add_ddp_prev_input:
      self.max_abs_ddp_prev = maxs_list[ch_idx]
      ch_idx += 1
    else:
      self.max_abs_ddp_prev = 1.0  # unused
    if add_div_ddu_input:
      self.max_abs_div_ddu = maxs_list[ch_idx]
      ch_idx += 1
    else:
      self.max_abs_div_ddu = 1.0  # unused
    if add_div_du_input:
      self.max_abs_div_du = maxs_list[ch_idx]
      ch_idx += 1
    else:
      self.max_abs_div_du = 1.0  # unused
    if add_div_u_input:
      self.max_abs_div_u = maxs_list[ch_idx]
      ch_idx += 1
    else:
      self.max_abs_div_u = 1.0  # unused
    self.max_abs_delta_delta_p = maxs_list[ch_idx]

    # Backward-compatible aliases
    self.max_abs_delta_Ux = self.max_abs_dddU_x
    self.max_abs_delta_Uy = self.max_abs_dddU_y
    self.max_abs_delta_Uz = self.max_abs_dddU_z
    self.last_tucker_rank = last_tucker_rank
    self.use_feature_decomposition = use_feature_decomposition

    with open(self.sample_indices_fn, 'rb') as f:
      self.sample_indices_per_sim_per_time = pk.load(f)

  def __call__(self, core_data_fn: str,  compute_tucker_factors=True, n_representative_blocks=7500) -> None:

    """
    Extract features and write to core data file.
    If use_feature_decomposition=True (default): apply Tucker decomposition to blocks.
    If use_feature_decomposition=False: write raw normalized blocks directly (no Tucker).
    """

    if not self.use_feature_decomposition:
      print("[FeatureExtractAndWrite] use_feature_decomposition=False: skipping Tucker, writing raw blocks.")
      self.write_raw_blocks_to_core_data(
        core_data_fn,
        self.chunk_size,
        self.sample_indices_per_sim_per_time,
      )
      print("Raw block writing complete.")
      return

    client = dask.distributed.Client(processes=False)
    
    if compute_tucker_factors:
        input_factors, output_factors = self.compute_representative_factors(
                                          self.sample_indices_per_sim_per_time,
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

    # Create the core data file and write transformed features to it
    with tables.open_file(core_data_fn, mode='w') as file:
        atom = tables.Float32Atom()
        if self.flatten_data:
          input_shape = (0, self.spatial_tucker_ranks[0] * self.spatial_tucker_ranks[1] * self.spatial_tucker_ranks[2] * self.last_tucker_rank)
          output_shape = (0, self.spatial_tucker_ranks[0] * self.spatial_tucker_ranks[1] * self.spatial_tucker_ranks[2])
        else:
          input_shape = (0, self.spatial_tucker_ranks[0], self.spatial_tucker_ranks[1], self.spatial_tucker_ranks[2], self.last_tucker_rank)
          output_shape = (0, self.spatial_tucker_ranks[0], self.spatial_tucker_ranks[1], self.spatial_tucker_ranks[2])
    
        file.create_earray(file.root, 'inputs', atom, input_shape)
        file.create_earray(file.root, 'outputs', atom, output_shape)

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
                                     n_representative_blocks: int = 5000,
                                    ):

    # Compute representative factors once for all sims.
    # Clamp to at least 1 so that each time step always contributes at least one
    # block even when n_representative_blocks < total number of time steps
    # (e.g. single-block-covers-full-domain mode).
    n_representative_blocks_per_frame = max(
        1,
        int(n_representative_blocks / (self.last_sim - self.first_sim + 1) / (self.last_t - self.first_t))
    )
    
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
          sample_indices=sample_indices_per_sim_per_time_representative,
          add_U_input=self.add_U_input,
          add_dU_input=self.add_dU_input,
          add_ddu_input=self.add_ddu_input,
          use_previous_dp_input=self.use_previous_dp_input,
          add_p_prev_input=self.add_p_prev_input,
          add_ddp_prev_input=self.add_ddp_prev_input,
          add_div_ddu_input=self.add_div_ddu_input,
          add_div_du_input=self.add_div_du_input,
          add_div_u_input=self.add_div_u_input,
        )
        all_inputs_u.append(inputs_u)
        all_inputs_obst.append(inputs_obst)
        all_outputs.append(outputs)

    all_inputs_u = np.concatenate(all_inputs_u)
    all_inputs_obst = np.concatenate(all_inputs_obst)
    all_outputs = np.concatenate(all_outputs)

    representative_blocks = (all_inputs_u, all_inputs_obst, all_outputs)
    input_factors, output_factors = self.get_representative_factors(representative_blocks, spatial_tucker_ranks=self.spatial_tucker_ranks)
    
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
                  self.gridded_h5_filenames[sim],
                  sim,
                  t_start=i_chunk * n_times_per_chunk,
                  t_end=min((i_chunk + 1) * n_times_per_chunk, self.last_t),
                  block_size=self.block_size,
                  first_sim=self.first_sim,
                  i_chunk=sub_chunk,
                  n_chunks=n_sub_chunks,
                  sample_indices=sample_indices_per_sim_per_time,
                  add_U_input=self.add_U_input,
                  add_dU_input=self.add_dU_input,
                  add_ddu_input=self.add_ddu_input,
                  use_previous_dp_input=self.use_previous_dp_input,
                  add_p_prev_input=self.add_p_prev_input,
                  add_ddp_prev_input=self.add_ddp_prev_input,
                  add_div_ddu_input=self.add_div_ddu_input,
                  add_div_du_input=self.add_div_du_input,
                  add_div_u_input=self.add_div_u_input,
                )

                print(f' -Transforming grid data to tensor cores for chunk {i_chunk + 1}/{n_chunks_per_sim} - subchunk {sub_chunk + 1}/{n_sub_chunks}', flush=True)
                in_features, out_features = self.transform_data_with_tucker(blocks_data, input_factors, output_factors)

                with tables.open_file(core_data_fn, mode='a') as f:
                    f.root.inputs.append(np.array(in_features))
                    f.root.outputs.append(np.array(out_features))

    print(f"All data transformed and written to {core_data_fn}.")

  def write_raw_blocks_to_core_data(self, core_data_fn, chunk_size, sample_indices_per_sim_per_time):
    """Write raw normalized blocks (no Tucker decomposition) to core_data HDF5 file."""
    blk_z = self.block_size_z
    blk_y = self.block_size_y
    blk_x = self.block_size_x
    # inputs: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU [p_prev if add_p_prev] [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] [div_ddu if add_div_ddu] [div_du if add_div_du] [div_u if add_div_u] + sdf
    # NOTE: normalize_raw_blocks concatenates [velocity | sdf], so add +1 for sdf channel
    n_in_ch = (int(self.add_U_input) * 3 + int(self.add_dU_input) * 3 + int(self.add_ddu_input) * 3 + 3 + int(self.add_p_prev_input) + int(self.use_previous_dp_input) + int(self.add_ddp_prev_input) + int(self.add_div_ddu_input) + int(self.add_div_du_input) + int(self.add_div_u_input)) + 1

    with tables.open_file(core_data_fn, mode='w') as file:
      atom = tables.Float32Atom()
      file.create_earray(file.root, 'inputs', atom, (0, blk_z, blk_y, blk_x, n_in_ch))
      file.create_earray(file.root, 'outputs', atom, (0, blk_z, blk_y, blk_x))

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
      print(f'Writing raw blocks from sim {sim + 1}/[{self.first_sim + 1}, {self.last_sim + 1}]...')
      for i_chunk in range(n_chunks_per_sim):
        for sub_chunk in range(n_sub_chunks):
          print(f' -Sampling block data for chunk {i_chunk + 1}/{n_chunks_per_sim} - subchunk {sub_chunk + 1}/{n_sub_chunks}', flush=True)
          blocks_data = self.sample_blocks_chunked(
            self.gridded_h5_filenames[sim],
            sim,
            t_start=i_chunk * n_times_per_chunk,
            t_end=min((i_chunk + 1) * n_times_per_chunk, self.last_t),
            block_size=self.block_size,
            first_sim=self.first_sim,
            i_chunk=sub_chunk,
            n_chunks=n_sub_chunks,
            sample_indices=sample_indices_per_sim_per_time,
            add_U_input=self.add_U_input,
            add_dU_input=self.add_dU_input,
            add_ddu_input=self.add_ddu_input,
            use_previous_dp_input=self.use_previous_dp_input,
            add_p_prev_input=self.add_p_prev_input,
            add_ddp_prev_input=self.add_ddp_prev_input,
            add_div_ddu_input=self.add_div_ddu_input,
            add_div_du_input=self.add_div_du_input,
            add_div_u_input=self.add_div_u_input,
          )
          in_features, out_features = self.normalize_raw_blocks(blocks_data)
          if in_features.shape[0] == 0:
            continue
          with tables.open_file(core_data_fn, mode='a') as f:
            f.root.inputs.append(in_features.astype(np.float32))
            f.root.outputs.append(out_features.astype(np.float32))

    print(f"All raw blocks written to {core_data_fn}.")

  def normalize_raw_blocks(self, blocks_data):
    """Normalize sampled blocks by max values without Tucker decomposition."""
    inputs_u, inputs_obst, outputs = blocks_data
    
    if inputs_u.ndim == 1 and inputs_u.size == 0:
      n_in_ch = int(self.add_U_input) * 3 + int(self.add_dU_input) * 3 + int(self.add_ddu_input) * 3 + 3 + int(self.add_p_prev_input) + int(self.use_previous_dp_input) + int(self.add_ddp_prev_input) + int(self.add_div_ddu_input) + int(self.add_div_du_input) + int(self.add_div_u_input)
      return (np.empty((0, self.block_size_z, self.block_size_y, self.block_size_x, n_in_ch), dtype=np.float32),
              np.empty((0, self.block_size_z, self.block_size_y, self.block_size_x), dtype=np.float32))
    
    # Build normalization factors in SAME ORDER as extraction in sample_blocks_chunked
    # inputs_u channels: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU [p_prev if add_p_prev] [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] [div_ddu if add_div_ddu] [div_du if add_div_du] [div_u if add_div_u]
    norm_parts = []
    if self.add_U_input:
      norm_parts.extend([self.max_abs_U_x, self.max_abs_U_y, self.max_abs_U_z])
    if self.add_dU_input:
      norm_parts.extend([self.max_abs_dU_x, self.max_abs_dU_y, self.max_abs_dU_z])
    if self.add_ddu_input:
      norm_parts.extend([self.max_abs_ddU_x, self.max_abs_ddU_y, self.max_abs_ddU_z])
    norm_parts.extend([self.max_abs_dddU_x, self.max_abs_dddU_y, self.max_abs_dddU_z])
    # Pressure-related inputs (NOT sdf - sdf is handled separately via inputs_obst)
    if self.add_p_prev_input:
      norm_parts.append(self.max_abs_p_prev)
    if self.use_previous_dp_input:
      norm_parts.append(self.max_abs_delta_p_prev)
    if self.add_ddp_prev_input:
      norm_parts.append(self.max_abs_ddp_prev)
    if self.add_div_ddu_input:
      norm_parts.append(self.max_abs_div_ddu)
    if self.add_div_du_input:
      norm_parts.append(self.max_abs_div_du)
    if self.add_div_u_input:
      norm_parts.append(self.max_abs_div_u)

    # Ensure norm_parts matches inputs_u channels
    if len(norm_parts) != inputs_u.shape[-1]:
      raise ValueError(f"Number of normalization values ({len(norm_parts)}) must match input channels ({inputs_u.shape[-1]}). "
                       f"Flags: add_U={self.add_U_input}, add_dU={self.add_dU_input}, add_ddu={self.add_ddu_input}, "
                       f"add_p_prev={self.add_p_prev_input}, use_prev_dp={self.use_previous_dp_input}, add_ddp_prev={self.add_ddp_prev_input}, "
                       f"add_div_ddu={self.add_div_ddu_input}, add_div_du={self.add_div_du_input}, add_div_u={self.add_div_u_input}")

    velocity = inputs_u / np.array(norm_parts)

    # Normalize obstacle/sdf and delta_p_prev separately
    sdf = inputs_obst[..., 0:1] / self.max_abs_dist  # First channel is always sdf
    obstacle = sdf
    pressure = outputs[..., 0] / self.max_abs_delta_delta_p
    in_features = np.concatenate([velocity, obstacle], axis=-1)  # (N, bsz, bsy, bsx, n_in_ch)
    return in_features, pressure  # pressure shape: (N, bsz, bsy, bsx)

  @staticmethod
  def sample_blocks_chunked(
    gridded_h5_fn_sim: str,
    sim: int,
    t_start: int,
    t_end: int,
    block_size,
    first_sim: int,
    n_chunks=False,
    i_chunk=None,
    sample_indices=None,
    add_U_input: bool = False,
    add_dU_input: bool = False,
    add_ddu_input: bool = True,
    use_previous_dp_input: bool = False,
    add_p_prev_input: bool = False,
    add_ddp_prev_input: bool = False,
    add_div_ddu_input: bool = False,
    add_div_du_input: bool = False,
    add_div_u_input: bool = False,
  ):
    """
    Static method to sample blocks chunked from gridded HDF5 simulation data.
    
    Channel extraction order (matches train_init.py gridding):
    [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU sdf [p_prev if add_p_prev] 
    [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] [div_ddu if add_div_ddu] 
    [div_du if add_div_du] [div_u if add_div_u] delta_delta_p
    """
    
    # Debug: print extraction flags
    print(f"[sample_blocks_chunked] Extraction flags: add_U={add_U_input}, add_dU={add_dU_input}, add_ddu={add_ddu_input}, "
          f"add_p_prev={add_p_prev_input}, use_prev_dp={use_previous_dp_input}, add_ddp_prev={add_ddp_prev_input}, "
          f"add_div_ddu={add_div_ddu_input}, add_div_du={add_div_du_input}, add_div_u={add_div_u_input}", flush=True)


    inputs_u_list = []
    inputs_obst_list = []
    outputs_list = []
    use_subchunks = n_chunks > 1
    count = 0

    sim_idx = sim - first_sim
    
    # Calculate channel indices based on flags: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU sdf [p_prev if add_p_prev] [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] ddp
    ch_idx = 0
    u_start = ch_idx if add_U_input else None
    u_end = (ch_idx + 3) if add_U_input else None
    ch_idx += 3 if add_U_input else 0
    dU_start = ch_idx if add_dU_input else None
    dU_end = (ch_idx + 3) if add_dU_input else None
    ch_idx += 3 if add_dU_input else 0
    ddu_start = ch_idx if add_ddu_input else None
    ddu_end = (ch_idx + 3) if add_ddu_input else None
    ch_idx += 3 if add_ddu_input else 0
    dddu_start = ch_idx
    dddu_end = ch_idx + 3
    ch_idx += 3
    sdf_start = ch_idx
    sdf_end = ch_idx + 1
    ch_idx += 1
    p_prev_sm_start = ch_idx if add_p_prev_input else None
    p_prev_sm_end = (ch_idx + 1) if add_p_prev_input else None
    ch_idx += 1 if add_p_prev_input else 0
    p_prev_start = ch_idx if use_previous_dp_input else None
    p_prev_end = (ch_idx + 1) if use_previous_dp_input else None
    ch_idx += 1 if use_previous_dp_input else 0
    ddp_prev_start = ch_idx if add_ddp_prev_input else None
    ddp_prev_end = (ch_idx + 1) if add_ddp_prev_input else None
    ch_idx += 1 if add_ddp_prev_input else 0
    div_ddu_start = ch_idx if add_div_ddu_input else None
    div_ddu_end = (ch_idx + 1) if add_div_ddu_input else None
    ch_idx += 1 if add_div_ddu_input else 0
    div_du_start = ch_idx if add_div_du_input else None
    div_du_end = (ch_idx + 1) if add_div_du_input else None
    ch_idx += 1 if add_div_du_input else 0
    div_u_start = ch_idx if add_div_u_input else None
    div_u_end = (ch_idx + 1) if add_div_u_input else None
    ch_idx += 1 if add_div_u_input else 0
    p_start = ch_idx
    p_end = ch_idx + 1

    for time in range(t_start, t_end):
      with tables.open_file(gridded_h5_fn_sim, mode='r') as f:
        grid = f.root.data[time, :, :, :, :]

      ZYX_indices = sample_indices[sim_idx][time]

      if use_subchunks:
        elements_per_sub_chunk = ceil(ZYX_indices.shape[0] / n_chunks)
        i_element_start = i_chunk * elements_per_sub_chunk
        i_element_end = (i_chunk + 1) * elements_per_sub_chunk
        ZYX_indices = ZYX_indices[i_element_start:i_element_end]

      # Determine block sizes
      if isinstance(block_size, int):
        block_size_z = block_size_y = block_size_x = block_size
      else:
        block_size_z, block_size_y, block_size_x = block_size

      for [ii, jj, kk] in ZYX_indices:
        i_idx_first = ii - block_size_z // 2
        i_idx_last = i_idx_first + block_size_z
        j_idx_first = jj - block_size_y // 2
        j_idx_last = j_idx_first + block_size_y
        k_idx_first = kk - block_size_x // 2
        k_idx_last = k_idx_first + block_size_x

        # Skip blocks that extend outside the grid
        if (i_idx_first < 0 or i_idx_last > grid.shape[0] or
              j_idx_first < 0 or j_idx_last > grid.shape[1] or
              k_idx_first < 0 or k_idx_last > grid.shape[2]):
          count += 1
          continue

        # Build input components based on flags
        vel_parts = []
        if add_U_input:
          vel_parts.append(grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, u_start:u_end])  # U
        if add_dU_input:
          vel_parts.append(grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, dU_start:dU_end])  # dU
        if add_ddu_input:
          vel_parts.append(grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, ddu_start:ddu_end])  # ddU
        vel_parts.append(grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, dddu_start:dddu_end])  # dddU

        if add_p_prev_input:
          p_prev_sm_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, p_prev_sm_start:p_prev_sm_end]  # p_prev
          vel_parts.append(p_prev_sm_sample)
        if use_previous_dp_input:
          delta_p_prev_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, p_prev_start:p_prev_end]  # delta_p_prev
          vel_parts.append(delta_p_prev_sample)
        if add_ddp_prev_input:
          ddp_prev_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, ddp_prev_start:ddp_prev_end]  # delta_delta_p_prev
          vel_parts.append(ddp_prev_sample)
        if add_div_ddu_input:
          div_ddu_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, div_ddu_start:div_ddu_end]  # div_delta_delta_U
          vel_parts.append(div_ddu_sample)
        if add_div_du_input:
          div_du_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, div_du_start:div_du_end]  # div_dU
          vel_parts.append(div_du_sample)
        if add_div_u_input:
          div_u_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, div_u_start:div_u_end]  # div_U
          vel_parts.append(div_u_sample)
        
        inputs_u_sample = np.concatenate(vel_parts, axis=-1)
        
        inputs_obst_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, sdf_start:sdf_end]  # sdf
        
        outputs_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, p_start:p_end]  # delta_p

        if not ((inputs_u_sample == 0).all() and (outputs_sample == 0).all()):
          inputs_u_list.append(inputs_u_sample)
          inputs_obst_list.append(inputs_obst_sample)
          outputs_list.append(outputs_sample)
        else:
          count += 1

    inputs_u = np.array(inputs_u_list)
    inputs_obst = np.array(inputs_obst_list)
    outputs = np.array(outputs_list)
    
    # Debug: print extracted shapes
    if inputs_u.ndim > 1:
      print(f"[sample_blocks_chunked] Extracted inputs_u shape: {inputs_u.shape} (last dim = {inputs_u.shape[-1]} channels)", flush=True)

    # No blocks were collected — return empty arrays with the correct shape
    if inputs_u.ndim == 1 and inputs_u.size == 0:
      if count > 0:
        print(f'    {count} blocks discarded')
      return inputs_u, inputs_obst, outputs

    for step in range(outputs.shape[0]):
      outputs[step, ...][inputs_obst[step, ...] != 0] -= np.mean(outputs[step, ...][inputs_obst[step, ...] != 0])

    # Remove per-block domain mean from pressure inputs (consistent with output mean removal)
    _vel_ch_base = 3 * (1 + int(add_U_input) + int(add_dU_input) + int(add_ddu_input))
    if add_p_prev_input:
      p_prev_sm_ch = _vel_ch_base
      for step in range(inputs_u.shape[0]):
        mask = inputs_obst[step, ..., 0] != 0
        if mask.any():
          inputs_u[step, ..., p_prev_sm_ch][mask] -= np.mean(inputs_u[step, ..., p_prev_sm_ch][mask])
    if use_previous_dp_input:
      dp_prev_ch = _vel_ch_base + int(add_p_prev_input)
      for step in range(inputs_u.shape[0]):
        mask = inputs_obst[step, ..., 0] != 0
        if mask.any():
          inputs_u[step, ..., dp_prev_ch][mask] -= np.mean(inputs_u[step, ..., dp_prev_ch][mask])
    if add_ddp_prev_input:
      ddp_prev_ch = _vel_ch_base + int(add_p_prev_input) + int(use_previous_dp_input)
      for step in range(inputs_u.shape[0]):
        mask = inputs_obst[step, ..., 0] != 0
        if mask.any():
          inputs_u[step, ..., ddp_prev_ch][mask] -= np.mean(inputs_u[step, ..., ddp_prev_ch][mask])

    array = np.c_[inputs_u, inputs_obst, outputs]
    reshaped_array = array.reshape(array.shape[0], -1)
    unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
    unique_array = array[unique_indices]
    
    # Extract based on total velocity channels
    vel_end = 3 * (1 + int(add_U_input) + int(add_dU_input) + int(add_ddu_input)) + int(add_p_prev_input) + int(use_previous_dp_input) + int(add_ddp_prev_input) + int(add_div_ddu_input) + int(add_div_du_input) + int(add_div_u_input)
    # inputs_obst now has sdf (1 channel) + delta_p_prev (1 channel if use_previous_dp_input)
    obst_end = vel_end + 1
    inputs_u = unique_array[..., 0:vel_end]
    inputs_obst = unique_array[..., vel_end:obst_end]
    outputs = unique_array[..., obst_end:obst_end+1]

    return inputs_u, inputs_obst, outputs

  def get_representative_factors(self, blocks_data: np.ndarray, spatial_tucker_ranks):
    # Support for non-cubic blocks
    inputs_u, inputs_obst, outputs = blocks_data
    chunk_size = inputs_u.shape[0]
    print(f'ACTUAL Chunk size: {chunk_size}')

    # Build normalization factors based on what's included
    norm_parts = []
    if self.add_U_input:
      norm_parts.extend([self.max_abs_U_x, self.max_abs_U_y, self.max_abs_U_z])
    if self.add_dU_input:
      norm_parts.extend([self.max_abs_dU_x, self.max_abs_dU_y, self.max_abs_dU_z])
    if self.add_ddu_input:
      norm_parts.extend([self.max_abs_ddU_x, self.max_abs_ddU_y, self.max_abs_ddU_z])

    norm_parts.extend([self.max_abs_dddU_x, self.max_abs_dddU_y, self.max_abs_dddU_z])

    if self.add_p_prev_input:
      norm_parts.append(self.max_abs_p_prev)
    if self.use_previous_dp_input:
      norm_parts.append(self.max_abs_delta_p_prev)
    if self.add_ddp_prev_input:
      norm_parts.append(self.max_abs_ddp_prev)

    velocity = inputs_u / np.array(norm_parts)
    
    # Normalize obstacle/sdf and delta_p_prev separately
    sdf = inputs_obst[..., 0:1] / self.max_abs_dist  # First channel is always sdf
    obstacle = sdf
    pressure = outputs[..., 0] / self.max_abs_delta_delta_p

    # Determine spatial ranks
    spatial_ranks = tuple(spatial_tucker_ranks)
    if hasattr(self, 'block_size_z'):
      block_shape = (self.block_size_z, self.block_size_y, self.block_size_x)
    else:
      block_shape = (self.block_size, self.block_size, self.block_size)

    print(f"Calculating representative Tucker factors ({chunk_size} samples) ...")
    input_tensor = np.concatenate([velocity, obstacle], axis=-1)
    last_rank = self.last_tucker_rank
    _, input_factors = tucker(input_tensor, rank=(chunk_size,) + spatial_ranks + (last_rank,))
    _, output_factors = tucker(pressure, rank=(chunk_size,) + spatial_ranks)

    with open(self.tucker_factors_fn, 'wb') as f:
      pk.dump({'input_factors': input_factors, 'output_factors': output_factors}, f)

    return input_factors, output_factors

  def transform_data_with_tucker(self, blocks_data: np.ndarray, input_factors, output_factors) -> np.ndarray:
    inputs_u, inputs_obst, outputs = blocks_data
    chunk_size = inputs_u.shape[0]
    print(f'ACTUAL Chunk size: {chunk_size}')

    # Build normalization factors based on what's included
    norm_parts = []
    if self.add_U_input:
      norm_parts.extend([self.max_abs_U_x, self.max_abs_U_y, self.max_abs_U_z])
    if self.add_dU_input:
      norm_parts.extend([self.max_abs_dU_x, self.max_abs_dU_y, self.max_abs_dU_z])
    if self.add_ddu_input:
      norm_parts.extend([self.max_abs_ddU_x, self.max_abs_ddU_y, self.max_abs_ddU_z])
    norm_parts.extend([self.max_abs_dddU_x, self.max_abs_dddU_y, self.max_abs_dddU_z])
    if self.add_p_prev_input:
      norm_parts.append(self.max_abs_p_prev)
    if self.use_previous_dp_input:
      norm_parts.append(self.max_abs_delta_p_prev)
    if self.add_ddp_prev_input:
      norm_parts.append(self.max_abs_ddp_prev)

    velocity = inputs_u / np.array(norm_parts)
    
    # Normalize obstacle/sdf and delta_p_prev separately
    sdf = inputs_obst[..., 0:1] / self.max_abs_dist  # First channel is always sdf
    obstacle = sdf
    pressure = outputs[..., 0] / self.max_abs_delta_p

    print("Transforming data using precomputed Tucker factors...")
    input_tensor = np.concatenate([velocity, obstacle], axis=-1)
    input_core = tl.tenalg.multi_mode_dot(input_tensor, input_factors[1:], modes=[1, 2, 3, 4], transpose=True)
    output_core = tl.tenalg.multi_mode_dot(pressure, output_factors[1:], modes=[1, 2, 3], transpose=True)

    if self.flatten_data:
        input_core = input_core.reshape(chunk_size, -1)
        output_core = output_core.reshape(chunk_size, -1)

    return input_core, output_core