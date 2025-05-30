# Standard library imports
import os
import random
import shutil
import time
import math
from math import ceil
import itertools
from ctypes import py_object
import pickle as pk
import scipy.ndimage as ndimage

# Set environment variable for TensorFlow deterministic operations (for reproducibility)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Third-party library imports
import numpy as np
import tables
import h5py
import matplotlib
import matplotlib.pyplot as plt
import scipy.spatial.qhull as qhull
from scipy.spatial import cKDTree as KDTree, distance
import matplotlib.path as mpltPath
from shapely.geometry import MultiPoint
from sklearn.decomposition import PCA, IncrementalPCA
import gc

# Set seeds for reproducibility across libraries
random.seed(0)
np.random.seed(0)

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose, 
    BatchNormalization, Activation, MaxPool2D, concatenate, Input
)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.gen_array_ops import inplace_update

# Enable deterministic random behavior in TensorFlow
tf.keras.utils.set_random_seed(0)

# Enable GPU memory growth for reproducibility and efficient resource use
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Dask and related imports
import dask
import dask.config
import dask.distributed
import dask_ml
import dask_ml.preprocessing
import dask_ml.decomposition

# Additional scientific computing and data processing libraries
from pyDOE import lhs

from . import utils
from .NNs import densePCA, densePCA_attention, conv1D_PCA
import warnings

warnings.filterwarnings("ignore", message="Unmanaged memory use is high")
warnings.filterwarnings("ignore", message="Sending large graph of size")
warnings.filterwarnings("ignore", message="full garbage collections took")

class Training:

  def __init__(self, grid_res, block_size, var_p, var_in, hdf5_paths, n_samples_per_frame, first_sim, last_sim, first_t, last_t, standardization_method, chunk_size):
    self.grid_res = grid_res
    self.block_size = block_size
    self.var_in = var_in
    self.var_p = var_p
    self.paths = hdf5_paths
    self.n_samples_per_frame = n_samples_per_frame
    self.first_sim = first_sim
    self.last_sim = last_sim
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.chunk_size = chunk_size

####################################################################
################## Processing Simulations ##########################
####################################################################

  def define_sample_indexes(self, n_samples_per_frame):

    # Check if self.x_min, self.x_max, etc. are not defined
    if not hasattr(self, 'x_min') or not hasattr(self, 'x_max') or not hasattr(self, 'y_min') or not hasattr(self, 'y_max') or not hasattr(self, 'z_min') or not hasattr(self, 'z_max'):
      # Needed for sampling indexes
      with h5py.File(self.dataset_path, "r") as f:
        data = np.array(f["sim_data"], dtype='float32')

      indice = utils.index(data[0, 0, :, 0], -100.0)[0]
      data_limited = data[0, 0, :indice, :]

      self.x_min = round(np.min(data_limited[..., 4]), 2)
      self.x_max = round(np.max(data_limited[..., 4]), 2)

      self.y_min = round(np.min(data_limited[..., 5]), 2)
      self.y_max = round(np.max(data_limited[..., 5]), 2)

      self.z_min = round(np.min(data_limited[..., 6]), 2)
      self.z_max = round(np.max(data_limited[..., 6]), 2)

    indices_per_sim_per_time = []
    for i_sim in range(self.first_sim, self.last_sim):
      indices_per_time = []
      for i_time in range(self.last_t - self.first_t):

        lower_bound = np.array(
          [0 + self.block_size * self.grid_res/2,
          0 + self.block_size * self.grid_res/2,
          0 + self.block_size * self.grid_res/2])

        upper_bound = np.array(
          [(self.z_max-self.z_min) - self.block_size * self.grid_res/2,
          (self.y_max-self.y_min) - self.block_size * self.grid_res/2,
          (self.x_max-self.x_min) - self.block_size * self.grid_res/2,
          ])
    
        ZYX = lower_bound + (upper_bound-lower_bound)*lhs(3, n_samples_per_frame)
        ZYX_indices = (np.round(ZYX/self.grid_res)).astype(int)
        ZYX_indices = np.unique([tuple(row) for row in ZYX_indices], axis=0)

        indices_per_time.append(ZYX_indices)
      indices_per_sim_per_time.append(indices_per_time)
    
        # Save indices_per_sim_per_time to a file
    with open('sample_indices_per_sim_per_time.pkl', 'wb') as f:
      pk.dump(indices_per_sim_per_time, f)

  def sample_blocks_chunked(self, sim, t_start, t_end, i_chunk=None, n_chunks=False):
    """Sample N blocks from each time step based on LHS"""

    inputs_u_list = []
    inputs_obst_list = []
    outputs_list = []
    use_subchunks = n_chunks > 1
    count=0

    sim = sim - self.first_sim
    for time in range(t_start, t_end):

      with tables.open_file(self.filename, mode='r') as f:
        grid = f.root.data[sim * (self.last_t - self.first_t) + time,:,:,:,:]

      ZYX_indices = self.sample_indices_per_sim_per_time[sim][time]

      if use_subchunks:
        elements_per_sub_chunk = ceil(ZYX_indices.shape[0]/n_chunks)
        i_element_start = i_chunk * elements_per_sub_chunk
        i_element_end = (i_chunk + 1) * elements_per_sub_chunk
        ZYX_indices = ZYX_indices[i_element_start:i_element_end]

      for [ii, jj, kk] in ZYX_indices:

        i_idx_fist  = int(ii - self.block_size/2)
        i_idx_last  = int(ii + self.block_size/2)

        j_idx_first = int(jj - self.block_size/2)
        j_idx_last  = int(jj + self.block_size/2)

        k_idx_first = int(kk - self.block_size/2)
        k_idx_last  = int(kk + self.block_size/2)

        inputs_u_sample      = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 0:3]
        inputs_obst_sample   = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 3:4]
        outputs_sample       = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 4:5]

        # Remove all the blocks with delta_U = 0 and delta_p = 0
        if not ((inputs_u_sample == 0).all() and (outputs_sample == 0).all()):
          inputs_u_list.append(inputs_u_sample)
          inputs_obst_list.append(inputs_obst_sample)
          outputs_list.append(outputs_sample)
        else:
          count += 1

    inputs_u = np.array(inputs_u_list)
    inputs_obst = np.array(inputs_obst_list)
    outputs = np.array(outputs_list)

    # Remove mean from each output block
    for step in range(outputs.shape[0]):
      outputs[step,...][inputs_obst[step,...] != 0] -= np.mean(outputs[step,...][inputs_obst[step,...] != 0])

    print('Removing duplicate blocks ...', flush=True)
    array = np.c_[inputs_u, inputs_obst, outputs]
    reshaped_array = array.reshape(array.shape[0], -1)
    # Find unique rows
    unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
    unique_array = array[unique_indices]
    inputs_u, inputs_obst, outputs = unique_array[...,0:3], unique_array[...,3:4], unique_array[...,4:5]

    if count > 0:
      print(f'    {count} blocks discarded')

    return inputs_u, inputs_obst, outputs


  def sample_blocks(self, sim, t_start, t_end, calculate_maxs=False):
    """Sample N blocks from each time step based on LHS"""

    inputs_u_list = []
    inputs_obst_list = []
    outputs_list = []

    count=0
    sim = sim - self.first_sim

    for time in range(t_start, t_end):

      with tables.open_file(self.filename, mode='r') as f:
        grid = f.root.data[sim * (self.last_t - self.first_t) + time,:,:,:,:]

      ZYX_indices = self.sample_indices_per_sim_per_time[sim][time]

      for [ii, jj, kk] in ZYX_indices:

        i_idx_fist  = int(ii - self.block_size/2)
        i_idx_last  = int(ii + self.block_size/2)

        j_idx_first = int(jj - self.block_size/2)
        j_idx_last  = int(jj + self.block_size/2)

        k_idx_first = int(kk - self.block_size/2)
        k_idx_last  = int(kk + self.block_size/2)

        inputs_u_sample      = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 0:3]
        inputs_obst_sample   = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 3:4]
        outputs_sample       = grid[i_idx_fist:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 4:5]

        # Remove all the blocks with delta_U = 0 and delta_p = 0
        if not ((inputs_u_sample == 0).all() and (outputs_sample == 0).all()):
          inputs_u_list.append(inputs_u_sample)
          inputs_obst_list.append(inputs_obst_sample)
          outputs_list.append(outputs_sample)
        else:
          count += 1

    inputs_u = np.array(inputs_u_list)
    inputs_obst = np.array(inputs_obst_list)
    outputs = np.array(outputs_list)

    # Remove mean from each output block
    for step in range(outputs.shape[0]):
      # print('before')
      # print(outputs[step,...].max())
      outputs[step,...][inputs_obst[step,...] != 0] -= np.mean(outputs[step,...][inputs_obst[step,...] != 0])
      # print('after')
      # print(outputs[step,...].mean())
      # print(np.abs(outputs[step,...]).max())
      #print('\n')    # Setting the average pressure in each block to 0
      
    print('Removing duplicate blocks ...', flush=True)
    array = np.c_[inputs_u, inputs_obst, outputs]
    reshaped_array = array.reshape(array.shape[0], -1)
    # Find unique rows
    unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
    unique_array = array[unique_indices]
    inputs_u, inputs_obst, outputs = unique_array[...,0:3], unique_array[...,3:4], unique_array[...,4:5]

    if calculate_maxs:
      self.max_abs_delta_Ux = max(np.abs(inputs_u[..., 0]).max(), self.max_abs_delta_Ux)
      self.max_abs_delta_Uy = max(np.abs(inputs_u[..., 1]).max(), self.max_abs_delta_Uy)
      self.max_abs_delta_Uz = max(np.abs(inputs_u[..., 2]).max(), self.max_abs_delta_Uz)
      self.max_abs_dist = max(np.abs(inputs_obst).max(), self.max_abs_dist)
      self.max_abs_delta_p = max(np.abs(outputs).max(), self.max_abs_delta_p)

    if count > 0:
      print(f'    {count} blocks discarded')

    return inputs_u, inputs_obst, outputs

  def write_sim_fields(self, i):
    """
    """
    with h5py.File(self.dataset_path, "r") as f:
      data          = np.array(f["sim_data"][i:i+1, self.first_t:(self.first_t + self.last_t), ...], dtype='float32')
      obst_boundary = np.array(f["obst_bound"][i, 0, ...], dtype='float32')
      y_bot_boundary = np.array(f["y_bot_bound"][i, 0, ...], dtype='float32')
      z_bot_boundary = np.array(f["z_bot_bound"][i, 0, ...], dtype='float32')
      y_top_boundary = np.array(f["y_top_bound"][i, 0, ...], dtype='float32')
      z_top_boundary = np.array(f["z_top_bound"][i, 0, ...], dtype='float32')

    indice = utils.index(data[0,0,:,0] , -100.0 )[0]
    data_limited = data[0,0,:indice,:]

    self.x_min = round(np.min(data_limited[...,4]),2)
    self.x_max = round(np.max(data_limited[...,4]),2)

    self.y_min = round(np.min(data_limited[...,5]),2) #- 0.1
    self.y_max = round(np.max(data_limited[...,5]),2) #+ 0.1

    self.z_min = round(np.min(data_limited[...,6]),2)
    self.z_max = round(np.max(data_limited[...,6]),2)

    ######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

    ## NOTE: On computational demand:
    # Equivalent to the 2D SM - block_size=128, grid_res=2.5e-4
    # grid_res 2.5e-4 -> X * Y * Z = 3000* 400 * 400 = 480M points
    # Currently manageble solution for 3D - block_size=32, grid_res=1e-3 - 1/4 the resolution
    # grid_res 1e-3 -> 600k points

    # Python regular grid x, and y coordinates
    X0, Y0, Z0 = utils.create_uniform_grid(self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max, self.grid_res)
    xyz0 = np.concatenate((np.expand_dims(X0, axis=1), np.expand_dims(Y0, axis=1), np.expand_dims(Z0, axis=1)), axis=-1)
    
    # CFD mesh cell centers x, y and z coordinates
    points = data_limited[...,4:7]

    # Problematic points for computational demand:
    # 1. utils.interp_weights
    #    a) Already gone from barycentric interp to IDW
    #    b) Is it necessary to use NN???
    # 2. utils.domain_dist
    #    a) Already using only 1/5 of the boundary points ...

    #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case
    vert, weights = utils.interp_weights(points, xyz0)

    boundaries_list = [obst_boundary, y_bot_boundary, z_bot_boundary, y_top_boundary, z_top_boundary]
    domain_bool, sdf = utils.domain_dist(0, boundaries_list, xyz0, self.grid_res)

    div = 1 #parameter defining the sliding window vertical and horizontal displacements
    
    self.grid_shape_z = int(round((self.z_max-self.z_min)/self.grid_res)) 
    self.grid_shape_y = int(round((self.y_max-self.y_min)/self.grid_res)) 
    self.grid_shape_x = int(round((self.x_max-self.x_min)/self.grid_res)) 

    count_ = data.shape[1]* int(self.grid_shape_y/div - self.block_size/div + 1 ) * int(self.grid_shape_x/div - self.block_size/div + 1 ) * int(self.grid_shape_z/div - self.block_size/div + 1 )

    count = 0
    cicle = 0

    #arrange data in array: #this can be put outside the j loop if the mesh is constant 
    x0 = np.min(X0)
    y0 = np.min(Y0)
    z0 = np.min(Z0)

    dx = self.grid_res
    dy = self.grid_res
    dz = self.grid_res

    indices= np.zeros((X0.shape[0],3))
    obst_bool = np.zeros((self.grid_shape_z, self.grid_shape_y,self.grid_shape_x, 1))
    sdfunct = obst_bool.copy()

    delta_p = data_limited[...,10:11]
    p_interp = utils.interpolate_fill(delta_p, vert, weights) 

    for (step, x_y_z) in enumerate(xyz0):
        if domain_bool[step] * (~np.isnan(p_interp[step])) :
            ii = int(round((x_y_z[...,2] - z0) / dz))
            jj = int(round((x_y_z[...,1] - y0) / dy))
            kk = int(round((x_y_z[...,0] - x0) / dx))

            indices[step,0] = ii
            indices[step,1] = jj
            indices[step,2] = kk

            sdfunct[ii, jj, kk, :] = sdf[step]
            obst_bool[ii, jj, kk, :]  = int(1)

    indices = indices.astype(int)

    #How many rotations to do:
    N_rotation = 1
    #N_blocks_per_frame = int(self.n_samples_per_frame/N_rotation/(self.last_t-self.first_t))
    #self.define_sample_indexes(self.n_samples_per_frame)

    # Number of subsquent t's with very small variations
    self.stationary_ts = 0
    for j in range(data.shape[1]):  #100 for both data and data_rect
      # go from the last time to the first to access if the simulation is stationary
      #j = (data.shape[1] -1) - j
      data_limited = data[0,j,:indice,:]
      self.write_time_step_fields(j, data_limited, vert, weights, indices, sdfunct)
      if self.stationary_ts > 5: 
        print('This simulation is stationary, ignoring it...')
        break


  def write_time_step_fields(self, j, data_limited, vert, weights, indices, sdfunct):
    """
    """

    Ux = data_limited[...,0:1]
    Uy = data_limited[...,1:2]
    Uz = data_limited[...,2:3]

    delta_p = data_limited[...,10:11]
    delta_Ux = data_limited[...,7:8]
    delta_Uy = data_limited[...,8:9]
    delta_Uz = data_limited[...,9:10]

    U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy) + np.square(Uz)))
    deltaU_max_norm = np.max(np.sqrt(np.square(delta_Ux) + np.square(delta_Uy), np.square(delta_Uz)))

    # Ignore time steps with minimal changes ...
    # there is not point in computing error metrics for these
    # it would exagerate the delta_p errors and give ~0% errors in p
    threshold = 1e-4
    print(f"deltaU_max_norm = {deltaU_max_norm}")
    print(f"U_max_norm      = {U_max_norm}")
    irrelevant_ts = (deltaU_max_norm/U_max_norm) < threshold or deltaU_max_norm < 1e-6 or U_max_norm < 1e-6

    if irrelevant_ts:
      print(f"\n\n Irrelevant time step, skipping it...")
      self.stationary_ts += 1
      return 0

    delta_p_adim = delta_p/pow(U_max_norm,2.0) 
    delta_Ux_adim = delta_Ux/U_max_norm 
    delta_Uy_adim = delta_Uy/U_max_norm
    delta_Uz_adim = delta_Uz/U_max_norm

    delta_p_interp = utils.interpolate_fill(delta_p_adim, vert, weights) #compared to the griddata interpolation 
    delta_Ux_interp = utils.interpolate_fill(delta_Ux_adim, vert, weights)#takes virtually no time  because "vert" and "weigths" where already calculated
    delta_Uy_interp = utils.interpolate_fill(delta_Uy_adim, vert, weights)
    delta_Uz_interp = utils.interpolate_fill(delta_Uz_adim, vert, weights)

    filter_tuple = (2, 2, 2)
    # 1D vectors to 3D arrays
    grid = np.zeros(shape=(self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 5))
    grid[:,:,:,0:1][tuple(indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0], 1)
    #grid[:,:,:,0] = ndimage.gaussian_filter(grid[:,:,:,0], sigma=filter_tuple, order=0)

    grid[:,:,:,1:2][tuple(indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0], 1)
    #grid[:,:,:,1] = ndimage.gaussian_filter(grid[:,:,:,1], sigma=filter_tuple, order=0)

    grid[:,:,:,2:3][tuple(indices.T)] = delta_Uz_interp.reshape(delta_Uz_interp.shape[0], 1)
    #grid[:,:,:,2] = ndimage.gaussian_filter(grid[:,:,:,2], sigma=filter_tuple, order=0)

    grid[:,:,:,3:4] = sdfunct
    grid[:,:,:,4:5][tuple(indices.T)] = delta_p_interp.reshape(delta_p_interp.shape[0], 1)

    # Setting any nan value to 0
    grid[np.isnan(grid)] = 0

    # Saving simulation data - not the blocks ... blocks data size can quickly become too much
    print(f"Writting t{j+self.first_t} to {self.filename}", flush=True)
    with tables.open_file(self.filename, mode='a') as file:
      file.root.data.append(np.array(np.expand_dims(grid, axis=0), dtype = 'float64'))

####################################################################
################## <Processing Simulations\> ##########################
####################################################################


####################################################################
############## <Processing data>
####################################################################

  def write_gridded_simulation_data(self) -> None:
    """
    """
    print(f'########## Writting CFD mesh data to a grid -> {self.filename} ############')
    #self.dataset_path = self.paths[0]

    NUM_COLUMNS = 5

    file = tables.open_file(self.filename, mode='w')
    atom = tables.Float32Atom()

    # Maybe I should have one array for sim, because the shape of the grid might be different..
    # For now I'll just assume...
    self.grid_shape_z = 0.1/self.grid_res
    self.grid_shape_y = 0.1/self.grid_res
    self.grid_shape_x = 1/self.grid_res

    array_c = file.create_earray(file.root, 'data', atom, (0, self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, NUM_COLUMNS))
    file.close()

    # Write simulation data in numpy grid to hdf5
    for i in range(self.first_sim, self.last_sim):
      print(f"\nProcessing sim {i+1}/{self.last_sim - self.first_sim}\n", flush=True)
      self.write_sim_fields(i)

  def calculate_block_abs_max(self) -> None:
    """
    """
    self.max_abs_delta_Ux = 0
    self.max_abs_delta_Uy = 0
    self.max_abs_delta_Uz = 0
    self.max_abs_dist = 0
    self.max_abs_delta_p = 0

    with open('sample_indices_per_sim_per_time.pkl', 'rb') as f:
      self.sample_indices_per_sim_per_time = pk.load(f)
      
    print('Calculating absolute maxs to normalize data...')
    for sim in range(self.first_sim, self.last_sim):
      for time in range(self.last_t - self.first_t):
        _ = self.sample_blocks(sim,
          t_start=time,
          t_end=time+1,
          calculate_maxs=True)   

    np.savetxt('maxs',
      [self.max_abs_delta_Ux,
        self.max_abs_delta_Uy,
        self.max_abs_delta_Uz,
        self.max_abs_dist,
        self.max_abs_delta_p])


  def fit_PCA_to_blocks(self, blocks_data: np.ndarray, client) -> None:
    """
    Fits Incremental PCA (IPCA) to input and output data blocks using Dask for distributed computation.

    Parameters:
    -----------
    blocks_data : np.NDArray
      A tuple containing three NumPy arrays:
      - inputs_u (np.NDArray): Input data representing velocity or similar features.
      - inputs_obst (np.NDArray): Input data representing obstacle or boundary features.
      - outputs (np.NDArray): Output data representing the target values.

    Returns:
    --------
    None
      This method updates the internal IPCA models (`self.ipca_input` and `self.ipca_p`) with the scaled and transformed data.

    Notes:
    ------
    - The method uses Dask for distributed computation to handle large datasets.
    - Data is scaled using `dask_ml.preprocessing.StandardScaler` before being passed to the IPCA models.
    - The method assumes that `self.ipca_input` and `self.ipca_p` are pre-initialized Incremental PCA models.
    """

    inputs_u, inputs_obst, outputs = blocks_data
    chunk_size = inputs_u.shape[0]
    print(f'Actual chunk size: {chunk_size}')

    # Flatten & Normalize to [-1, 1]
    x_array_flat = inputs_u.reshape((chunk_size, -1, 3)) / [
        self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz]
    obst_array_flat = inputs_obst.reshape((chunk_size, -1, 1)) / self.max_abs_dist
    y_array_flat = outputs.reshape((chunk_size, -1)) / self.max_abs_delta_p

    input_flat = np.concatenate((x_array_flat, obst_array_flat), axis=-1).reshape((chunk_size, -1))
    y_flat = y_array_flat.reshape((y_array_flat.shape[0], -1))

    input_flat = np.concatenate((inputs_u, inputs_obst), axis=-1).reshape((chunk_size, -1))
    y_flat = outputs.reshape((chunk_size, -1))

    # Scatter input and output data
    input_dask_future = client.scatter(input_flat)
    y_dask_future = client.scatter(y_flat)

    # Convert futures to Dask arrays
    input_dask = dask.array.from_delayed(input_dask_future, shape=input_flat.shape, dtype=input_flat.dtype)
    y_dask = dask.array.from_delayed(y_dask_future, shape=y_flat.shape, dtype=y_flat.dtype)

    del input_dask_future, y_dask_future

    # #scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask)
    # scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))

    # #scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask)
    # scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

    # inputScaled = scaler.transform(input_dask).compute()
    # yScaled = scaler1.transform(y_dask).compute()

    # self.ipca_input.partial_fit(inputScaled)
    # self.ipca_p.partial_fit(yScaled)

    # del input_dask, y_dask, inputScaled, yScaled
    # gc.collect()

    # Rechunk to avoid large memory spikes
    input_dask = input_dask.rechunk({0: "auto", 1: input_dask.shape[1]})
    y_dask = y_dask.rechunk({0: "auto", 1: y_dask.shape[1]})

    # Apply StandardScaler
    scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask)
    scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask)

    inputScaled = scaler.transform(input_dask)
    yScaled = scaler1.transform(y_dask)

    # Ensure proper chunking
    inputScaled = inputScaled.rechunk({0: "auto", 1: inputScaled.shape[1]})
    yScaled = yScaled.rechunk({0: "auto", 1: yScaled.shape[1]})

    gc.collect()
    # Incremental PCA
    self.ipca_input.partial_fit(inputScaled)
    self.ipca_p.partial_fit(yScaled)

    del input_dask, y_dask, inputScaled, yScaled, scaler, scaler1
    gc.collect()


  def fit_PCA(self, chunks_per_sim: int, n_times_per_chunk: int, n_sub_chunks: int) -> None:

    client = dask.distributed.Client(processes=False)

    for sim in range(self.first_sim, self.last_sim):
      print(f'Fitting sim {sim+1}/[{self.first_sim}, {self.last_sim}]...')
      for i_chunk in range(chunks_per_sim):

        # print(f' -Sampling block data for chunk {i_chunk+1}/{chunks_per_sim}', flush = True)
        # blocks_data = self.sample_blocks(sim,
        #   t_start=(i_chunk-1) * n_times_per_chunk,
        #   t_end=i_chunk * n_times_per_chunk)

        gc.collect()

        print(f' -Fitting chunk {i_chunk+1}/{chunks_per_sim} for sim {sim+1}/[{self.first_sim}, {self.last_sim}]', flush = True)

        # Create sub-chunks if necessary
        for sub_chunk in range(n_sub_chunks):
          gc.collect()

          # If there are subchunks, there is only 1 time at each chunk
          print(f' -Sampling block data for chunk {i_chunk+1}/{chunks_per_sim} - subchunk {sub_chunk+1}/{n_sub_chunks}', flush = True)
          blocks_data = self.sample_blocks_chunked(sim,
            t_start=(i_chunk-1) * n_times_per_chunk,
            t_end=i_chunk * n_times_per_chunk,
            i_chunk=sub_chunk,
            n_chunks=n_sub_chunks
            )

          # blocks_data_chunked = tuple(
          #     data[sub_chunk * elements_per_sub_chunk: (sub_chunk + 1) * elements_per_sub_chunk].copy()
          #     for data in blocks_data
          #   )

          print(f' -Fitting PCA for chunk {i_chunk+1}/{chunks_per_sim} - subchunk {sub_chunk+1}/{n_sub_chunks}', flush = True)
          self.fit_PCA_to_blocks(blocks_data, client)

    client.close()

  def transform_data_to_PC(self, blocks_data: np.ndarray, client) -> np.ndarray:

    inputs_u, inputs_obst, outputs = blocks_data
    chunk_size = inputs_u.shape[0]
    print(f'ACTUAL Chunk size: {chunk_size}')

    # Flatten & Normalize to [-1, 1]
    x_array_flat = inputs_u.reshape((chunk_size, -1, 3)) / [
        self.max_abs_delta_Ux, self.max_abs_delta_Uy, self.max_abs_delta_Uz]
    obst_array_flat = inputs_obst.reshape((chunk_size, -1, 1)) / self.max_abs_dist
    y_array_flat = outputs.reshape((chunk_size, -1)) / self.max_abs_delta_p

    input_flat = np.concatenate((x_array_flat, obst_array_flat), axis=-1).reshape((chunk_size, -1))
    y_flat = y_array_flat.reshape((y_array_flat.shape[0], -1))

    # Scatter input and output data
    input_dask_future = client.scatter(input_flat)
    y_dask_future = client.scatter(y_flat)

    # Convert futures to Dask arrays
    input_dask = dask.array.from_delayed(input_dask_future, shape=input_flat.shape, dtype=input_flat.dtype)
    y_dask = dask.array.from_delayed(y_dask_future, shape=y_flat.shape, dtype=y_flat.dtype)

    scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))
    scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

    inputScaled = scaler.transform(input_dask)
    yScaled = scaler1.transform(y_dask)

    input_transf = self.ipca_input.transform(inputScaled)
    y_transf = self.ipca_p.transform(yScaled)

    del input_dask, y_dask
    gc.collect()
    
    PC_data = np.concatenate((np.expand_dims(input_transf, axis=-1), np.expand_dims(y_transf, axis=-1)), axis=-1)

    return PC_data


  def transform_and_write_blocks_to_PC(
    self,
    filename_flat: str,
    chunks_per_sim: int,
    n_times_per_chunk: int,
    n_sub_chunks: int,
    max_num_PC: int = 256) -> None:
    
    with tables.open_file(filename_flat, mode='w') as file:
      atom = tables.Float32Atom()
      file.create_earray(file.root, 'data_flat', atom, (0, max_num_PC, 2))

    client = dask.distributed.Client(processes=False)

    for sim in range(self.first_sim, self.last_sim):
      print(f'Transforming data from sim {sim+1}/[{self.first_sim}, {self.last_sim}]...')
      for i_chunk in range(chunks_per_sim):

        # print(f' -Sampling block data for chunk {i_chunk+1}/{chunks_per_sim}', flush = True)
        # blocks_data = self.sample_blocks(sim,
        #   t_start= i_chunk * n_times,
        #   t_end= (i_chunk+1) * n_times)

        print(f' -Transforming chunk {i_chunk+1}/{chunks_per_sim} for sim {sim+1}/[{self.first_sim}, {self.last_sim}]', flush = True)
        # Create sub-chunks if necessary
        for sub_chunk in range(n_sub_chunks):
          print(f' -Sampling block data for chunk {i_chunk+1}/{chunks_per_sim} - subchunk {sub_chunk+1}/{n_sub_chunks}', flush = True)
          blocks_data = self.sample_blocks_chunked(sim,
            t_start=(i_chunk-1) * n_times_per_chunk,
            t_end=i_chunk * n_times_per_chunk,
            i_chunk=sub_chunk,
            n_chunks=n_sub_chunks
            )

          print(f' -Transforming grid data to PC for chunk {i_chunk+1}/{chunks_per_sim} - subchunk {sub_chunk+1}/{n_sub_chunks}', flush = True)
          PC_data = self.transform_data_to_PC(blocks_data, client)

          # Write Principal Component data
          with tables.open_file(filename_flat, mode='a') as f:
            f.root.data_flat.append(np.array(PC_data))

    client.close()

  def write_PC_to_h5(self, filename_flat: str, chunk_size: int = 500, max_num_PC: int = 256) -> None:
    
    with open('sample_indices_per_sim_per_time.pkl', 'rb') as f:
      self.sample_indices_per_sim_per_time = pk.load(f)
    
    total_times = self.last_t-self.first_t
    if chunk_size >  self.n_samples_per_frame:
      n_times_per_chunk = chunk_size // self.n_samples_per_frame
      n_chunks_per_sim = ceil(total_times/n_times_per_chunk)
      n_sub_chunks = 1
    # If each frame does not fit into the chunck, create sub-chunks
    else:
      n_sub_chunks = ceil(self.n_samples_per_frame / chunk_size)
      n_times_per_chunk = 1
      n_chunks_per_sim = total_times

    if not os.path.isfile('ipca_input.pkl'):
      self.ipca_p = dask_ml.decomposition.IncrementalPCA(max_num_PC)
      self.ipca_input = dask_ml.decomposition.IncrementalPCA(max_num_PC)
      self.fit_PCA(n_chunks_per_sim, n_times_per_chunk, n_sub_chunks)
    else:
      print('Loading PCA arrays, as those are already available', flush=True)
      self.ipca_p = pk.load(open("ipca_p.pkl",'rb'))
      self.ipca_input = pk.load(open("ipca_input.pkl",'rb'))

    pc_in_explained_var_cumulative = self.ipca_input.explained_variance_ratio_.cumsum()
    if pc_in_explained_var_cumulative[-1] < self.var_in:
      self.PC_input = max_num_PC
    else:
      self.PC_input = np.argmax(pc_in_explained_var_cumulative > self.var_in)

    pc_p_explained_var_cumulative = self.ipca_p.explained_variance_ratio_.cumsum()
    if pc_p_explained_var_cumulative[-1] < self.var_p:
      self.PC_p = max_num_PC
    else:
      self.PC_p = np.argmax(pc_p_explained_var_cumulative > self.var_p)
		
    print(f'PC_p : {self.PC_p}', flush = True)
    print(f'PC_input: {self.PC_input}', flush = True)

    print(' Total variance from input represented: ' + str(np.sum(self.ipca_input.explained_variance_ratio_[:self.PC_input])))
    pk.dump(self.ipca_input, open("ipca_input.pkl","wb"))

    print(' Total variance from p represented: ' + str(np.sum(self.ipca_p.explained_variance_ratio_[:self.PC_p])))
    pk.dump(self.ipca_p, open("ipca_p.pkl","wb"))

    self.transform_and_write_blocks_to_PC(filename_flat, n_chunks_per_sim, n_times_per_chunk, n_sub_chunks, max_num_PC)

####################################################################
############## <Processing data\>
####################################################################

  @tf.function
  def train_step(self, inputs, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(inputs, training=True)
      loss=self.loss_object(labels, predictions)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  #@tf.function
  def perform_validation(self):

    losses = []

    for (x_val, y_val) in self.test_dataset:
      x_val = tf.cast(x_val[...,0,0], dtype='float32')
      y_val = tf.cast(y_val[...,0,0], dtype='float32')

      val_logits = self.model(x_val)
      val_loss = self.loss_object(y_true = y_val , y_pred = val_logits)
      losses.append(val_loss)

    return losses
  
  def my_mse_loss(self):
    def loss_f(y_true, y_pred):

      loss = tf.reduce_mean(tf.square(y_true - y_pred) )

      return 1e6 * loss
    return loss_f

  def prepare_data_to_tf(self, hdf5_paths: str, max_num_PC: int, outarray_fn: str = 'gridded_sim_data.h5', outarray_flat_fn: str= 'PC_data.h5'):

    self.dataset_path = self.paths[0]
    self.filename = outarray_fn
    filename_flat = outarray_flat_fn
      
    if not os.path.isfile(self.filename):
      print('Numpy gridded data not available ... Reading original CFD simulations hdf5 dataset\n')
      self.write_gridded_simulation_data()


    self.define_sample_indexes(self.n_samples_per_frame)

    if not os.path.isfile('maxs'):
      self.calculate_block_abs_max()
    else:
      print('Numpy gridded data is available... loading it\n')
      maxs = np.loadtxt('maxs')
      self.max_abs_delta_Ux, \
        self.max_abs_delta_Uy, \
        self.max_abs_delta_Uz, \
        self.max_abs_dist, \
        self.max_abs_delta_p = maxs

    if not (os.path.isfile(filename_flat) and os.path.isfile('ipca_input.pkl') and os.path.isfile('ipca_p.pkl')):
      print('Data after PCA is not available... Applying PCA \n')
      self.write_PC_to_h5(filename_flat, self.chunk_size, max_num_PC)
    else:
      print('Data after PCA is available... loading it & stepping over PCA\n')
      self.ipca_p = pk.load(open("ipca_p.pkl",'rb'))
      self.ipca_input = pk.load(open("ipca_input.pkl",'rb'))

      # Determine the number of principal components for outputs (PC_p)
      self.PC_p = np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p)
      if self.PC_p == 0 or self.PC_p > max_num_PC:
          self.PC_p = max_num_PC

      # Determine the number of principal components for inputs (PC_input)
      self.PC_input = np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in)
      if self.PC_input == 0 or self.PC_input > max_num_PC:
          self.PC_input = max_num_PC

    print('Loading Blocks data\n')
    f = tables.open_file(filename_flat, mode='r')
    input = f.root.data_flat[...,:self.PC_input,0] 
    output = f.root.data_flat[...,:self.PC_p,1] 
    f.close()

    standardization_method="std"
    print(f'Normalizing PCA data based on standardization method: {standardization_method}')
    x, y = utils.normalize_PCA_data(input, output, standardization_method)
    x, y = utils.unison_shuffled_copies(x, y)
    print('Data shuffled \n')
    x = x.reshape((x.shape[0], x.shape[1], 1, 1))
    y = y.reshape((y.shape[0], y.shape[1], 1, 1))

    # Convert values to compatible tf Records - much faster
    split = 0.9
    if not (os.path.isfile('train_data.tfrecords') and os.path.isfile('test_data.tfrecords')):
      print("TFRecords train and test data is not available... writing it\n")
      count = utils.write_images_to_tfr_short(x[:int(split*x.shape[0]),...], y[:int(split*y.shape[0]),...], filename="train_data")
      count = utils.write_images_to_tfr_short(x[int(split*x.shape[0]):,...], y[int(split*y.shape[0]):,...], filename="test_data")
    else:
      print("TFRecords train and test data already available, using it... If you want to write new data, delete 'train_data.tfrecords' and 'test_data.tfrecords'!\n")
    self.len_train = int(split*x.shape[0])

    return 0 
   
  def load_data_and_train(self, lr, batch_size, max_num_PC, model_name, beta_1, num_epoch, n_layers, width, dropout_rate, regularization, model_architecture, new_model):

    train_path = 'train_data.tfrecords'
    test_path = 'test_data.tfrecords'

    self.train_dataset = utils.load_dataset_tf(filename = train_path, batch_size= batch_size, buffer_size=1024)
    self.test_dataset = utils.load_dataset_tf(filename = test_path, batch_size= batch_size, buffer_size=1024)

    # Training 

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)#, decay=0.45*lr, amsgrad=True)
    self.loss_object = self.my_mse_loss()

    if new_model:
      if (model_architecture=='MLP_small') or (model_architecture=='MLP_big') or (model_architecture=='MLP_small_unet') or (model_architecture=='MLP_huge') or (model_architecture=='MLP_huger'):
        self.model = densePCA(n_layers, width, self.PC_input, self.PC_p, dropout_rate, regularization)
      elif model_architecture == 'conv1D':
        self.model = self.conv1D_PCA(n_layers, width, self.PC_input, self.PC_p, dropout_rate, regularization)
      elif model_architecture == 'MLP_attention':
        self.model = self.densePCA_attention(n_layers, width, self.PC_input, self.PC_p, dropout_rate, regularization)
      else:
        raise ValueError('Invalid NN model type')
    else:
      print(f"Loading model:{'model_' + model_name + '.h5'}")
      self.model = tf.keras.models.load_model('model_' + model_name + '.h5')

    epochs_val_losses, epochs_train_losses = [], []

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999)#, epsilon=1e-08, decay=0.45*lr, amsgrad=True)

    min_yet = 1e9

    for epoch in range(num_epoch):
      progbar = tf.keras.utils.Progbar(math.ceil(self.len_train/batch_size))
      print('Start of epoch %d' %(epoch,))
      losses_train = []
      losses_test = []

      for step, (inputs, labels) in enumerate(self.train_dataset):

        inputs = tf.cast(inputs[...,0,0], dtype='float32')
        labels = tf.cast(labels[...,0,0], dtype='float32')
        loss = self.train_step(inputs, labels)
        losses_train.append(loss)

      losses_val  = self.perform_validation()

      losses_train_mean = np.mean(losses_train)
      losses_val_mean = np.mean(losses_val)

      epochs_train_losses.append(losses_train_mean)
      epochs_val_losses.append(losses_val_mean)
      print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean)))

      progbar.update(step+1)

      # It was found that if the min_delta is too small, or patience is too high it can cause overfitting
      stopEarly = utils.Callback_EarlyStopping(epochs_val_losses, min_delta=0.1/100, patience=100)
      if stopEarly:
        print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,num_epoch))
        break

      if epoch > 20:
        mod = 'model_' + model_name + '.h5'
        if losses_val_mean < min_yet:
          print(f'saving model: {mod}', flush=True)
          self.model.save(mod)
          min_yet = losses_val_mean
    
    print("Terminating training")
    mod = 'model_' + model_name + '.h5'
    ## Plot loss vs epoch
    plt.plot(list(range(len(epochs_train_losses))), epochs_train_losses, label ='train')
    plt.plot(list(range(len(epochs_val_losses))), epochs_val_losses, label ='val')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'loss_vs_epoch_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.png')

    ## Save losses data
    np.savetxt(f'train_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_train_losses, fmt='%d')
    np.savetxt(f'test_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_val_losses, fmt='%d')
        
    return 0

def main_train(dataset_path, first_sim, last_sim, first_t, last_t, num_epoch, lr, beta, batch_size, \
              standardization_method, n_samples_per_frame, block_size, grid_res, max_num_PC, \
              var_p, var_in, model_architecture, dropout_rate, outarray_fn, outarray_flat_fn, regularization, new_model, chunk_size):

  new_model = new_model.lower() == 'true'

  n_layers, width = utils.define_model_arch(model_architecture)

  paths = [dataset_path]

  model_name = f'{model_architecture}-{standardization_method}-{var_p}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'

  Train = Training(grid_res, block_size,var_p, var_in, paths, n_samples_per_frame, first_sim, last_sim, first_t, last_t, standardization_method, chunk_size)

  # If you want to read the crude dataset (hdf5) again, delete the 'gridded_sim_data.h5' file
  Train.prepare_data_to_tf(paths, max_num_PC, outarray_fn, outarray_flat_fn) #prepare and save data to tf records
  Train.load_data_and_train(lr, batch_size, max_num_PC, model_name, beta, num_epoch, n_layers, width, dropout_rate, regularization, model_architecture, new_model)

if __name__ == '__main__':

  path_placa = 'dataset_plate_deltas_5sim20t.hdf5'
  dataset_path = [path_placa]

  num_sims_placa = 5
  num_ts = [num_sims_placa]#, num_sims_rect, num_sims_tria, num_sims_placa]
  num_ts = 5

  # Training Parameters
  num_epoch = 5000
  lr = 1e-5
  beta = 0.99
  batch_size = 1024 #*8
  ## Possible methods:
  ## 'std', 'min_max' or 'max_abs'
  standardization_method = 'std'

  # Data-Processing Parameters
  n_samples_per_frame = int(1e4) #no. of samples per simulation
  block_size = 128
  grid_res = 2.5e-4
  max_num_PC = 512 # to not exceed the width of the NN
  var_p = 0.95
  var_in = 0.95

  model_architecture = 'MLP_small'
  dropout_rate = 0.1
  regularization = 1e-4

  outarray_fn = '../blocks_dataset/gridded_sim_data.h5'
  outarray_flat_fn = '../blocks_dataset/PC_data.h5'

  new_model = True

  main_train(dataset_path, first_sim, last_sim, num_ts, num_epoch, lr, beta, batch_size, standardization_method, \
    n_samples_per_frame, block_size, grid_res, max_num_PC, var_p, var_in, model_architecture, dropout_rate, outarray_fn, outarray_flat_fn, regularization, new_model)
