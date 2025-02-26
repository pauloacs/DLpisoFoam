# Standard library imports
import os
import random
import shutil
import time
import math
import itertools
from ctypes import py_object
import pickle as pk

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

class Training:

  def __init__(self, delta, block_size, var_p, var_in, hdf5_paths, n_samples, num_sims, first_t, last_t, standardization_method, n_chunks):
    self.delta = delta
    self.block_size = block_size
    self.var_in = var_in
    self.var_p = var_p
    self.paths = hdf5_paths
    self.n_samples = n_samples
    self.num_sims = num_sims[0]
    self.first_t = first_t
    self.last_t = last_t
    self.standardization_method = standardization_method
    self.n_chunks = n_chunks

####################################################################
################## Processing Simulations ##########################
####################################################################

  def sample_blocks(self, grid, x_list, obst_list, y_list, N):
    """Sample N blocks from each time step based on LHS"""

    lb = np.array([0 + self.block_size * self.delta/2 , 0 + self.block_size * self.delta/2 ])
    ub = np.array([(self.x_max-self.x_min) - self.block_size * self.delta/2, (self.y_max-self.y_min) - self.block_size * self.delta/2])

    XY = lb + (ub-lb)*lhs(2,N)
    XY_indices = (np.round(XY/self.delta)).astype(int)

    new_XY_indices = [tuple(row) for row in XY_indices]
    XY_indices = np.unique(new_XY_indices, axis=0)
    count=0
    for [jj, ii] in XY_indices:

            i_range = [int(ii - self.block_size/2), int( ii + self.block_size/2) ]
            j_range = [int(jj - self.block_size/2), int( jj + self.block_size/2) ]

            x_u = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 0:2 ]
            x_obst = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 2:3 ]
            y = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 3:4 ]

            # Remove all the blocks with delta_U = 0 and delta_p = 0
            if not ((x_u == 0).all() and (y == 0).all()):
              x_list.append(x_u)
              obst_list.append(x_obst)
              y_list.append(y)
            else:
              count += 1

    print(f'{count} blocks discarded')
    return x_list, obst_list, y_list

  def process_sim(self, i):
    """
    """

    hdf5_file = h5py.File(self.dataset_path, "r")
    data = np.array(hdf5_file["sim_data"][i:i+1, self.first_t:(self.first_t + self.last_t), ...], dtype='float32')
    top_boundary = hdf5_file["top_bound"][i:i+1, self.first_t:(self.first_t + self.last_t), ...]
    obst_boundary = hdf5_file["obst_bound"][i:i+1, self.first_t:(self.first_t + self.last_t),  ...]
    hdf5_file.close()          

    indice = utils.index(data[0,0,:,0] , -100.0 )[0]
    data_limited = data[0,0,:indice,:]

    self.x_min = round(np.min(data_limited[...,3]),2)
    self.x_max = round(np.max(data_limited[...,3]),2)

    self.y_min = round(np.min(data_limited[...,4]),2) #- 0.1
    self.y_max = round(np.max(data_limited[...,4]),2) #+ 0.1

    ######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

    X0, Y0 = utils.create_uniform_grid(self.x_min, self.x_max, self.y_min, self.y_max, self.delta)
    xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
    points = data_limited[...,3:5] #coordinates

    vert, weights = utils.interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case

    domain_bool, sdf = utils.domain_dist(0, top_boundary, obst_boundary, xy0)

    div = 1 #parameter defining the sliding window vertical and horizontal displacements
    
    self.grid_shape_y = int(round((self.y_max-self.y_min)/self.delta)) 
    self.grid_shape_x = int(round((self.x_max-self.x_min)/self.delta)) 

    count_ = data.shape[1]* int(self.grid_shape_y/div - self.block_size/div + 1 ) * int(self.grid_shape_x/div - self.block_size/div + 1 )

    count = 0
    cicle = 0

    #arrange data in array: #this can be put outside the j loop if the mesh is constant 

    x0 = np.min(X0)
    y0 = np.min(Y0)
    dx = self.delta
    dy = self.delta

    indices= np.zeros((X0.shape[0],2))
    obst_bool  = np.zeros((self.grid_shape_y,self.grid_shape_x,1))
    sdfunct = np.zeros((self.grid_shape_y,self.grid_shape_x,1))

    delta_p = data_limited[...,7:8] #values
    p_interp = utils.interpolate_fill(delta_p, vert, weights) 
  
    for (step, x_y) in enumerate(xy0):  
        if domain_bool[step] * (~np.isnan(p_interp[step])) :
            jj = int(round((x_y[...,0] - x0) / dx))
            ii = int(round((x_y[...,1] - y0) / dy))

            indices[step,0] = ii
            indices[step,1] = jj
            sdfunct[ii,jj,:] = sdf[step]
            obst_bool[ii,jj,:]  = int(1)

    indices = indices.astype(int)

    # Number of subsquent t's with very small variations
    self.stationary_ts = 0
    for j in range(data.shape[1]):  #100 for both data and data_rect
      # go from the last time to the first to access if the simulation is stationary
      #j = (data.shape[1] -1) - j
      data_limited = data[0,j,:indice,:]
      self.process_time_step(j, data_limited, vert, weights, indices, sdfunct)
      if self.stationary_ts > 5: 
        print('This simulation is stationary, ignoring it...')
        break

  def process_time_step(self, j, data_limited, vert, weights, indices, sdfunct):
    """
    """

    Ux = data_limited[...,0:1] #values
    Uy = data_limited[...,1:2] #values

    delta_p = data_limited[...,7:8] #values
    delta_Ux = data_limited[...,5:6] #values
    delta_Uy = data_limited[...,6:7] #values

    U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy)))
    deltaU_max_norm = np.max(np.sqrt(np.square(delta_Ux) + np.square(delta_Uy)))
    # Ignore time steps with minimal changes ...
    # there is not point in computing error metrics for these
    # it would exagerate the delta_p errors and give ~0% errors in p
    threshold = 1e-4
    print(deltaU_max_norm)
    print(U_max_norm)
    irrelevant_ts = (deltaU_max_norm/U_max_norm) < threshold or deltaU_max_norm < 1e-6 or U_max_norm < 1e-6

    if irrelevant_ts:
       print(f"\n\n Irrelevant time step, skipping it...")
       self.stationary_ts += 1
       return 0

    delta_p_adim = delta_p/pow(U_max_norm,2.0) 
    delta_Ux_adim = delta_Ux/U_max_norm 
    delta_Uy_adim = delta_Uy/U_max_norm 

    delta_p_interp = utils.interpolate_fill(delta_p_adim, vert, weights) #compared to the griddata interpolation 
    delta_Ux_interp = utils.interpolate_fill(delta_Ux_adim, vert, weights)#takes virtually no time  because "vert" and "weigths" where already calculated
    delta_Uy_interp = utils.interpolate_fill(delta_Uy_adim, vert, weights)

    # 1D vectors to 2D arrays
    grid = np.zeros(shape=(1, self.grid_shape_y, self.grid_shape_x, 4))
    grid[0,:,:,0:1][tuple(indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0],1)
    grid[0,:,:,1:2][tuple(indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0],1)
    grid[0,:,:,2:3] = sdfunct
    grid[0,:,:,3:4][tuple(indices.T)] = delta_p_interp.reshape(delta_p_interp.shape[0],1)

    x_list = []
    obst_list = []
    y_list = []

    # Setting any nan value to 0
    grid[np.isnan(grid)] = 0

    #How many rotations to do:
    N_rotation = 2
    N = int(self.n_samples/N_rotation/(self.last_t-self.first_t))

    x_list, obst_list, y_list = self.sample_blocks(grid, x_list, obst_list, y_list, N)

    # Rotate and sample
    grid_y_inverted = grid[:, ::-1, :, :]
    x_list, obst_list, y_list = self.sample_blocks(grid_y_inverted, x_list, obst_list, y_list, N)

    # Rotate and sample
#    grid_y_inverted = grid[:, :, ::-1, :]
#    x_list, obst_list, y_list = self.sample_blocks(grid_y_inverted, x_list, obst_list, y_list, N)

    # Rotate and sample
#    grid_y_inverted = grid[:, ::-1, ::-1, :]
#    x_list, obst_list, y_list = self.sample_blocks(grid_y_inverted, x_list, obst_list, y_list, N)

    if len(x_list) == 0:
       print('All blocks have been discarded, skipping time step')
       return 0

    x_array = np.array(x_list, dtype = 'float32')
    obst_array = np.array(obst_list, dtype = 'float32')
    y_array = np.array(y_list, dtype = 'float32')

    self.max_abs_Ux_list.append(np.max(np.abs(x_array[...,0])))
    self.max_abs_Uy_list.append(np.max(np.abs(x_array[...,1])))
    self.max_abs_dist_list.append(np.max(np.abs(obst_array[...,0])))

    # Setting the average pressure in each block to 0
    for step in range(y_array.shape[0]):
      y_array[step,...][obst_array[step,...] != 0] -= np.mean(y_array[step,...][obst_array[step,...] != 0])
    
    self.max_abs_p_list.append(np.max(np.abs(y_array[...,0])))

    array = np.c_[x_array,obst_array,y_array]
    
    # Removing duplicate data
    reshaped_array = array.reshape(array.shape[0], -1)
    # Find unique rows
    unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
    unique_array = array[unique_indices]

    print(f"Writting t{j+self.first_t} to {self.filename}", flush=True)
    file = tables.open_file(self.filename, mode='a')
    file.root.data.append(np.array(unique_array, dtype = 'float16'))
    file.close()

####################################################################
################## <Processing Simulations\> ##########################
####################################################################


####################################################################
############## <Processing data>
####################################################################

  def read_dataset(self):
    """
    """
    #pathCil, pathRect, pathTria , pathPlate = self.paths[0], self.paths[1], self.paths[2], self.paths[3]
    self.dataset_path = self.paths[0]

    NUM_COLUMNS = 4

    file = tables.open_file(self.filename, mode='w')
    atom = tables.Float32Atom()

    array_c = file.create_earray(file.root, 'data', atom, (0, self.block_size, self.block_size, NUM_COLUMNS))
    file.close()

    self.max_abs_Ux_list = []
    self.max_abs_Uy_list  = []
    self.max_abs_dist_list  = []
    self.max_abs_p_list  = []

    for i in range(np.sum(self.num_sims)):
      print(f"\nProcessing sim {i}/{np.sum(self.num_sims)}\n", flush=True)
      self.process_sim(i)

    self.max_abs_Ux = np.max(np.abs(self.max_abs_Ux_list))
    self.max_abs_Uy = np.max(np.abs(self.max_abs_Uy_list))
    self.max_abs_dist = np.max(np.abs(self.max_abs_dist_list))
    self.max_abs_p = np.max(np.abs(self.max_abs_p_list))

    np.savetxt('maxs', [self.max_abs_Ux, self.max_abs_Uy, self.max_abs_dist, self.max_abs_p] )

    return 0

  def apply_PCA(self, filename_flat: str, max_num_PC: int):

    file = tables.open_file(filename_flat, mode='w')
    atom = tables.Float32Atom()

    file.create_earray(file.root, 'data_flat', atom, (0, max_num_PC, 2))
    file.close()

    client = dask.distributed.Client(processes=False)

    N = int(self.n_samples * (self.num_sims))

    chunk_size = int(N/self.n_chunks)
    print('Applying incremental PCA ' + str(N//chunk_size) + ' times', flush = True)

    if (not os.path.isfile(filename_flat)) or (not os.path.isfile('ipca_input.pkl')):
      self.ipca_p = dask_ml.decomposition.IncrementalPCA(max_num_PC)
      self.ipca_input = dask_ml.decomposition.IncrementalPCA(max_num_PC)

      for i in range(int(N//chunk_size)):

        f = tables.open_file(self.filename, mode='r')
        x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:2]
        obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,2:3]
        y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,3:4]
        f.close()

        if x_array.shape[0] < max_num_PC:
          print('This chunck is too small ... skipping')
          break

        x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], 2 ))

        # Normalize to [-1,1]
        x_array_flat1 = x_array_flat[...,0:1]/self.max_abs_Ux
        x_array_flat2 = x_array_flat[...,1:2]/self.max_abs_Uy
        obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/self.max_abs_dist

        y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/self.max_abs_p

        input_flat = np.concatenate((x_array_flat1,x_array_flat2,obst_array_flat) , axis = -1)
        input_flat = input_flat.reshape((input_flat.shape[0],-1))
        y_flat = y_array_flat.reshape((y_array_flat.shape[0],-1)) 

        # Scatter input and output data
        input_dask_future = client.scatter(input_flat)
        y_dask_future = client.scatter(y_array_flat)

        # Convert futures to Dask arrays
        input_dask = dask.array.from_delayed(input_dask_future, shape=input_flat.shape, dtype=input_flat.dtype)
        y_dask = dask.array.from_delayed(y_dask_future, shape=y_array_flat.shape, dtype=y_array_flat.dtype)

        #scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask)
        scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))

        #scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask)
        scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

        inputScaled = scaler.transform(input_dask)
        yScaled = scaler1.transform(y_dask)

        self.ipca_input.partial_fit(inputScaled)
        self.ipca_p.partial_fit(yScaled)

        print('Fitted ' + str(i+1) + '/' + str(N//chunk_size), flush = True)

    else:
      print('Loading PCA arrays, as those are already available', flush=True)
      self.ipca_p = pk.load(open("ipca_p.pkl",'rb'))
      self.ipca_input = pk.load(open("ipca_input.pkl",'rb'))

    self.PC_p = np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC  #max defined to be 32 here
    self.PC_input = np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) if np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) > 1 and np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) <= max_num_PC else max_num_PC

    print('PC_p :' + str(self.PC_p), flush = True)
    print('PC_input :' + str(self.PC_input), flush = True)

    print(' Total variance from input represented: ' + str(np.sum(self.ipca_input.explained_variance_ratio_[:self.PC_input])))
    pk.dump(self.ipca_input, open("ipca_input.pkl","wb"))

    print(' Total variance from p represented: ' + str(np.sum(self.ipca_p.explained_variance_ratio_[:self.PC_p])))
    pk.dump(self.ipca_p, open("ipca_p.pkl","wb"))

    for i in range(int(N//chunk_size)):

      f = tables.open_file(self.filename, mode='r')
      x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:2] # e.g. read from disk only this part of the dataset
      obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,2:3]
      y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,3:4]
      f.close()

      if x_array.shape[0] < max_num_PC:
        print('This chunck is too small ... skipping')
        break

      x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], 2 ))

      # Normalize to [-1,1]
      x_array_flat1 = x_array_flat[...,0:1]/self.max_abs_Ux
      x_array_flat2 = x_array_flat[...,1:2]/self.max_abs_Uy
      obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/self.max_abs_dist

      input_flat = np.concatenate((x_array_flat1,x_array_flat2,obst_array_flat) , axis = -1)
      input_flat = input_flat.reshape((input_flat.shape[0],-1))

      y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/self.max_abs_p

      # Scatter input and output data
      input_dask_future = client.scatter(input_flat)
      y_dask_future = client.scatter(y_array_flat)
   
      # Convert futures to Dask arrays
      input_dask = dask.array.from_delayed(input_dask_future, shape=input_flat.shape, dtype=input_flat.dtype)
      y_dask = dask.array.from_delayed(y_dask_future, shape=y_array_flat.shape, dtype=y_array_flat.dtype)

      scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))
      scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

      inputScaled = scaler.transform(input_dask)
      yScaled = scaler1.transform(y_dask)

      input_transf = self.ipca_input.transform(input_flat)#[:,:self.PC_input]
      y_transf = self.ipca_p.transform(y_array_flat)#[:,:self.PC_input]

      array_image = np.concatenate((np.expand_dims(input_transf, axis=-1) , np.expand_dims(y_transf, axis=-1)), axis = -1)#, y_array]
      print(array_image.shape, flush = True)

      f = tables.open_file(filename_flat, mode='a')
      f.root.data_flat.append(np.array(array_image))
      f.close()
      
      print('transformed ' + str(i+1) + '/' + str(N//chunk_size), flush = True)

    client.close()

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

  def prepare_data (self, hdf5_paths: str, max_num_PC: int, outarray_fn: str = 'outarray.h5', outarray_flat_fn: str= 'outarray_flat.h5'):

    self.filename = outarray_fn
    filename_flat = outarray_flat_fn
      
    if not (os.path.isfile(outarray_fn) and os.path.isfile('maxs')):
      print('Blocks data is not available... Reading hdf5 dataset again\n')
      self.read_dataset()
    else:
      print('Blocks data is available... loading it\n')
      maxs = np.loadtxt('maxs')
      self.max_abs_Ux, self.max_abs_Uy, self.max_abs_dist, self.max_abs_p = maxs[0], maxs[1], maxs[2], maxs[3]

    if not (os.path.isfile(filename_flat) and os.path.isfile('ipca_input.pkl') and os.path.isfile('ipca_p.pkl')):
      print('Data after PCA is not available... Applying PCA \n')
      self.apply_PCA(filename_flat, max_num_PC)
    else:
      print('Data after PCA is available... loading it & stepping over PCA\n')
      self.ipca_p = pk.load(open("ipca_p.pkl",'rb'))
      self.ipca_input = pk.load(open("ipca_input.pkl",'rb'))

      self.PC_p = np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC  #max defined to be 32 here
      self.PC_input = int(np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in))

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
   
  def load_data_And_train(self, lr, batch_size, max_num_PC, model_name, beta_1, num_epoch, n_layers, width, dropout_rate, regularization, model_architecture, new_model):

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

def main_train(dataset_path, num_sims, first_t, last_t, num_epoch, lr, beta, batch_size, \
              standardization_method, n_samples, block_size, delta, max_num_PC, \
              var_p, var_in, model_architecture, dropout_rate, outarray_fn, outarray_flat_fn, regularization, new_model, n_chunks):

  new_model = new_model.lower() == 'true'

  n_layers, width = utils.define_model_arch(model_architecture)

  paths = [dataset_path]
  num_sims = [num_sims]

  model_name = f'{model_architecture}-{standardization_method}-{var_p}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'

  Train = Training(delta, block_size,var_p, var_in, paths, n_samples, num_sims, first_t, last_t, standardization_method, n_chunks)

  # If you want to read the crude dataset (hdf5) again, delete the 'outarray.h5' file
  Train.prepare_data (paths, max_num_PC, outarray_fn, outarray_flat_fn) #prepare and save data to tf records
  Train.load_data_And_train(lr, batch_size, max_num_PC, model_name, beta, num_epoch, n_layers, width, dropout_rate, regularization, model_architecture, new_model)

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
  n_samples = int(1e4) #no. of samples per simulation
  block_size = 128
  delta = 5e-3
  max_num_PC = 512 # to not exceed the width of the NN
  var_p = 0.95
  var_in = 0.95

  model_architecture = 'MLP_small'
  dropout_rate = 0.1
  regularization = 1e-4

  outarray_fn = '../blocks_dataset/outarray.h5'
  outarray_flat_fn = '../blocks_dataset/outarray_flat.h5'

  new_model = True

  main_train(dataset_path, num_sims, num_ts, num_epoch, lr, beta, batch_size, standardization_method, \
    n_samples, block_size, delta, max_num_PC, var_p, var_in, model_architecture, dropout_rate, outarray_fn, outarray_flat_fn, regularization, new_model)
