# Standard library imports
import os
import random
import math
import pickle as pk
import scipy.ndimage as ndimage

# Set environment variable for TensorFlow deterministic operations (for reproducibility)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Third-party library imports
import numpy as np
import tables
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree, distance
import gc

# Set seeds for reproducibility across libraries
random.seed(0)
np.random.seed(0)

# TensorFlow imports
import tensorflow as tf

# Enable deterministic random behavior in TensorFlow
tf.keras.utils.set_random_seed(0)

# Enable GPU memory growth for reproducibility and efficient resource use
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from . import utils
import warnings
from .data_processor import CFDDataProcessor, FeatureExtractAndWrite
from .train import Training

warnings.filterwarnings("ignore", message="Unmanaged memory use is high")
warnings.filterwarnings("ignore", message="Sending large graph of size")
warnings.filterwarnings("ignore", message="full garbage collections took")

def main_train(
  dataset_path,
  first_sim,
  last_sim,
  first_t,
  last_t,
  num_epoch,
  lr,
  beta,
  batch_size,
  standardization_method,
  n_samples_per_frame,
  block_size,
  grid_res,
  ranks,
  var_p,
  var_in,
  model_architecture,
  dropout_rate,
  gridded_h5_fn,
  outarray_flat_fn,
  regularization,
  new_model,
  chunk_size
):

  new_model = new_model.lower() == 'true'

  if 'mlp' in model_architecture.lower():
    flatten_data = True
  else:
    flatten_data = False

  n_layers, width = utils.define_model_arch(model_architecture)

  model_name = f'{model_architecture}-{standardization_method}-{var_p}-drop{dropout_rate}-lr{lr}-reg{regularization}-batch{batch_size}'

  if not os.path.isfile(gridded_h5_fn):
    print('Numpy gridded data not available ... Reading original CFD simulations hdf5 dataset\n')
    processor = CFDDataProcessor(
      grid_res=grid_res,
      block_size=block_size,
      var_p=var_p,
      var_in=var_in,
      original_dataset_path=dataset_path,
      n_samples_per_frame=n_samples_per_frame,
      first_sim=first_sim,
      last_sim=last_sim,
      first_t=first_t,
      last_t=last_t,
      standardization_method=standardization_method,
      chunk_size=chunk_size,
      gridded_h5_fn=gridded_h5_fn
    )
    processor.write_gridded_simulation_data()

  sample_indices_fn = 'sample_indices_per_sim_per_time.pkl'
  _ = utils.define_sample_indexes(
      n_samples_per_frame,
      block_size,
      grid_res,
      first_sim,
      last_sim,
      first_t,
      last_t,
      dataset_path,
      sample_indices_fn
      )

  Train = Training(var_p, var_in, standardization_method)

  if not os.path.isfile('maxs'):
    utils.calculate_and_save_block_abs_max(
      first_sim,
      last_sim,
      last_t,
      sample_indices_fn,
      gridded_h5_fn,
      block_size
    )
  else:
    print('Numpy gridded data is available... loading it\n')
    maxs = np.loadtxt('maxs')
    Train.max_abs_delta_Ux, \
      Train.max_abs_delta_Uy, \
      Train.max_abs_delta_Uz, \
      Train.max_abs_dist, \
      Train.max_abs_delta_p = maxs

  if not os.path.isfile(outarray_flat_fn):
    print('Data after tucker decomposition is not available... Applying Tucker decomposition \n')
    feature_writer = FeatureExtractAndWrite(
      grid_res=grid_res,
      block_size=block_size,
      var_p=var_p,
      var_in=var_in,
      original_dataset_path=dataset_path,
      n_samples_per_frame=n_samples_per_frame,
      first_sim=first_sim,
      last_sim=last_sim,
      first_t=first_t,
      last_t=last_t,
      standardization_method=standardization_method,
      chunk_size=chunk_size,
      gridded_h5_fn=gridded_h5_fn,
      flatten_data=flatten_data
    )
    feature_writer.write_features_to_h5(outarray_flat_fn, chunk_size, ranks)

  # If you want to read the crude dataset (hdf5) again, delete the gridded_h5_fn file
  Train.prepare_data_to_tf(gridded_h5_fn, outarray_flat_fn, flatten_data) #prepare and save data to tf records
  Train.load_data_and_train(lr, batch_size, model_name, beta, num_epoch, n_layers, width, dropout_rate, regularization, model_architecture, new_model, ranks, flatten_data)

if __name__ == '__main__':

  original_dataset_path = 'dataset_plate_deltas_5sim20t.hdf5'

  num_sims_placa = 5
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

  main_train(original_dataset_path, first_sim, last_sim, num_ts, num_epoch, lr, beta, batch_size, standardization_method, \
    n_samples_per_frame, block_size, grid_res, max_num_PC, var_p, var_in, model_architecture, dropout_rate, outarray_fn, outarray_flat_fn, regularization, new_model)
