# Standard library imports
import os
import random
import math

# Set environment variable for TensorFlow deterministic operations (for reproducibility)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Third-party library imports
import numpy as np
import tables
import matplotlib.pyplot as plt

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
from .neural_networks import (
    MLP, dense_attention, conv1D, FNO3d, GNN, MLP_Mixer_3D, SimpleCNN3D, SimpleCNN3D_SmoothSpecialized, Simple_multi_layer_3D, UNet3D, UNet3D_deep, UNet3D_attention, SymmetricPadding3D
)
from .utils import io_operations as utils_io
from .utils import model_utils as utils_model
from .utils import data_processing as utils_data
import warnings

warnings.filterwarnings("ignore", message="Unmanaged memory use is high")
warnings.filterwarnings("ignore", message="Sending large graph of size")
warnings.filterwarnings("ignore", message="full garbage collections took")

class Training:

  def __init__(self,
                standardization_method: str = "std",
                train_tfrecord_fn: str = 'train_data.tfrecords',
                test_tfrecord_fn: str = 'test_data.tfrecords'):

    self.standardization_method = standardization_method
    self.train_tfrecord_fn = train_tfrecord_fn
    self.test_tfrecord_fn = test_tfrecord_fn  
  
  @tf.function
  def train_step(self, inputs, labels):
    with tf.GradientTape() as tape:
      outputs = self.model(inputs, training=True)
      # Handle both tensor and dict outputs
      if isinstance(outputs, dict):
          loss = self.loss_object(labels, outputs)
      else:
          loss = self.loss_object(labels, outputs)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  #@tf.function
  def perform_validation(self, flatten_data: bool) -> list:

    losses = []

    for (x_val, y_val) in self.test_dataset:
      if flatten_data:
        x_val = tf.cast(x_val[...,0,0], dtype='float32')
        y_val = tf.cast(y_val[...,0,0], dtype='float32')

      outputs = self.model(x_val)
      # Handle both tensor and dict outputs
      if isinstance(outputs, dict):
          val_loss = self.loss_object(y_val, outputs)
      else:
          val_loss = self.loss_object(y_true=y_val, y_pred=outputs)
      losses.append(val_loss)

    return losses
  
  def my_mse_loss(self):
    def loss_f(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

      loss = tf.reduce_mean(tf.square(y_true - y_pred) )

      return 100 * loss
    return loss_f


  #  TESTING
  def my_mse_energy_loss(self, alpha_energy=0.2):
      def loss_f(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

          # Pointwise MSE
          mse = tf.reduce_mean(tf.square(y_true - y_pred))

          # Per-sample RMS amplitude
          rms_true = tf.sqrt(
              tf.reduce_mean(tf.square(y_true), axis=(1, 2, 3)) + 1e-8
          )

          rms_pred = tf.sqrt(
              tf.reduce_mean(tf.square(y_pred), axis=(1, 2, 3)) + 1e-8
          )

          # Match global amplitude / energy
          energy_loss = tf.reduce_mean(tf.square(rms_true - rms_pred))

          return 100.0 * (mse + alpha_energy * energy_loss)

      return loss_f



  def my_mixed_weighted_mse_loss(self, beta=1.0, cap=3.0, alpha=0.5):
      def loss_f(y_true, y_pred):

          error2 = tf.square(y_true - y_pred)

          mse = tf.reduce_mean(error2)

          mean_abs = tf.reduce_mean(
              tf.abs(y_true),
              axis=(1, 2, 3),
              keepdims=True
          ) + 1e-8

          rel_amp = tf.abs(y_true) / mean_abs
          rel_amp = tf.clip_by_value(rel_amp, 0.0, cap)

          weights = 1.0 + beta * rel_amp

          weighted_mse = tf.reduce_mean(weights * error2)

          loss = (1.0 - alpha) * mse + alpha * weighted_mse

          return 100.0 * loss

      return loss_f


  def lowpass_3d(self, y, pool_size=(3, 5, 11)):
      """Extract low-frequency (smooth) component via 3D average pooling."""
      return tf.nn.avg_pool3d(
          y[..., tf.newaxis],
          ksize=pool_size,
          strides=(1, 1, 1),
          padding='SAME',
          data_format='NDHWC'
      )[..., 0]


  def my_localized_decomposition_loss(self,
                                       lambda_smooth=0.3,
                                       lambda_local=0.2,
                                       pool_size=(3, 5, 11),
                                       beta=0.5,
                                       cap=2.0,
                                       alpha=0.5):
      """
      Localized loss that separately supervises smooth and local components.
      
      - Supervises p_smooth to match low-frequency target
      - Supervises p_local to match high-frequency residual
      - Main supervision on p_total with amplitude weighting
      
      Args:
          lambda_smooth: Weight for smooth head loss
          lambda_local: Weight for local head loss
          pool_size: Kernel size for low-pass filtering (z, y, x)
          beta, cap, alpha: Weighted MSE parameters for main loss
      """
      def loss_f(y_true, y_pred_dict):
          # Extract components from dict
          p_total = y_pred_dict['p_total']
          p_smooth = y_pred_dict['p_smooth']
          p_local = y_pred_dict['p_local']

          # Decompose target into smooth and local
          y_smooth = self.lowpass_3d(y_true, pool_size=pool_size)
          y_local = y_true - y_smooth

          # ---- Main loss: weighted MSE on total ----
          error2 = tf.square(y_true - p_total)
          mse = tf.reduce_mean(error2)

          mean_abs = tf.reduce_mean(
              tf.abs(y_true),
              axis=(1, 2, 3),
              keepdims=True
          ) + 1e-8

          rel_amp = tf.abs(y_true) / mean_abs
          rel_amp = tf.clip_by_value(rel_amp, 0.0, cap)
          weights = 1.0 + beta * rel_amp

          weighted_mse = tf.reduce_mean(weights * error2)
          main_loss = (1.0 - alpha) * mse + alpha * weighted_mse

          # ---- Smooth head auxiliary loss ----
          smooth_loss = tf.reduce_mean(tf.square(y_smooth - p_smooth))

          # ---- Local head auxiliary loss ----
          local_loss = tf.reduce_mean(tf.square(y_local - p_local))

          # Combined loss
          total_loss = (
              main_loss +
              lambda_smooth * smooth_loss +
              lambda_local * local_loss
          )

          return 100.0 * total_loss

      return loss_f



  def prepare_data_to_tf(
    self,
    outarray_flat_fn: str= 'features_data.h5',
    normalization_factors_fn: str = 'mean_std.npz',
    flatten_data: bool = False,
    load_existing_normalization: bool = False):

    filename_flat = outarray_flat_fn
     
    print('Loading feature data (tucker cores extracted from blocks)\n')
    with tables.open_file(filename_flat, mode='r') as f:
      input = f.root.inputs[...] 
      output = f.root.outputs[...] 

    standardization_method="std"
    print(f'Normalizing feature data based on standardization method: {standardization_method}')
    x, y = utils_data.normalize_feature_data(input, output, standardization_method, normalization_factors_fn=normalization_factors_fn, load_existing=load_existing_normalization)

    split = 0.9

    # Shuffle data
    #x, y = utils_data.unison_shuffled_copies(x, y)
    #print('Data shuffled \n')

    n_train = int(split * x.shape[0])

    # Chronological split:
    # train = earlier samples
    # validation = last samples
    print("Using chronological split: validation uses last time samples\n")


    if flatten_data:
      x = x.reshape((x.shape[0], x.shape[1], 1, 1))
      y = y.reshape((y.shape[0], y.shape[1], 1, 1))

    # Convert values to compatible tf Records - much faster
    split = 0.9
    if not (os.path.isfile(self.train_tfrecord_fn) and os.path.isfile(self.test_tfrecord_fn)):
      print("TFRecords train and test data is not available... writing it\n")

      utils_io.write_images_to_tfr_short(
          x[:n_train, ...],
          y[:n_train, ...],
          filename=self.train_tfrecord_fn,
      )

      utils_io.write_images_to_tfr_short(
          x[n_train:, ...],
          y[n_train:, ...],
          filename=self.test_tfrecord_fn,
      )

    else:
      print(f"TFRecords train and test data already available, using it... If you want to write new data, delete '{self.train_tfrecord_fn}' and '{self.test_tfrecord_fn}'!\n")
    self.len_train = n_train

    return 0 
  
  def load_data_and_train(self,
    lr: float,
    batch_size: int,
    model_name: str,
    beta_1: float,
    num_epoch: int,
    n_layers: int,
    width: int,
    dropout_rate: float,
    regularization: float,
    model_architecture: str,
    new_model: bool,
    spatial_tucker_ranks: tuple,
    flatten_data: bool,
    weights_fn: str,
    model_h5_path: str='',
    last_tucker_rank: int=4,
    use_feature_decomposition: bool=True,
    block_size=None) -> None:

    train_path = self.train_tfrecord_fn
    test_path = self.test_tfrecord_fn

    self.train_dataset = utils_io.load_dataset_tf(filename = train_path, batch_size = batch_size, buffer_size=1024)
    self.test_dataset = utils_io.load_dataset_tf(filename = test_path, batch_size = batch_size, buffer_size=1024)

    # Training 

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)#, decay=0.45*lr, amsgrad=True)
    
    # Set loss based on model architecture
    # Models that use decomposition loss
    _DECOMPOSITION_MODELS = ('cnn_smooth_specialized',)
    model_arch_lower = model_architecture.lower()
    
    if model_arch_lower in _DECOMPOSITION_MODELS:
        self.loss_object = self.my_localized_decomposition_loss(
            lambda_smooth=0.3,
            lambda_local=0.2,
            pool_size=(3, 5, 11),
            beta=0.5,
            cap=2.0,
            alpha=0.5
        )
    else:
        # Default weighted MSE loss for standard models
        self.loss_object = self.my_mixed_weighted_mse_loss(beta=0.5, cap=2.0, alpha=0.5)

    print(model_architecture)
    def prod(iterable):
        from functools import reduce
        from operator import mul
        return reduce(mul, iterable, 1)

    # When feature decomposition is disabled, flat models (MLP, conv1d, etc.) are not applicable;
    # default to CNN which operates directly on the 3D block.
    _FLAT_MODELS = ('mlp_small', 'mlp_big', 'mlp_small_unet', 'mlp_huge', 'mlp_huger', 'conv1d', 'mlp_attention')
    effective_model_arch = model_architecture
    if not use_feature_decomposition and model_architecture.lower() in _FLAT_MODELS:
      print(f"[use_feature_decomposition=False] model_architecture='{model_architecture}' is a flat "
            "model — overriding to 'cnn'. Set model_architecture to a 3D model to suppress this.")
      effective_model_arch = 'cnn'

    # Determine spatial dims to pass to 3D models
    spatial_dims = spatial_tucker_ranks if use_feature_decomposition else block_size

    input_features_size = prod(spatial_dims) * last_tucker_rank
    output_features_size = prod(spatial_dims)
    if new_model:
      model_architecture_norm = effective_model_arch.lower()
      match model_architecture_norm:
        case 'mlp_small' | 'mlp_big' | 'mlp_small_unet' | 'mlp_huge' | 'mlp_huger':
          self.model = MLP(
          n_layers, width,
          input_features_size,
          output_features_size,
          dropout_rate, regularization
          )
        case 'conv1d':
          self.model = conv1D(
          n_layers, width,
          input_features_size,
          output_features_size,
          dropout_rate, regularization
          )
        case 'mlp_attention':
          self.model = dense_attention(
          n_layers, width,
          input_features_size,
          output_features_size,
          dropout_rate, regularization
          )
        case 'gnn':
            self.model = GNN(spatial_dims)  # GNN does not expose in_channels
        case 'fno3d':
          self.model = FNO3d(spatial_dims, in_channels=last_tucker_rank)
        case 'mixer':
          self.model = MLP_Mixer_3D(n_layers, spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
        case 'cnn':
          self.model = SimpleCNN3D(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
        case 'cnn_smooth_specialized':
          self.model = SimpleCNN3D_SmoothSpecialized(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
        case 'unet3d':
          self.model = UNet3D(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
        case 'unet3d_deep':
          self.model = UNet3D_deep(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization, base_filters=4, n_levels=3)
        case 'unet3d_attention':
          self.model = UNet3D_attention(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
        case 'multi_layer_3d':
          self.model = Simple_multi_layer_3D(
            spatial_dims, in_channels=last_tucker_rank, n_layers=n_layers, width=width,
            dropout_rate=dropout_rate, regularization=regularization
          )
        case _:
          raise ValueError('Invalid NN model type')
        ## ANOTHER IDEA? WHAT ABOUT A TIME-SERIES BASED NN? IF THE PATTERN REPEATS ITSELF, IT MIGHT HELP...
    else:
      model_path = f"{model_h5_path}/model_{model_name}.h5"
      print(f"Loading model: {model_path}")
      self.model = tf.keras.models.load_model(
          model_path,
          custom_objects={"SymmetricPadding3D": SymmetricPadding3D},
          compile=False,
      )

    epochs_val_losses, epochs_train_losses = [], []

    min_yet = 1e9

    for epoch in range(num_epoch):
      progbar = tf.keras.utils.Progbar(math.ceil(self.len_train/batch_size))
      print('Start of epoch %d' %(epoch,))
      losses_train = []
      losses_test = []

      for step, (inputs, labels) in enumerate(self.train_dataset):

        if flatten_data:
          inputs = inputs[..., 0, 0]
          labels = labels[..., 0, 0]
        inputs = tf.cast(inputs, dtype='float32')
        labels = tf.cast(labels, dtype='float32')

        loss = self.train_step(inputs, labels)
        losses_train.append(loss)

      losses_val  = self.perform_validation(flatten_data)

      losses_train_mean = np.mean(losses_train)
      losses_val_mean = np.mean(losses_val)

      epochs_train_losses.append(losses_train_mean)
      epochs_val_losses.append(losses_val_mean)
      print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean)))

      progbar.update(step+1)

      # It was found that if the min_delta is too small, or patience is too high it can cause overfitting
      stopEarly = utils_model.Callback_EarlyStopping(epochs_val_losses, min_delta=0.1/100, patience=20)
      if stopEarly:
        print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,num_epoch))
        break

      if epoch > 5:
        model_fn = f'./{model_h5_path}/model_{model_name}.h5'
        if losses_val_mean < min_yet:
          print(f'saving model: {model_fn}', flush=True)
          self.model.save(model_fn)
          self.model.save_weights(weights_fn)
          min_yet = losses_val_mean
    
    print("Terminating training")
    ## Plot loss vs epoch
    plt.plot(list(range(len(epochs_train_losses))), epochs_train_losses, label ='train')
    plt.plot(list(range(len(epochs_val_losses))), epochs_val_losses, label ='val')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{model_h5_path}/loss_vs_epoch_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.png')

    ## Save losses data
    np.savetxt(f'{model_h5_path}/train_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_train_losses, fmt='%d')
    np.savetxt(f'{model_h5_path}/test_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_val_losses, fmt='%d')