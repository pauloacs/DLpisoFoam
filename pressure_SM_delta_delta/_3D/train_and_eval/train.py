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
    MLP, dense_attention, conv1D, FNO3d, GNN, MLP_Mixer_3D, SimpleCNN3D, Simple_multi_layer_3D, UNet3D, UNet3D_deep, UNet3D_attention, SymmetricPadding3D, SimpleCNN3D_two_heads
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
      predictions = self.model(inputs, training=True)
      loss=self.loss_object(labels, predictions)

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

      val_logits = self.model(x_val, training=False)
      val_loss = self.loss_object(y_true = y_val , y_pred = val_logits)
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

          if isinstance(y_pred, dict):
              y_pred = y_pred["p_total"]

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

  @staticmethod
  def lowpass_3d(y, pool_size=(3, 7, 15)):
    """
    y shape: [B, Z, Y, X]
    """
    y5 = y[..., None]

    y_smooth = tf.nn.avg_pool3d(
        y5,
        ksize=[1, pool_size[0], pool_size[1], pool_size[2], 1],
        strides=[1, 1, 1, 1, 1],
        padding="SAME",
    )

    return tf.squeeze(y_smooth, axis=-1)



  @staticmethod
  def smoothness_loss_3d(p, wz=0.25, wy=1.0, wx=4.0):
      """
      p shape: [B, Z, Y, X]

      Higher wx penalizes vertical banding / rapid x variation.
      """
      dz = p[:, 1:, :, :] - p[:, :-1, :, :]
      dy = p[:, :, 1:, :] - p[:, :, :-1, :]
      dx = p[:, :, :, 1:] - p[:, :, :, :-1]

      return (
          wz * tf.reduce_mean(tf.square(dz))
          + wy * tf.reduce_mean(tf.square(dy))
          + wx * tf.reduce_mean(tf.square(dx))
      )


  def my_two_head_loss(
      self,
      beta=1.0,
      cap=3.0,
      alpha=0.25,
      lambda_smooth=0.1,
      lambda_local=0.0,
      lambda_smoothness=0.0,
      pool_size=(1, 3, 9),
  ):
      def loss_f(y_true, y_pred):

          p_total = y_pred["p_total"]
          p_smooth = y_pred["p_smooth"]
          p_local = y_pred["p_local"]

          # Make target gauge-consistent with p_total
          y_true = y_true - tf.reduce_mean(
              y_true,
              axis=(1, 2, 3),
              keepdims=True,
          )

          # Main mixed loss on final prediction
          error2 = tf.square(y_true - p_total)
          mse = tf.reduce_mean(error2)

          mean_abs = tf.reduce_mean(
              tf.abs(y_true),
              axis=(1, 2, 3),
              keepdims=True,
          ) + 1e-8

          rel_amp = tf.abs(y_true) / mean_abs
          rel_amp = tf.clip_by_value(rel_amp, 0.0, cap)

          weights = 1.0 + beta * rel_amp
          weighted_mse = tf.reduce_mean(weights * error2)

          main_loss = (1.0 - alpha) * mse + alpha * weighted_mse

          # Low-pass target for smooth head
          y_smooth = self.lowpass_3d(y_true, pool_size=pool_size)

          y_smooth = y_smooth - tf.reduce_mean(
              y_smooth,
              axis=(1, 2, 3),
              keepdims=True,
          )

          p_smooth_centered = p_smooth - tf.reduce_mean(
              p_smooth,
              axis=(1, 2, 3),
              keepdims=True,
          )

          smooth_loss = tf.reduce_mean(
              tf.square(y_smooth - p_smooth_centered)
          )

          total_loss = main_loss + lambda_smooth * smooth_loss

          if lambda_local > 0.0:
              y_local = y_true - y_smooth

              p_local_centered = p_local - tf.reduce_mean(
                  p_local,
                  axis=(1, 2, 3),
                  keepdims=True,
              )

              local_loss = tf.reduce_mean(
                  tf.square(y_local - p_local_centered)
              )

              total_loss += lambda_local * local_loss

          if lambda_smoothness > 0.0:
              smooth_reg = self.smoothness_loss_3d(
                  p_smooth_centered,
                  wz=0.25,
                  wy=1.0,
                  wx=1.0,
              )

              total_loss += lambda_smoothness * smooth_reg

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

    split = 0.8

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


  def plot_decomposed_predictions(self, model_h5_path: str, flatten_data: bool = False):
    """
    Visualize p_smooth and p_local decomposition for cnn_smooth_specialized model.
    Plots slices at different dimensions and saves to model_h5_path.
    """
    print("\n=== Plotting decomposed predictions (p_smooth, p_local, p_total) ===")
    
    # Get a batch from test data
    sample_batch = None
    sample_labels = None
    
    for (x_batch, y_batch) in self.test_dataset:
      if flatten_data:
        x_batch = tf.cast(x_batch[..., 0, 0], dtype='float32')
        y_batch = tf.cast(y_batch[..., 0, 0], dtype='float32')
      else:
        x_batch = tf.cast(x_batch, dtype='float32')
        y_batch = tf.cast(y_batch, dtype='float32')
      
      sample_batch = x_batch
      sample_labels = y_batch
      break  # Just take first batch
    
    if sample_batch is None:
      print("ERROR: Could not load test batch for visualization")
      return
    
    # Get predictions (dict with p_smooth, p_local, p_total)
    predictions = self.model(sample_batch, training=False)
    
    p_smooth = predictions['p_smooth'].numpy()  # (batch, z, y, x)
    p_local = predictions['p_local'].numpy()
    p_total = predictions['p_total'].numpy()
    y_true = sample_labels.numpy()  # (batch, z, y, x)
    
    # Plot for first sample in batch
    sample_idx = 0
    z_slice = p_smooth.shape[1] // 2  # Middle Z
    y_slice = p_smooth.shape[2] // 2  # Middle Y
    x_slice = p_smooth.shape[3] // 2  # Middle X
    
    # =========================================================================
    # Plot Z-slices (middle Z plane)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Z-Slice (at z={z_slice}): Comparing Components', fontsize=14, fontweight='bold')
    
    # True
    im = axes[0, 0].imshow(y_true[sample_idx, z_slice, :, :], cmap='RdBu_r')
    axes[0, 0].set_title('Ground Truth (y_true)')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im, ax=axes[0, 0])
    
    # p_smooth
    im = axes[0, 1].imshow(p_smooth[sample_idx, z_slice, :, :], cmap='RdBu_r')
    axes[0, 1].set_title('p_smooth (Far-field)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # p_local
    im = axes[0, 2].imshow(p_local[sample_idx, z_slice, :, :], cmap='RdBu_r')
    axes[0, 2].set_title('p_local (Obstacle)')
    plt.colorbar(im, ax=axes[0, 2])
    
    # p_total
    im = axes[0, 3].imshow(p_total[sample_idx, z_slice, :, :], cmap='RdBu_r')
    axes[0, 3].set_title('p_total (prediction)')
    plt.colorbar(im, ax=axes[0, 3])
    
    # Error maps
    im = axes[1, 0].imshow(np.abs(y_true[sample_idx, z_slice, :, :] - p_total[sample_idx, z_slice, :, :]), cmap='Reds')
    axes[1, 0].set_title('|Error| (total)')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(np.abs(y_true[sample_idx, z_slice, :, :] - p_smooth[sample_idx, z_slice, :, :]), cmap='Reds')
    axes[1, 1].set_title('|Error| (smooth)')
    plt.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].imshow(np.abs(y_true[sample_idx, z_slice, :, :] - p_local[sample_idx, z_slice, :, :]), cmap='Reds')
    axes[1, 2].set_title('|Error| (local)')
    plt.colorbar(im, ax=axes[1, 2])
    
    # Decomposition check: p_smooth + p_local
    im = axes[1, 3].imshow((p_smooth + p_local)[sample_idx, z_slice, :, :], cmap='RdBu_r')
    axes[1, 3].set_title('p_smooth + p_local')
    axes[1, 3].set_xlabel('X')
    plt.colorbar(im, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig(f'{model_h5_path}/decomposed_slices_z.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_h5_path}/decomposed_slices_z.png")
    
    # =========================================================================
    # Plot Y-slices (middle Y plane)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Y-Slice (at y={y_slice}): Comparing Components', fontsize=14, fontweight='bold')
    
    # True
    im = axes[0, 0].imshow(y_true[sample_idx, :, y_slice, :], cmap='RdBu_r')
    axes[0, 0].set_title('Ground Truth (y_true)')
    axes[0, 0].set_ylabel('Z')
    plt.colorbar(im, ax=axes[0, 0])
    
    # p_smooth
    im = axes[0, 1].imshow(p_smooth[sample_idx, :, y_slice, :], cmap='RdBu_r')
    axes[0, 1].set_title('p_smooth (Far-field)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # p_local
    im = axes[0, 2].imshow(p_local[sample_idx, :, y_slice, :], cmap='RdBu_r')
    axes[0, 2].set_title('p_local (Obstacle)')
    plt.colorbar(im, ax=axes[0, 2])
    
    # p_total
    im = axes[0, 3].imshow(p_total[sample_idx, :, y_slice, :], cmap='RdBu_r')
    axes[0, 3].set_title('p_total (prediction)')
    plt.colorbar(im, ax=axes[0, 3])
    
    # Error maps
    im = axes[1, 0].imshow(np.abs(y_true[sample_idx, :, y_slice, :] - p_total[sample_idx, :, y_slice, :]), cmap='Reds')
    axes[1, 0].set_title('|Error| (total)')
    axes[1, 0].set_ylabel('Z')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(np.abs(y_true[sample_idx, :, y_slice, :] - p_smooth[sample_idx, :, y_slice, :]), cmap='Reds')
    axes[1, 1].set_title('|Error| (smooth)')
    plt.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].imshow(np.abs(y_true[sample_idx, :, y_slice, :] - p_local[sample_idx, :, y_slice, :]), cmap='Reds')
    axes[1, 2].set_title('|Error| (local)')
    plt.colorbar(im, ax=axes[1, 2])
    
    # Decomposition check
    im = axes[1, 3].imshow((p_smooth + p_local)[sample_idx, :, y_slice, :], cmap='RdBu_r')
    axes[1, 3].set_title('p_smooth + p_local')
    axes[1, 3].set_xlabel('X')
    plt.colorbar(im, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig(f'{model_h5_path}/decomposed_slices_y.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_h5_path}/decomposed_slices_y.png")
    
    # =========================================================================
    # Plot X-slices (middle X plane)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'X-Slice (at x={x_slice}): Comparing Components', fontsize=14, fontweight='bold')
    
    # True
    im = axes[0, 0].imshow(y_true[sample_idx, :, :, x_slice], cmap='RdBu_r')
    axes[0, 0].set_title('Ground Truth (y_true)')
    axes[0, 0].set_ylabel('Z')
    plt.colorbar(im, ax=axes[0, 0])
    
    # p_smooth
    im = axes[0, 1].imshow(p_smooth[sample_idx, :, :, x_slice], cmap='RdBu_r')
    axes[0, 1].set_title('p_smooth (Far-field)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # p_local
    im = axes[0, 2].imshow(p_local[sample_idx, :, :, x_slice], cmap='RdBu_r')
    axes[0, 2].set_title('p_local (Obstacle)')
    plt.colorbar(im, ax=axes[0, 2])
    
    # p_total
    im = axes[0, 3].imshow(p_total[sample_idx, :, :, x_slice], cmap='RdBu_r')
    axes[0, 3].set_title('p_total (prediction)')
    plt.colorbar(im, ax=axes[0, 3])
    
    # Error maps
    im = axes[1, 0].imshow(np.abs(y_true[sample_idx, :, :, x_slice] - p_total[sample_idx, :, :, x_slice]), cmap='Reds')
    axes[1, 0].set_title('|Error| (total)')
    axes[1, 0].set_ylabel('Z')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(np.abs(y_true[sample_idx, :, :, x_slice] - p_smooth[sample_idx, :, :, x_slice]), cmap='Reds')
    axes[1, 1].set_title('|Error| (smooth)')
    plt.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].imshow(np.abs(y_true[sample_idx, :, :, x_slice] - p_local[sample_idx, :, :, x_slice]), cmap='Reds')
    axes[1, 2].set_title('|Error| (local)')
    plt.colorbar(im, ax=axes[1, 2])
    
    # Decomposition check
    im = axes[1, 3].imshow((p_smooth + p_local)[sample_idx, :, :, x_slice], cmap='RdBu_r')
    axes[1, 3].set_title('p_smooth + p_local')
    axes[1, 3].set_xlabel('Y')
    plt.colorbar(im, ax=axes[1, 3])
    
    plt.tight_layout()
    plt.savefig(f'{model_h5_path}/decomposed_slices_x.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {model_h5_path}/decomposed_slices_x.png")
    
    print("=== Visualization complete ===\n")


  def plot_test_predictions_z_slices(self, model_h5_path: str, flatten_data: bool = False, n_z_slices: int = 5):
      """Plot 5 Z-slices (truth / prediction / |error|) for every test sample."""
      print(f"\n=== Plotting test predictions ({n_z_slices} Z-slices per sample) ===")
      os.makedirs(model_h5_path, exist_ok=True)

      global_idx = 0
      for (x_batch, y_batch) in self.test_dataset:
          if flatten_data:
              x_batch = tf.cast(x_batch[..., 0, 0], dtype='float32')
              y_batch = tf.cast(y_batch[..., 0, 0], dtype='float32')
          else:
              x_batch = tf.cast(x_batch, dtype='float32')
              y_batch = tf.cast(y_batch, dtype='float32')

          predictions = self.model(x_batch, training=False)
          y_pred = predictions['p_total'].numpy() if isinstance(predictions, dict) else predictions.numpy()
          y_true = y_batch.numpy()

          nz = y_true.shape[1]
          z_indices = [int(i * (nz - 1) / (n_z_slices - 1)) for i in range(n_z_slices)]
          row_labels = ['Ground Truth', 'Prediction', '|Error|']

          for b in range(y_true.shape[0]):
              fig, axes = plt.subplots(3, n_z_slices, figsize=(8 * n_z_slices, 10))
              fig.suptitle(f'Test sample {global_idx}', fontsize=14, fontweight='bold')

              for col, z_idx in enumerate(z_indices):
                  sl_true = y_true[b, z_idx]
                  sl_pred = y_pred[b, z_idx]
                  sl_err  = np.abs(sl_true - sl_pred)

                  vmax_col = float(max(np.nanmax(np.abs(sl_true)), np.nanmax(np.abs(sl_pred)))) or 1.0
                  vmax_err = float(np.nanmax(sl_err)) or 1.0

                  slices   = [sl_true, sl_pred, sl_err]
                  vmaxes   = [vmax_col, vmax_col, vmax_err]

                  for row, (sl, vm) in enumerate(zip(slices, vmaxes)):
                      cmap = 'RdBu_r' if row < 2 else 'Reds'
                      vmin = -vm if row < 2 else 0.0
                      im = axes[row, col].imshow(sl, cmap=cmap, vmin=vmin, vmax=vm, aspect='auto')
                      title = f'{row_labels[row]}\nz={z_idx}' if col == 0 else f'z={z_idx}'
                      axes[row, col].set_title(title, fontsize=8)
                      axes[row, col].axis('off')
                      plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

              plt.tight_layout()
              path = f'{model_h5_path}/test_pred_{global_idx:04d}.png'
              plt.savefig(path, dpi=80, bbox_inches='tight')
              plt.close()
              print(f"  Saved: {path}")
              global_idx += 1

      print(f"=== Done: {global_idx} test samples plotted ===\n")


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
    #self.loss_object = self.my_mse_loss()
    #self.loss_object = self.my_mse_energy_loss(alpha_energy=0.2)
    

    if model_architecture == 'cnn_two_heads':
      #self.loss_object = self.my_two_head_loss(
      #    beta=1.0,
      #    cap=3.0,
      #    alpha=0.25, # 0.25
      #    lambda_smooth=0.1,
      #    lambda_local=0.0,
      #    lambda_smoothness=0.0,
      #    pool_size=(1, 3, 9),
      #)
      self.loss_object = self.my_mixed_weighted_mse_loss(beta=0.5, cap=2.0, alpha=0.25)
    else:
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
        case 'cnn_two_heads':
          self.model = SimpleCNN3D_two_heads(spatial_dims,
                                             in_channels=last_tucker_rank,
                                             return_heads=True,
                                             dropout_rate=dropout_rate, regularization=regularization)
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
      stopEarly = utils_model.Callback_EarlyStopping(epochs_val_losses, min_delta=0.1/100, patience=50)
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

    # Plot Z-slice predictions for all test samples
    self.plot_test_predictions_z_slices(model_h5_path, flatten_data, n_z_slices=5)

    # Plot decomposed predictions (p_smooth, p_local) if using cnn_smooth_specialized
    if model_architecture.lower() == 'cnn_smooth_specialized':
        self.plot_decomposed_predictions(model_h5_path, flatten_data)