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
    MLP, dense_attention, conv1D, FNO3d, GNN, MLP_Mixer_3D, SimpleCNN3D, Simple_multi_layer_3D, UNet3D, UNet3D_deep, UNet3D_attention, SymmetricPadding3D, SimpleCNN3D_two_heads_smooth, SimpleCNN3D_two_heads, SimpleCNN3D_multi_out, SimpleCNN3D_multi_out_divU
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

  def my_mixed_weighted_mse_loss_masked(self, beta=1.0, cap=3.0, alpha=0.5):
      def loss_f(y_true, y_pred):
          if isinstance(y_pred, dict):
              y_pred = y_pred["p_total"]

          # If y_true is multi-channel (predict_ddUCorr_output=True) but the model only
          # predicts a single scalar field, restrict y_true to the pressure channel (0).
          if len(y_true.shape) == 4 and len(y_pred.shape) == 1:
              y_true = y_true[..., 0]

          # Create domain mask: True where inside domain (obst_bool != 0)
          # obst_bool shape: (z, y, x, 1), y_true/y_pred shape: (batch, z, y, x) or (batch, z, y, x, n_ch)
          if self.obst_bool is not None:
              mask = tf.cast(self.obst_bool[:, :, :, 0] != 0, dtype=tf.float32)  # (z, y, x)
              mask = tf.expand_dims(mask, axis=0)  # (1, z, y, x), broadcasts with batch
              # If y_pred has an extra channel dim (multi-output), expand mask to broadcast over it
              if len(y_pred.shape) == 5:
                  mask = tf.expand_dims(mask, axis=-1)  # (1, z, y, x, 1)
          else:
              mask = tf.ones_like(y_true)
          
          # Mask out outside-domain values
          y_true_masked = y_true * mask
          y_pred_masked = y_pred * mask
          
          error2 = tf.square(y_true_masked - y_pred_masked) * mask
          
          # Count valid elements for normalization
          n_valid = tf.reduce_sum(mask)
          
          # MSE with mask normalization
          mse = tf.reduce_sum(error2) / (n_valid + 1e-8)
          
          # Mean amplitude (per-batch) with mask
          mean_abs = tf.reduce_sum(
              tf.abs(y_true_masked),
              axis=(1, 2, 3),
              keepdims=True
          ) / (tf.reduce_sum(mask, axis=(1, 2, 3), keepdims=True) + 1e-8)
          
          rel_amp = tf.abs(y_true_masked) / (mean_abs + 1e-8)
          rel_amp = tf.clip_by_value(rel_amp, 0.0, cap)
          
          weights = (1.0 + beta * rel_amp) * mask
          weighted_mse = tf.reduce_sum(weights * error2) / (n_valid + 1e-8)
          
          loss = (1.0 - alpha) * mse + alpha * weighted_mse
          
          return 100.0 * loss
      
      return loss_f


  def my_weighted_loss_split(
        self,
        w_p=1.0,
        w_u=1.0,
        w_cont=0.1,
        beta=1.0,
        cap=3.0,
        alpha=0.5,
        div_u=None,
        mean_out_vel=None,
        std_out_vel=None,
        grid_res=1.0,
    ):
        if isinstance(grid_res, (tuple, list)):
            _dx, _dy, _dz = (
                float(grid_res[0]),
                float(grid_res[1]),
                float(grid_res[2]),
            )
        else:
            _dx = _dy = _dz = float(grid_res)

        def loss_f(y_true, y_pred):

            # =========================================================
            # 1) DOMAIN MASK (fluid cells only)
            # =========================================================
            if self.obst_bool is not None:
                mask = tf.cast(
                    self.obst_bool[:, :, :, 0] != 0,
                    tf.float32,
                )
                mask = tf.expand_dims(mask, axis=0)  # (1,Z,Y,X)
            else:
                mask = tf.ones_like(y_true[..., 0])

            mask_c = tf.expand_dims(mask, axis=-1)

            # =========================================================
            # 2) div(U) ACTIVITY MASK
            # =========================================================
            _div_u = (
                self.div_u_batch
                if hasattr(self, "div_u_batch")
                and self.div_u_batch is not None
                else div_u
            )

            if _div_u is not None:

                div_u_tf = tf.cast(_div_u, y_pred.dtype)

                if len(div_u_tf.shape) == 3:
                    div_u_tf = tf.expand_dims(div_u_tf, axis=0)

                abs_div = tf.abs(div_u_tf)

                # ---------------------------------------------
                # p10 threshold (per sample)
                # ---------------------------------------------
                flat = tf.reshape(abs_div, [tf.shape(abs_div)[0], -1])
                sorted_flat = tf.sort(flat, axis=-1)

                n = tf.shape(sorted_flat)[-1]

                p10_idx = tf.cast(
                    0.10 * tf.cast(n, tf.float32),
                    tf.int32,
                )

                p10 = sorted_flat[:, p10_idx]
                p10 = p10[:, None, None, None]

                # ---------------------------------------------
                # active div(U) mask
                # ---------------------------------------------
                div_mask = tf.cast(
                    abs_div > p10,
                    y_pred.dtype,
                )

                # ---------------------------------------------
                # optional dilation
                # ---------------------------------------------
                div_mask = tf.nn.max_pool3d(
                    div_mask[..., None],
                    ksize=[1, 2, 2, 2, 1],
                    strides=[1, 1, 1, 1, 1],
                    padding="SAME",
                )[..., 0]

                # combine with obstacle mask
                div_mask = div_mask * mask

            else:
                div_mask = mask

            div_mask_c = tf.expand_dims(div_mask, axis=-1)

            # =========================================================
            # 3) PRESSURE LOSS
            # =========================================================
            p_true = y_true[..., 0]
            p_pred = y_pred[..., 0]

            err_p = tf.square(p_true - p_pred) * mask

            n_valid = tf.reduce_sum(mask)

            mse_p = tf.reduce_sum(err_p) / (n_valid + 1e-8)

            mean_abs_p = (
                tf.reduce_sum(tf.abs(p_true) * mask)
                / (n_valid + 1e-8)
            )

            rel_amp_p = tf.clip_by_value(
                tf.abs(p_true) / (mean_abs_p + 1e-8),
                0.0,
                cap,
            )

            weights_p = (1.0 + beta * rel_amp_p) * mask

            weighted_mse_p = (
                tf.reduce_sum(weights_p * err_p)
                / (n_valid + 1e-8)
            )

            loss_p = (
                (1.0 - alpha) * mse_p
                + alpha * weighted_mse_p
            )

            # =========================================================
            # 4) VELOCITY LOSS (masked by active div(U))
            # =========================================================
            u_true = y_true[..., 1:4]
            u_pred = y_pred[..., 1:4]

            err_u = tf.square(u_true - u_pred) * div_mask_c

            n_valid_u = tf.reduce_sum(div_mask_c)

            mse_u = tf.reduce_sum(err_u) / (n_valid_u + 1e-8)

            mean_abs_u = (
                tf.reduce_sum(tf.abs(u_true) * div_mask_c)
                / (n_valid_u + 1e-8)
            )

            rel_amp_u = tf.clip_by_value(
                tf.abs(u_true) / (mean_abs_u + 1e-8),
                0.0,
                cap,
            )

            weights_u = (
                1.0 + beta * rel_amp_u
            ) * div_mask_c

            weighted_mse_u = (
                tf.reduce_sum(weights_u * err_u)
                / (n_valid_u + 1e-8)
            )

            loss_u = (
                (1.0 - alpha) * mse_u
                + alpha * weighted_mse_u
            )

            # =========================================================
            # 5) CONTINUITY LOSS
            # =========================================================
            cont_loss = 0.0

            if _div_u is not None:

                # ---------------------------------------------
                # denormalize velocity
                # ---------------------------------------------
                if (
                    std_out_vel is not None
                    and mean_out_vel is not None
                ):
                    _s = tf.reshape(
                        tf.cast(std_out_vel, y_pred.dtype),
                        [1, 1, 1, 1, 3],
                    )

                    _m = tf.reshape(
                        tf.cast(mean_out_vel, y_pred.dtype),
                        [1, 1, 1, 1, 3],
                    )

                    vel_pred = y_pred[..., 1:4] * _s + _m

                else:
                    vel_pred = y_pred[..., 1:4]

                # apply active-region mask
                vel_pred = vel_pred * div_mask_c

                ddU_x = vel_pred[..., 0]
                ddU_y = vel_pred[..., 1]
                ddU_z = vel_pred[..., 2]

                # ---------------------------------------------
                # finite-difference divergence
                # ---------------------------------------------
                dUx_dx = (
                    ddU_x[:, 1:-1, 1:-1, 2:]
                    - ddU_x[:, 1:-1, 1:-1, :-2]
                ) / (2.0 * _dx)

                dUy_dy = (
                    ddU_y[:, 1:-1, 2:, 1:-1]
                    - ddU_y[:, 1:-1, :-2, 1:-1]
                ) / (2.0 * _dy)

                dUz_dz = (
                    ddU_z[:, 2:, 1:-1, 1:-1]
                    - ddU_z[:, :-2, 1:-1, 1:-1]
                ) / (2.0 * _dz)

                div_ddU = dUx_dx + dUy_dy + dUz_dz

                div_u_interior = div_u_tf[
                    :,
                    1:-1,
                    1:-1,
                    1:-1,
                ]

                cont_residual = div_u_interior + div_ddU

                # ---------------------------------------------
                # continuity mask
                # ---------------------------------------------
                cont_mask = div_mask[
                    :,
                    1:-1,
                    1:-1,
                    1:-1,
                ]

                cont_loss = (
                    tf.reduce_sum(
                        tf.square(cont_residual) * cont_mask
                    )
                    / (tf.reduce_sum(cont_mask) + 1e-8)
                )

            # =========================================================
            # 6) FINAL LOSS
            # =========================================================
            total_loss = (
                w_p * loss_p
                + w_u * loss_u
                + w_cont * cont_loss
            )

            return 100.0 * total_loss

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

  ## LOSS TO TEST

  def my_multihead_loss_from_total_only(
        self,
        lambda_total=1.0,
        lambda_smooth=0.3,
        lambda_local=0.7,
        lambda_grad=0.05,
        lambda_local_grad=0.5,
        smooth_ksize=(3, 3, 5),
        smooth_passes=2,
        scale=100.0,
  ):
        """
        Multi-head loss that builds y_smooth and y_local from the total target internally.

        Expected y_true:
            - (batch, z, y, x)
            or
            - (batch, z, y, x, 1)

        Expected y_pred:
            dict with keys:
                "p_total"  -> (batch, z, y, x)
                "p_smooth" -> (batch, z, y, x)
                "p_local"  -> (batch, z, y, x)

        The smooth target is built inside the loss using masked normalized average pooling.
        """

        def masked_mse(y_true_f, y_pred_f, mask):
            sq_err = tf.square(y_true_f - y_pred_f) * mask
            return tf.reduce_sum(sq_err) / (tf.reduce_sum(mask) + 1e-8)

        def masked_gradient_loss_3d(y_true_f, y_pred_f, mask):
            # z gradients
            dz_true = y_true_f[:, 1:, :, :] - y_true_f[:, :-1, :, :]
            dz_pred = y_pred_f[:, 1:, :, :] - y_pred_f[:, :-1, :, :]
            dz_mask = mask[:, 1:, :, :] * mask[:, :-1, :, :]

            # y gradients
            dy_true = y_true_f[:, :, 1:, :] - y_true_f[:, :, :-1, :]
            dy_pred = y_pred_f[:, :, 1:, :] - y_pred_f[:, :, :-1, :]
            dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]

            # x gradients
            dx_true = y_true_f[:, :, :, 1:] - y_true_f[:, :, :, :-1]
            dx_pred = y_pred_f[:, :, :, 1:] - y_pred_f[:, :, :, :-1]
            dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]

            loss_z = tf.reduce_sum(tf.abs(dz_true - dz_pred) * dz_mask) / (tf.reduce_sum(dz_mask) + 1e-8)
            loss_y = tf.reduce_sum(tf.abs(dy_true - dy_pred) * dy_mask) / (tf.reduce_sum(dy_mask) + 1e-8)
            loss_x = tf.reduce_sum(tf.abs(dx_true - dx_pred) * dx_mask) / (tf.reduce_sum(dx_mask) + 1e-8)

            return loss_z + loss_y + loss_x

        def masked_smooth_target(y_total, mask):
            """
            Build a smooth target from y_total using mask-normalized average pooling:
                smooth = avg(y * mask) / avg(mask)

            This avoids biasing the smooth field near obstacles/outside-domain zeros.
            """
            # Expand channel dimension for avg_pool3d: (B, Z, Y, X, 1)
            y = tf.expand_dims(y_total, axis=-1)
            m = tf.expand_dims(mask, axis=-1)

            ksize = [1, smooth_ksize[0], smooth_ksize[1], smooth_ksize[2], 1]
            strides = [1, 1, 1, 1, 1]

            s_num = y * m
            s_den = m

            for _ in range(smooth_passes):
                s_num = tf.nn.avg_pool3d(s_num, ksize=ksize, strides=strides, padding="SAME")
                s_den = tf.nn.avg_pool3d(s_den, ksize=ksize, strides=strides, padding="SAME")

            s = s_num / (s_den + 1e-8)

            # Remove channel dim -> (B, Z, Y, X)
            s = tf.squeeze(s, axis=-1)
            return s

        def loss_f(y_true, y_pred):
            # ------------------------------------------------------------
            # Predictions
            # ------------------------------------------------------------
            if not isinstance(y_pred, dict):
                raise ValueError(
                    "This loss expects model outputs as a dict with keys "
                    "'p_total', 'p_smooth', 'p_local'."
                )

            p_total_pred = y_pred["p_total"]
            p_smooth_pred = y_pred["p_smooth"]
            p_local_pred = y_pred["p_local"]

            # ------------------------------------------------------------
            # y_true -> total target only
            # Accept (B,Z,Y,X) or (B,Z,Y,X,1)
            # ------------------------------------------------------------
            if len(y_true.shape) == 5:
                y_total = y_true[..., 0]
            elif len(y_true.shape) == 4:
                y_total = y_true
            else:
                raise ValueError(
                    "y_true must have shape (batch, z, y, x) or (batch, z, y, x, 1)."
                )

            # ------------------------------------------------------------
            # Domain mask
            # obst_bool expected shape: (z, y, x, 1)
            # final mask shape: (1, z, y, x)
            # ------------------------------------------------------------
            if self.obst_bool is not None:
                mask = tf.cast(self.obst_bool[:, :, :, 0] != 0, dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=0)  # (1, z, y, x)
            else:
                mask = tf.ones_like(y_total)

            # ------------------------------------------------------------
            # Build smooth/local targets internally
            # ------------------------------------------------------------
            y_smooth = masked_smooth_target(y_total, mask)
            y_local = y_total - y_smooth

            # ------------------------------------------------------------
            # Main MSE terms
            # ------------------------------------------------------------
            loss_total = masked_mse(y_total, p_total_pred, mask)
            loss_smooth = masked_mse(y_smooth, p_smooth_pred, mask)
            loss_local = masked_mse(y_local, p_local_pred, mask)

            # ------------------------------------------------------------
            # Gradient loss
            # total field + smaller local-field term
            # ------------------------------------------------------------
            grad_total = masked_gradient_loss_3d(y_total, p_total_pred, mask)
            grad_local = masked_gradient_loss_3d(y_local, p_local_pred, mask)

            grad_loss = grad_total + lambda_local_grad * grad_local

            # ------------------------------------------------------------
            # Final loss
            # ------------------------------------------------------------
            loss = (
                lambda_total * loss_total
                + lambda_smooth * loss_smooth
                + lambda_local * loss_local
                + lambda_grad * grad_loss
            )

            return scale * loss

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

    dp_prev = x[..., 14:15]
    y = np.concatenate([y, dp_prev], axis=-1)

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


  def plot_test_predictions_z_slices(self, model_h5_path: str, flatten_data: bool = False, n_z_slices: int = 5, obst_bool=None, mean_std_fn=None, predict_ddUCorr_output: bool = False, div_u_ch_idx: int = None, maxs_fn: str = None):
      """Plot 5 Z-slices (truth / prediction / |error|) for every test sample.
      When predict_ddUCorr_output=True, also plots 3 extra figures for ddU components (x, y, z).
      Post-processing mirrors main_mpi: destandardize (* std + mean), scale by max_abs,
      then enforce zero mean at the outlet (Ref_BC=0)."""
      print(f"\n=== Plotting test predictions ({n_z_slices} Z-slices per sample) ===")
      os.makedirs(model_h5_path, exist_ok=True)
      ddp_dir = os.path.join(model_h5_path, 'ddp')
      ddU_dir = os.path.join(model_h5_path, 'ddU')
      os.makedirs(ddp_dir, exist_ok=True)
      os.makedirs(ddU_dir, exist_ok=True)

      # Load standardization factors if provided
      denorm_mean_out = None
      denorm_std_out = None
      if mean_std_fn is not None and os.path.exists(mean_std_fn):
          data = np.load(mean_std_fn)
          denorm_mean_out = data['mean_out']  # shape: (output_channels,) or scalar
          denorm_std_out = data['std_out']    # shape: (output_channels,) or scalar
          print(f"Loaded standardization factors from {mean_std_fn}")

      # Load max_abs scaling factors from the maxs file.
      # Layout (from data_processor): [..., ddp, ddUx, ddUy, ddUz] when predict_ddUCorr_output=True
      #                               [..., ddp]                    otherwise
      max_abs_ddp = None
      max_abs_ddU = None  # np.array([x, y, z]) when predict_ddUCorr_output=True
      _maxs_fn = maxs_fn if maxs_fn is not None else os.path.join(model_h5_path, 'maxs')
      if os.path.exists(_maxs_fn):
          _maxs = np.loadtxt(_maxs_fn)
          if predict_ddUCorr_output and len(_maxs) >= 4:
              max_abs_ddp = float(_maxs[-4])
              max_abs_ddU = np.array([float(_maxs[-3]), float(_maxs[-2]), float(_maxs[-1])])
          else:
              max_abs_ddp = float(_maxs[-1])
          print(f"Loaded max_abs_ddp={max_abs_ddp:.6g} from {_maxs_fn}")
      else:
          print(f"[plot_test_predictions_z_slices] maxs file not found at {_maxs_fn}, skipping max_abs scaling")

      def _plot_field_z_slices(y_true_f, y_pred_f, global_idx, b, z_indices, obst_bool, row_labels, n_z_slices, title_prefix, filename):
          """Helper: plot 3-row x n_z_slices figure for a single scalar field."""
          fig, axes = plt.subplots(3, n_z_slices, figsize=(8 * n_z_slices, 10))
          fig.suptitle(f'{title_prefix} — sample {global_idx}', fontsize=14, fontweight='bold')
          for col, z_idx in enumerate(z_indices):
              sl_true = y_true_f[b, z_idx]
              sl_pred = y_pred_f[b, z_idx]
              sl_err  = np.abs(sl_true - sl_pred)
              vmax_col = float(max(np.nanmax(np.abs(sl_true)), np.nanmax(np.abs(sl_pred)))) or 1.0
              vmax_err = float(np.nanmax(sl_err)) or 1.0
              slices = [sl_true, sl_pred, sl_err]
              vmaxes = [vmax_col, vmax_col, vmax_err]
              for row, (sl, vm) in enumerate(zip(slices, vmaxes)):
                  cmap = 'RdBu_r' if row < 2 else 'Reds'
                  vmin = -vm if row < 2 else 0.0
                  if obst_bool is not None:
                      sl = np.ma.array(sl, mask=obst_bool[z_idx, ..., 0] == 0)
                  im = axes[row, col].imshow(sl, cmap=cmap, vmin=vmin, vmax=vm, aspect='auto', interpolation='none')
                  title = f'{row_labels[row]}\nz={z_idx}' if col == 0 else f'z={z_idx}'
                  axes[row, col].set_title(title, fontsize=8)
                  axes[row, col].axis('off')
                  plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
          plt.tight_layout()
          plt.savefig(filename, dpi=80, bbox_inches='tight')
          plt.close()
          print(f"  Saved: {filename}")

      global_idx = 0
      _mask_check_done = False
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

          # On the first batch, run div(U*) mask diagnostics if applicable
          if not _mask_check_done:
              _mask_check_done = True
              try:
                  from pressure_SM_delta_delta._3D.train_and_eval.neural_networks import debug_div_mask
                  debug_div_mask(self.model, x_batch, div_u_ch_idx)
              except Exception as _dbg_e:
                  print(f"[plot_test_predictions_z_slices] Mask debug skipped: {_dbg_e}")

          # Denormalize y_true and y_pred if standardization factors available
          if denorm_mean_out is not None and denorm_std_out is not None:
              y_true = y_true * denorm_std_out + denorm_mean_out
              y_pred = y_pred * denorm_std_out + denorm_mean_out

          # Split channels when predict_ddUCorr_output=True: shape (batch, z, y, x, 4)
          multi_out = predict_ddUCorr_output and y_pred.ndim == 5
          if multi_out:
              y_true_ddp  = y_true[..., 0]
              y_pred_ddp  = y_pred[..., 0]
              y_true_ddU  = [y_true[..., c] for c in (1, 2, 3)]  # [x, y, z components]
              y_pred_ddU  = [y_pred[..., c] for c in (1, 2, 3)]
          else:
              y_true_ddp = y_true
              y_pred_ddp = y_pred
              y_true_ddU = y_pred_ddU = None

          # --- max_abs inverse scaling (undo per-channel normalization from data prep) ---
          if max_abs_ddp is not None:
              y_true_ddp = y_true_ddp * max_abs_ddp
              y_pred_ddp = y_pred_ddp * max_abs_ddp
          if max_abs_ddU is not None and multi_out:
              y_true_ddU = [y_true_ddU[c] * float(max_abs_ddU[c]) for c in range(3)]
              y_pred_ddU = [y_pred_ddU[c] * float(max_abs_ddU[c]) for c in range(3)]

          # --- BC enforcement: mean of top-X slice of prediction = 0 (Ref_BC=0, as in main_mpi) ---
          outlet_offset = np.mean(y_pred_ddp[:, :, :, -2:], axis=(1, 2, 3))  # (batch,)
          y_pred_ddp = y_pred_ddp - outlet_offset[:, np.newaxis, np.newaxis, np.newaxis]

          nz = y_true_ddp.shape[1]
          z_indices = [int(i * (nz - 1) / (n_z_slices - 1)) for i in range(n_z_slices)]
          row_labels = ['Ground Truth', 'Prediction', '|Error|']

          for b in range(y_true_ddp.shape[0]):
              # --- ddp plot ---
              _plot_field_z_slices(
                  y_true_ddp, y_pred_ddp, global_idx, b, z_indices, obst_bool,
                  row_labels, n_z_slices,
                  title_prefix='ddp',
                  filename=os.path.join(ddp_dir, f'test_pred_{global_idx:04d}.png'),
              )

              # --- ddU component plots (only when predict_ddUCorr_output=True) ---
              if multi_out:
                  for comp_idx, comp_name in enumerate(('ddU_x', 'ddU_y', 'ddU_z')):
                      _plot_field_z_slices(
                          y_true_ddU[comp_idx], y_pred_ddU[comp_idx],
                          global_idx, b, z_indices, obst_bool,
                          row_labels, n_z_slices,
                          title_prefix=comp_name,
                          filename=os.path.join(ddU_dir, f'test_pred_{global_idx:04d}_{comp_name}.png'),
                      )

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
    block_size=None,
    obst_bool=None,
    predict_ddUCorr_output: bool=False,
    div_u_ch_idx: int=None,
    div_u_grid=None,
    lambda_cont: float=0.02,
    grid_res=1.0) -> None:

    train_path = self.train_tfrecord_fn
    test_path = self.test_tfrecord_fn
    
    self.obst_bool = obst_bool
    self.train_dataset = utils_io.load_dataset_tf(filename = train_path, batch_size = batch_size, buffer_size=1024)
    self.test_dataset = utils_io.load_dataset_tf(filename = test_path, batch_size = batch_size, buffer_size=1024)

    # Load normalization statistics (written by prepare_data_to_tf) for denormalization in loss/mask.
    _norm_fn = os.path.join(model_h5_path, 'mean_std.npz')
    _mean_in = _std_in = _mean_out = _std_out = None
    if os.path.exists(_norm_fn):
        _nd = np.load(_norm_fn)
        if 'mean_in' in _nd:
            _mean_in  = _nd['mean_in']
            _std_in   = _nd['std_in']
            _mean_out = _nd['mean_out']
            _std_out  = _nd['std_out']
    # Scalar normalization factors for the divU input channel
    _div_u_mean = float(_mean_in.flat[div_u_ch_idx]) if (_mean_in is not None and div_u_ch_idx is not None) else 0.0
    _div_u_std  = float(_std_in.flat[div_u_ch_idx])  if (_std_in  is not None and div_u_ch_idx is not None) else 1.0
    # Per-channel normalization factors for the velocity output (ddUx, ddUy, ddUz → channels 1-3)
    if _mean_out is not None and predict_ddUCorr_output:
        _mean_out_vel = _mean_out.flat[1:4] if _mean_out.size >= 4 else np.zeros(3)
        _std_out_vel  = _std_out.flat[1:4]  if _std_out.size  >= 4 else np.ones(3)
    else:
        _mean_out_vel = _std_out_vel = None

    # =========================================================
    # Architecture and loss selection driven by predict_ddUCorr_output
    # ---------------------------------------------------------
    # predict_ddUCorr_output=True  → cnn_multi_out_divu
    #                                + my_weighted_loss_split
    #                                (requires use_feature_decomposition=False
    #                                 and add_divUStar_input=True so div_u_ch_idx is set)
    # predict_ddUCorr_output=False → cnn_two_heads
    #                                + my_mixed_weighted_mse_loss_masked
    # =========================================================

    if predict_ddUCorr_output:
        if use_feature_decomposition:
            raise ValueError(
                "[predict_ddUCorr_output=True] 'cnn_multi_out_divu' requires "
                "use_feature_decomposition=False. "
                "Set 'use_feature_decomposition = False' in python_module."
            )
        if div_u_ch_idx is None:
            raise ValueError(
                "[predict_ddUCorr_output=True] 'cnn_multi_out_divu' requires the divU input "
                "channel. Set 'add_divUStar_input = True' in python_module so that "
                "div_u_ch_idx is resolved."
            )
        effective_model_arch = 'cnn_multi_out_divu'
        self.loss_object = self.my_weighted_loss_split(
            w_p=1.0,
            w_u=3.5,
            w_cont=0.0,
            beta=0.5,
            cap=2.0,
            alpha=0.0,
            div_u=div_u_grid,
            mean_out_vel=_mean_out_vel,
            std_out_vel=_std_out_vel,
            grid_res=grid_res,
        )
    else:
        if model_architecture.lower() not in ['cnn_two_heads', 'cnn_two_heads_smooth']:
            effective_model_arch = 'cnn_two_heads'
        else:
            effective_model_arch = model_architecture.lower()
        # best so far..
        #self.loss_object = self.my_mixed_weighted_mse_loss_masked(beta=1.0, cap=3.0, alpha=0.0)


        consider_dp_loss = True
        if consider_dp_loss:
        
        else:
            self.loss_object = self.my_multihead_loss_from_total_only(
                lambda_total=1,
                lambda_smooth=0.1,
                lambda_local=1,
                lambda_grad=0.03,
                lambda_local_grad=1.0,
                smooth_ksize=(3, 3, 3),
                smooth_passes=1,
                scale=100.0,
            )


    print(
        f"[Config] predict_ddUCorr_output={predict_ddUCorr_output} → "
        f"model='{effective_model_arch}', "
        f"loss='{'my_weighted_loss_split' if predict_ddUCorr_output else 'my_mixed_weighted_mse_loss_masked'}'"
    )

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)

    def prod(iterable):
        from functools import reduce
        from operator import mul
        return reduce(mul, iterable, 1)

    # Determine spatial dims to pass to 3D models.
    # cnn_multi_out_divu uses raw block_size (use_feature_decomposition=False enforced above).
    # cnn_two_heads      uses spatial_tucker_ranks (Tucker decomposition).
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
          _n_out_ch = 4 if predict_ddUCorr_output else 1
          self.model = SimpleCNN3D(spatial_dims, in_channels=last_tucker_rank, out_channels=_n_out_ch, dropout_rate=dropout_rate, regularization=regularization)
        case 'cnn_two_heads':
          self.model = SimpleCNN3D_two_heads(spatial_dims,
                                             in_channels=last_tucker_rank,
                                             return_heads=True,
                                             dropout_rate=dropout_rate, regularization=regularization)
        case 'cnn_two_heads_smooth':
            self.model = SimpleCNN3D_two_heads_smooth(spatial_dims,
                                                  in_channels=last_tucker_rank,
                                                  return_heads=True,
                                                  dropout_rate=dropout_rate, regularization=regularization)
        case 'cnn_multi_out':
          _n_out_ch = 4 if predict_ddUCorr_output else 1
          self.model = SimpleCNN3D_multi_out(spatial_dims, in_channels=last_tucker_rank, out_channels=_n_out_ch, dropout_rate=dropout_rate, regularization=regularization)
        case 'cnn_multi_out_divu':
          _n_out_ch = 4 if predict_ddUCorr_output else 1
          self.model = SimpleCNN3D_multi_out_divU(spatial_dims, in_channels=last_tucker_rank, out_channels=_n_out_ch, dropout_rate=dropout_rate, regularization=regularization, div_u_ch_idx=div_u_ch_idx, div_u_mean=_div_u_mean, div_u_std=_div_u_std)
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
    # Denormalize using mean_std.npz and max_abs from maxs file (both live in model_h5_path)
    mean_std_path = os.path.join(model_h5_path, 'mean_std.npz')
    self.plot_test_predictions_z_slices(model_h5_path, flatten_data, n_z_slices=5, obst_bool=obst_bool, mean_std_fn=mean_std_path, predict_ddUCorr_output=predict_ddUCorr_output, div_u_ch_idx=div_u_ch_idx, maxs_fn=os.path.join(model_h5_path, 'maxs'))

    # Plot decomposed predictions (p_smooth, p_local, p_total) for cnn_two_heads
    if not predict_ddUCorr_output:
        self.plot_decomposed_predictions(model_h5_path, flatten_data)