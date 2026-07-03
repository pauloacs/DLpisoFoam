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
from .neural_networks_shifter import (
    SimpleCNN3D_ddp_shifter,
    SimpleCNN3D_ddp_shifter_lightweight,
    SimpleCNN3D_ddp_shifter_velocity,
)
from .shifter_loss import ShifterLoss, _build_roi_mask_from_gradDpPrevMag, _build_upstream_soft_roi_mask
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
    self.obst_bool = None

  @tf.function
  def train_step(self, inputs, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(inputs, training=True)
      loss = self.loss_object(labels, predictions)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  def perform_validation(self, flatten_data: bool) -> list:
    losses = []

    for (x_val, y_val) in self.test_dataset:
      if flatten_data:
        x_val = tf.cast(x_val[..., 0, 0], dtype='float32')
        y_val = tf.cast(y_val[..., 0, 0], dtype='float32')

      val_logits = self.model(x_val, training=False)
      val_loss = self.loss_object(y_true=y_val, y_pred=val_logits)
      losses.append(val_loss)

    return losses

  def my_mse_loss(self):
    def loss_f(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
      loss = tf.reduce_mean(tf.square(y_true - y_pred))
      return 100 * loss
    return loss_f

  def my_mse_energy_loss(self, alpha_energy=0.2):
      def loss_f(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
          mse = tf.reduce_mean(tf.square(y_true - y_pred))
          rms_true = tf.sqrt(tf.reduce_mean(tf.square(y_true), axis=(1, 2, 3)) + 1e-8)
          rms_pred = tf.sqrt(tf.reduce_mean(tf.square(y_pred), axis=(1, 2, 3)) + 1e-8)
          energy_loss = tf.reduce_mean(tf.square(rms_true - rms_pred))
          return 100.0 * (mse + alpha_energy * energy_loss)
      return loss_f

  def my_mixed_weighted_mse_loss(self, beta=1.0, cap=3.0, alpha=0.5):
      def loss_f(y_true, y_pred):
          if isinstance(y_pred, dict):
              y_pred = y_pred["p_total"]

          error2 = tf.square(y_true - y_pred)
          mse = tf.reduce_mean(error2)
          mean_abs = tf.reduce_mean(tf.abs(y_true), axis=(1, 2, 3), keepdims=True) + 1e-8
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

          if len(y_true.shape) == 5 and len(y_pred.shape) == 4:
              y_true = y_true[..., 0]

          if self.obst_bool is not None:
              mask = tf.cast(self.obst_bool[:, :, :, 0] != 0, dtype=tf.float32)
              mask = tf.expand_dims(mask, axis=0)
              if len(y_pred.shape) == 5:
                  mask = tf.expand_dims(mask, axis=-1)
          else:
              mask = tf.ones_like(y_true)

          y_true_masked = y_true * mask
          y_pred_masked = y_pred * mask
          error2 = tf.square(y_true_masked - y_pred_masked) * mask
          n_valid = tf.reduce_sum(mask)
          mse = tf.reduce_sum(error2) / (n_valid + 1e-8)

          mean_abs = tf.reduce_sum(tf.abs(y_true_masked), axis=(1, 2, 3), keepdims=True) / (tf.reduce_sum(mask, axis=(1, 2, 3), keepdims=True) + 1e-8)
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
            if self.obst_bool is not None:
                mask = tf.cast(self.obst_bool[:, :, :, 0] != 0, tf.float32)
                mask = tf.expand_dims(mask, axis=0)
            else:
                mask = tf.ones_like(y_true[..., 0])

            mask_c = tf.expand_dims(mask, axis=-1)
            _div_u = self.div_u_batch if hasattr(self, "div_u_batch") and self.div_u_batch is not None else div_u

            if _div_u is not None:
                div_u_tf = tf.cast(_div_u, y_pred.dtype)
                if len(div_u_tf.shape) == 3:
                    div_u_tf = tf.expand_dims(div_u_tf, axis=0)
                abs_div = tf.abs(div_u_tf)
                flat = tf.reshape(abs_div, [tf.shape(abs_div)[0], -1])
                sorted_flat = tf.sort(flat, axis=-1)
                n = tf.shape(sorted_flat)[-1]
                p10_idx = tf.cast(0.10 * tf.cast(n, tf.float32), tf.int32)
                p10 = sorted_flat[:, p10_idx]
                p10 = p10[:, None, None, None]
                div_mask = tf.cast(abs_div > p10, y_pred.dtype)
                div_mask = tf.nn.max_pool3d(div_mask[..., None], ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding="SAME")[..., 0]
                div_mask = div_mask * mask
            else:
                div_mask = mask

            div_mask_c = tf.expand_dims(div_mask, axis=-1)

            p_true = y_true[..., 0]
            p_pred = y_pred[..., 0]
            err_p = tf.square(p_true - p_pred) * mask
            n_valid = tf.reduce_sum(mask)
            mse_p = tf.reduce_sum(err_p) / (n_valid + 1e-8)
            mean_abs_p = tf.reduce_sum(tf.abs(p_true) * mask) / (n_valid + 1e-8)
            rel_amp_p = tf.clip_by_value(tf.abs(p_true) / (mean_abs_p + 1e-8), 0.0, cap)
            weights_p = (1.0 + beta * rel_amp_p) * mask
            weighted_mse_p = tf.reduce_sum(weights_p * err_p) / (n_valid + 1e-8)
            loss_p = (1.0 - alpha) * mse_p + alpha * weighted_mse_p

            u_true = y_true[..., 1:4]
            u_pred = y_pred[..., 1:4]
            err_u = tf.square(u_true - u_pred) * div_mask_c
            n_valid_u = tf.reduce_sum(div_mask_c)
            mse_u = tf.reduce_sum(err_u) / (n_valid_u + 1e-8)
            mean_abs_u = tf.reduce_sum(tf.abs(u_true) * div_mask_c) / (n_valid_u + 1e-8)
            rel_amp_u = tf.clip_by_value(tf.abs(u_true) / (mean_abs_u + 1e-8), 0.0, cap)
            weights_u = (1.0 + beta * rel_amp_u) * div_mask_c
            weighted_mse_u = tf.reduce_sum(weights_u * err_u) / (n_valid_u + 1e-8)
            loss_u = (1.0 - alpha) * mse_u + alpha * weighted_mse_u

            cont_loss = 0.0
            if _div_u is not None:
                if std_out_vel is not None and mean_out_vel is not None:
                    _s = tf.reshape(tf.cast(std_out_vel, y_pred.dtype), [1, 1, 1, 1, 3])
                    _m = tf.reshape(tf.cast(mean_out_vel, y_pred.dtype), [1, 1, 1, 1, 3])
                    vel_pred = y_pred[..., 1:4] * _s + _m
                else:
                    vel_pred = y_pred[..., 1:4]

                vel_pred = vel_pred * div_mask_c
                ddU_x = vel_pred[..., 0]
                ddU_y = vel_pred[..., 1]
                ddU_z = vel_pred[..., 2]

                dUx_dx = (ddU_x[:, 1:-1, 1:-1, 2:] - ddU_x[:, 1:-1, 1:-1, :-2]) / (2.0 * _dx)
                dUy_dy = (ddU_y[:, 1:-1, 2:, 1:-1] - ddU_y[:, 1:-1, :-2, 1:-1]) / (2.0 * _dy)
                dUz_dz = (ddU_z[:, 2:, 1:-1, 1:-1] - ddU_z[:, :-2, 1:-1, 1:-1]) / (2.0 * _dz)
                div_ddU = dUx_dx + dUy_dy + dUz_dz
                div_u_interior = div_u_tf[:, 1:-1, 1:-1, 1:-1]
                cont_residual = div_u_interior + div_ddU
                cont_mask = div_mask[:, 1:-1, 1:-1, 1:-1]
                cont_loss = tf.reduce_sum(tf.square(cont_residual) * cont_mask) / (tf.reduce_sum(cont_mask) + 1e-8)

            total_loss = w_p * loss_p + w_u * loss_u + w_cont * cont_loss
            return 100.0 * total_loss

        return loss_f

  @staticmethod
  def lowpass_3d(y, pool_size=(3, 7, 15)):
    y5 = y[..., None]
    y_smooth = tf.nn.avg_pool3d(y5, ksize=[1, pool_size[0], pool_size[1], pool_size[2], 1], strides=[1, 1, 1, 1, 1], padding="SAME")
    return tf.squeeze(y_smooth, axis=-1)

  @staticmethod
  def smoothness_loss_3d(p, wz=0.25, wy=1.0, wx=4.0):
      dz = p[:, 1:, :, :] - p[:, :-1, :, :]
      dy = p[:, :, 1:, :] - p[:, :, :-1, :]
      dx = p[:, :, :, 1:] - p[:, :, :, :-1]
      return wz * tf.reduce_mean(tf.square(dz)) + wy * tf.reduce_mean(tf.square(dy)) + wx * tf.reduce_mean(tf.square(dx))

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

          y_true = y_true - tf.reduce_mean(y_true, axis=(1, 2, 3), keepdims=True)
          error2 = tf.square(y_true - p_total)
          mse = tf.reduce_mean(error2)
          mean_abs = tf.reduce_mean(tf.abs(y_true), axis=(1, 2, 3), keepdims=True) + 1e-8
          rel_amp = tf.abs(y_true) / mean_abs
          rel_amp = tf.clip_by_value(rel_amp, 0.0, cap)
          weights = 1.0 + beta * rel_amp
          weighted_mse = tf.reduce_mean(weights * error2)
          main_loss = (1.0 - alpha) * mse + alpha * weighted_mse

          y_smooth = self.lowpass_3d(y_true, pool_size=pool_size)
          y_smooth = y_smooth - tf.reduce_mean(y_smooth, axis=(1, 2, 3), keepdims=True)
          p_smooth_centered = p_smooth - tf.reduce_mean(p_smooth, axis=(1, 2, 3), keepdims=True)
          smooth_loss = tf.reduce_mean(tf.square(y_smooth - p_smooth_centered))
          total_loss = main_loss + lambda_smooth * smooth_loss

          if lambda_local > 0.0:
              y_local = y_true - y_smooth
              p_local_centered = p_local - tf.reduce_mean(p_local, axis=(1, 2, 3), keepdims=True)
              local_loss = tf.reduce_mean(tf.square(y_local - p_local_centered))
              total_loss += lambda_local * local_loss

          if lambda_smoothness > 0.0:
              smooth_reg = self.smoothness_loss_3d(p_smooth_centered, wz=0.25, wy=1.0, wx=1.0)
              total_loss += lambda_smoothness * smooth_reg

          return 100.0 * total_loss
      return loss_f

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
        def masked_mse(y_true_f, y_pred_f, mask):
            sq_err = tf.square(y_true_f - y_pred_f) * mask
            return tf.reduce_sum(sq_err) / (tf.reduce_sum(mask) + 1e-8)

        def masked_gradient_loss_3d(y_true_f, y_pred_f, mask):
            dz_true = y_true_f[:, 1:, :, :] - y_true_f[:, :-1, :, :]
            dz_pred = y_pred_f[:, 1:, :, :] - y_pred_f[:, :-1, :, :]
            dz_mask = mask[:, 1:, :, :] * mask[:, :-1, :, :]

            dy_true = y_true_f[:, :, 1:, :] - y_true_f[:, :, :-1, :]
            dy_pred = y_pred_f[:, :, 1:, :] - y_pred_f[:, :, :-1, :]
            dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]

            dx_true = y_true_f[:, :, :, 1:] - y_true_f[:, :, :, :-1]
            dx_pred = y_pred_f[:, :, :, 1:] - y_pred_f[:, :, :, :-1]
            dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]

            loss_z = tf.reduce_sum(tf.abs(dz_true - dz_pred) * dz_mask) / (tf.reduce_sum(dz_mask) + 1e-8)
            loss_y = tf.reduce_sum(tf.abs(dy_true - dy_pred) * dy_mask) / (tf.reduce_sum(dy_mask) + 1e-8)
            loss_x = tf.reduce_sum(tf.abs(dx_true - dx_pred) * dx_mask) / (tf.reduce_sum(dx_mask) + 1e-8)
            return loss_z + loss_y + loss_x

        def masked_smooth_target(y_total, mask):
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
            s = tf.squeeze(s, axis=-1)
            return s

        def loss_f(y_true, y_pred):
            if not isinstance(y_pred, dict):
                raise ValueError("This loss expects model outputs as a dict with keys 'p_total', 'p_smooth', 'p_local'.")

            p_total_pred = y_pred["p_total"]
            p_smooth_pred = y_pred["p_smooth"]
            p_local_pred = y_pred["p_local"]

            if len(y_true.shape) == 5:
                y_total = y_true[..., 0]
            elif len(y_true.shape) == 4:
                y_total = y_true
            else:
                raise ValueError("y_true must have shape (batch, z, y, x) or (batch, z, y, x, 1).")

            if self.obst_bool is not None:
                mask = tf.cast(self.obst_bool[:, :, :, 0] != 0, dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=0)
            else:
                mask = tf.ones_like(y_total)

            y_smooth = masked_smooth_target(y_total, mask)
            y_local = y_total - y_smooth

            loss_total = masked_mse(y_total, p_total_pred, mask)
            loss_smooth = masked_mse(y_smooth, p_smooth_pred, mask)
            loss_local = masked_mse(y_local, p_local_pred, mask)

            grad_total = masked_gradient_loss_3d(y_total, p_total_pred, mask)
            grad_local = masked_gradient_loss_3d(y_local, p_local_pred, mask)
            grad_loss = grad_total + lambda_local_grad * grad_local

            loss = (
                lambda_total * loss_total
                + lambda_smooth * loss_smooth
                + lambda_local * loss_local
                + lambda_grad * grad_loss
            )
            return scale * loss

        return loss_f


  def my_multihead_loss_dp_main_from_total_only(
        self,
        mean_out_ddp,
        std_out_ddp,
        max_abs_ddp,
        mean_in_dpprev,
        std_in_dpprev,
        max_abs_dpprev,
        lambda_res=1.0,              # main residual fit
        lambda_smooth=0.0,           # residual smooth-head supervision
        lambda_local=0.0,            # residual local-head supervision
        lambda_local_grad=0.0,       # gradient-difference on local residual
        lambda_dp_smooth_abs=0.0,    # NEW: absolute smoothness penalty on final dP_pred
        lambda_improve=0.0,          # NEW: penalize being worse than dP_prev baseline
        improve_margin=0.0,          # optional margin in baseline comparison
        smooth_ksize=(3, 3, 5),
        smooth_passes=1,
        scale=100.0,
        wz=0.25, wy=1.0, wx=4.0,     # anisotropic smoothness weights for dP_pred
    ):
        """
        Residual loss with NON-EQUIVALENT dP-based regularization.

        Expected y_true:
            y_true[..., 0] = ddP_true_norm
            y_true[..., 1] = dP_prev_norm

        Expected y_pred:
            dict with keys:
                "p_total"  -> ddP_pred_norm
                "p_smooth" -> smooth residual component
                "p_local"  -> local residual component

        IMPORTANT:
        - MSE(dP_true, dP_pred) == MSE(ddP_true, ddP_pred), so we do NOT use loss_dp here.
        - Instead, we add:
            1) absolute smoothness penalty on dP_pred
            2) baseline-improvement penalty w.r.t. dP_prev
        """

        mean_out_ddp_t   = tf.constant(float(mean_out_ddp), dtype=tf.float32)
        std_out_ddp_t    = tf.constant(float(std_out_ddp), dtype=tf.float32)
        max_abs_ddp_t    = tf.constant(float(max_abs_ddp), dtype=tf.float32)

        mean_in_dpprev_t = tf.constant(float(mean_in_dpprev), dtype=tf.float32)
        std_in_dpprev_t  = tf.constant(float(std_in_dpprev), dtype=tf.float32)
        max_abs_dpprev_t = tf.constant(float(max_abs_dpprev), dtype=tf.float32)

        def masked_mse(y_true_f, y_pred_f, mask):
            sq_err = tf.square(y_true_f - y_pred_f) * mask
            return tf.reduce_sum(sq_err) / (tf.reduce_sum(mask) + 1e-8)

        def masked_gradient_difference_loss_3d(y_true_f, y_pred_f, mask):
            # z
            dz_true = y_true_f[:, 1:, :, :] - y_true_f[:, :-1, :, :]
            dz_pred = y_pred_f[:, 1:, :, :] - y_pred_f[:, :-1, :, :]
            dz_mask = mask[:, 1:, :, :] * mask[:, :-1, :, :]

            # y
            dy_true = y_true_f[:, :, 1:, :] - y_true_f[:, :, :-1, :]
            dy_pred = y_pred_f[:, :, 1:, :] - y_pred_f[:, :, :-1, :]
            dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]

            # x
            dx_true = y_true_f[:, :, :, 1:] - y_true_f[:, :, :, :-1]
            dx_pred = y_pred_f[:, :, :, 1:] - y_pred_f[:, :, :, :-1]
            dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]

            loss_z = tf.reduce_sum(tf.abs(dz_true - dz_pred) * dz_mask) / (tf.reduce_sum(dz_mask) + 1e-8)
            loss_y = tf.reduce_sum(tf.abs(dy_true - dy_pred) * dy_mask) / (tf.reduce_sum(dy_mask) + 1e-8)
            loss_x = tf.reduce_sum(tf.abs(dx_true - dx_pred) * dx_mask) / (tf.reduce_sum(dx_mask) + 1e-8)

            return loss_z + loss_y + loss_x

        def masked_absolute_smoothness_loss_3d(p, mask, wz=0.25, wy=1.0, wx=4.0):
            """
            Absolute smoothness on p itself (NOT difference to truth).
            This is what makes the dP-based regularization genuinely different.
            """
            # z
            dz = p[:, 1:, :, :] - p[:, :-1, :, :]
            dz_mask = mask[:, 1:, :, :] * mask[:, :-1, :, :]
            loss_z = tf.reduce_sum(tf.square(dz) * dz_mask) / (tf.reduce_sum(dz_mask) + 1e-8)

            # y
            dy = p[:, :, 1:, :] - p[:, :, :-1, :]
            dy_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            loss_y = tf.reduce_sum(tf.square(dy) * dy_mask) / (tf.reduce_sum(dy_mask) + 1e-8)

            # x
            dx = p[:, :, :, 1:] - p[:, :, :, :-1]
            dx_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            loss_x = tf.reduce_sum(tf.square(dx) * dx_mask) / (tf.reduce_sum(dx_mask) + 1e-8)

            return wz * loss_z + wy * loss_y + wx * loss_x

        def baseline_improvement_loss(dp_true, dp_pred, dpprev, mask, margin=0.0):
            """
            Penalize the model when it is worse than simply using dP_prev.

            err_pred = |dp_true - dp_pred|
            err_base = |dp_true - dpprev|

            loss = relu(err_pred - err_base + margin)
            """
            err_pred = tf.abs(dp_true - dp_pred)
            err_base = tf.abs(dp_true - dpprev)

            worse_than_baseline = tf.nn.relu(err_pred - err_base + margin)
            return tf.reduce_sum(worse_than_baseline * mask) / (tf.reduce_sum(mask) + 1e-8)

        def masked_smooth_target(y_total, mask):
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
            s = tf.squeeze(s, axis=-1)
            return s

        def loss_f(y_true, y_pred):
            if not isinstance(y_pred, dict):
                raise ValueError(
                    "This loss expects model outputs as a dict with keys "
                    "'p_total', 'p_smooth', 'p_local'."
                )

            ddp_pred_norm   = y_pred["p_total"]
            ddp_smooth_norm = y_pred["p_smooth"]
            ddp_local_norm  = y_pred["p_local"]

            if len(y_true.shape) != 5 or y_true.shape[-1] < 2:
                raise ValueError(
                    "y_true must have shape (batch, z, y, x, 2): "
                    "[ddP_true_norm, dP_prev_norm]"
                )

            ddp_true_norm = y_true[..., 0]
            dpprev_norm   = y_true[..., 1]

            if self.obst_bool is not None:
                mask = tf.cast(self.obst_bool[:, :, :, 0] != 0, dtype=tf.float32)
                mask = tf.expand_dims(mask, axis=0)
            else:
                mask = tf.ones_like(ddp_true_norm)

            # ------------------------------------------------------------
            # Undo normalization to common non-dimensional pressure space
            # ------------------------------------------------------------
            ddp_true = (ddp_true_norm * std_out_ddp_t + mean_out_ddp_t) * max_abs_ddp_t
            ddp_pred = (ddp_pred_norm * std_out_ddp_t + mean_out_ddp_t) * max_abs_ddp_t

            ddp_smooth_pred = (ddp_smooth_norm * std_out_ddp_t + mean_out_ddp_t) * max_abs_ddp_t
            ddp_local_pred  = (ddp_local_norm  * std_out_ddp_t + mean_out_ddp_t) * max_abs_ddp_t

            dpprev = (dpprev_norm * std_in_dpprev_t + mean_in_dpprev_t) * max_abs_dpprev_t

            # Final fields
            dp_true = dpprev + ddp_true
            dp_pred = dpprev + ddp_pred

            # Residual decomposition targets
            ddp_smooth_true = masked_smooth_target(ddp_true, mask)
            ddp_local_true  = ddp_true - ddp_smooth_true

            # ------------------------------------------------------------
            # Loss terms
            # ------------------------------------------------------------
            loss_res = masked_mse(ddp_true, ddp_pred, mask)

            loss_smooth = masked_mse(ddp_smooth_true, ddp_smooth_pred, mask) if lambda_smooth > 0.0 else 0.0
            loss_local  = masked_mse(ddp_local_true,  ddp_local_pred,  mask) if lambda_local  > 0.0 else 0.0

            grad_local = masked_gradient_difference_loss_3d(ddp_local_true, ddp_local_pred, mask) if lambda_local_grad > 0.0 else 0.0

            # NEW: genuinely different terms
            loss_dp_smooth_abs = masked_absolute_smoothness_loss_3d(dp_pred, mask, wz=wz, wy=wy, wx=wx) if lambda_dp_smooth_abs > 0.0 else 0.0
            loss_improve = baseline_improvement_loss(dp_true, dp_pred, dpprev, mask, margin=improve_margin) if lambda_improve > 0.0 else 0.0

            # ------------------------------------------------------------
            # Final loss
            # ------------------------------------------------------------
            loss = (
                lambda_res * loss_res
                + lambda_smooth * loss_smooth
                + lambda_local * loss_local
                + lambda_local_grad * grad_local
                + lambda_dp_smooth_abs * loss_dp_smooth_abs
                + lambda_improve * loss_improve
            )

            tf.print(
                "loss_res:", loss_res,
                "loss_smooth:", loss_smooth,
                "loss_local:", loss_local,
                "grad_local:", grad_local,
                "loss_dp_smooth_abs:", loss_dp_smooth_abs,
                "loss_improve:", loss_improve
            )

            return scale * loss

        return loss_f


  def plot_decomposed_predictions(self, model_h5_path: str, flatten_data: bool = False):
    print("\n=== Plotting decomposed predictions (p_smooth, p_local, p_total) ===")
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
      break

    if sample_batch is None:
      print("ERROR: Could not load test batch for visualization")
      return

    predictions = self.model(sample_batch, training=False)
    p_smooth = predictions['p_smooth'].numpy()
    p_local = predictions['p_local'].numpy()
    p_total = predictions['p_total'].numpy()
    y_true = sample_labels.numpy()
    if y_true.ndim == 5 and y_true.shape[-1] >= 1:
      y_true = y_true[..., 0]

    sample_idx = 0
    z_slice = p_smooth.shape[1] // 2
    y_slice = p_smooth.shape[2] // 2
    x_slice = p_smooth.shape[3] // 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Z-Slice (at z={z_slice}): Comparing Components', fontsize=14, fontweight='bold')
    im = axes[0, 0].imshow(y_true[sample_idx, z_slice, :, :], cmap='RdBu_r'); axes[0, 0].set_title('Ground Truth (y_true)'); axes[0, 0].set_ylabel('Y'); plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(p_smooth[sample_idx, z_slice, :, :], cmap='RdBu_r'); axes[0, 1].set_title('p_smooth (Far-field)'); plt.colorbar(im, ax=axes[0, 1])
    im = axes[0, 2].imshow(p_local[sample_idx, z_slice, :, :], cmap='RdBu_r'); axes[0, 2].set_title('p_local (Obstacle)'); plt.colorbar(im, ax=axes[0, 2])
    im = axes[0, 3].imshow(p_total[sample_idx, z_slice, :, :], cmap='RdBu_r'); axes[0, 3].set_title('p_total (prediction)'); plt.colorbar(im, ax=axes[0, 3])
    im = axes[1, 0].imshow(np.abs(y_true[sample_idx, z_slice, :, :] - p_total[sample_idx, z_slice, :, :]), cmap='Reds'); axes[1, 0].set_title('|Error| (total)'); axes[1, 0].set_ylabel('Y'); plt.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].imshow(np.abs(y_true[sample_idx, z_slice, :, :] - p_smooth[sample_idx, z_slice, :, :]), cmap='Reds'); axes[1, 1].set_title('|Error| (smooth)'); plt.colorbar(im, ax=axes[1, 1])
    im = axes[1, 2].imshow(np.abs(y_true[sample_idx, z_slice, :, :] - p_local[sample_idx, z_slice, :, :]), cmap='Reds'); axes[1, 2].set_title('|Error| (local)'); plt.colorbar(im, ax=axes[1, 2])
    im = axes[1, 3].imshow((p_smooth + p_local)[sample_idx, z_slice, :, :], cmap='RdBu_r'); axes[1, 3].set_title('p_smooth + p_local'); axes[1, 3].set_xlabel('X'); plt.colorbar(im, ax=axes[1, 3])
    plt.tight_layout(); plt.savefig(f'{model_h5_path}/decomposed_slices_z.png', dpi=100, bbox_inches='tight'); plt.close(); print(f"Saved: {model_h5_path}/decomposed_slices_z.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Y-Slice (at y={y_slice}): Comparing Components', fontsize=14, fontweight='bold')
    im = axes[0, 0].imshow(y_true[sample_idx, :, y_slice, :], cmap='RdBu_r'); axes[0, 0].set_title('Ground Truth (y_true)'); axes[0, 0].set_ylabel('Z'); plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(p_smooth[sample_idx, :, y_slice, :], cmap='RdBu_r'); axes[0, 1].set_title('p_smooth (Far-field)'); plt.colorbar(im, ax=axes[0, 1])
    im = axes[0, 2].imshow(p_local[sample_idx, :, y_slice, :], cmap='RdBu_r'); axes[0, 2].set_title('p_local (Obstacle)'); plt.colorbar(im, ax=axes[0, 2])
    im = axes[0, 3].imshow(p_total[sample_idx, :, y_slice, :], cmap='RdBu_r'); axes[0, 3].set_title('p_total (prediction)'); plt.colorbar(im, ax=axes[0, 3])
    im = axes[1, 0].imshow(np.abs(y_true[sample_idx, :, y_slice, :] - p_total[sample_idx, :, y_slice, :]), cmap='Reds'); axes[1, 0].set_title('|Error| (total)'); axes[1, 0].set_ylabel('Z'); plt.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].imshow(np.abs(y_true[sample_idx, :, y_slice, :] - p_smooth[sample_idx, :, y_slice, :]), cmap='Reds'); axes[1, 1].set_title('|Error| (smooth)'); plt.colorbar(im, ax=axes[1, 1])
    im = axes[1, 2].imshow(np.abs(y_true[sample_idx, :, y_slice, :] - p_local[sample_idx, :, y_slice, :]), cmap='Reds'); axes[1, 2].set_title('|Error| (local)'); plt.colorbar(im, ax=axes[1, 2])
    im = axes[1, 3].imshow((p_smooth + p_local)[sample_idx, :, y_slice, :], cmap='RdBu_r'); axes[1, 3].set_title('p_smooth + p_local'); axes[1, 3].set_xlabel('X'); plt.colorbar(im, ax=axes[1, 3])
    plt.tight_layout(); plt.savefig(f'{model_h5_path}/decomposed_slices_y.png', dpi=100, bbox_inches='tight'); plt.close(); print(f"Saved: {model_h5_path}/decomposed_slices_y.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'X-Slice (at x={x_slice}): Comparing Components', fontsize=14, fontweight='bold')
    im = axes[0, 0].imshow(y_true[sample_idx, :, :, x_slice], cmap='RdBu_r'); axes[0, 0].set_title('Ground Truth (y_true)'); axes[0, 0].set_ylabel('Z'); plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].imshow(p_smooth[sample_idx, :, :, x_slice], cmap='RdBu_r'); axes[0, 1].set_title('p_smooth (Far-field)'); plt.colorbar(im, ax=axes[0, 1])
    im = axes[0, 2].imshow(p_local[sample_idx, :, :, x_slice], cmap='RdBu_r'); axes[0, 2].set_title('p_local (Obstacle)'); plt.colorbar(im, ax=axes[0, 2])
    im = axes[0, 3].imshow(p_total[sample_idx, :, :, x_slice], cmap='RdBu_r'); axes[0, 3].set_title('p_total (prediction)'); plt.colorbar(im, ax=axes[0, 3])
    im = axes[1, 0].imshow(np.abs(y_true[sample_idx, :, :, x_slice] - p_total[sample_idx, :, :, x_slice]), cmap='Reds'); axes[1, 0].set_title('|Error| (total)'); axes[1, 0].set_ylabel('Z'); plt.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].imshow(np.abs(y_true[sample_idx, :, :, x_slice] - p_smooth[sample_idx, :, :, x_slice]), cmap='Reds'); axes[1, 1].set_title('|Error| (smooth)'); plt.colorbar(im, ax=axes[1, 1])
    im = axes[1, 2].imshow(np.abs(y_true[sample_idx, :, :, x_slice] - p_local[sample_idx, :, :, x_slice]), cmap='Reds'); axes[1, 2].set_title('|Error| (local)'); plt.colorbar(im, ax=axes[1, 2])
    im = axes[1, 3].imshow((p_smooth + p_local)[sample_idx, :, :, x_slice], cmap='RdBu_r'); axes[1, 3].set_title('p_smooth + p_local'); axes[1, 3].set_xlabel('Y'); plt.colorbar(im, ax=axes[1, 3])
    plt.tight_layout(); plt.savefig(f'{model_h5_path}/decomposed_slices_x.png', dpi=100, bbox_inches='tight'); plt.close(); print(f"Saved: {model_h5_path}/decomposed_slices_x.png")
    print("=== Visualization complete ===\n")



  def plot_test_predictions_z_slices(
        self,
        model_h5_path: str,
        flatten_data: bool = False,
        n_z_slices: int = 5,
        obst_bool=None,
        mean_std_fn=None,
        predict_ddUCorr_output: bool = False,
        div_u_ch_idx: int = None,
        maxs_fn: str = None,
        consider_dp_loss: bool = True,
        dp_prev_input_ch_idx: int = 14,
        dp_prev_maxs_idx: int = None,
):
    """
    Plot Z-slices for:
        - ddP (always)
        - dP   (when consider_dp_loss=True and y_true includes dP_prev as channel 1)

    For dP:
        dP_true = dP_prev + ddP_true
        dP_pred = dP_prev + ddP_pred

    Also plots baseline error:
        |dP_true - dP_prev|
    so you can compare the surrogate against simply reusing dP_prev.
    """

    print(f"\n=== Plotting test predictions ({n_z_slices} Z-slices per sample) ===")
    os.makedirs(model_h5_path, exist_ok=True)

    ddp_dir = os.path.join(model_h5_path, 'ddp')
    dp_dir = os.path.join(model_h5_path, 'dP')
    ddU_dir = os.path.join(model_h5_path, 'ddU')

    os.makedirs(ddp_dir, exist_ok=True)
    os.makedirs(dp_dir, exist_ok=True)
    os.makedirs(ddU_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load normalization stats
    # ------------------------------------------------------------
    denorm_mean_out = None
    denorm_std_out = None
    denorm_mean_in = None
    denorm_std_in = None

    if mean_std_fn is not None and os.path.exists(mean_std_fn):
        data = np.load(mean_std_fn)
        denorm_mean_out = data['mean_out']
        denorm_std_out = data['std_out']
        if 'mean_in' in data:
            denorm_mean_in = data['mean_in']
            denorm_std_in = data['std_in']
        print(f"Loaded standardization factors from {mean_std_fn}")

    # ------------------------------------------------------------
    # Load max_abs scaling factors
    # ------------------------------------------------------------
    max_abs_ddp = None
    max_abs_ddU = None
    max_abs_dpprev = None

    _maxs_fn = maxs_fn if maxs_fn is not None else os.path.join(model_h5_path, 'maxs')
    if os.path.exists(_maxs_fn):
        _maxs = np.loadtxt(_maxs_fn)

        if predict_ddUCorr_output and len(_maxs) >= 4:
            max_abs_ddp = float(_maxs[-4])
            max_abs_ddU = np.array([
                float(_maxs[-3]),
                float(_maxs[-2]),
                float(_maxs[-1])
            ])
        else:
            max_abs_ddp = float(_maxs[-1])

        if consider_dp_loss:
            dp_prev_maxs_idx = (dp_prev_input_ch_idx + 1) if dp_prev_maxs_idx is None else dp_prev_maxs_idx  # +1: SDF at _vel_end shifts pressure channels by 1 in maxs file
            max_abs_dpprev = float(np.ravel(_maxs)[dp_prev_maxs_idx])

        print(f"Loaded max_abs_ddp={max_abs_ddp:.6g} from {_maxs_fn}")
        if max_abs_dpprev is not None:
            print(f"Loaded max_abs_dpprev={max_abs_dpprev:.6g} from {_maxs_fn}")
    else:
        print(f"[plot_test_predictions_z_slices] maxs file not found at {_maxs_fn}, skipping max_abs scaling")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _plot_field_z_slices_3rows(
        y_true_f, y_pred_f, global_idx, b, z_indices, obst_bool,
        row_labels, n_z_slices, title_prefix, filename
    ):
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

                im = axes[row, col].imshow(
                    sl,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vm,
                    aspect='auto',
                    interpolation='none'
                )

                title = f'{row_labels[row]}\nz={z_idx}' if col == 0 else f'z={z_idx}'
                axes[row, col].set_title(title, fontsize=8)
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(filename, dpi=80, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

    def _plot_field_z_slices_4rows(
        y_true_f, y_pred_f, y_base_f, global_idx, b, z_indices, obst_bool,
        row_labels, n_z_slices, title_prefix, filename
    ):
        """
        4-row plot for dP:
            row 0: ground truth dP
            row 1: predicted dP
            row 2: |pred error|
            row 3: |baseline error| where baseline = dP_prev
        """
        fig, axes = plt.subplots(4, n_z_slices, figsize=(8 * n_z_slices, 13))
        fig.suptitle(f'{title_prefix} — sample {global_idx}', fontsize=14, fontweight='bold')

        for col, z_idx in enumerate(z_indices):
            sl_true = y_true_f[b, z_idx]
            sl_pred = y_pred_f[b, z_idx]
            sl_base = y_base_f[b, z_idx]

            sl_err_pred = np.abs(sl_true - sl_pred)
            sl_err_base = np.abs(sl_true - sl_base)

            vmax_col = float(max(
                np.nanmax(np.abs(sl_true)),
                np.nanmax(np.abs(sl_pred)),
                np.nanmax(np.abs(sl_base))
            )) or 1.0

            vmax_err = float(max(
                np.nanmax(sl_err_pred),
                np.nanmax(sl_err_base)
            )) or 1.0

            slices = [sl_true, sl_pred, sl_err_pred, sl_err_base]
            vmaxes = [vmax_col, vmax_col, vmax_err, vmax_err]

            for row, (sl, vm) in enumerate(zip(slices, vmaxes)):
                cmap = 'RdBu_r' if row < 2 else 'Reds'
                vmin = -vm if row < 2 else 0.0

                if obst_bool is not None:
                    sl = np.ma.array(sl, mask=obst_bool[z_idx, ..., 0] == 0)

                im = axes[row, col].imshow(
                    sl,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vm,
                    aspect='auto',
                    interpolation='none'
                )

                title = f'{row_labels[row]}\nz={z_idx}' if col == 0 else f'z={z_idx}'
                axes[row, col].set_title(title, fontsize=8)
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(filename, dpi=80, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------
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

        # ============================================================
        # CASE 1: multi-output ddP + ddU
        # ============================================================
        multi_out = predict_ddUCorr_output and y_pred.ndim == 5

        if multi_out:
            if denorm_mean_out is not None and denorm_std_out is not None:
                y_true = y_true * denorm_std_out + denorm_mean_out
                y_pred = y_pred * denorm_std_out + denorm_mean_out

            y_true_ddp = y_true[..., 0]
            y_pred_ddp = y_pred[..., 0]
            y_true_ddU = [y_true[..., c] for c in (1, 2, 3)]
            y_pred_ddU = [y_pred[..., c] for c in (1, 2, 3)]

            if max_abs_ddp is not None:
                y_true_ddp = y_true_ddp * max_abs_ddp
                y_pred_ddp = y_pred_ddp * max_abs_ddp
            if max_abs_ddU is not None:
                y_true_ddU = [y_true_ddU[c] * float(max_abs_ddU[c]) for c in range(3)]
                y_pred_ddU = [y_pred_ddU[c] * float(max_abs_ddU[c]) for c in range(3)]

            outlet_offset = np.mean(y_pred_ddp[:, :, :, -2:], axis=(1, 2, 3))
            y_pred_ddp = y_pred_ddp - outlet_offset[:, np.newaxis, np.newaxis, np.newaxis]

            nz = y_true_ddp.shape[1]
            z_indices = [int(i * (nz - 1) / (n_z_slices - 1)) for i in range(n_z_slices)]
            row_labels = ['Ground Truth', 'Prediction', '|Error|']

            for b in range(y_true_ddp.shape[0]):
                _plot_field_z_slices_3rows(
                    y_true_ddp, y_pred_ddp,
                    global_idx, b, z_indices, obst_bool,
                    row_labels, n_z_slices,
                    title_prefix='ddp',
                    filename=os.path.join(ddp_dir, f'test_pred_{global_idx:04d}.png'),
                )

                for comp_idx, comp_name in enumerate(('ddU_x', 'ddU_y', 'ddU_z')):
                    _plot_field_z_slices_3rows(
                        y_true_ddU[comp_idx], y_pred_ddU[comp_idx],
                        global_idx, b, z_indices, obst_bool,
                        row_labels, n_z_slices,
                        title_prefix=comp_name,
                        filename=os.path.join(ddU_dir, f'test_pred_{global_idx:04d}_{comp_name}.png'),
                    )

                global_idx += 1

            continue

        # ============================================================
        # CASE 2: pressure-only
        # ============================================================
        has_dp_prev_in_y = (y_true.ndim == 5 and y_true.shape[-1] >= 2 and consider_dp_loss)

        if y_true.ndim == 5:
            y_true_ddp_norm = y_true[..., 0]
        else:
            y_true_ddp_norm = y_true

        y_pred_ddp_norm = y_pred

        if denorm_mean_out is not None and denorm_std_out is not None:
            y_true_ddp = y_true_ddp_norm * denorm_std_out + denorm_mean_out
            y_pred_ddp = y_pred_ddp_norm * denorm_std_out + denorm_mean_out
        else:
            y_true_ddp = y_true_ddp_norm
            y_pred_ddp = y_pred_ddp_norm

        if max_abs_ddp is not None:
            y_true_ddp = y_true_ddp * max_abs_ddp
            y_pred_ddp = y_pred_ddp * max_abs_ddp

        outlet_offset = np.mean(y_pred_ddp[:, :, :, -2:], axis=(1, 2, 3))
        y_pred_ddp = y_pred_ddp - outlet_offset[:, np.newaxis, np.newaxis, np.newaxis]

        nz = y_true_ddp.shape[1]
        z_indices = [int(i * (nz - 1) / (n_z_slices - 1)) for i in range(n_z_slices)]
        row_labels_3 = ['Ground Truth', 'Prediction', '|Error|']

        if has_dp_prev_in_y:
            if denorm_mean_in is None or denorm_std_in is None or max_abs_dpprev is None:
                raise ValueError(
                    "consider_dp_loss=True plotting requires mean_in/std_in and max_abs_dpprev."
                )

            dpprev_norm = y_true[..., 1]

            mean_in_dpprev = float(np.ravel(denorm_mean_in)[dp_prev_input_ch_idx])
            std_in_dpprev = float(np.ravel(denorm_std_in)[dp_prev_input_ch_idx])

            dpprev = (dpprev_norm * std_in_dpprev + mean_in_dpprev) * max_abs_dpprev

            dp_true = dpprev + y_true_ddp
            dp_pred_f10 = dpprev + y_pred_ddp * 0.1
            dp_pred_f25 = dpprev + y_pred_ddp * 0.25
            dp_pred_f50 = dpprev + y_pred_ddp * 0.5
            dp_pred = dpprev + y_pred_ddp

            row_labels_4 = [
                'Ground Truth dP',
                'Prediction dP',
                '|Error| pred',
                '|Error| dP_prev baseline'
            ]

        for b in range(y_true_ddp.shape[0]):
            _plot_field_z_slices_3rows(
                y_true_ddp, y_pred_ddp,
                global_idx, b, z_indices, obst_bool,
                row_labels_3, n_z_slices,
                title_prefix='ddp',
                filename=os.path.join(ddp_dir, f'test_pred_{global_idx:04d}.png'),
            )

            if has_dp_prev_in_y:
                pred_err = np.mean(np.abs(dp_true[b] - dp_pred[b]))
                pred_err_f10 = np.mean(np.abs(dp_true[b] - dp_pred_f10[b]))
                pred_err_f25 = np.mean(np.abs(dp_true[b] - dp_pred_f25[b]))
                pred_err_f50 = np.mean(np.abs(dp_true[b] - dp_pred_f50[b]))
                base_err = np.mean(np.abs(dp_true[b] - dpprev[b]))

                improvement = 100.0 * (1.0 - pred_err / (base_err + 1e-12))
                improvement_f10 = 100.0 * (1.0 - pred_err_f10 / (base_err + 1e-12))
                improvement_f25 = 100.0 * (1.0 - pred_err_f25 / (base_err + 1e-12))
                improvement_f50 = 100.0 * (1.0 - pred_err_f50 / (base_err + 1e-12))

                print(f"[dP plot] sample {global_idx}: "
                    f"baseline MAE={base_err:.3e}, "
                    f"pred MAE={pred_err:.3e}, "
                    f"improvement={improvement:.2f}%, "
                    f"pred_f10 MAE={pred_err_f10:.3e}, "
                    f"improvement_f10={improvement_f10:.2f}%, "
                    f"pred_f25 MAE={pred_err_f25:.3e}, "
                    f"improvement_f25={improvement_f25:.2f}%, "
                    f"pred_f50 MAE={pred_err_f50:.3e}, "
                    f"improvement_f50={improvement_f50:.2f}%"
                )


                _plot_field_z_slices_4rows(
                    dp_true, dp_pred, dpprev,
                    global_idx, b, z_indices, obst_bool,
                    row_labels_4, n_z_slices,
                    title_prefix='dP',
                    filename=os.path.join(dp_dir, f'test_pred_{global_idx:04d}_dP.png'),
                )

            global_idx += 1

    print(f"=== Done: {global_idx} test samples plotted ===\n")


  def plot_shifter_roi_debug(
      self,
      model_h5_path: str,
      flatten_data: bool = False,
      n_z_slices: int = 5,
      obst_bool=None,
      mean_std_fn=None,
      maxs_fn: str = None,
      gradDpPrev_input_ch_idxs: tuple = None,
      sample_limit: int = 1,
  ):
      """
      Visualize ROI mask used by ShifterLoss from gradDpPrev magnitude.

      Saves per-sample images with 3 rows:
          1) gradDpPrevMag
          2) soft ROI mask (0-1)
          3) overlay (grad magnitude + soft ROI contour)
      """
      if gradDpPrev_input_ch_idxs is None or len(gradDpPrev_input_ch_idxs) != 3:
          print("[ROI debug] Skipping ROI plot: gradDpPrev_input_ch_idxs must be a 3-tuple.")
          return

      roi_dir = os.path.join(model_h5_path, 'roi')
      os.makedirs(roi_dir, exist_ok=True)

      positive_eps = float(getattr(self.loss_object, 's_roi_positive_eps', 1e-12))
      dilation_radius = int(getattr(self.loss_object, 's_roi_dilation_radius', 3))

      denorm_mean_in = None
      denorm_std_in = None
      if mean_std_fn is not None and os.path.exists(mean_std_fn):
          data = np.load(mean_std_fn)
          if 'mean_in' in data and 'std_in' in data:
              denorm_mean_in = np.ravel(data['mean_in'])
              denorm_std_in = np.ravel(data['std_in'])

      _maxs_fn = maxs_fn if maxs_fn is not None else os.path.join(model_h5_path, 'maxs')
      maxs_flat = None
      if os.path.exists(_maxs_fn):
          maxs_flat = np.ravel(np.loadtxt(_maxs_fn))

      gx_idx, gy_idx, gz_idx = [int(i) for i in gradDpPrev_input_ch_idxs]
      plotted = 0

      print("\n=== Plotting ROI debug slices ===")
      print(
          f"[ROI debug] threshold=0.1*mean(positive_grad), positive_eps={positive_eps:.3e}, "
          f"dilation_radius={dilation_radius}"
      )

      for (x_batch, _) in self.test_dataset:
          if flatten_data:
              x_batch = tf.cast(x_batch[..., 0, 0], dtype='float32')
          else:
              x_batch = tf.cast(x_batch, dtype='float32')

          x_np = x_batch.numpy()
          if x_np.ndim != 5:
              print(f"[ROI debug] Unexpected x shape: {x_np.shape}. Expected (B,Z,Y,X,C).")
              return

          n_channels = x_np.shape[-1]
          if max(gx_idx, gy_idx, gz_idx) >= n_channels:
              print(
                  f"[ROI debug] grad indices {gradDpPrev_input_ch_idxs} out of bounds "
                  f"for x with {n_channels} channels."
              )
              return

          gx = x_np[..., gx_idx]
          gy = x_np[..., gy_idx]
          gz = x_np[..., gz_idx]

          if denorm_mean_in is not None and denorm_std_in is not None and maxs_flat is not None:
              gx = (gx * denorm_std_in[gx_idx] + denorm_mean_in[gx_idx]) * maxs_flat[gx_idx + 1]
              gy = (gy * denorm_std_in[gy_idx] + denorm_mean_in[gy_idx]) * maxs_flat[gy_idx + 1]
              gz = (gz * denorm_std_in[gz_idx] + denorm_mean_in[gz_idx]) * maxs_flat[gz_idx + 1]

          grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz + 1e-16).astype(np.float32)

          if obst_bool is None:
              fluid_mask = np.ones((1, grad_mag.shape[1], grad_mag.shape[2], grad_mag.shape[3], 1), dtype=np.float32)
          else:
              fluid_mask = (np.asarray(obst_bool)[..., 0] != 0).astype(np.float32)[None, ..., None]

          grad_mag_tf = tf.convert_to_tensor(grad_mag[..., None], dtype=tf.float32)
          fluid_mask_tf = tf.convert_to_tensor(fluid_mask, dtype=tf.float32)

          hard_dilation_radius = max(1, int(dilation_radius // 2))

          roi_mask_tf, grad_threshold_tf = _build_roi_mask_from_gradDpPrevMag(
              grad_mag_tf,
              fluid_mask_tf,
              positive_eps=positive_eps,
              dilation_radius=hard_dilation_radius,
          )

          roi_soft_mask_tf = _build_upstream_soft_roi_mask(
              roi_mask_tf,
              fluid_mask_tf,
              taper_width=max(1, 10 * int(dilation_radius)),
          )

          roi_mask = roi_mask_tf.numpy()[..., 0]
          roi_soft_mask = roi_soft_mask_tf.numpy()[..., 0]
          grad_threshold = float(grad_threshold_tf.numpy())
          nz = grad_mag.shape[1]
          z_indices = [int(i * (nz - 1) / (max(n_z_slices, 2) - 1)) for i in range(max(n_z_slices, 2))]

          batch_to_plot = min(int(sample_limit), int(grad_mag.shape[0]))
          for b in range(batch_to_plot):
              fig, axes = plt.subplots(3, len(z_indices), figsize=(6 * len(z_indices), 9))
              fig.suptitle(
                  f"ROI debug - sample {plotted} | threshold={grad_threshold:.3e}",
                  fontsize=12,
                  fontweight='bold'
              )

              fluid_slice = fluid_mask[0, ..., 0]
              fluid_cells = np.sum(fluid_slice > 0.5)
              roi_cells = np.sum((roi_soft_mask[b] > 0.5) & (fluid_slice > 0.5))
              roi_fraction = float(roi_cells / (fluid_cells + 1e-12))

              for col, z_idx in enumerate(z_indices):
                  sl_grad = grad_mag[b, z_idx]
                  sl_roi = roi_soft_mask[b, z_idx]
                  sl_fluid = fluid_slice[z_idx]

                  sl_grad_ma = np.ma.array(sl_grad, mask=sl_fluid < 0.5)
                  sl_roi_ma = np.ma.array(sl_roi, mask=sl_fluid < 0.5)

                  im0 = axes[0, col].imshow(sl_grad_ma, cmap='magma', aspect='auto', interpolation='none')
                  axes[0, col].set_title(f'gradDpPrevMag z={z_idx}', fontsize=8)
                  axes[0, col].axis('off')
                  plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

                  im1 = axes[1, col].imshow(sl_roi_ma, cmap='viridis', vmin=0.0, vmax=1.0, aspect='auto', interpolation='none')
                  axes[1, col].set_title(f'Soft ROI mask z={z_idx}', fontsize=8)
                  axes[1, col].axis('off')
                  plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

                  im2 = axes[2, col].imshow(sl_grad_ma, cmap='magma', aspect='auto', interpolation='none')
                  axes[2, col].imshow(sl_roi_ma, cmap='viridis', vmin=0.0, vmax=1.0, alpha=0.35, aspect='auto', interpolation='none')
                  axes[2, col].set_title(f'Overlay z={z_idx}', fontsize=8)
                  axes[2, col].axis('off')
                  plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

              plt.tight_layout()
              fname = os.path.join(roi_dir, f'roi_debug_{plotted:04d}.png')
              plt.savefig(fname, dpi=90, bbox_inches='tight')
              plt.close()

              print(f"  Saved: {fname} (roi_fraction={roi_fraction:.3f})")
              plotted += 1

          break

      if plotted == 0:
          print("[ROI debug] No samples plotted.")
      else:
          print(f"=== ROI debug complete: {plotted} sample(s) ===\n")



  def plot_shifter_predictions_z_slices(
      self,
      model_h5_path: str,
      flatten_data: bool = False,
      n_z_slices: int = 5,
      obst_bool=None,
      mean_std_fn=None,
      maxs_fn: str = None,
      consider_dp_loss: bool = True,
      dp_prev_input_ch_idx: int = 14,
      dp_prev_maxs_idx: int = None,
      gradDpPrev_input_ch_idxs: tuple = None,
      U_input_ch_idxs: tuple = None,
      uDotGradDpPrev_input_ch_idx: int = None,
  ):
      """
      Plot Z-slices for shifter model.

      Supported latent outputs:
          vector shifter: [ux, uy, uz, s]
          scalar-velocity shifter: [a, s]

      Supported y_true layouts:
          vector shifter:
              [ddP_true, grad_x, grad_y, grad_z]
              [ddP_true, dP_prev, grad_x, grad_y, grad_z]
          scalar-velocity shifter:
              [ddP_true, uDotGradDpPrev]
              [ddP_true, dP_prev, uDotGradDpPrev]
      """

      print(f"\n=== Plotting shifter predictions ({n_z_slices} Z-slices per sample) ===")
      os.makedirs(model_h5_path, exist_ok=True)

      ddp_dir = os.path.join(model_h5_path, 'ddp')
      dp_dir = os.path.join(model_h5_path, 'dP')
      latent_dir = os.path.join(model_h5_path, 'latent')
      shifter_dir = os.path.join(model_h5_path, 'shifter_terms')

      os.makedirs(ddp_dir, exist_ok=True)
      os.makedirs(dp_dir, exist_ok=True)
      os.makedirs(latent_dir, exist_ok=True)
      os.makedirs(shifter_dir, exist_ok=True)

      # ------------------------------------------------------------
      # Load normalization stats
      # ------------------------------------------------------------
      denorm_mean_out = None
      denorm_std_out = None
      denorm_mean_in = None
      denorm_std_in = None

      if mean_std_fn is not None and os.path.exists(mean_std_fn):
          data = np.load(mean_std_fn)
          denorm_mean_out = data['mean_out']
          denorm_std_out = data['std_out']
          if 'mean_in' in data:
              denorm_mean_in = data['mean_in']
              denorm_std_in = data['std_in']
          print(f"Loaded standardization factors from {mean_std_fn}")

      # ------------------------------------------------------------
      # Load max_abs scaling factors
      # ------------------------------------------------------------
      max_abs_ddp = None
      max_abs_dpprev = None

      _maxs_fn = maxs_fn if maxs_fn is not None else os.path.join(model_h5_path, 'maxs')
      if os.path.exists(_maxs_fn):
          _maxs = np.loadtxt(_maxs_fn)
          max_abs_ddp = float(np.ravel(_maxs)[-1])

          if consider_dp_loss:
              dp_prev_maxs_idx = (dp_prev_input_ch_idx + 1) if dp_prev_maxs_idx is None else dp_prev_maxs_idx  # +1: SDF at _vel_end shifts pressure channels by 1 in maxs file
              max_abs_dpprev = float(np.ravel(_maxs)[dp_prev_maxs_idx])

          print(f"Loaded max_abs_ddp={max_abs_ddp:.6g} from {_maxs_fn}")
          if max_abs_dpprev is not None:
              print(f"Loaded max_abs_dpprev={max_abs_dpprev:.6g} from {_maxs_fn}")
      else:
          print(f"[plot_shifter_predictions_z_slices] maxs file not found at {_maxs_fn}, skipping max_abs scaling")

      # ------------------------------------------------------------
      # Helpers
      # ------------------------------------------------------------
      def _mask_slice_if_needed(sl, z_idx, obst_bool):
          if obst_bool is not None:
              return np.ma.array(sl, mask=obst_bool[z_idx, ..., 0] == 0)
          return sl

      def _plot_rows_signed(
          data_list,
          row_labels,
          global_idx,
          b,
          z_indices,
          obst_bool,
          title_prefix,
          filename
      ):
          """
          Generic multi-row signed plot with common symmetric color scale.
          """
          n_rows = len(data_list)
          fig, axes = plt.subplots(n_rows, len(z_indices), figsize=(8 * len(z_indices), 3.2 * n_rows))
          if n_rows == 1:
              axes = np.expand_dims(axes, axis=0)
          fig.suptitle(f'{title_prefix} - sample {global_idx}', fontsize=14, fontweight='bold')

          vmax_signed = 1.0
          for arr in data_list:
              vmax_signed = max(vmax_signed, float(np.nanmax(np.abs(arr[b]))))

          for col, z_idx in enumerate(z_indices):
              for row, arr in enumerate(data_list):
                  sl = arr[b, z_idx]
                  sl = _mask_slice_if_needed(sl, z_idx, obst_bool)

                  im = axes[row, col].imshow(
                      sl,
                      cmap='RdBu_r',
                      vmin=-vmax_signed,
                      vmax=vmax_signed,
                      aspect='auto',
                      interpolation='none'
                  )

                  title = f'{row_labels[row]}\nz={z_idx}' if col == 0 else f'z={z_idx}'
                  axes[row, col].set_title(title, fontsize=8)
                  axes[row, col].axis('off')
                  plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

          plt.tight_layout()
          plt.savefig(filename, dpi=80, bbox_inches='tight')
          plt.close()
          print(f"  Saved: {filename}")

      def _plot_field_z_slices_3rows(
          y_true_f, y_pred_f, global_idx, b, z_indices, obst_bool,
          row_labels, title_prefix, filename
      ):
          fig, axes = plt.subplots(3, len(z_indices), figsize=(8 * len(z_indices), 10))
          fig.suptitle(f'{title_prefix} - sample {global_idx}', fontsize=14, fontweight='bold')

          for col, z_idx in enumerate(z_indices):
              sl_true = y_true_f[b, z_idx]
              sl_pred = y_pred_f[b, z_idx]
              sl_err = np.abs(sl_true - sl_pred)

              vmax_col = float(max(np.nanmax(np.abs(sl_true)), np.nanmax(np.abs(sl_pred)))) or 1.0
              vmax_err = float(np.nanmax(sl_err)) or 1.0

              slices = [sl_true, sl_pred, sl_err]
              vmaxes = [vmax_col, vmax_col, vmax_err]

              for row, (sl, vm) in enumerate(zip(slices, vmaxes)):
                  cmap = 'RdBu_r' if row < 2 else 'Reds'
                  vmin = -vm if row < 2 else 0.0

                  sl = _mask_slice_if_needed(sl, z_idx, obst_bool)

                  im = axes[row, col].imshow(
                      sl,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vm,
                      aspect='auto',
                      interpolation='none'
                  )

                  title = f'{row_labels[row]}\nz={z_idx}' if col == 0 else f'z={z_idx}'
                  axes[row, col].set_title(title, fontsize=8)
                  axes[row, col].axis('off')
                  plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

          plt.tight_layout()
          plt.savefig(filename, dpi=80, bbox_inches='tight')
          plt.close()
          print(f"  Saved: {filename}")

      def _plot_field_z_slices_4rows(
          y_true_f, y_pred_f, y_base_f, global_idx, b, z_indices, obst_bool,
          row_labels, title_prefix, filename
      ):
          fig, axes = plt.subplots(4, len(z_indices), figsize=(8 * len(z_indices), 13))
          fig.suptitle(f'{title_prefix} - sample {global_idx}', fontsize=14, fontweight='bold')

          for col, z_idx in enumerate(z_indices):
              sl_true = y_true_f[b, z_idx]
              sl_pred = y_pred_f[b, z_idx]
              sl_base = y_base_f[b, z_idx]

              sl_err_pred = np.abs(sl_true - sl_pred)
              sl_err_base = np.abs(sl_true - sl_base)

              vmax_col = float(max(
                  np.nanmax(np.abs(sl_true)),
                  np.nanmax(np.abs(sl_pred)),
                  np.nanmax(np.abs(sl_base))
              )) or 1.0

              vmax_err = float(max(
                  np.nanmax(sl_err_pred),
                  np.nanmax(sl_err_base)
              )) or 1.0

              slices = [sl_true, sl_pred, sl_err_pred, sl_err_base]
              vmaxes = [vmax_col, vmax_col, vmax_err, vmax_err]

              for row, (sl, vm) in enumerate(zip(slices, vmaxes)):
                  cmap = 'RdBu_r' if row < 2 else 'Reds'
                  vmin = -vm if row < 2 else 0.0

                  sl = _mask_slice_if_needed(sl, z_idx, obst_bool)

                  im = axes[row, col].imshow(
                      sl,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vm,
                      aspect='auto',
                      interpolation='none'
                  )

                  title = f'{row_labels[row]}\nz={z_idx}' if col == 0 else f'z={z_idx}'
                  axes[row, col].set_title(title, fontsize=8)
                  axes[row, col].axis('off')
                  plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

          plt.tight_layout()
          plt.savefig(filename, dpi=80, bbox_inches='tight')
          plt.close()
          print(f"  Saved: {filename}")

      # ------------------------------------------------------------
      # Main loop
      # ------------------------------------------------------------
      global_idx = 0

      for (x_batch, y_batch) in self.test_dataset:
          if flatten_data:
              x_batch = tf.cast(x_batch[..., 0, 0], dtype='float32')
              y_batch = tf.cast(y_batch[..., 0, 0], dtype='float32')
          else:
              x_batch = tf.cast(x_batch, dtype='float32')
              y_batch = tf.cast(y_batch, dtype='float32')

          x_full = x_batch.numpy()
          y_pred = self.model(x_batch, training=False).numpy()
          y_true_full = y_batch.numpy()

          # --------------------------------------------------------
          # Predicted latent fields and target decoding
          # --------------------------------------------------------
          if y_pred.ndim != 5:
              raise ValueError(
                  f"Shifter model must output a 5D tensor. Got {y_pred.shape}"
              )
          if y_true_full.ndim != 5 or y_true_full.shape[-1] < 2:
              raise ValueError(
                  f"Expected y_true shape (B,Z,Y,X,C>=2) for shifter plotting. Got {y_true_full.shape}"
              )

          latent_channels = int(y_pred.shape[-1])
          is_velocity_shifter = latent_channels == 4 and y_true_full.shape[-1] >= 7
          ddp_true_norm = y_true_full[..., 0]

          if is_velocity_shifter:
              a_x = y_pred[..., 0]
              a_y = y_pred[..., 1]
              a_z = y_pred[..., 2]
              src = y_pred[..., 3]

              if y_true_full.shape[-1] >= 8 and consider_dp_loss:
                  has_dp_prev_in_y = True
                  dpprev_norm_from_y = y_true_full[..., 1]
                  u_x_norm = y_true_full[..., 2]
                  u_y_norm = y_true_full[..., 3]
                  u_z_norm = y_true_full[..., 4]
                  grad_x_norm = y_true_full[..., 5]
                  grad_y_norm = y_true_full[..., 6]
                  grad_z_norm = y_true_full[..., 7]
              else:
                  has_dp_prev_in_y = False
                  dpprev_norm_from_y = None
                  u_x_norm = y_true_full[..., 1]
                  u_y_norm = y_true_full[..., 2]
                  u_z_norm = y_true_full[..., 3]
                  grad_x_norm = y_true_full[..., 4]
                  grad_y_norm = y_true_full[..., 5]
                  grad_z_norm = y_true_full[..., 6]

              if denorm_mean_out is not None and denorm_std_out is not None and max_abs_ddp is not None:
                  ddp_true = (ddp_true_norm * denorm_std_out + denorm_mean_out) * max_abs_ddp
                  src_common = src * denorm_std_out * max_abs_ddp

                  if (
                      U_input_ch_idxs is not None and len(U_input_ch_idxs) == 3 and
                      gradDpPrev_input_ch_idxs is not None and len(gradDpPrev_input_ch_idxs) == 3 and
                      denorm_mean_in is not None and denorm_std_in is not None and
                      maxs_fn is not None and os.path.exists(maxs_fn)
                  ):
                      _maxs = np.loadtxt(maxs_fn)
                      mean_in_flat = np.ravel(denorm_mean_in)
                      std_in_flat = np.ravel(denorm_std_in)
                      maxs_flat = np.ravel(_maxs)
                      ux_idx, uy_idx, uz_idx = [int(i) for i in U_input_ch_idxs]
                      gx_idx, gy_idx, gz_idx = [int(i) for i in gradDpPrev_input_ch_idxs]
                      u_x_phys = (u_x_norm * std_in_flat[ux_idx] + mean_in_flat[ux_idx]) * maxs_flat[ux_idx]
                      u_y_phys = (u_y_norm * std_in_flat[uy_idx] + mean_in_flat[uy_idx]) * maxs_flat[uy_idx]
                      u_z_phys = (u_z_norm * std_in_flat[uz_idx] + mean_in_flat[uz_idx]) * maxs_flat[uz_idx]
                      grad_x_phys = (grad_x_norm * std_in_flat[gx_idx] + mean_in_flat[gx_idx]) * maxs_flat[gx_idx + 1]
                      grad_y_phys = (grad_y_norm * std_in_flat[gy_idx] + mean_in_flat[gy_idx]) * maxs_flat[gy_idx + 1]
                      grad_z_phys = (grad_z_norm * std_in_flat[gz_idx] + mean_in_flat[gz_idx]) * maxs_flat[gz_idx + 1]
                  else:
                      u_x_phys = u_x_norm
                      u_y_phys = u_y_norm
                      u_z_phys = u_z_norm
                      grad_x_phys = grad_x_norm
                      grad_y_phys = grad_y_norm
                      grad_z_phys = grad_z_norm

                  shift_x_norm = -(a_x * u_x_norm * grad_x_norm)
                  shift_y_norm = -(a_y * u_y_norm * grad_y_norm)
                  shift_z_norm = -(a_z * u_z_norm * grad_z_norm)
                  shift_sum_norm = shift_x_norm + shift_y_norm + shift_z_norm
                  ddp_pred = (
                      -(a_x * u_x_phys * grad_x_phys)
                      -(a_y * u_y_phys * grad_y_phys)
                      -(a_z * u_z_phys * grad_z_phys)
                      + src_common
                  )
                  mean_out_scalar = float(np.ravel(denorm_mean_out)[0])
                  std_out_scalar = float(np.ravel(denorm_std_out)[0])
                  ddp_pred_norm_plot = ((ddp_pred / max_abs_ddp) - mean_out_scalar) / std_out_scalar
              else:
                  ddp_true = ddp_true_norm
                  shift_x_norm = -(a_x * u_x_norm * grad_x_norm)
                  shift_y_norm = -(a_y * u_y_norm * grad_y_norm)
                  shift_z_norm = -(a_z * u_z_norm * grad_z_norm)
                  shift_sum_norm = shift_x_norm + shift_y_norm + shift_z_norm
                  ddp_pred = shift_sum_norm + src
                  ddp_pred_norm_plot = ddp_pred
          else:
              if latent_channels < 4:
                  raise ValueError(
                      f"Vector shifter model must output shape (B,Z,Y,X,4). Got {y_pred.shape}"
                  )

              ux = y_pred[..., 0]
              uy = y_pred[..., 1]
              uz = y_pred[..., 2]
              src = y_pred[..., 3]

              if y_true_full.shape[-1] < 4:
                  raise ValueError(
                      f"Expected y_true shape (B,Z,Y,X,C>=4) for vector shifter plotting. Got {y_true_full.shape}"
                  )

              if y_true_full.shape[-1] >= 5 and consider_dp_loss:
                  has_dp_prev_in_y = True
                  dpprev_norm_from_y = y_true_full[..., 1]
                  grad_x_norm = y_true_full[..., 2]
                  grad_y_norm = y_true_full[..., 3]
                  grad_z_norm = y_true_full[..., 4]
              else:
                  has_dp_prev_in_y = False
                  dpprev_norm_from_y = None
                  grad_x_norm = y_true_full[..., 1]
                  grad_y_norm = y_true_full[..., 2]
                  grad_z_norm = y_true_full[..., 3]

              if denorm_mean_out is not None and denorm_std_out is not None and max_abs_ddp is not None:
                  ddp_true = (ddp_true_norm * denorm_std_out + denorm_mean_out) * max_abs_ddp
                  src_common = src * denorm_std_out * max_abs_ddp

                  if gradDpPrev_input_ch_idxs is not None and denorm_mean_in is not None and denorm_std_in is not None:
                      grad_idx_x, grad_idx_y, grad_idx_z = [int(i) for i in gradDpPrev_input_ch_idxs]
                      _maxs = np.loadtxt(maxs_fn) if maxs_fn else None
                      _maxs_flat = np.ravel(_maxs)

                      mean_in_flat = np.ravel(denorm_mean_in)
                      std_in_flat = np.ravel(denorm_std_in)

                      grad_x_phys = (grad_x_norm * std_in_flat[grad_idx_x] + mean_in_flat[grad_idx_x]) * _maxs_flat[grad_idx_x + 1]
                      grad_y_phys = (grad_y_norm * std_in_flat[grad_idx_y] + mean_in_flat[grad_idx_y]) * _maxs_flat[grad_idx_y + 1]
                      grad_z_phys = (grad_z_norm * std_in_flat[grad_idx_z] + mean_in_flat[grad_idx_z]) * _maxs_flat[grad_idx_z + 1]
                  else:
                      grad_x_phys = grad_x_norm
                      grad_y_phys = grad_y_norm
                      grad_z_phys = grad_z_norm

                  shift_sum_phys = -ux * grad_x_phys - uy * grad_y_phys - uz * grad_z_phys
                  ddp_pred = shift_sum_phys + src_common

                  shift_x_norm = -ux * grad_x_norm
                  shift_y_norm = -uy * grad_y_norm
                  shift_z_norm = -uz * grad_z_norm
                  shift_sum_norm = shift_x_norm + shift_y_norm + shift_z_norm
                  mean_out_scalar = float(np.ravel(denorm_mean_out)[0])
                  std_out_scalar = float(np.ravel(denorm_std_out)[0])
                  ddp_pred_norm_plot = ((ddp_pred / max_abs_ddp) - mean_out_scalar) / std_out_scalar
              else:
                  ddp_true = ddp_true_norm
                  shift_x_norm = -ux * grad_x_norm
                  shift_y_norm = -uy * grad_y_norm
                  shift_z_norm = -uz * grad_z_norm
                  shift_sum_norm = shift_x_norm + shift_y_norm + shift_z_norm
                  ddp_pred = shift_sum_norm + src
                  ddp_pred_norm_plot = ddp_pred

          # CFD-style outlet correction for final ddP/dP comparison
          ddp_pred_for_dp = ddp_pred.copy()
          outlet_offset = np.mean(ddp_pred_for_dp[:, :, :, -2:], axis=(1, 2, 3))
          ddp_pred_for_dp = ddp_pred_for_dp - outlet_offset[:, np.newaxis, np.newaxis, np.newaxis]

          nz = ddp_true.shape[1]
          z_indices = [int(i * (nz - 1) / (n_z_slices - 1)) for i in range(n_z_slices)]

          # --------------------------------------------------------
          # Resolve dP_prev if available either in y_true or x
          # --------------------------------------------------------
          has_dp_prev_in_x = (
              consider_dp_loss and
              x_full.ndim == 5 and
              x_full.shape[-1] > dp_prev_input_ch_idx
          )

          dpprev = None
          if has_dp_prev_in_y or has_dp_prev_in_x:
              if denorm_mean_in is None or denorm_std_in is None or max_abs_dpprev is None:
                  print('[plot_shifter_predictions_z_slices] Missing mean/std/max_abs for dP_prev. Skipping dP plots.')
              else:
                  mean_in_dpprev = float(np.ravel(denorm_mean_in)[dp_prev_input_ch_idx])
                  std_in_dpprev = float(np.ravel(denorm_std_in)[dp_prev_input_ch_idx])

                  if has_dp_prev_in_y:
                      dpprev_norm = dpprev_norm_from_y
                  else:
                      dpprev_norm = x_full[..., dp_prev_input_ch_idx]

                  dpprev = (dpprev_norm * std_in_dpprev + mean_in_dpprev) * max_abs_dpprev

          # --------------------------------------------------------
          # Per-sample plots
          # --------------------------------------------------------
          row_labels_3 = ['Ground Truth', 'Prediction', '|Error|']

          for b in range(ddp_true.shape[0]):

              # ---------------- ddP final plot ----------------
              _plot_field_z_slices_3rows(
                  ddp_true,
                  ddp_pred_for_dp,
                  global_idx,
                  b,
                  z_indices,
                  obst_bool,
                  row_labels_3,
                  title_prefix='ddP',
                  filename=os.path.join(ddp_dir, f'test_pred_{global_idx:04d}.png'),
              )

              # ---------------- latent fields ----------------
              if is_velocity_shifter:
                  latent_components = [
                      (a_x, 'ax'),
                      (a_y, 'ay'),
                      (a_z, 'az'),
                      (src, 's'),
                  ]
              else:
                  latent_components = [
                      (ux, 'ux'),
                      (uy, 'uy'),
                      (uz, 'uz'),
                      (src, 's'),
                  ]

              for comp_data, comp_name in latent_components:
                  _plot_field_z_slices_3rows(
                      np.zeros_like(comp_data),
                      comp_data,
                      global_idx,
                      b,
                      z_indices,
                      obst_bool,
                      ['Zero', comp_name, '|Value|'],
                      title_prefix=f'latent_{comp_name}',
                      filename=os.path.join(latent_dir, f'test_pred_{global_idx:04d}_{comp_name}.png'),
                  )

              # ---------------- shifter decomposition ----------------
              mean_abs_shift = np.mean(np.abs(shift_sum_norm[b]))
              mean_abs_s = np.mean(np.abs(src[b]))
              s_fraction = mean_abs_s / (mean_abs_shift + mean_abs_s + 1e-12)

              print(
                  f"[shifter terms] sample {global_idx}: "
                  f"mean|shift_sum|={mean_abs_shift:.3e}, "
                  f"mean|s|={mean_abs_s:.3e}, "
                  f"s_fraction={100.0 * s_fraction:.2f}%"
              )

              if is_velocity_shifter:
                  _plot_rows_signed(
                      data_list=[
                          shift_x_norm,
                          shift_y_norm,
                          shift_z_norm,
                          shift_sum_norm,
                          src,
                          ddp_pred_norm_plot,
                          ddp_true_norm,
                          np.abs(ddp_true_norm - ddp_pred_norm_plot),
                      ],
                      row_labels=[
                          'shift_x = -(ax*Ux)*grad_x',
                          'shift_y = -(ay*Uy)*grad_y',
                          'shift_z = -(az*Uz)*grad_z',
                          'shift_sum',
                          'source term s',
                          'ddP_pred_norm (reconstructed)',
                          'ddP_true_norm',
                          '|ddP_true - ddP_pred| (norm)',
                      ],
                      global_idx=global_idx,
                      b=b,
                      z_indices=z_indices,
                      obst_bool=obst_bool,
                      title_prefix='shifter_terms',
                      filename=os.path.join(shifter_dir, f'test_pred_{global_idx:04d}_shifter.png'),
                  )
              else:
                  _plot_rows_signed(
                      data_list=[
                          shift_x_norm,
                          shift_y_norm,
                          shift_z_norm,
                          shift_sum_norm,
                          src,
                          ddp_pred_norm_plot,
                          ddp_true_norm,
                          np.abs(ddp_true_norm - ddp_pred_norm_plot),
                      ],
                      row_labels=[
                          'shift_x = -ux*grad_x',
                          'shift_y = -uy*grad_y',
                          'shift_z = -uz*grad_z',
                          'shift_sum',
                          'source term s',
                          'ddP_pred_norm (reconstructed)',
                          'ddP_true_norm',
                          '|ddP_true - ddP_pred| (norm)',
                      ],
                      global_idx=global_idx,
                      b=b,
                      z_indices=z_indices,
                      obst_bool=obst_bool,
                      title_prefix='shifter_terms',
                      filename=os.path.join(shifter_dir, f'test_pred_{global_idx:04d}_shifter.png'),
                  )

              # ---------------- dP final plot ----------------
              if dpprev is not None:
                  dp_true = dpprev + ddp_true
                  dp_pred_f10 = dpprev + ddp_pred_for_dp * 0.1
                  dp_pred_f25 = dpprev + ddp_pred_for_dp * 0.25
                  dp_pred_f50 = dpprev + ddp_pred_for_dp * 0.5
                  dp_pred = dpprev + ddp_pred_for_dp

                  pred_err = np.mean(np.abs(dp_true[b] - dp_pred[b]))
                  pred_err_f10 = np.mean(np.abs(dp_true[b] - dp_pred_f10[b]))
                  pred_err_f25 = np.mean(np.abs(dp_true[b] - dp_pred_f25[b]))
                  pred_err_f50 = np.mean(np.abs(dp_true[b] - dp_pred_f50[b]))
                  base_err = np.mean(np.abs(dp_true[b] - dpprev[b]))

                  improvement = 100.0 * (1.0 - pred_err / (base_err + 1e-12))
                  improvement_f10 = 100.0 * (1.0 - pred_err_f10 / (base_err + 1e-12))
                  improvement_f25 = 100.0 * (1.0 - pred_err_f25 / (base_err + 1e-12))
                  improvement_f50 = 100.0 * (1.0 - pred_err_f50 / (base_err + 1e-12))

                  print(
                      f"[dP plot] sample {global_idx}: "
                      f"baseline MAE={base_err:.3e}, "
                      f"pred MAE={pred_err:.3e}, "
                      f"improvement={improvement:.2f}%, "
                      f"pred_f10 MAE={pred_err_f10:.3e}, "
                      f"improvement_f10={improvement_f10:.2f}%, "
                      f"pred_f25 MAE={pred_err_f25:.3e}, "
                      f"improvement_f25={improvement_f25:.2f}%, "
                      f"pred_f50 MAE={pred_err_f50:.3e}, "
                      f"improvement_f50={improvement_f50:.2f}%"
                  )

                  row_labels_4 = [
                      'Ground Truth dP',
                      'Prediction dP',
                      '|Error| pred',
                      '|Error| dP_prev baseline',
                  ]

                  _plot_field_z_slices_4rows(
                      dp_true,
                      dp_pred,
                      dpprev,
                      global_idx,
                      b,
                      z_indices,
                      obst_bool,
                      row_labels_4,
                      title_prefix='dP',
                      filename=os.path.join(dp_dir, f'test_pred_{global_idx:04d}_dP.png'),
                  )

              global_idx += 1

      print(f"=== Done: {global_idx} test samples plotted ===\n")



  def prepare_data_to_tf(
    self,
    outarray_flat_fn: str = 'features_data.h5',
    normalization_factors_fn: str = 'mean_std.npz',
    flatten_data: bool = False,
    load_existing_normalization: bool = False,
    include_dp_prev_in_y: bool = True,
    include_gradDpPrev_in_y: bool = False,
    include_velocity_components_in_y: bool = False,
    include_uDotGradDpPrev_in_y: bool = False,
    dp_prev_input_ch_idx: int = 14,
    gradDpPrev_input_ch_idxs: tuple = None,
    U_input_ch_idxs: tuple = None,
    uDotGradDpPrev_input_ch_idx: int = None,
    force_rewrite_tfrecords: bool = False,
  ):

    filename_flat = outarray_flat_fn

    print('Loading feature data (tucker cores extracted from blocks)\n')
    with tables.open_file(filename_flat, mode='r') as f:
      input = f.root.inputs[...]
      output = f.root.outputs[...]

    standardization_method = "std"
    print(f'Normalizing feature data based on standardization method: {standardization_method}')

    x, y = utils_data.normalize_feature_data(
        input,
        output,
        standardization_method,
        normalization_factors_fn=normalization_factors_fn,
        load_existing=load_existing_normalization
    )

    # ------------------------------------------------------------
    # OPTIONAL: append normalized dP_prev from input channel
    # Resulting y:
    #   y[..., 0] = ddP_true_norm
    #   y[..., 1] = dP_prev_norm
    # ------------------------------------------------------------
    if include_dp_prev_in_y:
      if x.shape[-1] <= dp_prev_input_ch_idx:
        raise ValueError(
            f"Requested dp_prev_input_ch_idx={dp_prev_input_ch_idx}, "
            f"but x has only {x.shape[-1]} channels."
        )

      dp_prev = x[..., dp_prev_input_ch_idx]   # shape: (N, Z, Y, X)

      if y.ndim == dp_prev.ndim:
        # y currently (N, Z, Y, X)
        y = np.stack([y, dp_prev], axis=-1)

      elif y.ndim == dp_prev.ndim + 1:
        # y currently (N, Z, Y, X, C)
        y = np.concatenate([y, dp_prev[..., None]], axis=-1)

      else:
        raise ValueError(
            f"Unexpected y shape {y.shape} for dp_prev shape {dp_prev.shape}"
        )

      print(
          f'[prepare_data_to_tf] Appended normalized dP_prev from '
          f'x[..., {dp_prev_input_ch_idx}] to y.'
      )
      print(f'[prepare_data_to_tf] New y shape: {y.shape}')

    if include_gradDpPrev_in_y:
            if gradDpPrev_input_ch_idxs is None or len(gradDpPrev_input_ch_idxs) != 3:
                raise ValueError(
                        'include_gradDpPrev_in_y=True requires gradDpPrev_input_ch_idxs=(ix, iy, iz).'
                )

            gx_idx, gy_idx, gz_idx = [int(i) for i in gradDpPrev_input_ch_idxs]
            max_idx = max(gx_idx, gy_idx, gz_idx)
            if x.shape[-1] <= max_idx:
                raise ValueError(
                        f'gradDpPrev_input_ch_idxs={gradDpPrev_input_ch_idxs} out of bounds for x with {x.shape[-1]} channels.'
                )

            grad_x = x[..., gx_idx]
            grad_y = x[..., gy_idx]
            grad_z = x[..., gz_idx]

            if y.ndim == grad_x.ndim:
                # y currently (N, Z, Y, X)
                y = np.stack([y, grad_x, grad_y, grad_z], axis=-1)
            elif y.ndim == grad_x.ndim + 1:
                # y currently (N, Z, Y, X, C)
                y = np.concatenate([y, grad_x[..., None], grad_y[..., None], grad_z[..., None]], axis=-1)
            else:
                raise ValueError(
                        f'Unexpected y shape {y.shape} for gradDpPrev shape {grad_x.shape}'
                )

            print(
                    f'[prepare_data_to_tf] Appended normalized gradDpPrev from '
                    f'x[..., {gx_idx}], x[..., {gy_idx}], x[..., {gz_idx}] to y.'
            )
            print(f'[prepare_data_to_tf] New y shape: {y.shape}')

    if include_velocity_components_in_y:
            if U_input_ch_idxs is None or len(U_input_ch_idxs) != 3:
                raise ValueError(
                    'include_velocity_components_in_y=True requires U_input_ch_idxs=(ix, iy, iz).'
                )
            if gradDpPrev_input_ch_idxs is None or len(gradDpPrev_input_ch_idxs) != 3:
                raise ValueError(
                    'include_velocity_components_in_y=True requires gradDpPrev_input_ch_idxs=(ix, iy, iz).'
                )

            ux_idx, uy_idx, uz_idx = [int(i) for i in U_input_ch_idxs]
            gx_idx, gy_idx, gz_idx = [int(i) for i in gradDpPrev_input_ch_idxs]
            max_idx = max(ux_idx, uy_idx, uz_idx, gx_idx, gy_idx, gz_idx)
            if x.shape[-1] <= max_idx:
                raise ValueError(
                    f'Velocity component indices out of bounds for x with {x.shape[-1]} channels.'
                )

            u_x = x[..., ux_idx]
            u_y = x[..., uy_idx]
            u_z = x[..., uz_idx]
            grad_x = x[..., gx_idx]
            grad_y = x[..., gy_idx]
            grad_z = x[..., gz_idx]

            if y.ndim == u_x.ndim:
                y = np.stack([y, u_x, u_y, u_z, grad_x, grad_y, grad_z], axis=-1)
            elif y.ndim == u_x.ndim + 1:
                y = np.concatenate([
                    y,
                    u_x[..., None], u_y[..., None], u_z[..., None],
                    grad_x[..., None], grad_y[..., None], grad_z[..., None],
                ], axis=-1)
            else:
                raise ValueError(
                    f'Unexpected y shape {y.shape} for velocity component shape {u_x.shape}'
                )

            print(
                f'[prepare_data_to_tf] Appended normalized U and gradDpPrev components from '
                f'x[..., {ux_idx}:{uz_idx + 1}] and x[..., {gx_idx}:{gz_idx + 1}] to y.'
            )
            print(f'[prepare_data_to_tf] New y shape: {y.shape}')

    if include_uDotGradDpPrev_in_y:
            if uDotGradDpPrev_input_ch_idx is None:
                raise ValueError(
                    'include_uDotGradDpPrev_in_y=True requires uDotGradDpPrev_input_ch_idx.'
                )

            if x.shape[-1] <= int(uDotGradDpPrev_input_ch_idx):
                raise ValueError(
                    f'uDotGradDpPrev_input_ch_idx={uDotGradDpPrev_input_ch_idx} out of bounds for x with {x.shape[-1]} channels.'
                )

            u_dot_grad = x[..., int(uDotGradDpPrev_input_ch_idx)]

            if y.ndim == u_dot_grad.ndim:
                y = np.stack([y, u_dot_grad], axis=-1)
            elif y.ndim == u_dot_grad.ndim + 1:
                y = np.concatenate([y, u_dot_grad[..., None]], axis=-1)
            else:
                raise ValueError(
                    f'Unexpected y shape {y.shape} for uDotGradDpPrev shape {u_dot_grad.shape}'
                )

            print(
                f'[prepare_data_to_tf] Appended normalized U.dot(gradDpPrev) from '
                f'x[..., {int(uDotGradDpPrev_input_ch_idx)}] to y.'
            )
            print(f'[prepare_data_to_tf] New y shape: {y.shape}')


    split = 0.8
    n_samples = x.shape[0]
    n_train = int(split * n_samples)

    # Shuffle indices before splitting
    rng = np.random.default_rng(seed=42)   # fixed seed for reproducibility
    perm = rng.permutation(n_samples)

    train_idx = perm[:n_train]
    test_idx  = perm[n_train:]

    print("Using shuffled split: validation uses random samples\n")

    if flatten_data:
        x = x.reshape((x.shape[0], x.shape[1], 1, 1))
        y = y.reshape((y.shape[0], y.shape[1], 1, 1))

    # ------------------------------------------------------------
    # TFRecords
    # Force rewrite when target format changes
    # ------------------------------------------------------------
    rewrite_now = force_rewrite_tfrecords or include_dp_prev_in_y or include_gradDpPrev_in_y or include_velocity_components_in_y or include_uDotGradDpPrev_in_y

    if rewrite_now or not (
        os.path.isfile(self.train_tfrecord_fn) and
        os.path.isfile(self.test_tfrecord_fn)
    ):
        print("Writing TFRecords train/test data...\n")

        utils_io.write_images_to_tfr_short(
            x[train_idx, ...],
            y[train_idx, ...],
            filename=self.train_tfrecord_fn,
        )

        utils_io.write_images_to_tfr_short(
            x[test_idx, ...],
            y[test_idx, ...],
            filename=self.test_tfrecord_fn,
        )

    else:
      print(
          f"TFRecords train and test data already available, using them.\n"
          f"If the target format changed, delete '{self.train_tfrecord_fn}' "
          f"and '{self.test_tfrecord_fn}' or set force_rewrite_tfrecords=True.\n"
      )

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
    block_size=None,
    obst_bool=None,
    predict_ddUCorr_output: bool=False,
    div_u_ch_idx: int=None,
    div_u_grid=None,
    lambda_cont: float=0.02,
    grid_res=1.0,
    consider_dp_loss: bool=True,
    dp_prev_input_ch_idx: int=14,
    dp_prev_maxs_idx: int=None,
    gradDpPrev_input_ch_idxs: tuple=None,
    U_input_ch_idxs: tuple=None,
    uDotGradDpPrev_input_ch_idx: int=None,
    use_s_roi_penalty: bool=False) -> None:

    train_path = self.train_tfrecord_fn
    test_path = self.test_tfrecord_fn

    self.obst_bool = obst_bool
    self.train_dataset = utils_io.load_dataset_tf(filename=train_path, batch_size=batch_size, buffer_size=1024)
    self.test_dataset = utils_io.load_dataset_tf(filename=test_path, batch_size=batch_size, buffer_size=1024)

    _norm_fn = os.path.join(model_h5_path, 'mean_std.npz')
    _mean_in = _std_in = _mean_out = _std_out = None
    if os.path.exists(_norm_fn):
        _nd = np.load(_norm_fn)
        if 'mean_in' in _nd:
            _mean_in = _nd['mean_in']
            _std_in = _nd['std_in']
            _mean_out = _nd['mean_out']
            _std_out = _nd['std_out']

    _div_u_mean = float(_mean_in.flat[div_u_ch_idx]) if (_mean_in is not None and div_u_ch_idx is not None) else 0.0
    _div_u_std = float(_std_in.flat[div_u_ch_idx]) if (_std_in is not None and div_u_ch_idx is not None) else 1.0
    if _mean_out is not None and predict_ddUCorr_output:
        _mean_out_vel = _mean_out.flat[1:4] if _mean_out.size >= 4 else np.zeros(3)
        _std_out_vel = _std_out.flat[1:4] if _std_out.size >= 4 else np.ones(3)
    else:
        _mean_out_vel = _std_out_vel = None

    mean_out_ddp = std_out_ddp = None
    mean_in_dpprev = std_in_dpprev = None
    max_abs_ddp = max_abs_dpprev = None
    mean_in_grads = std_in_grads = max_abs_grads = None
    mean_in_u = std_in_u = max_abs_u = None
    mean_in_u_dot_grad = std_in_u_dot_grad = max_abs_u_dot_grad = None
    requested_arch = model_architecture.lower()
    is_shifter_arch = requested_arch in [
        'cnn_shifter',
        'cnn_shifter_lightweight',
        'simplecnn3d_ddp_shifter',
        'simplecnn3d_ddp_shifter_lightweight',
        'cnn_shifter_velocity',
        'simplecnn3d_ddp_shifter_velocity',
    ]
    is_velocity_shifter_arch = requested_arch in [
        'cnn_shifter_velocity',
        'simplecnn3d_ddp_shifter_velocity',
    ]

    _maxs = None
    if consider_dp_loss and (not predict_ddUCorr_output):
        _maxs_fn = os.path.join(model_h5_path, 'maxs')
        if not os.path.exists(_maxs_fn):
            raise FileNotFoundError(f"consider_dp_loss=True requires maxs file, but not found: {_maxs_fn}")
        _maxs = np.loadtxt(_maxs_fn)
        dp_prev_maxs_idx = (dp_prev_input_ch_idx + 1) if dp_prev_maxs_idx is None else dp_prev_maxs_idx  # +1: SDF at _vel_end shifts pressure channels by 1 in maxs file
        if _mean_in is None or _std_in is None or _mean_out is None or _std_out is None:
            raise ValueError("consider_dp_loss=True requires mean/std statistics loaded from mean_std.npz")
        mean_in_dpprev = float(np.ravel(_mean_in)[dp_prev_input_ch_idx])
        std_in_dpprev = float(np.ravel(_std_in)[dp_prev_input_ch_idx])
        mean_out_ddp = float(np.ravel(_mean_out)[0])
        std_out_ddp = float(np.ravel(_std_out)[0])
        max_abs_dpprev = float(np.ravel(_maxs)[dp_prev_maxs_idx])
        max_abs_ddp = float(np.ravel(_maxs)[-1])
        print(f"[dP-main loss] dp_prev_input_ch_idx = {dp_prev_input_ch_idx}")
        print(f"[dP-main loss] dp_prev_maxs_idx     = {dp_prev_maxs_idx}")
        print(f"[dP-main loss] mean_in_dpprev       = {mean_in_dpprev}")
        print(f"[dP-main loss] std_in_dpprev        = {std_in_dpprev}")
        print(f"[dP-main loss] max_abs_dpprev       = {max_abs_dpprev}")
        print(f"[dP-main loss] mean_out_ddp         = {mean_out_ddp}")
        print(f"[dP-main loss] std_out_ddp          = {std_out_ddp}")
        print(f"[dP-main loss] max_abs_ddp          = {max_abs_ddp}")

    if is_shifter_arch:
        _maxs_fn = os.path.join(model_h5_path, 'maxs')
        if not os.path.exists(_maxs_fn):
            raise FileNotFoundError(f"[cnn_shifter] requires maxs file, but not found: {_maxs_fn}")
        if _mean_in is None or _std_in is None or _mean_out is None or _std_out is None:
            raise ValueError("[cnn_shifter] requires mean/std statistics loaded from mean_std.npz")
        if _maxs is None:
            _maxs = np.loadtxt(_maxs_fn)

        if use_feature_decomposition:
            raise ValueError("[cnn_shifter] requires use_feature_decomposition=False.")
        if is_velocity_shifter_arch:
            if U_input_ch_idxs is None or len(U_input_ch_idxs) != 3:
                raise ValueError("[cnn_shifter_velocity] requires U_input_ch_idxs=(ix, iy, iz).")
            if gradDpPrev_input_ch_idxs is None or len(gradDpPrev_input_ch_idxs) != 3:
                raise ValueError("[cnn_shifter_velocity] requires gradDpPrev_input_ch_idxs=(ix, iy, iz).")
        else:
            if gradDpPrev_input_ch_idxs is None or len(gradDpPrev_input_ch_idxs) != 3:
                raise ValueError("[cnn_shifter] requires gradDpPrev_input_ch_idxs=(ix, iy, iz).")

        _mean_in_flat = np.ravel(_mean_in)
        _std_in_flat = np.ravel(_std_in)
        _mean_out_flat = np.ravel(_mean_out)
        _std_out_flat = np.ravel(_std_out)
        _maxs_flat = np.ravel(_maxs)

        mean_out_ddp = float(_mean_out_flat[0])
        std_out_ddp = float(_std_out_flat[0])
        max_abs_ddp = float(_maxs_flat[-1])
        if is_velocity_shifter_arch:
            u_idx_x, u_idx_y, u_idx_z = [int(i) for i in U_input_ch_idxs]
            grad_idx_x, grad_idx_y, grad_idx_z = [int(i) for i in gradDpPrev_input_ch_idxs]
            mean_in_u = [
                float(_mean_in_flat[u_idx_x]),
                float(_mean_in_flat[u_idx_y]),
                float(_mean_in_flat[u_idx_z]),
            ]
            std_in_u = [
                float(_std_in_flat[u_idx_x]),
                float(_std_in_flat[u_idx_y]),
                float(_std_in_flat[u_idx_z]),
            ]
            max_abs_u = [
                float(_maxs_flat[u_idx_x]),
                float(_maxs_flat[u_idx_y]),
                float(_maxs_flat[u_idx_z]),
            ]
            mean_in_grads = [
                float(_mean_in_flat[grad_idx_x]),
                float(_mean_in_flat[grad_idx_y]),
                float(_mean_in_flat[grad_idx_z]),
            ]
            std_in_grads = [
                float(_std_in_flat[grad_idx_x]),
                float(_std_in_flat[grad_idx_y]),
                float(_std_in_flat[grad_idx_z]),
            ]
            max_abs_grads = [
                float(_maxs_flat[grad_idx_x + 1]),  # +1: SDF at _vel_end shifts pressure channels by 1 in maxs file
                float(_maxs_flat[grad_idx_y + 1]),
                float(_maxs_flat[grad_idx_z + 1]),
            ]

            print(f"[cnn_shifter_velocity loss] mean_out_ddp        = {mean_out_ddp}")
            print(f"[cnn_shifter_velocity loss] std_out_ddp         = {std_out_ddp}")
            print(f"[cnn_shifter_velocity loss] max_abs_ddp         = {max_abs_ddp}")
            print(f"[cnn_shifter_velocity loss] U idxs             = {U_input_ch_idxs}")
            print(f"[cnn_shifter_velocity loss] grad idxs          = {gradDpPrev_input_ch_idxs}")
            print(f"[cnn_shifter_velocity loss] mean_in_u          = {mean_in_u}")
            print(f"[cnn_shifter_velocity loss] std_in_u           = {std_in_u}")
            print(f"[cnn_shifter_velocity loss] max_abs_u          = {max_abs_u}")
            print(f"[cnn_shifter_velocity loss] mean_in_grads      = {mean_in_grads}")
            print(f"[cnn_shifter_velocity loss] std_in_grads       = {std_in_grads}")
            print(f"[cnn_shifter_velocity loss] max_abs_grads      = {max_abs_grads}")

            effective_model_arch = 'cnn_shifter_velocity'
        else:
            grad_idx_x, grad_idx_y, grad_idx_z = [int(i) for i in gradDpPrev_input_ch_idxs]
            mean_in_grads = [
                float(_mean_in_flat[grad_idx_x]),
                float(_mean_in_flat[grad_idx_y]),
                float(_mean_in_flat[grad_idx_z]),
            ]
            std_in_grads = [
                float(_std_in_flat[grad_idx_x]),
                float(_std_in_flat[grad_idx_y]),
                float(_std_in_flat[grad_idx_z]),
            ]
            max_abs_grads = [
                float(_maxs_flat[grad_idx_x + 1]),  # +1: SDF at _vel_end shifts pressure channels by 1 in maxs file
                float(_maxs_flat[grad_idx_y + 1]),
                float(_maxs_flat[grad_idx_z + 1]),
            ]

            print(f"[cnn_shifter loss] mean_out_ddp  = {mean_out_ddp}")
            print(f"[cnn_shifter loss] std_out_ddp   = {std_out_ddp}")
            print(f"[cnn_shifter loss] max_abs_ddp   = {max_abs_ddp}")
            print(f"[cnn_shifter loss] grad idxs     = {gradDpPrev_input_ch_idxs}")
            print(f"[cnn_shifter loss] mean_in_grads = {mean_in_grads}")
            print(f"[cnn_shifter loss] std_in_grads  = {std_in_grads}")
            print(f"[cnn_shifter loss] max_abs_grads = {max_abs_grads}")

            effective_model_arch = 'cnn_shifter_lightweight' if 'lightweight' in requested_arch else 'cnn_shifter'
        
        if isinstance(grid_res, (tuple, list)):
            _dx, _dy, _dz = float(grid_res[0]), float(grid_res[1]), float(grid_res[2])
        else:
            _dx = _dy = _dz = float(grid_res)


        self.loss_object = ShifterLoss(
            lambda_res=1.0,
            lambda_u_smooth=0.0,
            lambda_u_mag=0.0,
            lambda_s=0.0,

            lambda_s_mean=0.0,
            lambda_s_lowfreq=0.0,
            s_lowfreq_pool_size=(3, 6, 15),
            s_lowfreq_passes=1,

            use_s_roi_penalty=use_s_roi_penalty,
            lambda_s_outside_roi=100.0,      # Key lever
            
            residual_loss_mode="weighted_huber",
            beta_amp=2.0,
            delta_huber=2e-5,

            mean_out_ddp=mean_out_ddp,
            std_out_ddp=std_out_ddp,
            max_abs_ddp=max_abs_ddp,
            formulation='velocity' if is_velocity_shifter_arch else 'vector',
            mean_in_grads=mean_in_grads,
            std_in_grads=std_in_grads,
            max_abs_grads=max_abs_grads,
            mean_in_u=mean_in_u,
            std_in_u=std_in_u,
            max_abs_u=max_abs_u,
            mean_in_u_dot_grad=mean_in_u_dot_grad,
            std_in_u_dot_grad=std_in_u_dot_grad,
            max_abs_u_dot_grad=max_abs_u_dot_grad,
            dx=_dx,
            dy=_dy,
            dz=_dz,
            obst_bool=self.obst_bool,
            debug_print=True,
        )


    elif predict_ddUCorr_output:
        if use_feature_decomposition:
            raise ValueError("[predict_ddUCorr_output=True] 'cnn_multi_out_divu' requires use_feature_decomposition=False. Set 'use_feature_decomposition = False' in python_module.")
        if div_u_ch_idx is None:
            raise ValueError("[predict_ddUCorr_output=True] 'cnn_multi_out_divu' requires the divU input channel. Set 'add_divUStar_input = True' in python_module so that div_u_ch_idx is resolved.")
        effective_model_arch = 'cnn_multi_out_divu'
        self.loss_object = self.my_weighted_loss_split(
            w_p=1.0, w_u=3.5, w_cont=0.0,
            beta=0.5, cap=2.0, alpha=0.0,
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

        if consider_dp_loss:
            # self.loss_object = self.my_multihead_loss_dp_main_from_total_only(
            #     mean_out_ddp=mean_out_ddp,
            #     std_out_ddp=std_out_ddp,
            #     max_abs_ddp=max_abs_ddp,
            #     mean_in_dpprev=mean_in_dpprev,
            #     std_in_dpprev=std_in_dpprev,
            #     max_abs_dpprev=max_abs_dpprev,
            #     lambda_dp=1.0,
            #     lambda_res=0.2,
            #     lambda_smooth=0,
            #     lambda_local=0,
            #     lambda_grad=0,
            #     lambda_local_grad=0,
            #     smooth_ksize=(3, 3, 3),
            #     smooth_passes=1,
            #     scale=10000.0,
            # )


            self.loss_object = self.my_multihead_loss_dp_main_from_total_only(
                mean_out_ddp=mean_out_ddp,
                std_out_ddp=std_out_ddp,
                max_abs_ddp=max_abs_ddp,
                mean_in_dpprev=mean_in_dpprev,
                std_in_dpprev=std_in_dpprev,
                max_abs_dpprev=max_abs_dpprev,
                lambda_res=100.0,
                lambda_smooth=0.0,
                lambda_local=0.0,
                lambda_local_grad=0.0,
                lambda_dp_smooth_abs=0,   # start small
                lambda_improve=0,          # also small
                improve_margin=0.0,
                smooth_ksize=(3, 3, 5),
                smooth_passes=1,
                scale=10000.0,
                wz=1.0,
                wy=1.0,
                wx=1.0,
            )


        else:
            self.loss_object = self.my_multihead_loss_from_total_only(
                lambda_total=1.0,
                lambda_smooth=0.15,
                lambda_local=1.0,
                lambda_grad=0.03,
                lambda_local_grad=1.0,
                smooth_ksize=(3, 3, 5),
                smooth_passes=1,
                scale=100.0,
            )

    if is_shifter_arch:
        _loss_name = 'ShifterLoss'
    elif predict_ddUCorr_output:
        _loss_name = 'my_weighted_loss_split'
    else:
        _loss_name = 'my_multihead_loss_dp_main_from_total_only' if consider_dp_loss else 'my_multihead_loss_from_total_only'

    print(f"[Config] predict_ddUCorr_output={predict_ddUCorr_output} → model='{effective_model_arch}', loss='{_loss_name}', consider_dp_loss={consider_dp_loss}")

    if is_shifter_arch and getattr(self.loss_object, 'use_s_roi_penalty', False):
        self.plot_shifter_roi_debug(
            model_h5_path=model_h5_path,
            flatten_data=flatten_data,
            n_z_slices=5,
            obst_bool=obst_bool,
            mean_std_fn=os.path.join(model_h5_path, 'mean_std.npz'),
            maxs_fn=os.path.join(model_h5_path, 'maxs'),
            gradDpPrev_input_ch_idxs=gradDpPrev_input_ch_idxs,
            sample_limit=1,
        )

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)

    def prod(iterable):
        from functools import reduce
        from operator import mul
        return reduce(mul, iterable, 1)

    spatial_dims = spatial_tucker_ranks if use_feature_decomposition else block_size
    input_features_size = prod(spatial_dims) * last_tucker_rank
    output_features_size = prod(spatial_dims)
    if new_model:
            model_architecture_norm = effective_model_arch.lower()
            match model_architecture_norm:
                case 'mlp_small' | 'mlp_big' | 'mlp_small_unet' | 'mlp_huge' | 'mlp_huger':
                    self.model = MLP(n_layers, width, input_features_size, output_features_size, dropout_rate, regularization)
                case 'conv1d':
                    self.model = conv1D(n_layers, width, input_features_size, output_features_size, dropout_rate, regularization)
                case 'mlp_attention':
                    self.model = dense_attention(n_layers, width, input_features_size, output_features_size, dropout_rate, regularization)
                case 'gnn':
                    self.model = GNN(spatial_dims)
                case 'fno3d':
                    self.model = FNO3d(spatial_dims, in_channels=last_tucker_rank)
                case 'mixer':
                    self.model = MLP_Mixer_3D(n_layers, spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
                case 'cnn':
                    _n_out_ch = 4 if predict_ddUCorr_output else 1
                    self.model = SimpleCNN3D(spatial_dims, in_channels=last_tucker_rank, out_channels=_n_out_ch, dropout_rate=dropout_rate, regularization=regularization)
                case 'cnn_two_heads':
                    self.model = SimpleCNN3D_two_heads(spatial_dims, in_channels=last_tucker_rank, return_heads=True, dropout_rate=dropout_rate, regularization=regularization)
                case 'cnn_two_heads_smooth':
                    self.model = SimpleCNN3D_two_heads_smooth(spatial_dims, in_channels=last_tucker_rank, return_heads=True, dropout_rate=dropout_rate, regularization=regularization)
                case 'cnn_multi_out':
                    _n_out_ch = 4 if predict_ddUCorr_output else 1
                    self.model = SimpleCNN3D_multi_out(spatial_dims, in_channels=last_tucker_rank, out_channels=_n_out_ch, dropout_rate=dropout_rate, regularization=regularization)
                case 'cnn_multi_out_divu':
                    _n_out_ch = 4 if predict_ddUCorr_output else 1
                    self.model = SimpleCNN3D_multi_out_divU(spatial_dims, in_channels=last_tucker_rank, out_channels=_n_out_ch, dropout_rate=dropout_rate, regularization=regularization, div_u_ch_idx=div_u_ch_idx, div_u_mean=_div_u_mean, div_u_std=_div_u_std)
                case 'cnn_shifter':
                    self.model = SimpleCNN3D_ddp_shifter(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
                case 'cnn_shifter_lightweight':
                    self.model = SimpleCNN3D_ddp_shifter_lightweight(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
                case 'cnn_shifter_velocity':
                    self.model = SimpleCNN3D_ddp_shifter_velocity(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
                case 'unet3d':
                    self.model = UNet3D(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
                case 'unet3d_deep':
                    self.model = UNet3D_deep(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization, base_filters=4, n_levels=3)
                case 'unet3d_attention':
                    self.model = UNet3D_attention(spatial_dims, in_channels=last_tucker_rank, dropout_rate=dropout_rate, regularization=regularization)
                case 'multi_layer_3d':
                    self.model = Simple_multi_layer_3D(spatial_dims, in_channels=last_tucker_rank, n_layers=n_layers, width=width, dropout_rate=dropout_rate, regularization=regularization)
                case _:
                    raise ValueError('Invalid NN model type')
    else:
      model_path = f"{model_h5_path}/model_{model_name}.h5"
      print(f"Loading model: {model_path}")
      self.model = tf.keras.models.load_model(model_path, custom_objects={"SymmetricPadding3D": SymmetricPadding3D}, compile=False)

    epochs_val_losses, epochs_train_losses = [], []
    min_yet = 1e9

    for epoch in range(num_epoch):
      progbar = tf.keras.utils.Progbar(math.ceil(self.len_train / batch_size))
      print('Start of epoch %d' % (epoch,))
      losses_train = []

      for step, (inputs, labels) in enumerate(self.train_dataset):
        if flatten_data:
          inputs = inputs[..., 0, 0]
          labels = labels[..., 0, 0]
        inputs = tf.cast(inputs, dtype='float32')
        labels = tf.cast(labels, dtype='float32')
        loss = self.train_step(inputs, labels)
        losses_train.append(loss)

      losses_val = self.perform_validation(flatten_data)
      losses_train_mean = np.mean(losses_train)
      losses_val_mean = np.mean(losses_val)
      epochs_train_losses.append(losses_train_mean)
      epochs_val_losses.append(losses_val_mean)
      print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f \n' % (epoch, float(losses_train_mean), float(losses_val_mean)))
      progbar.update(step + 1)

      stopEarly = utils_model.Callback_EarlyStopping(epochs_val_losses, min_delta=0.1 / 100, patience=20)
      if stopEarly:
        print("Callback_EarlyStopping signal received at epoch= %d/%d" % (epoch, num_epoch))
        break

      if epoch > 5:
        model_fn = f'./{model_h5_path}/model_{model_name}.h5'
        if losses_val_mean < min_yet:
          print(f'saving model: {model_fn}', flush=True)
          self.model.save(model_fn)
          self.model.save_weights(weights_fn)
          min_yet = losses_val_mean

    print("Terminating training")
    plt.plot(list(range(len(epochs_train_losses))), epochs_train_losses, label='train')
    plt.plot(list(range(len(epochs_val_losses))), epochs_val_losses, label='val')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{model_h5_path}/loss_vs_epoch_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.png')

    np.savetxt(f'{model_h5_path}/train_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_train_losses, fmt='%d')
    np.savetxt(f'{model_h5_path}/test_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_val_losses, fmt='%d')

    mean_std_path = os.path.join(model_h5_path, 'mean_std.npz')
    if is_shifter_arch:
        self.plot_shifter_predictions_z_slices(
            model_h5_path,
            flatten_data,
            n_z_slices=5,
            obst_bool=obst_bool,
            mean_std_fn=mean_std_path,
            maxs_fn=os.path.join(model_h5_path, 'maxs'),
            consider_dp_loss=consider_dp_loss,
            dp_prev_input_ch_idx=dp_prev_input_ch_idx,
            dp_prev_maxs_idx=dp_prev_maxs_idx,
            gradDpPrev_input_ch_idxs=gradDpPrev_input_ch_idxs,
            U_input_ch_idxs=U_input_ch_idxs,
            uDotGradDpPrev_input_ch_idx=uDotGradDpPrev_input_ch_idx,
        )
    else:
        self.plot_test_predictions_z_slices(model_h5_path, flatten_data, n_z_slices=5, obst_bool=obst_bool, mean_std_fn=mean_std_path, predict_ddUCorr_output=predict_ddUCorr_output, div_u_ch_idx=div_u_ch_idx, maxs_fn=os.path.join(model_h5_path, 'maxs'))

    if (not predict_ddUCorr_output) and (not is_shifter_arch):
        self.plot_decomposed_predictions(model_h5_path, flatten_data)
