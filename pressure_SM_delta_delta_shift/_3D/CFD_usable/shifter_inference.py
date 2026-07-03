"""
Runtime inference for shifter surrogate model.

Key innovation: The model outputs [ux, uy, uz, s], which are reconstructed to ddP:
  ddP_pred = -ux * ∂x(dP_prev) - uy * ∂y(dP_prev) - uz * ∂z(dP_prev) + s

Then ddP_pred is assembled and returned to OpenFOAM exactly as before.
No solver change required — the C++ interface still receives only ddP_pred.
"""

import os
import numpy as np
import pickle as pk
import tensorflow as tf
from scipy.ndimage import central_diff_estimator

# Assume you can import from the parent pressure_SM_delta_delta package utilities
# (or replicate the key functions here)


def central_diff_x_np(field, dx=1.0):
    """Compute ∂x field using central differences (numpy)."""
    # field shape: (d, h, w) or (d, h, w, c)
    if field.ndim == 3:
        field = field[..., np.newaxis]
    padded = np.pad(field, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='reflect')
    grad_x = (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * dx)
    return grad_x


def central_diff_y_np(field, dy=1.0):
    """Compute ∂y field using central differences (numpy)."""
    if field.ndim == 3:
        field = field[..., np.newaxis]
    padded = np.pad(field, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='reflect')
    grad_y = (padded[:, 2:, :, :] - padded[:, :-2, :, :]) / (2.0 * dy)
    return grad_y


def central_diff_z_np(field, dz=1.0):
    """Compute ∂z field using central differences (numpy)."""
    if field.ndim == 3:
        field = field[..., np.newaxis]
    padded = np.pad(field, ((1, 1), (0, 0), (0, 0), (0, 0)), mode='reflect')
    grad_z = (padded[2:, :, :, :] - padded[:-2, :, :, :]) / (2.0 * dz)
    return grad_z


def reconstruct_ddp_from_shifter(ux, uy, uz, s, dpPrev_grid, dx=1.0, dy=1.0, dz=1.0):
    """
    Reconstruct ddP from shifter outputs [ux, uy, uz, s] and dpPrev field.
    
    Computes:
      ddP_pred = -ux * ∂x(dP_prev) - uy * ∂y(dP_prev) - uz * ∂z(dP_prev) + s
    
    Args:
        ux, uy, uz, s: (d, h, w) predicted shift/source fields
        dpPrev_grid: (d, h, w) previous pressure increment field (on model grid)
        dx, dy, dz: grid spacings
    
    Returns:
        ddP_pred: (d, h, w) reconstructed pressure increment
    """
    
    # Compute gradients of dpPrev
    grad_dpPrev_x = central_diff_x_np(dpPrev_grid, dx)  # (d, h, w-2)
    grad_dpPrev_y = central_diff_y_np(dpPrev_grid, dy)  # (d, h-2, w)
    grad_dpPrev_z = central_diff_z_np(dpPrev_grid, dz)  # (d-2, h, w)
    
    # Crop all to common size (center region)
    min_d = min(grad_dpPrev_z.shape[0], ux.shape[0])
    min_h = min(grad_dpPrev_y.shape[1], uy.shape[1])
    min_w = min(grad_dpPrev_x.shape[2], uz.shape[2])
    
    d_margin = (ux.shape[0] - min_d) // 2
    h_margin = (uy.shape[1] - min_h) // 2
    w_margin = (uz.shape[2] - min_w) // 2
    
    ux_c = ux[d_margin:d_margin+min_d, h_margin:h_margin+min_h, w_margin:w_margin+min_w]
    uy_c = uy[d_margin:d_margin+min_d, h_margin:h_margin+min_h, w_margin:w_margin+min_w]
    uz_c = uz[d_margin:d_margin+min_d, h_margin:h_margin+min_h, w_margin:w_margin+min_w]
    s_c  = s[d_margin:d_margin+min_d, h_margin:h_margin+min_h, w_margin:w_margin+min_w]
    
    gx_c = grad_dpPrev_x[:min_d, :min_h, :min_w]
    gy_c = grad_dpPrev_y[:min_d, :min_h, :min_w]
    gz_c = grad_dpPrev_z[:min_d, :min_h, :min_w]
    
    ddp_pred = - ux_c * gx_c - uy_c * gy_c - uz_c * gz_c + s_c
    
    return ddp_pred


class ShifterInferenceEngine:
    """
    Handles runtime inference with shifter reconstruction.
    
    Wraps model loading, input assembly, inference, and reconstruction.
    Intended to be instantiated once at simulation start (e.g., in OpenFOAM init_func).
    """
    
    def __init__(
        self,
        model_path,
        maxs_path,
        mean_std_path,
        grid_res=1e-3,
        dx=1.0, dy=1.0, dz=1.0,
        verbose=True,
    ):
        """
        Args:
            model_path: Path to trained Keras model
            maxs_path: Path to maxs file (channel normalization)
            mean_std_path: Path to mean_std.npz (data standardization)
            grid_res: Grid resolution (for interpolation setup)
            dx, dy, dz: Grid spacings (for gradient computation)
            verbose: Print diagnostic info
        """
        
        self.model_path = model_path
        self.maxs_path = maxs_path
        self.mean_std_path = mean_std_path
        self.grid_res = grid_res
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.verbose = verbose
        
        # Load model, maxs, and normalization stats
        self.model = tf.keras.models.load_model(model_path)
        if verbose:
            print(f"[ShifterInferenceEngine] Loaded model from {model_path}")
            print(f"[ShifterInferenceEngine] Model outputs shape: {self.model.output_shape}")
        
        # Load maxs for denormalization
        if maxs_path.endswith('.npy'):
            self.maxs = np.load(maxs_path)
        else:
            self.maxs = np.loadtxt(maxs_path)
        
        # Load standardization stats
        std_data = np.load(mean_std_path)
        self.mean_in = std_data['mean_in']
        self.std_in = std_data['std_in']
        self.mean_out = std_data['mean_out']
        self.std_out = std_data['std_out']
        
        self.max_ux = self.maxs[0] if len(self.maxs) > 0 else 1.0
        self.max_uy = self.maxs[1] if len(self.maxs) > 1 else 1.0
        self.max_uz = self.maxs[2] if len(self.maxs) > 2 else 1.0
        self.max_s  = self.maxs[3] if len(self.maxs) > 3 else 1.0
        self.max_ddp = self.maxs[4] if len(self.maxs) > 4 else 1.0
        
        if verbose:
            print(f"[ShifterInferenceEngine] Loaded maxs: ux={self.max_ux:.4f}, uy={self.max_uy:.4f}, uz={self.max_uz:.4f}, s={self.max_s:.4f}, ddp={self.max_ddp:.4f}")
    
    def predict_and_reconstruct(
        self,
        grid_block,
        dpPrev_block,
    ):
        """
        Run inference on grid block and reconstruct ddP using shifter formula.
        
        Args:
            grid_block: (batch, d, h, w, n_channels) input features (normalized)
            dpPrev_block: (batch, d, h, w, 1) previous pressure increment on model grid (raw, not normalized)
        
        Returns:
            ddp_pred: (batch, d, h, w, 1) reconstructed pressure increment (raw, denormalized)
        """
        
        # Run model inference
        output = self.model(grid_block, training=False)  # (batch, d, h, w, 4) [ux, uy, uz, s]
        
        # Denormalize outputs
        output_denorm = (output * self.std_out) + self.mean_out
        
        ux_raw = output_denorm[..., 0]  # (batch, d, h, w)
        uy_raw = output_denorm[..., 1]  # (batch, d, h, w)
        uz_raw = output_denorm[..., 2]  # (batch, d, h, w)
        s_raw  = output_denorm[..., 3]  # (batch, d, h, w)
        
        # Scale to physical units based on maxs
        ux = ux_raw * self.max_ux
        uy = uy_raw * self.max_uy
        uz = uz_raw * self.max_uz
        s  = s_raw * self.max_s
        
        # Denormalize dpPrev if it was normalized
        dpPrev_denorm = dpPrev_block[..., 0]  # (batch, d, h, w)
        
        # Reconstruct ddP for each batch sample
        batch_size = ux.shape[0]
        ddp_batch = []
        
        for b in range(batch_size):
            ddp_b = reconstruct_ddp_from_shifter(
                ux[b].numpy() if hasattr(ux[b], 'numpy') else ux[b],
                uy[b].numpy() if hasattr(uy[b], 'numpy') else uy[b],
                uz[b].numpy() if hasattr(uz[b], 'numpy') else uz[b],
                s[b].numpy() if hasattr(s[b], 'numpy') else s[b],
                dpPrev_denorm[b].numpy() if hasattr(dpPrev_denorm[b], 'numpy') else dpPrev_denorm[b],
                self.dx, self.dy, self.dz,
            )
            ddp_batch.append(ddp_b)
        
        ddp_pred = np.array(ddp_batch)  # (batch, d, h, w)
        
        # Scale to physical units
        ddp_physical = ddp_pred * self.max_ddp
        
        return ddp_physical[..., np.newaxis]  # (batch, d, h, w, 1)
