
"""
Masked shifter loss for latent reconstruction of ddP.

Supported y_true layouts
------------------------
1) 4 channels:
    y_true[..., 0] = ddP_true
    y_true[..., 1] = gradDpPrev_x
    y_true[..., 2] = gradDpPrev_y
    y_true[..., 3] = gradDpPrev_z

2) 5 channels:
    y_true[..., 0] = ddP_true
    y_true[..., 1] = dP_prev
    y_true[..., 2] = gradDpPrev_x
    y_true[..., 3] = gradDpPrev_y
    y_true[..., 4] = gradDpPrev_z

Predicted latent channels
-------------------------
vector formulation:
    y_pred[..., 0] = ux
    y_pred[..., 1] = uy
    y_pred[..., 2] = uz
    y_pred[..., 3] = s

Reconstruction
--------------
vector:
    ddP_pred = -ux * gradDpPrev_x - uy * gradDpPrev_y - uz * gradDpPrev_z + s

velocity:
    ddP_pred = -(ax * Ux) * gradDpPrev_x
               -(ay * Uy) * gradDpPrev_y
               -(az * Uz) * gradDpPrev_z
               + s
"""

import tensorflow as tf


def _build_domain_mask(obst_bool, dtype=tf.float32):
    """
    Convert obst_bool to mask with shape (1, Z, Y, X, 1).

    Supported obst_bool shapes:
        (Z, Y, X, 1)
        (Z, Y, X)
    """
    if obst_bool is None:
        return None

    mask = tf.convert_to_tensor(obst_bool)

    if len(mask.shape) == 4:
        mask = mask[..., 0]
    elif len(mask.shape) != 3:
        raise ValueError(
            f"obst_bool must have shape (Z,Y,X,1) or (Z,Y,X), got {mask.shape}"
        )

    mask = tf.cast(mask != 0, dtype)
    mask = mask[None, ..., None]  # (1, Z, Y, X, 1)
    return mask


def _masked_mean(x, mask=None, eps=1e-8):
    """
    Mean over valid cells only if mask is provided.

    x shape:
        (B, Z, Y, X, C) or broadcast-compatible
    mask shape:
        (1, Z, Y, X, 1)
    """
    if mask is None:
        return tf.reduce_mean(x)

    return tf.reduce_sum(x * mask) / (tf.reduce_sum(mask) + eps)


def _masked_weighted_huber(err, mask=None, weights=None, delta=2e-5, eps=1e-8):
    """
    Weighted masked Huber loss.

    err shape:
        (B, Z, Y, X, 1)
    mask shape:
        (1, Z, Y, X, 1) or None
    weights:
        broadcast-compatible with err or None
    """
    abs_err = tf.abs(err)
    quad = tf.minimum(abs_err, delta)
    lin = abs_err - quad
    huber = 0.5 * tf.square(quad) + delta * lin

    if weights is None:
        weights = 1.0

    if mask is None:
        return tf.reduce_mean(weights * huber)

    num = tf.reduce_sum(weights * huber * mask)
    den = tf.reduce_sum(mask) + eps
    return num / den


def _masked_second_derivative_smoothness_loss_3d(
    field,
    mask=None,
    dx=1.0,
    dy=1.0,
    dz=1.0,
    wz=1.0,
    wy=1.0,
    wx=1.0,
    eps=1e-8,
):
    """
    Mask-aware second-derivative smoothness penalty on a scalar field.

    field shape:
        (B, Z, Y, X, 1)

    Uses central second differences:
        d2/dz2, d2/dy2, d2/dx2
    """
    if len(field.shape) != 5 or field.shape[-1] != 1:
        raise ValueError(f"Expected field shape (B,Z,Y,X,1), got {field.shape}")

    dtype = field.dtype

    if mask is None:
        mask = tf.ones_like(field, dtype=dtype)
    else:
        mask = tf.cast(mask, dtype)

    # z second derivative
    d2z = (field[:, 2:, :, :, :] - 2.0 * field[:, 1:-1, :, :, :] + field[:, :-2, :, :, :]) / (dz ** 2)
    mz = mask[:, 2:, :, :, :] * mask[:, 1:-1, :, :, :] * mask[:, :-2, :, :, :]
    loss_z = tf.reduce_sum(tf.square(d2z) * mz) / (tf.reduce_sum(mz) + eps)

    # y second derivative
    d2y = (field[:, :, 2:, :, :] - 2.0 * field[:, :, 1:-1, :, :] + field[:, :, :-2, :, :]) / (dy ** 2)
    my = mask[:, :, 2:, :, :] * mask[:, :, 1:-1, :, :] * mask[:, :, :-2, :, :]
    loss_y = tf.reduce_sum(tf.square(d2y) * my) / (tf.reduce_sum(my) + eps)

    # x second derivative
    d2x = (field[:, :, :, 2:, :] - 2.0 * field[:, :, :, 1:-1, :] + field[:, :, :, :-2, :]) / (dx ** 2)
    mx = mask[:, :, :, 2:, :] * mask[:, :, :, 1:-1, :] * mask[:, :, :, :-2, :]
    loss_x = tf.reduce_sum(tf.square(d2x) * mx) / (tf.reduce_sum(mx) + eps)

    return wz * loss_z + wy * loss_y + wx * loss_x


def _masked_lowpass_3d(
    field,
    mask=None,
    pool_size=(1, 7, 21),
    passes=1,
    eps=1e-8,
):
    """
    Mask-aware low-pass filter for a scalar 3D field.

    field shape:
        (B, Z, Y, X, 1)

    Returns a smoothed / low-frequency version of the field.
    """
    if len(field.shape) != 5 or field.shape[-1] != 1:
        raise ValueError(f"Expected field shape (B,Z,Y,X,1), got {field.shape}")

    if mask is None:
        m = tf.ones_like(field)
    else:
        m = tf.cast(mask, field.dtype)

    num = field * m
    den = m

    ksize = [1, pool_size[0], pool_size[1], pool_size[2], 1]
    strides = [1, 1, 1, 1, 1]

    for _ in range(passes):
        num = tf.nn.avg_pool3d(num, ksize=ksize, strides=strides, padding="SAME")
        den = tf.nn.avg_pool3d(den, ksize=ksize, strides=strides, padding="SAME")

    return num / (den + eps)


def _compute_gradDpPrevMag(grad_x, grad_y, grad_z):
    """
    Compute magnitude of gradDpPrev from components.

    Each shape: (B, Z, Y, X, 1)
    """
    mag_sq = tf.square(grad_x) + tf.square(grad_y) + tf.square(grad_z)
    mag = tf.sqrt(mag_sq + 1e-16)
    return mag


def _build_roi_mask_from_gradDpPrevMag(
    grad_mag,
    fluid_mask,
    quantile=0.10,
    positive_eps=1e-12,
    dilation_radius=1,
    eps=1e-8,
):
    """
    Build a ROI (region of interest) mask for the `s` field based on gradDpPrevMag.

    Detects regions with significant local pressure gradients and creates an expanded
    mask that allows `s` to be non-zero mainly around obstacles/wakes with high gradients.

    Steps:
    1. Compute threshold as 10% of the mean of positive gradient values
    2. Create active mask where gradDpPrevMag exceeds threshold
    3. Dilate the active mask to include local neighborhoods
    4. Intersect with fluid_mask

    Args:
        grad_mag: (B, Z, Y, X, 1) gradient magnitude (already in physical space)
        fluid_mask: (1, Z, Y, X, 1) or (B, Z, Y, X, 1) fluid domain mask
        quantile: kept for backward compatibility (unused)
        positive_eps: float, gradient values below this are ignored
        dilation_radius: int, max pooling radius for dilation
        eps: float, small positive for numerical stability

    Returns:
        roi_mask: (B, Z, Y, X, 1) binary mask indicating allowed region for s
        grad_threshold: scalar, the threshold used
    """
    dtype = grad_mag.dtype
    grad_mag_np = grad_mag  # Keep as tensor
    
    # Ensure fluid_mask has batch dimension
    if len(fluid_mask.shape) == 4:
        # (1, Z, Y, X, 1) -> expand to batch
        fluid_mask_expanded = fluid_mask
    else:
        fluid_mask_expanded = fluid_mask

    # Find positive gradient values inside fluid
    fluid_mask_cast = tf.cast(fluid_mask_expanded > 0.5, dtype)
    valid_mask = tf.cast(grad_mag_np > positive_eps, dtype) * fluid_mask_cast

    # Compute threshold as 10% of mean positive gradient in fluid cells.
    num_valid_f = tf.reduce_sum(valid_mask)
    grad_sum_valid = tf.reduce_sum(grad_mag_np * valid_mask)

    grad_threshold = tf.cond(
        num_valid_f > 0.0,
        lambda: tf.maximum(
            tf.constant(0.1, dtype=dtype) * (grad_sum_valid / (num_valid_f + eps)),
            tf.constant(positive_eps, dtype=dtype),
        ),
        lambda: tf.constant(positive_eps, dtype=dtype),
    )

    # Create active mask
    active_mask = tf.cast(grad_mag_np > grad_threshold, dtype) * fluid_mask_cast

    # Dilate only in cross-stream directions (Z,Y), not streamwise X.
    # This avoids adding upstream/downstream hard-buffer; X transition is handled
    # by the soft taper mask.
    dilate_r = max(0, int(dilation_radius))
    ksize = [1, 2 * dilate_r + 1, 2 * dilate_r + 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    
    # Reshape for pooling: (B, Z, Y, X, 1)
    dilated = tf.nn.max_pool3d(active_mask, ksize=ksize, strides=strides, padding="SAME")
    
    # Final ROI: dilated mask intersected with fluid
    roi_mask = dilated * fluid_mask_cast
    
    return roi_mask, grad_threshold


def _build_upstream_soft_roi_mask(roi_mask, fluid_mask, taper_width=5, eps=1e-8):
    """
    Convert a binary ROI mask into a soft 0-1 mask with a smooth upstream edge.

    The taper is applied along the X axis (streamwise direction) and is anchored
    to the leading-obstacle plane: the mask is 1 inside the ROI from the first
    obstacles downstream, ramps linearly from 1 -> 0 over ``taper_width`` cells
    just upstream of the leading obstacle plane, and is 0 further upstream. This
    keeps the transition located near the first obstacles instead of extending to
    the far-upstream edge of the ROI.

    Args:
        roi_mask: (B, Z, Y, X, 1) binary ROI mask.
        fluid_mask: (1, Z, Y, X, 1) or (B, Z, Y, X, 1) fluid domain mask.
        taper_width: positive integer controlling how far the upstream ramp extends.
        eps: numerical stability constant.

    Returns:
        soft_roi_mask: (B, Z, Y, X, 1) float mask in [0, 1].
    """
    dtype = roi_mask.dtype
    roi_binary = tf.cast(roi_mask > 0.5, dtype)
    fluid_bin = tf.cast(fluid_mask > 0.5, dtype)

    if taper_width is None or int(taper_width) <= 0:
        return tf.clip_by_value(roi_binary * fluid_bin, 0.0, 1.0)

    tw = int(taper_width)

    shape = tf.shape(roi_binary)  # (B, Z, Y, X, 1)
    x_size = shape[3]

    # Streamwise index grid, broadcastable to (B, Z, Y, X, 1).
    x_idx = tf.reshape(tf.range(x_size), [1, 1, 1, x_size, 1])
    x_idx_f = tf.cast(x_idx, dtype)

    # Interior obstacles = solid cells that have some fluid strictly upstream of
    # them on the same streamwise line. This excludes the outer inlet/left wall
    # so the leading plane corresponds to the first real obstacles.
    solid = 1.0 - fluid_bin
    fluid_upstream = tf.cast(tf.cumsum(fluid_bin, axis=3, exclusive=True) > 0.5, dtype)
    interior_solid = solid * fluid_upstream

    # Per-sample leading-obstacle plane = smallest X index containing an interior
    # obstacle anywhere in the cross-stream plane.
    big = x_size + 1
    x_idx_b = tf.broadcast_to(x_idx, shape)
    masked_idx = tf.where(interior_solid > 0.5, x_idx_b, tf.fill(shape, big))
    lead_x = tf.reduce_min(masked_idx, axis=[1, 2, 3, 4], keepdims=True)  # (B,1,1,1,1)
    has_obs = tf.cast(lead_x < x_size, dtype)
    lead_x_f = tf.cast(lead_x, dtype)

    # Linear ramp: 1 at/downstream of the leading plane, decaying to 0 over the
    # taper_width cells immediately upstream of it.
    dist_upstream = lead_x_f - x_idx_f  # > 0 upstream of the plane
    ramp = tf.clip_by_value(1.0 - dist_upstream / (tw + 1.0), 0.0, 1.0)

    # Fallback: if no interior obstacle was found, keep the ROI as-is.
    ramp = has_obs * ramp + (1.0 - has_obs)

    soft_roi = roi_binary * ramp
    soft_roi = tf.clip_by_value(soft_roi * fluid_bin, 0.0, 1.0)
    return soft_roi


@tf.keras.utils.register_keras_serializable(package="pressure_SM_delta_delta")
class ShifterLoss(tf.keras.losses.Loss):
    """
    Masked loss for shifter outputs.

    Supported formulations:
        vector:
            y_pred = [ux, uy, uz, s]
            ddP_pred = -ux * grad_x - uy * grad_y - uz * grad_z + s

        velocity:
            y_pred = [ax, ay, az, s]
            ddP_pred = -(ax * Ux) * grad_x - (ay * Uy) * grad_y - (az * Uz) * grad_z + s

    Optional regularization:
        - smoothness of ux, uy, uz (or ax, ay, az)
        - magnitude of ux, uy, uz (or ax, ay, az)
        - magnitude of s
        - mean of s
        - low-frequency part of s

    Residual loss modes:
        - "mse"           : original masked MSE
        - "weighted_mse"  : amplitude-weighted masked MSE
        - "weighted_huber": amplitude-weighted masked Huber
    """

    def __init__(
        self,
        lambda_res=1.0,
        lambda_u_smooth=0.0,
        lambda_u_mag=1e-3,
        lambda_s=1e-3,

        # Source regularization terms
        lambda_s_mean=0.0,
        lambda_s_lowfreq=0.0,
        s_lowfreq_pool_size=(1, 7, 21),
        s_lowfreq_passes=1,

        # NEW: ROI-based regularization for s
        use_s_roi_penalty=False,
        lambda_s_outside_roi=0.0,
        lambda_s_inside_roi=0.0,
        s_roi_threshold_quantile=0.10,
        s_roi_positive_eps=1e-12,
        s_roi_dilation_radius=3,

        # residual loss mode
        residual_loss_mode="mse",   # "mse", "weighted_mse", "weighted_huber"
        beta_amp=2.0,
        delta_huber=2e-5,

        mean_out_ddp=None,
        std_out_ddp=None,
        max_abs_ddp=None,
        formulation="vector",

        mean_in_grads=None,
        std_in_grads=None,
        max_abs_grads=None,

        mean_in_u=None,
        std_in_u=None,
        max_abs_u=None,

        mean_in_u_dot_grad=None,
        std_in_u_dot_grad=None,
        max_abs_u_dot_grad=None,

        dx=1.0,
        dy=1.0,
        dz=1.0,
        obst_bool=None,
        smooth_wz=1.0,
        smooth_wy=1.0,
        smooth_wx=1.0,
        debug_print=True,
        name="shifter_loss",
    ):
        super().__init__(name=name)

        self.lambda_res = float(lambda_res)
        self.lambda_u_smooth = float(lambda_u_smooth)
        self.lambda_u_mag = float(lambda_u_mag)
        self.lambda_s = float(lambda_s)

        self.lambda_s_mean = float(lambda_s_mean)
        self.lambda_s_lowfreq = float(lambda_s_lowfreq)
        self.s_lowfreq_pool_size = tuple(int(v) for v in s_lowfreq_pool_size)
        self.s_lowfreq_passes = int(s_lowfreq_passes)

        # ROI penalty parameters
        self.use_s_roi_penalty = bool(use_s_roi_penalty)
        self.lambda_s_outside_roi = float(lambda_s_outside_roi)
        self.lambda_s_inside_roi = float(lambda_s_inside_roi)
        self.s_roi_threshold_quantile = float(s_roi_threshold_quantile)
        self.s_roi_positive_eps = float(s_roi_positive_eps)
        self.s_roi_dilation_radius = int(s_roi_dilation_radius)

        self.residual_loss_mode = str(residual_loss_mode).lower()
        self.formulation = str(formulation).lower()
        self.beta_amp = float(beta_amp)
        self.delta_huber = float(delta_huber)

        if self.residual_loss_mode not in ["mse", "weighted_mse", "weighted_huber"]:
            raise ValueError(
                f"Invalid residual_loss_mode='{residual_loss_mode}'. "
                f"Choose from ['mse', 'weighted_mse', 'weighted_huber']"
            )

        if mean_out_ddp is None or std_out_ddp is None or max_abs_ddp is None:
            raise ValueError(
                "ShifterLoss requires mean_out_ddp, std_out_ddp, and max_abs_ddp "
                "to reconstruct ddP in common physical/reference space."
            )

        if self.formulation == "vector":
            if mean_in_grads is None or std_in_grads is None or max_abs_grads is None:
                raise ValueError(
                    "ShifterLoss(vector) requires mean_in_grads, std_in_grads, and max_abs_grads "
                    "for the three gradDpPrev input channels."
                )

            if len(mean_in_grads) != 3 or len(std_in_grads) != 3 or len(max_abs_grads) != 3:
                raise ValueError(
                    "ShifterLoss(vector) expects exactly three gradient statistics for "
                    "[grad_x, grad_y, grad_z]."
                )

        elif self.formulation in ["velocity", "scalar_velocity"]:
            if mean_in_u is None or std_in_u is None or max_abs_u is None:
                raise ValueError(
                    "ShifterLoss(velocity) requires mean/std/max_abs for the three U input channels."
                )
            if mean_in_grads is None or std_in_grads is None or max_abs_grads is None:
                raise ValueError(
                    "ShifterLoss(velocity) requires mean/std/max_abs for the three gradDpPrev input channels."
                )
            if len(mean_in_u) != 3 or len(std_in_u) != 3 or len(max_abs_u) != 3:
                raise ValueError(
                    "ShifterLoss(velocity) expects exactly three U statistics for [Ux, Uy, Uz]."
                )
            if len(mean_in_grads) != 3 or len(std_in_grads) != 3 or len(max_abs_grads) != 3:
                raise ValueError(
                    "ShifterLoss(velocity) expects exactly three gradient statistics for [grad_x, grad_y, grad_z]."
                )
            self.formulation = "velocity"
        else:
            raise ValueError(
                f"Invalid ShifterLoss formulation='{self.formulation}'."
            )

        self.mean_out_ddp = float(mean_out_ddp)
        self.std_out_ddp = float(std_out_ddp)
        self.max_abs_ddp = float(max_abs_ddp)

        self.mean_in_grads = None if mean_in_grads is None else [float(v) for v in mean_in_grads]
        self.std_in_grads = None if std_in_grads is None else [float(v) for v in std_in_grads]
        self.max_abs_grads = None if max_abs_grads is None else [float(v) for v in max_abs_grads]

        self.mean_in_u = None if mean_in_u is None else [float(v) for v in mean_in_u]
        self.std_in_u = None if std_in_u is None else [float(v) for v in std_in_u]
        self.max_abs_u = None if max_abs_u is None else [float(v) for v in max_abs_u]

        self.mean_in_u_dot_grad = None if mean_in_u_dot_grad is None else float(mean_in_u_dot_grad)
        self.std_in_u_dot_grad = None if std_in_u_dot_grad is None else float(std_in_u_dot_grad)
        self.max_abs_u_dot_grad = None if max_abs_u_dot_grad is None else float(max_abs_u_dot_grad)

        self.dx = float(dx)
        self.dy = float(dy)
        self.dz = float(dz)

        self.smooth_wz = float(smooth_wz)
        self.smooth_wy = float(smooth_wy)
        self.smooth_wx = float(smooth_wx)

        self.debug_print = bool(debug_print)

        self._obst_bool_raw = obst_bool
        self._mask = _build_domain_mask(obst_bool, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lambda_res": self.lambda_res,
            "lambda_u_smooth": self.lambda_u_smooth,
            "lambda_u_mag": self.lambda_u_mag,
            "lambda_s": self.lambda_s,

            "lambda_s_mean": self.lambda_s_mean,
            "lambda_s_lowfreq": self.lambda_s_lowfreq,
            "s_lowfreq_pool_size": self.s_lowfreq_pool_size,
            "s_lowfreq_passes": self.s_lowfreq_passes,

            "use_s_roi_penalty": self.use_s_roi_penalty,
            "lambda_s_outside_roi": self.lambda_s_outside_roi,
            "lambda_s_inside_roi": self.lambda_s_inside_roi,
            "s_roi_threshold_quantile": self.s_roi_threshold_quantile,
            "s_roi_positive_eps": self.s_roi_positive_eps,
            "s_roi_dilation_radius": self.s_roi_dilation_radius,

            "residual_loss_mode": self.residual_loss_mode,
            "beta_amp": self.beta_amp,
            "delta_huber": self.delta_huber,

            "mean_out_ddp": self.mean_out_ddp,
            "std_out_ddp": self.std_out_ddp,
            "max_abs_ddp": self.max_abs_ddp,
            "formulation": self.formulation,

            "mean_in_grads": self.mean_in_grads,
            "std_in_grads": self.std_in_grads,
            "max_abs_grads": self.max_abs_grads,

            "mean_in_u": self.mean_in_u,
            "std_in_u": self.std_in_u,
            "max_abs_u": self.max_abs_u,

            "mean_in_u_dot_grad": self.mean_in_u_dot_grad,
            "std_in_u_dot_grad": self.std_in_u_dot_grad,
            "max_abs_u_dot_grad": self.max_abs_u_dot_grad,

            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "obst_bool": self._obst_bool_raw,

            "smooth_wz": self.smooth_wz,
            "smooth_wy": self.smooth_wy,
            "smooth_wx": self.smooth_wx,
            "debug_print": self.debug_print,
        })
        return config

    def _parse_targets(self, y_true):
        """Parse formulation-specific targets from y_true."""
        if len(y_true.shape) != 5:
            raise ValueError(
                f"ShifterLoss expects y_true shape (B,Z,Y,X,C), got {y_true.shape}"
            )

        n_ch = y_true.shape[-1]
        if n_ch is None:
            raise ValueError(
                "ShifterLoss requires statically known y_true channel count "
                "(4 or 5)."
            )

        if self.formulation == "vector":
            if n_ch == 4:
                ddp_true = y_true[..., 0:1]
                grad_x = y_true[..., 1:2]
                grad_y = y_true[..., 2:3]
                grad_z = y_true[..., 3:4]
            elif n_ch >= 5:
                ddp_true = y_true[..., 0:1]
                grad_x = y_true[..., 2:3]
                grad_y = y_true[..., 3:4]
                grad_z = y_true[..., 4:5]
            else:
                raise ValueError(
                    f"ShifterLoss(vector) expects y_true with 4 or 5 channels, got {n_ch}"
                )
            return ddp_true, grad_x, grad_y, grad_z

        # velocity formulation
        if n_ch == 7:
            ddp_true = y_true[..., 0:1]
            u_x = y_true[..., 1:2]
            u_y = y_true[..., 2:3]
            u_z = y_true[..., 3:4]
            grad_x = y_true[..., 4:5]
            grad_y = y_true[..., 5:6]
            grad_z = y_true[..., 6:7]
        elif n_ch >= 8:
            ddp_true = y_true[..., 0:1]
            u_x = y_true[..., 2:3]
            u_y = y_true[..., 3:4]
            u_z = y_true[..., 4:5]
            grad_x = y_true[..., 5:6]
            grad_y = y_true[..., 6:7]
            grad_z = y_true[..., 7:8]
        else:
            raise ValueError(
                f"ShifterLoss(velocity) expects y_true with 7 or 8 channels, got {n_ch}"
            )

        return ddp_true, u_x, u_y, u_z, grad_x, grad_y, grad_z

    def _to_common_space(self, value, mean, std, max_abs):
        dtype = value.dtype
        mean_t = tf.cast(mean, dtype)
        std_t = tf.cast(std, dtype)
        max_abs_t = tf.cast(max_abs, dtype)
        return (value * std_t + mean_t) * max_abs_t

    def _compute_residual_loss(self, ddp_true_common, ddp_pred_common, mask=None):
        """
        Residual loss with selectable mode:
          - mse
          - weighted_mse
          - weighted_huber
        """
        err = ddp_true_common - ddp_pred_common
        if mask is None:
            mask = self._mask
        eps = 1e-8

        if self.residual_loss_mode == "mse":
            return _masked_mean(tf.square(err), mask), None

        mean_abs_ddp = _masked_mean(tf.abs(ddp_true_common), mask) + eps
        weights = 1.0 + self.beta_amp * tf.abs(ddp_true_common) / mean_abs_ddp

        if self.residual_loss_mode == "weighted_mse":
            loss = _masked_mean(weights * tf.square(err), mask)
            return loss, weights

        if self.residual_loss_mode == "weighted_huber":
            loss = _masked_weighted_huber(
                err, mask=mask, weights=weights, delta=self.delta_huber
            )
            return loss, weights

        raise ValueError(f"Unsupported residual_loss_mode={self.residual_loss_mode}")

    def call(self, y_true, y_pred):
        """
        Physics-based reconstruction in common physical/reference space.

        y_pred expected shape:
            vector:   (B, Z, Y, X, 4) = [ux, uy, uz, s]
            velocity: (B, Z, Y, X, 4) = [ax, ay, az, s]
        """
        if len(y_pred.shape) != 5:
            raise ValueError(f"ShifterLoss expects 5D y_pred, got {y_pred.shape}")

        if self.formulation == "vector":
            if y_pred.shape[-1] < 4:
                raise ValueError(
                    f"ShifterLoss(vector) expects y_pred shape (B,Z,Y,X,4), got {y_pred.shape}"
                )

            ddp_true, grad_x, grad_y, grad_z = self._parse_targets(y_true)
            ux = y_pred[..., 0:1]
            uy = y_pred[..., 1:2]
            uz = y_pred[..., 2:3]
            src = y_pred[..., 3:4]

        else:
            if y_pred.shape[-1] < 4:
                raise ValueError(
                    f"ShifterLoss(velocity) expects y_pred shape (B,Z,Y,X,4), got {y_pred.shape}"
                )

            ddp_true, u_x, u_y, u_z, grad_x, grad_y, grad_z = self._parse_targets(y_true)
            a_x = y_pred[..., 0:1]
            a_y = y_pred[..., 1:2]
            a_z = y_pred[..., 2:3]
            src = y_pred[..., 3:4]

        # ----------------------------------------------------------
        # Denormalize all variables to common physical/reference space
        # ----------------------------------------------------------
        ddp_true_common = self._to_common_space(
            ddp_true, self.mean_out_ddp, self.std_out_ddp, self.max_abs_ddp
        )
        grad_x_common = self._to_common_space(
            grad_x, self.mean_in_grads[0], self.std_in_grads[0], self.max_abs_grads[0]
        )
        grad_y_common = self._to_common_space(
            grad_y, self.mean_in_grads[1], self.std_in_grads[1], self.max_abs_grads[1]
        )
        grad_z_common = self._to_common_space(
            grad_z, self.mean_in_grads[2], self.std_in_grads[2], self.max_abs_grads[2]
        )

        if self.formulation == "velocity":
            u_x_common = self._to_common_space(
                u_x, self.mean_in_u[0], self.std_in_u[0], self.max_abs_u[0]
            )
            u_y_common = self._to_common_space(
                u_y, self.mean_in_u[1], self.std_in_u[1], self.max_abs_u[1]
            )
            u_z_common = self._to_common_space(
                u_z, self.mean_in_u[2], self.std_in_u[2], self.max_abs_u[2]
            )

        # ----------------------------------------------------------
        # Source term scaling
        # ----------------------------------------------------------
        src_common = src * self.std_out_ddp * self.max_abs_ddp

        # ----------------------------------------------------------
        # Physics-based reconstruction
        # ----------------------------------------------------------
        if self.formulation == "vector":
            shift_term = -ux * grad_x_common - uy * grad_y_common - uz * grad_z_common
        else:
            shift_term = (
                -a_x * u_x_common * grad_x_common
                -a_y * u_y_common * grad_y_common
                -a_z * u_z_common * grad_z_common
            )

        ddp_pred_common = shift_term + src_common

        # ----------------------------------------------------------
        # Main residual loss
        # ----------------------------------------------------------
        loss_res, res_weights = self._compute_residual_loss(ddp_true_common, ddp_pred_common)

        # ----------------------------------------------------------
        # Smoothness penalty on ux,uy,uz or ax,ay,az
        # ----------------------------------------------------------
        if self.lambda_u_smooth > 0.0:
            if self.formulation == "vector":
                c1, c2, c3 = ux, uy, uz
            else:
                c1, c2, c3 = a_x, a_y, a_z

            loss_u_smooth = (
                _masked_second_derivative_smoothness_loss_3d(
                    c1, self._mask,
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    wz=self.smooth_wz, wy=self.smooth_wy, wx=self.smooth_wx
                )
                + _masked_second_derivative_smoothness_loss_3d(
                    c2, self._mask,
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    wz=self.smooth_wz, wy=self.smooth_wy, wx=self.smooth_wx
                )
                + _masked_second_derivative_smoothness_loss_3d(
                    c3, self._mask,
                    dx=self.dx, dy=self.dy, dz=self.dz,
                    wz=self.smooth_wz, wy=self.smooth_wy, wx=self.smooth_wx
                )
            )
        else:
            loss_u_smooth = tf.constant(0.0, dtype=ddp_true.dtype)

        # ----------------------------------------------------------
        # Magnitude penalty on ux,uy,uz or ax,ay,az
        # ----------------------------------------------------------
        if self.lambda_u_mag > 0.0:
            if self.formulation == "vector":
                loss_u_mag = _masked_mean(
                    tf.square(ux) + tf.square(uy) + tf.square(uz),
                    self._mask
                )
            else:
                loss_u_mag = _masked_mean(
                    tf.square(a_x) + tf.square(a_y) + tf.square(a_z),
                    self._mask
                )
        else:
            loss_u_mag = tf.constant(0.0, dtype=ddp_true.dtype)

        # ----------------------------------------------------------
        # Source regularization terms (basic)
        # ----------------------------------------------------------
        if self.lambda_s > 0.0:
            loss_s = _masked_mean(tf.square(src_common), self._mask)
        else:
            loss_s = tf.constant(0.0, dtype=ddp_true.dtype)

        if self.lambda_s_mean > 0.0:
            mean_s = _masked_mean(src_common, self._mask)
            loss_s_mean = tf.square(mean_s)
        else:
            loss_s_mean = tf.constant(0.0, dtype=ddp_true.dtype)

        if self.lambda_s_lowfreq > 0.0:
            src_lp = _masked_lowpass_3d(
                src_common,
                mask=self._mask,
                pool_size=self.s_lowfreq_pool_size,
                passes=self.s_lowfreq_passes,
            )
            loss_s_lowfreq = _masked_mean(tf.square(src_lp), self._mask)
        else:
            loss_s_lowfreq = tf.constant(0.0, dtype=ddp_true.dtype)

        # ----------------------------------------------------------
        # ROI-based spatial regularization for s
        # ----------------------------------------------------------
        loss_s_outside_roi = tf.constant(0.0, dtype=ddp_true.dtype)
        loss_s_inside_roi = tf.constant(0.0, dtype=ddp_true.dtype)
        roi_mask = None
        grad_threshold = tf.constant(self.s_roi_positive_eps, dtype=ddp_true.dtype)

        if self.use_s_roi_penalty and (self.lambda_s_outside_roi > 0.0 or self.lambda_s_inside_roi > 0.0):
            # Compute gradDpPrevMag in physical space
            grad_mag_phys = _compute_gradDpPrevMag(grad_x_common, grad_y_common, grad_z_common)

            # Build ROI mask detached from backprop.
            # tf.stop_gradient is an op, not a context manager.
            grad_mag_phys_detached = tf.stop_gradient(grad_mag_phys)
            # Keep hard ROI compact (small geometric buffer), and let soft taper
            # handle the long upstream transition.
            hard_dilation_radius = max(1, int(self.s_roi_dilation_radius // 2))

            roi_mask, grad_threshold = _build_roi_mask_from_gradDpPrevMag(
                grad_mag_phys_detached,
                self._mask,
                positive_eps=self.s_roi_positive_eps,
                dilation_radius=hard_dilation_radius,
            )
            roi_mask = tf.stop_gradient(roi_mask)
            grad_threshold = tf.stop_gradient(grad_threshold)

            # Smooth upstream edge of the ROI with a 0-1 mask.
            roi_soft_mask = _build_upstream_soft_roi_mask(
                roi_mask,
                self._mask,
                taper_width=max(1, 4 * int(self.s_roi_dilation_radius)),
            )
            roi_soft_mask = tf.stop_gradient(roi_soft_mask)

            # Residual should only contribute inside the ROI.
            residual_mask = self._mask * roi_soft_mask

            # Forbidden mask = outside ROI
            s_forbidden_mask = self._mask * (1.0 - roi_soft_mask)

            if self.lambda_s_outside_roi > 0.0:
                loss_s_outside_roi = _masked_mean(tf.square(src_common), s_forbidden_mask)

            if self.lambda_s_inside_roi > 0.0:
                s_allowed_mask = self._mask * roi_soft_mask
                loss_s_inside_roi = _masked_mean(tf.square(src_common), s_allowed_mask)

            loss_res, res_weights = self._compute_residual_loss(
                ddp_true_common,
                ddp_pred_common,
                mask=residual_mask,
            )
        else:
            loss_res, res_weights = self._compute_residual_loss(ddp_true_common, ddp_pred_common)

        # ----------------------------------------------------------
        # Debug / diagnostics
        # ----------------------------------------------------------
        if self.debug_print:
            mean_abs_shift = _masked_mean(tf.abs(shift_term), self._mask)
            mean_abs_s = _masked_mean(tf.abs(src_common), self._mask)
            mean_abs_ddp_true = _masked_mean(tf.abs(ddp_true_common), self._mask)
            mean_abs_ddp_pred = _masked_mean(tf.abs(ddp_pred_common), self._mask)

            rmse_res = tf.sqrt(
                _masked_mean(tf.square(ddp_true_common - ddp_pred_common), self._mask) + 1e-16
            )
            rel_rmse = rmse_res / (mean_abs_ddp_true + 1e-12)
            s_fraction = mean_abs_s / (mean_abs_shift + mean_abs_s + 1e-12)

            if res_weights is not None:
                mean_w = _masked_mean(res_weights, self._mask)
                max_w = tf.reduce_max(res_weights)
            else:
                mean_w = tf.constant(1.0, dtype=ddp_true.dtype)
                max_w = tf.constant(1.0, dtype=ddp_true.dtype)

            total_raw = (
                self.lambda_res * loss_res
                + self.lambda_u_smooth * loss_u_smooth
                + self.lambda_u_mag * loss_u_mag
                + self.lambda_s * loss_s
                + self.lambda_s_mean * loss_s_mean
                + self.lambda_s_lowfreq * loss_s_lowfreq
                + self.lambda_s_outside_roi * loss_s_outside_roi
                + self.lambda_s_inside_roi * loss_s_inside_roi
            )
            total_scaled = 1e6 * total_raw

            tf.print(
                "[ShifterLoss raw physical/reference-space]",
                "mode:", self.residual_loss_mode,
                "formulation:", self.formulation,
                "res:", loss_res,
                "u_smooth:", loss_u_smooth,
                "u_mag:", loss_u_mag,
                "s:", loss_s,
                "s_mean:", loss_s_mean,
                "s_lowfreq:", loss_s_lowfreq,
                "s_outside_roi:", loss_s_outside_roi,
                "s_inside_roi:", loss_s_inside_roi
            )

            tf.print(
                "[ShifterLoss weighted]",
                "res:", self.lambda_res * loss_res,
                "u_smooth:", self.lambda_u_smooth * loss_u_smooth,
                "u_mag:", self.lambda_u_mag * loss_u_mag,
                "s:", self.lambda_s * loss_s,
                "s_mean:", self.lambda_s_mean * loss_s_mean,
                "s_lowfreq:", self.lambda_s_lowfreq * loss_s_lowfreq,
                "s_outside_roi:", self.lambda_s_outside_roi * loss_s_outside_roi,
                "s_inside_roi:", self.lambda_s_inside_roi * loss_s_inside_roi
            )

            # ROI statistics
            if self.use_s_roi_penalty and roi_mask is not None:
                roi_fraction = _masked_mean(roi_soft_mask, self._mask)
                forbidden_fraction = 1.0 - roi_fraction
                
                s_allowed_mask = self._mask * roi_soft_mask
                s_forbidden_mask = self._mask * (1.0 - roi_soft_mask)
                
                mean_abs_s_inside = _masked_mean(tf.abs(src_common), s_allowed_mask)
                mean_abs_s_outside = _masked_mean(tf.abs(src_common), s_forbidden_mask)
                rms_s_inside = tf.sqrt(_masked_mean(tf.square(src_common), s_allowed_mask) + 1e-16)
                rms_s_outside = tf.sqrt(_masked_mean(tf.square(src_common), s_forbidden_mask) + 1e-16)

                tf.print(
                    "[ShifterLoss ROI stats]",
                    "roi_fraction:", roi_fraction,
                    "forbidden_fraction:", forbidden_fraction,
                    "grad_threshold:", grad_threshold,
                    "mean|s|_inside_roi:", mean_abs_s_inside,
                    "mean|s|_outside_roi:", mean_abs_s_outside,
                    "rms_s_inside_roi:", rms_s_inside,
                    "rms_s_outside_roi:", rms_s_outside
                )

            if self.formulation == "vector":
                tf.print(
                    "[ShifterLoss stats]",
                    "mean|ux|:", _masked_mean(tf.abs(ux), self._mask),
                    "mean|uy|:", _masked_mean(tf.abs(uy), self._mask),
                    "mean|uz|:", _masked_mean(tf.abs(uz), self._mask),
                    "mean|ddP_true_phys|:", mean_abs_ddp_true,
                    "mean|ddP_pred_phys|:", mean_abs_ddp_pred,
                    "mean|shift_phys|:", mean_abs_shift,
                    "mean|s_phys|:", mean_abs_s,
                    "rmse:", rmse_res,
                    "rel_rmse:", rel_rmse,
                    "mean_w:", mean_w,
                    "max_w:", max_w,
                    "s_fraction:", s_fraction,
                    "huber_delta:", self.delta_huber,
                    "total_raw:", total_raw,
                    "total_scaled(1e6x):", total_scaled
                )
            else:
                tf.print(
                    "[ShifterLoss stats]",
                    "mean|ax|:", _masked_mean(tf.abs(a_x), self._mask),
                    "mean|ay|:", _masked_mean(tf.abs(a_y), self._mask),
                    "mean|az|:", _masked_mean(tf.abs(a_z), self._mask),
                    "mean|Ux_phys|:", _masked_mean(tf.abs(u_x_common), self._mask),
                    "mean|Uy_phys|:", _masked_mean(tf.abs(u_y_common), self._mask),
                    "mean|Uz_phys|:", _masked_mean(tf.abs(u_z_common), self._mask),
                    "mean|ddP_true_phys|:", mean_abs_ddp_true,
                    "mean|ddP_pred_phys|:", mean_abs_ddp_pred,
                    "mean|shift_phys|:", mean_abs_shift,
                    "mean|s_phys|:", mean_abs_s,
                    "rmse:", rmse_res,
                    "rel_rmse:", rel_rmse,
                    "mean_w:", mean_w,
                    "max_w:", max_w,
                    "s_fraction:", s_fraction,
                    "huber_delta:", self.delta_huber,
                    "total_raw:", total_raw,
                    "total_scaled(1e6x):", total_scaled
                )

        return 1e6 * (
            self.lambda_res * loss_res
            + self.lambda_u_smooth * loss_u_smooth
            + self.lambda_u_mag * loss_u_mag
            + self.lambda_s * loss_s
            + self.lambda_s_mean * loss_s_mean
            + self.lambda_s_lowfreq * loss_s_lowfreq
            + self.lambda_s_outside_roi * loss_s_outside_roi
            + self.lambda_s_inside_roi * loss_s_inside_roi
        )
