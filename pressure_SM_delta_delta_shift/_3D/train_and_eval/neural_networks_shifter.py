"""
Shifter-specific CNN architectures.

Supported formulations:
    - vector shifter: outputs [ux, uy, uz, s]
            ddP_pred = -ux * ∂x(dP_prev) - uy * ∂y(dP_prev) - uz * ∂z(dP_prev) + s
    - scalar-velocity shifter: outputs [a, s]
            ddP_pred = -a * (U · ∇dP_prev) + s
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, Activation, MaxPooling3D,
    UpSampling3D, Concatenate, Dropout, Add, Lambda
)

from tensorflow.keras.initializers import RandomNormal, Zeros

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L2


import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Layer,
    Conv3D,
    BatchNormalization,
    LayerNormalization,
    Activation,
    Dropout,
    MaxPooling3D,
    UpSampling3D,
    Concatenate,
)
from tensorflow.keras.regularizers import L2


@tf.keras.utils.register_keras_serializable(package="pressure_SM_delta_delta")
class MatchSpatialDims(Layer):
    """
    Pad the upsampled tensor so its spatial dimensions match the target tensor.
    Safe for serialization (unlike Lambda closures).
    """

    def call(self, inputs):
        upsampled, target = inputs
        target_shape = tf.shape(target)
        up_shape = tf.shape(upsampled)

        pads = [
            [0, 0],  # batch
            [0, target_shape[1] - up_shape[1]],  # z
            [0, target_shape[2] - up_shape[2]],  # y
            [0, target_shape[3] - up_shape[3]],  # x
            [0, 0],  # channels
        ]
        return tf.pad(upsampled, pads)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="pressure_SM_delta_delta")
class ScaledTanh(Layer):
    """
    Output = scale * tanh(x)
    Good for bounded shift coefficients ux, uy, uz.
    Safe for serialization.
    """

    def __init__(self, scale=0.3, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        return self.scale * tf.tanh(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


def SimpleCNN3D_ddp_shifter(
    input_shape=(32, 32, 32),
    in_channels=7,
    dropout_rate=0.1,
    regularization=None,
    u_scale=0.3,
    norm_type="layer",   # "batch" or "layer"
):
    """
    3D CNN that outputs 4 channels [ux, uy, uz, s] for shifter formulation.

    Improvements vs Lambda version:
    - serializable custom layers
    - zero-initialized output heads so ddP_pred starts near 0
    - explicit bounded scaling for ux, uy, uz
    - optional LayerNorm if batch size is small

    Output shape:
        (batch, z, y, x, 4)
        channels = [ux, uy, uz, s]
    """

    l2_reg = L2(regularization) if regularization else None

    def Norm():
        if norm_type.lower() == "layer":
            return LayerNormalization(axis=-1)
        return BatchNormalization()

    def conv_block(x, filters, name_prefix):
        x = Conv3D(
            filters, 3, padding="same",
            kernel_regularizer=l2_reg,
            name=f"{name_prefix}_conv1"
        )(x)
        x = Norm()(x)
        x = Activation("relu", name=f"{name_prefix}_relu1")(x)

        x = Conv3D(
            filters, 3, padding="same",
            kernel_regularizer=l2_reg,
            name=f"{name_prefix}_conv2"
        )(x)
        x = Norm()(x)
        x = Activation("relu", name=f"{name_prefix}_relu2")(x)
        return x

    # ===== Input =====
    inputs = Input(shape=(*input_shape, in_channels), name="inputs")

    # ===== Encoder =====
    res1 = conv_block(inputs, 32, "enc1")
    x = MaxPooling3D(2, name="pool1")(res1)

    res2 = conv_block(x, 64, "enc2")
    x = MaxPooling3D(2, name="pool2")(res2)

    x = conv_block(x, 128, "bottleneck")
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name="bottleneck_dropout")(x)

    # ===== Decoder =====
    x = UpSampling3D(2, name="up1")(x)
    x = MatchSpatialDims(name="match1")([x, res2])
    x = Concatenate(name="concat1")([x, res2])
    x = conv_block(x, 64, "dec1")

    x = UpSampling3D(2, name="up2")(x)
    x = MatchSpatialDims(name="match2")([x, res1])
    x = Concatenate(name="concat2")([x, res1])
    x = conv_block(x, 32, "dec2")


    #Output heads =====
    # Idea:
    # - ux, uy, uz: small random init so shift branch is "alive" from the start
    # - s: zero init so the network does not immediately abuse the source shortcut

    head_kwargs_shift = dict(
        padding="same",
        kernel_regularizer=l2_reg,
        kernel_initializer=RandomNormal(stddev=1e-3),
        bias_initializer=Zeros(),
    )

    head_kwargs_source = dict(
        padding="same",
        kernel_regularizer=l2_reg,
        kernel_initializer=Zeros(),
        bias_initializer=Zeros(),
    )

    ux_raw = Conv3D(1, 1, name="ux_raw", **head_kwargs_shift)(x)
    uy_raw = Conv3D(1, 1, name="uy_raw", **head_kwargs_shift)(x)
    uz_raw = Conv3D(1, 1, name="uz_raw", **head_kwargs_shift)(x)
    s_raw  = Conv3D(1, 1, name="s_raw",  **head_kwargs_source)(x)

    ux = ScaledTanh(scale=u_scale, name="ux")(ux_raw)
    uy = ScaledTanh(scale=u_scale, name="uy")(uy_raw)
    uz = ScaledTanh(scale=u_scale, name="uz")(uz_raw)

    # Keep s linear; regularize s in the loss
    s = Activation("linear", name="s")(s_raw)


    outputs = Concatenate(axis=-1, name="shift_and_source")([ux, uy, uz, s])

    model = Model(inputs=inputs, outputs=outputs, name="SimpleCNN3D_ddp_shifter")
    model.summary()

    return model


def SimpleCNN3D_ddp_shifter_velocity(
    input_shape=(32, 32, 32),
    in_channels=7,
    dropout_rate=0.1,
    regularization=None,
    a_scale=0.3,
    norm_type="layer",
):
    """
    3D CNN that outputs 4 channels [ax, ay, az, s] for the velocity shifter formulation.

    Reconstruction:
        ddP_pred = -(ax * Ux) * grad_x - (ay * Uy) * grad_y - (az * Uz) * grad_z + s
    """

    l2_reg = L2(regularization) if regularization else None

    def Norm():
        if norm_type.lower() == "layer":
            return LayerNormalization(axis=-1)
        return BatchNormalization()

    def conv_block(x, filters, name_prefix):
        x = Conv3D(
            filters, 3, padding="same",
            kernel_regularizer=l2_reg,
            name=f"{name_prefix}_conv1"
        )(x)
        x = Norm()(x)
        x = Activation("relu", name=f"{name_prefix}_relu1")(x)

        x = Conv3D(
            filters, 3, padding="same",
            kernel_regularizer=l2_reg,
            name=f"{name_prefix}_conv2"
        )(x)
        x = Norm()(x)
        x = Activation("relu", name=f"{name_prefix}_relu2")(x)
        return x

    inputs = Input(shape=(*input_shape, in_channels), name="inputs")

    res1 = conv_block(inputs, 32, "enc1")
    x = MaxPooling3D(2, name="pool1")(res1)

    res2 = conv_block(x, 64, "enc2")
    x = MaxPooling3D(2, name="pool2")(res2)

    x = conv_block(x, 128, "bottleneck")
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name="bottleneck_dropout")(x)

    x = UpSampling3D(2, name="up1")(x)
    x = MatchSpatialDims(name="match1")([x, res2])
    x = Concatenate(name="concat1")([x, res2])
    x = conv_block(x, 64, "dec1")

    x = UpSampling3D(2, name="up2")(x)
    x = MatchSpatialDims(name="match2")([x, res1])
    x = Concatenate(name="concat2")([x, res1])
    x = conv_block(x, 32, "dec2")

    head_kwargs_shift = dict(
        padding="same",
        kernel_regularizer=l2_reg,
        kernel_initializer=RandomNormal(stddev=1e-3),
        bias_initializer=Zeros(),
    )

    head_kwargs_source = dict(
        padding="same",
        kernel_regularizer=l2_reg,
        kernel_initializer=Zeros(),
        bias_initializer=Zeros(),
    )

    a_raw = Conv3D(3, 1, name="a_raw", **head_kwargs_shift)(x)
    s_raw = Conv3D(1, 1, name="s_raw", **head_kwargs_source)(x)

    a = ScaledTanh(scale=a_scale, name="a")(a_raw)
    s = Activation("linear", name="s")(s_raw)

    outputs = Concatenate(axis=-1, name="velocity_shift_and_source")([a, s])

    model = Model(inputs=inputs, outputs=outputs, name="SimpleCNN3D_ddp_shifter_velocity")
    model.summary()
    return model




def SimpleCNN3D_ddp_shifter_original(
    input_shape=(32, 32, 32),
    in_channels=7,
    dropout_rate=0.1,
    regularization=None,
):
    """
    3D CNN that outputs 4 channels [ux, uy, uz, s] for shifter formulation.
    
    Same proven residual trunk as SimpleCNN3D_multi_out, but specifically
    designed to produce shift coefficients + amplitude correction.
    
    Args:
        input_shape: spatial shape (d, h, w)
        in_channels: number of input feature channels
        dropout_rate: dropout probability
        regularization: L2 regularization strength (or None)
    
    Returns:
        TensorFlow/Keras Model that outputs (batch, d, h, w, 4)
    """
    
    l2_reg = L2(regularization) if regularization else None
    
    # ===== Input =====
    inputs = Input(shape=(*input_shape, in_channels))
    
    # ===== Encoder (downsampling) =====
    # Conv block 1
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    res1 = Activation('relu')(x)
    x = MaxPooling3D(2)(res1)
    
    # Conv block 2
    x = Conv3D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    res2 = Activation('relu')(x)
    x = MaxPooling3D(2)(res2)
    
    # Conv block 3 (bottleneck)
    x = Conv3D(128, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Conv3D(128, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # ===== Decoder (upsampling) =====
    # Helper: pad upsampled output to match residual shape (gradient-safe, handles odd dimensions)
    def match_spatial_dims(args):
        upsampled, target = args
        target_shape = tf.shape(target)
        up_shape = tf.shape(upsampled)
        # Pad upsampled to match target's spatial dims (always 0 or 1 padding per dim)
        pads = [
            [0, 0],  # batch dimension
            [0, target_shape[1] - up_shape[1]],  # depth
            [0, target_shape[2] - up_shape[2]],  # height
            [0, target_shape[3] - up_shape[3]],  # width
            [0, 0],  # channels
        ]
        return tf.pad(upsampled, pads)
    
    # Upsample block 1
    x = UpSampling3D(2)(x)
    x = Lambda(match_spatial_dims)([x, res2])
    x = Concatenate()([x, res2])
    x = Conv3D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Upsample block 2
    x = UpSampling3D(2)(x)
    x = Lambda(match_spatial_dims)([x, res1])
    x = Concatenate()([x, res1])
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # ===== Output head: 4 channels for [ux, uy, uz, s] =====
    # Use tanh for shift coefficients (should be small, roughly [-0.5, 0.5])
    # and linear for source term s (can be more flexible)
    ux = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='tanh', name='ux')(x)
    uy = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='tanh', name='uy')(x)
    uz = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='tanh', name='uz')(x)
    s  = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='linear', name='s')(x)
    
    # Concatenate all 4 channels: (batch, d, h, w, 1) x4 -> (batch, d, h, w, 4)
    outputs = Concatenate(axis=-1)([ux, uy, uz, s])
    
    model = Model(inputs=inputs, outputs=outputs, name='SimpleCNN3D_ddp_shifter')
    return model


def SimpleCNN3D_ddp_shifter_lightweight(
    input_shape=(32, 32, 32),
    in_channels=7,
    dropout_rate=0.1,
    regularization=None,
):
    """
    Lightweight shifter model (fewer filters) for quick iteration or smaller GPUs.
    
    Same architecture as SimpleCNN3D_ddp_shifter but with 50% fewer filters.
    """
    
    l2_reg = L2(regularization) if regularization else None
    
    inputs = Input(shape=(*input_shape, in_channels))
    
    # Encoder
    x = Conv3D(16, 3, padding='same', kernel_regularizer=l2_reg)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    res1 = Activation('relu')(x)
    x = MaxPooling3D(2)(res1)
    
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    res2 = Activation('relu')(x)
    x = MaxPooling3D(2)(res2)
    
    x = Conv3D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Conv3D(64, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Decoder
    # Helper: pad upsampled output to match residual shape (gradient-safe, handles odd dimensions)
    def match_spatial_dims(args):
        upsampled, target = args
        target_shape = tf.shape(target)
        up_shape = tf.shape(upsampled)
        # Pad upsampled to match target's spatial dims (always 0 or 1 padding per dim)
        pads = [
            [0, 0],  # batch dimension
            [0, target_shape[1] - up_shape[1]],  # depth
            [0, target_shape[2] - up_shape[2]],  # height
            [0, target_shape[3] - up_shape[3]],  # width
            [0, 0],  # channels
        ]
        return tf.pad(upsampled, pads)
    
    x = UpSampling3D(2)(x)
    x = Lambda(match_spatial_dims)([x, res2])
    x = Concatenate()([x, res2])
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(32, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling3D(2)(x)
    x = Lambda(match_spatial_dims)([x, res1])
    x = Concatenate()([x, res1])
    x = Conv3D(16, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, 3, padding='same', kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output: [ux, uy, uz, s] with same constraints
    ux = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='tanh', name='ux')(x)
    uy = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='tanh', name='uy')(x)
    uz = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='tanh', name='uz')(x)
    s  = Conv3D(1, 1, padding='same', kernel_regularizer=l2_reg, activation='linear', name='s')(x)
    
    outputs = Concatenate(axis=-1)([ux, uy, uz, s])
    
    model = Model(inputs=inputs, outputs=outputs, name='SimpleCNN3D_ddp_shifter_lightweight')
    return model
