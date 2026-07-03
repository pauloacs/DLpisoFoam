from email.mime import base

import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from spektral.layers import GCNConv

################################################################################
## Custom Layers
################################################################################


################################################################################
## Custom Layers
################################################################################

class SymmetricPadding3D(Layer):
    """Custom layer for mirror-like padding in 3D.

    Supports:
      - mode='SYMMETRIC': repeats the boundary value
      - mode='REFLECT': reflects without repeating the boundary value

    For your boundary artifact test, try mode='REFLECT' next.
    """

    def __init__(self, padding, mode="SYMMETRIC", **kwargs):
        super().__init__(**kwargs)

        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = tuple(padding)

        self.mode = mode.upper()

        if self.mode not in ["SYMMETRIC", "REFLECT"]:
            raise ValueError(
                "Padding mode must be either 'SYMMETRIC' or 'REFLECT'. "
                f"Received: {mode}"
            )

    def call(self, x):
        pz, py, px = self.padding

        return tf.pad(
            x,
            paddings=[
                [0, 0],      # batch
                [pz, pz],    # z
                [py, py],    # y
                [px, px],    # x
                [0, 0],      # channels
            ],
            mode=self.mode,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "padding": self.padding,
                "mode": self.mode,
            }
        )
        return config


def padded_conv3d(
    x,
    filters,
    kernel_size=3,
    dilation_rate=(1, 1, 1),
    regularization=1e-5,
    use_bias=True,
    pad_mode="REFLECT",
):
    """Helper: Conv3D with explicit mirror padding instead of zero padding.

    For kernel_size=3, the required padding equals the dilation_rate.

    Recommended first tests:
      pad_mode='REFLECT'    -> usually less boundary repetition
      pad_mode='SYMMETRIC'  -> repeats boundary value
    """

    if kernel_size != 3:
        raise NotImplementedError("This helper assumes kernel_size=3.")

    # For kernel_size=3, required padding equals dilation_rate
    padding = dilation_rate

    x = SymmetricPadding3D(
        padding=padding,
        mode=pad_mode,
    )(x)

    x = layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        padding="valid",
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_regularizer=regularizers.l2(regularization),
    )(x)

    return x


################################################################################
## Neural Networks architectures
################################################################################

def GNN(
    rank,
    n_gnn_layers=3,
    gnn_units=64,
    dropout_rate=None,
    regularization=None
):
    """
    Creates a GNN for features prediction.
    Inputs:
        - Input shape: (4, 4, 4, 4)  (grid: 4x4x4, 4 features per node)
        - Output shape: (4, 4, 4)    (grid: 4x4x4, 1 output per node)
    """

    n_nodes = rank * rank * rank
    node_features = 4
    output_dim = 1

    # Input: (rank,rank,rank,noide_features)
    X_in = Input(shape=(rank, rank, rank, node_features), name='X_in')  # (4,4,4,4) input
    # Reshape to (n_nodes, node_features)
    x = layers.Reshape((n_nodes, node_features))(X_in)

    # Adjacency matrix input (n_nodes, n_nodes)
    A_in = Input(shape=(n_nodes, n_nodes), name='A_in')

    reg = regularizers.l2(regularization) if regularization else None

    for _ in range(n_gnn_layers):
        x = GCNConv(gnn_units, activation='relu', kernel_regularizer=reg)([x, A_in])
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(output_dim)(x)  # (n_nodes, 1)
    # Reshape back to (rank,rank,rank)
    outputs = layers.Reshape((rank, rank, rank))(x)

    model = Model(inputs=[X_in, A_in], outputs=outputs, name="GNN")
    print(model.summary())
    return model

class SpectralConv3D(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # (mx, my, mz)
        initializer = tf.keras.initializers.GlorotUniform()
        self.weight_real = self.add_weight(
            shape=(in_channels, out_channels, *self.modes),
            initializer=initializer,
            trainable=True,
            name="w_real"
        )
        self.weight_imag = self.add_weight(
            shape=(in_channels, out_channels, *self.modes),
            initializer=initializer,
            trainable=True,
            name="w_imag"
        )

    def call(self, x):
        # x: (B, X, Y, Z, C)
        x_ft = tf.signal.fft3d(tf.cast(x, tf.complex64))  # (B, X, Y, Z, C)
        x_ft = tf.transpose(x_ft, [0, 4, 1, 2, 3])  # (B, C, X, Y, Z)
        mx, my, mz = self.modes
        x_ft_crop = x_ft[:, :, :mx, :my, :mz]
        w_complex = tf.complex(self.weight_real, self.weight_imag)
        out_ft = tf.einsum("bixyz,ioxyz->boxyz", x_ft_crop, w_complex)
        # Pad to original size
        pad = [[0, 0], [0, 0], [0, x.shape[1] - mx], [0, x.shape[2] - my], [0, x.shape[3] - mz]]
        out_ft = tf.pad(out_ft, pad)
        out_ft = tf.transpose(out_ft, [0, 2, 3, 4, 1])  # (B, X, Y, Z, Cout)
        x_out = tf.signal.ifft3d(out_ft)
        x_out_real = tf.math.real(x_out)
        return x_out_real

def FNO3d_old(rank, out_channels=1, width=8, n_layers=3):
    """
    Simpler FNO 3D model.
    """
    input_shape = (rank, rank, rank, 4)
    modes = (rank, rank, rank)
    inputs = Input(shape=input_shape)
    x = Conv3D(width, kernel_size=1)(inputs)
    for _ in range(n_layers):
        x1 = SpectralConv3D(width, width, modes)(x)
        x2 = Conv3D(width, kernel_size=1)(x)
        x = Add()([x1, x2])
        x = Activation('gelu')(x)
    x = Conv3D(out_channels, kernel_size=1)(x)
    if out_channels == 1:
        x = tf.squeeze(x, axis=-1)
    model = Model(inputs, x, name="FNO3D")
    model.summary()
    return model



# Fourier Neural Operator (FNO) for 3D flow data, inspired by Li et al. (2021)
# Reference: https://arxiv.org/abs/2010.08895

class FNOBlock3D(tf.keras.layers.Layer):
    """
    A single 3D FNO block: spectral convolution + pointwise convolution + skip connection.
    """
    def __init__(self, width, modes, activation='gelu'):
        super().__init__()
        self.width = width
        self.modes = modes  # (mx, my, mz)
        self.activation = tf.keras.layers.Activation(activation)
        # Spectral weights (real and imag) for each input/output channel and mode
        self.weight_real = self.add_weight(
            shape=(width, width, *modes),
            initializer='glorot_uniform',
            trainable=True,
            name="fno_weight_real"
        )
        self.weight_imag = self.add_weight(
            shape=(width, width, *modes),
            initializer='glorot_uniform',
            trainable=True,
            name="fno_weight_imag"
        )
        # Pointwise (1x1x1) convolution
        self.w_conv = tf.keras.layers.Conv3D(width, kernel_size=1)

    def call(self, x):
        # x: (B, X, Y, Z, width)
        x_ft = tf.signal.fft3d(tf.cast(x, tf.complex64))  # (B, X, Y, Z, width)
        x_ft = tf.transpose(x_ft, [0, 4, 1, 2, 3])  # (B, width, X, Y, Z)
        # Dynamically determine the number of modes to use based on input shape
        X, Y, Z = x.shape[1], x.shape[2], x.shape[3]
        mx = min(self.modes[0], X)
        my = min(self.modes[1], Y)
        mz = min(self.modes[2], Z)
        # Truncate high-frequency modes
        x_ft_crop = x_ft[:, :, :mx, :my, :mz]  # (B, width, mx, my, mz)
        w_complex = tf.complex(
            self.weight_real[:, :, :mx, :my, :mz],
            self.weight_imag[:, :, :mx, :my, :mz]
        )
        out_ft = tf.einsum("bixyz,ioxyz->boxyz", x_ft_crop, w_complex)
        # Pad back to original size
        pad = [[0, 0], [0, 0], [0, X - mx], [0, Y - my], [0, Z - mz]]
        out_ft = tf.pad(out_ft, pad)
        out_ft = tf.transpose(out_ft, [0, 2, 3, 4, 1])  # (B, X, Y, Z, width)
        x_ifft = tf.signal.ifft3d(out_ft)
        x_ifft = tf.math.real(x_ifft)
        # Pointwise conv and skip connection
        x_pw = self.w_conv(x)
        return self.activation(x_ifft + x_pw)

def FNO3d(
    rank,
    in_channels=4,
    out_channels=1,
    width=32,
    modes=(12, 12, 12),
    n_layers=2,
    activation='gelu'
):
    """
    FNO-3D model for flow data, following Li et al. (2021).
    Args:
        rank: spatial size (e.g., 4 for 4x4x4)
        in_channels: input features per voxel
        out_channels: output features per voxel
        width: number of channels in FNO layers
        modes: number of Fourier modes to keep (mx, my, mz)
        n_layers: number of FNO blocks
    Returns:
        Keras Model mapping (rank, rank, rank, in_channels) -> (rank, rank, rank) or (..., out_channels)
    """
    input_shape = (rank, rank, rank, in_channels)
    inputs = tf.keras.Input(shape=input_shape)
    # Initial projection to width channels
    x = tf.keras.layers.Conv3D(width, kernel_size=1)(inputs)
    # Stack FNO blocks
    for _ in range(n_layers):
        x = FNOBlock3D(width, modes, activation=activation)(x)
    # Final projection to output channels
    x = tf.keras.layers.Conv3D(out_channels, kernel_size=1)(x)
    if out_channels == 1:
        x = tf.squeeze(x, axis=-1)
    model = tf.keras.Model(inputs, x, name="FNO3D")
    model.summary()
    return model


from tensorflow.keras.layers import Dropout, LayerNormalization, Add, Reshape, Permute

def MLP_Mixer_3D(n_layers, rank, in_channels=4, token_mlp_dim=128, channel_mlp_dim=128, 
                 dropout_rate=None, regularization=None):
    """
    Creates a simple MLP-Mixer for 3D CFD blocks of shape (rank, rank, rank, in_channels).
    Outputs shape: (rank, rank, rank)
    """

    n_tokens = rank ** 3
    input_shape = (rank, rank, rank, in_channels)
    inputs = Input(shape=input_shape)

    # Flatten spatial dims but keep feature dim
    x = Reshape((n_tokens, in_channels))(inputs)

    if regularization is not None:
        regularizer = regularizers.l2(regularization)
        print(f'\nUsing L2 regularization. Value: {regularization}\n')
    else:
        regularizer = None

    for _ in range(n_layers):
        # Token mixing
        y = LayerNormalization()(x)
        y = Permute((2, 1))(y)  # (batch, channels, tokens)
        y = Dense(token_mlp_dim, activation='gelu', kernel_regularizer=regularizer)(y)
        y = Dense(n_tokens, kernel_regularizer=regularizer)(y)
        y = Permute((2, 1))(y)
        if dropout_rate:
            y = Dropout(dropout_rate)(y)
        x = Add()([x, y])

        # Channel mixing
        y = LayerNormalization()(x)
        y = Dense(channel_mlp_dim, activation='gelu', kernel_regularizer=regularizer)(y)
        y = Dense(in_channels, kernel_regularizer=regularizer)(y)
        if dropout_rate:
            y = Dropout(dropout_rate)(y)
        x = Add()([x, y])

    # Project to scalar output per token (voxel)
    x = Dense(1)(x)  # (batch, n_tokens, 1)
    x = Reshape((rank, rank, rank, 1))(x)  # Keep the last dimension for channel

    # Optionally squeeze the last dimension if you want (batch, rank, rank, rank)
    x = Lambda(lambda t: tf.squeeze(t, axis=-1))(x)

    model = Model(inputs, x, name="MLP_Mixer_3D")
    print(model.summary())
    return model


def Simple_multi_layer_3D(rank, in_channels=4, width=64, n_layers=2, dropout_rate=None, regularization=None):
    """
    Simple MLP for 3D CFD blocks.
    Input: (rank, rank, rank, in_channels)
    Output: (rank, rank, rank)
    """
    n_tokens = rank ** 3
    input_shape = (rank, rank, rank, in_channels)
    inputs = Input(shape=input_shape)
    x = Reshape((n_tokens, in_channels))(inputs)

    if regularization is not None:
        regularizer = regularizers.l2(regularization)
    else:
        regularizer = None

    for _ in range(n_layers):
        x = Dense(width, activation='relu', kernel_regularizer=regularizer)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    x = Dense(1, kernel_regularizer=regularizer)(x)
    x = Lambda(lambda t: tf.squeeze(t, axis=-1))(x)  # (batch, n_tokens)
    outputs = Reshape((rank, rank, rank))(x)

    model = Model(inputs, outputs, name="SimpleMLP3D")
    print(model.summary())
    return model


from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, Activation, Add, Dropout, LayerNormalization
from tensorflow.keras.layers import Dense, Lambda


def SimpleCNN3D_two_heads_improve(
    rank,
    in_channels=4,
    base_filters=8,
    dropout_rate=0.05,
    regularization=1e-5,
    smooth_base_filters=4,
    return_heads=True,
):
    """
    Two-head 3D CNN for delta_delta_p prediction.

    Head 1: local head
        - Same as the original/local CNN output.
        - Good for near-obstacle peaks and sharp structures.

    Head 2: smooth head
        - Smooth-biased shallow UNet/coarse branch.
        - Downsamples only in y/x, not z.
        - Uses weak coarse skip connections.
        - Predicts smooth pressure at coarse resolution, then upsamples.
        - Avoids full-resolution convolutions after p_smooth output.

    Outputs:
        if return_heads=True:
            {
                "p_total":  [B, Z, Y, X],
                "p_smooth": [B, Z, Y, X],
                "p_local":  [B, Z, Y, X],
            }
        else:
            p_total only.
    """

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels))

    reg = regularizers.l2(regularization) if regularization else None

    # ---------------------------------------------------------------------
    # Shared trunk: your current good local/dilated CNN trunk
    # ---------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 4),
        (1, 2, 6),
        (1, 2, 4),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for dilation in dilations:
        x_res = x

        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)

        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)

        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)

        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    # ---------------------------------------------------------------------
    # Head 1: local / peak head
    # ---------------------------------------------------------------------
    # Keep this intentionally very close to your original final projection.
    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_local",
    )(x)

    # ---------------------------------------------------------------------
    # Head 2: smooth / far-field head
    # ---------------------------------------------------------------------
    # Important:
    #   - no z-downsampling
    #   - average pooling instead of max pooling
    #   - weak/coarse skips only
    #   - pressure predicted at coarse resolution, then upsampled
    #   - no full-resolution conv after p_smooth is produced
    # ---------------------------------------------------------------------

    def smooth_conv_block(t, filters, name_prefix):
        t = padded_conv3d(
            t,
            filters=filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        t = layers.LayerNormalization(name=f"{name_prefix}_ln1")(t)
        t = layers.Activation("relu", name=f"{name_prefix}_relu1")(t)

        if dropout_rate:
            t = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(t)

        t = padded_conv3d(
            t,
            filters=filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        t = layers.LayerNormalization(name=f"{name_prefix}_ln2")(t)
        t = layers.Activation("relu", name=f"{name_prefix}_relu2")(t)

        return t

    # Start smooth branch from shared trunk, but reduce capacity.
    s0 = layers.Conv3D(
        filters=smooth_base_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="smooth_input_reduce",
    )(x)
    s0 = layers.LayerNormalization(name="smooth_input_ln")(s0)
    s0 = layers.Activation("relu", name="smooth_input_relu")(s0)

    # Encoder level 0
    e0 = smooth_conv_block(
        s0,
        filters=smooth_base_filters,
        name_prefix="smooth_enc0",
    )

    # Downsample only y/x.
    # For (50,80,250): -> (50,40,50)
    d1 = layers.AveragePooling3D(
        pool_size=(1, 2, 5),
        strides=(1, 2, 5),
        padding="valid",
        name="smooth_down1",
    )(e0)

    # Encoder level 1
    e1 = smooth_conv_block(
        d1,
        filters=smooth_base_filters * 2,
        name_prefix="smooth_enc1",
    )

    # Downsample only y/x again.
    # For (50,40,50): -> (50,20,10)
    d2 = layers.AveragePooling3D(
        pool_size=(1, 2, 5),
        strides=(1, 2, 5),
        padding="valid",
        name="smooth_down2",
    )(e1)

    # Bottleneck
    b = smooth_conv_block(
        d2,
        filters=smooth_base_filters * 4,
        name_prefix="smooth_bottleneck",
    )

    # Decoder level 1
    u1 = layers.UpSampling3D(
        size=(1, 2, 5),
        name="smooth_up1",
    )(b)

    # Weak coarse skip from e1.
    # Reduce skip channels so it cannot copy too much high-frequency detail.
    skip1 = layers.Conv3D(
        filters=smooth_base_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="smooth_skip1_reduce",
    )(e1)

    u1 = layers.Concatenate(name="smooth_concat1")([u1, skip1])

    u1 = smooth_conv_block(
        u1,
        filters=smooth_base_filters * 2,
        name_prefix="smooth_dec1",
    )

    # Decoder level 0
    u0 = layers.UpSampling3D(
        size=(1, 2, 5),
        name="smooth_up0",
    )(u1)

    # Very weak high-resolution skip.
    # This is intentionally tiny to avoid copying local peaks.
    skip0 = layers.Conv3D(
        filters=max(1, smooth_base_filters // 2),
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="smooth_skip0_reduce",
    )(e0)

    u0 = layers.Concatenate(name="smooth_concat0")([u0, skip0])

    u0 = smooth_conv_block(
        u0,
        filters=smooth_base_filters,
        name_prefix="smooth_dec0",
    )

    # Smooth output
    p_smooth = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_smooth_raw",
    )(u0)

    # Anti-ribbing / anti-blocking smoothing on smooth head only.
    # This should suppress x-direction striping without destroying the local head.
    p_smooth = layers.AveragePooling3D(
        pool_size=(1, 3, 9),
        strides=(1, 1, 1),
        padding="same",
        name="p_smooth_antiblock",
    )(p_smooth)

    # ---------------------------------------------------------------------
    # Combine heads and enforce zero-mean gauge on total
    # ---------------------------------------------------------------------
    p_total_raw = layers.Add(name="p_total_raw")([p_smooth, p_local])

    p_mean = layers.Lambda(
        lambda t: tf.reduce_mean(
            t,
            axis=(1, 2, 3, 4),
            keepdims=True,
        ),
        name="p_total_mean",
    )(p_total_raw)

    # Important:
    # apply gauge correction to p_smooth, then return corrected p_smooth.
    # This makes p_smooth consistent with the loss target.
    p_smooth = layers.Subtract(name="p_smooth_gauge_corrected")(
        [p_smooth, p_mean]
    )

    p_total = layers.Add(name="p_total")([p_smooth, p_local])

    # Remove channel dimension safely
    p_total = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_total_squeeze")(p_total)
    p_smooth = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_smooth_squeeze")(p_smooth)
    p_local = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_local_squeeze")(p_local)

    if return_heads:
        outputs = {
            "p_total": p_total,
            "p_smooth": p_smooth,
            "p_local": p_local,
        }
    else:
        outputs = p_total

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN3D_two_heads_UNET_smooth_biased",
    )

    model.summary()
    return model



def SimpleCNN3D_two_heads2(
    rank,
    in_channels=4,
    base_filters=8,
    dropout_rate=0.05,
    regularization=1e-5,
    smooth_base_filters=4,
    smooth_levels=3,
    return_heads=True,
):
    """
    Two-head 3D CNN for delta_delta_p prediction.

    Head 1 (local): same idea as current local head for near-obstacle peaks.
    Head 2 (smooth): simple UNet-style branch (similar spirit to UNet3D_deep).
    """

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels))
    reg = regularizers.l2(regularization) if regularization else None

    # ---------------------------------------------------------------------
    # Shared trunk (same as SimpleCNN3D_two_heads)
    # ---------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 4),
        (1, 2, 6),
        (1, 2, 4),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for dilation in dilations:
        x_res = x

        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)

        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)

        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)

        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    # ---------------------------------------------------------------------
    # Head 1: local / peak head
    # ---------------------------------------------------------------------
    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_local",
    )(x)

    # ---------------------------------------------------------------------
    # Head 2: smooth / far-field head (UNet-like, simple)
    # ---------------------------------------------------------------------
    def smooth_conv_block(t, filters, name_prefix):
        t = layers.Conv3D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name_prefix}_conv1")(t)
        t = layers.BatchNormalization(name=f"{name_prefix}_bn1")(t)
        t = layers.Activation("relu", name=f"{name_prefix}_relu1")(t)
        if dropout_rate:
            t = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop")(t)
        t = layers.Conv3D(filters, 3, padding="same", kernel_regularizer=reg, name=f"{name_prefix}_conv2")(t)
        t = layers.BatchNormalization(name=f"{name_prefix}_bn2")(t)
        t = layers.Activation("relu", name=f"{name_prefix}_relu2")(t)
        return t

    def crop_to_match(x_up, x_ref, name_prefix):
        def _crop(inp):
            x_u, x_r = inp
            z_start = (tf.shape(x_u)[1] - tf.shape(x_r)[1]) // 2
            y_start = (tf.shape(x_u)[2] - tf.shape(x_r)[2]) // 2
            x_start = (tf.shape(x_u)[3] - tf.shape(x_r)[3]) // 2
            z_end = z_start + tf.shape(x_r)[1]
            y_end = y_start + tf.shape(x_r)[2]
            x_end = x_start + tf.shape(x_r)[3]
            return x_u[:, z_start:z_end, y_start:y_end, x_start:x_end, :]
        return layers.Lambda(_crop, name=f"{name_prefix}_crop")([x_up, x_ref])

    # Encoder
    skips = []
    s = x
    filters = smooth_base_filters
    for lvl in range(smooth_levels):
        s = smooth_conv_block(s, filters, name_prefix=f"smooth_enc{lvl}")
        skips.append(s)
        s = layers.MaxPooling3D(pool_size=2, padding="same", name=f"smooth_pool{lvl}")(s)
        filters = min(filters * 2, 128)

    # Bottleneck
    s = smooth_conv_block(s, filters, name_prefix="smooth_bottleneck")

    # Decoder
    for lvl in range(smooth_levels - 1, -1, -1):
        filters = max(smooth_base_filters, filters // 2)
        s = layers.UpSampling3D(size=2, name=f"smooth_up{lvl}")(s)
        s = crop_to_match(s, skips[lvl], name_prefix=f"smooth_dec{lvl}")
        s = layers.Concatenate(name=f"smooth_concat{lvl}")([s, skips[lvl]])
        s = smooth_conv_block(s, filters, name_prefix=f"smooth_dec{lvl}_blk")

    p_smooth = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_smooth",
    )(s)

    # ---------------------------------------------------------------------
    # Combine heads and enforce zero-mean on total
    # ---------------------------------------------------------------------
    p_total_raw = layers.Add(name="p_total_raw")([p_smooth, p_local])

    p_mean = tf.reduce_mean(
        p_total_raw,
        axis=(1, 2, 3, 4),
        keepdims=True,
    )
    p_total = layers.Subtract(name="p_total_gauge_corrected")([p_total_raw, p_mean])

    # Remove channel dimension
    p_total = tf.squeeze(p_total, axis=-1)
    p_smooth = tf.squeeze(p_smooth, axis=-1)
    p_local = tf.squeeze(p_local, axis=-1)

    if return_heads:
        outputs = {
            "p_total": p_total,
            "p_smooth": p_smooth,
            "p_local": p_local,
        }
    else:
        outputs = p_total

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN3D_two_heads_UNET",
    )
    model.summary()
    return model


def SimpleCNN3D_two_heads_(
    rank,
    in_channels=12,
    base_filters=16,
    smooth_filters=32,
    dropout_rate=0.05,
    regularization=1e-5,
    return_heads=False,
):
    """
    Parallel two-branch 3D CNN for delta_delta_p prediction.

    Branch 1: p_high
        - Full-resolution dilated residual CNN.
        - Similar to the previous successful model.
        - Intended to capture local/high-frequency pressure structures.

    Branch 2: p_smooth
        - Independent smooth branch directly from the input.
        - Uses average pooling, coarse convolutions, feature upsampling,
          and full-resolution refinement.
        - Intended to capture broad/low-frequency pressure structures.

    Final:
        p_total = gauge_correct(p_high + p_smooth)

    Parameters
    ----------
    rank : tuple/list or int
        If tuple/list: (sz, sy, sx).
        If int: uses same size in all directions.

    in_channels : int
        Number of input channels.

    base_filters : int
        Number of filters in the high-frequency branch.

    smooth_filters : int
        Number of filters in the coarse smooth branch.

    dropout_rate : float
        Dropout rate. Use 0.0 to disable dropout.

    regularization : float
        L2 regularization coefficient.

    return_heads : bool
        If False:
            returns only p_total.
        If True:
            returns dict with p_total, p_high, p_smooth.

    Returns
    -------
    model : keras.Model
    """

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:    
        sz = sy = sx = int(rank)

    inputs = Input(
        shape=(sz, sy, sx, in_channels),
        name="input",
    )

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def conv_norm_relu(
        x,
        filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
    ):
        x = padded_conv3d(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def residual_block(
        x,
        filters,
        dilation_rate,
        block_id,
    ):
        x_res = x

        y = padded_conv3d(
            x,
            filters=filters,
            kernel_size=3,
            dilation_rate=dilation_rate,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)

        if dropout_rate is not None and dropout_rate > 0.0:
            y = layers.Dropout(
                dropout_rate,
                name=f"high_dropout_{block_id}",
            )(y)

        y = padded_conv3d(
            y,
            filters=filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)

        x = layers.Add(name=f"high_res_add_{block_id}")([x_res, y])
        x = layers.Activation(
            "relu",
            name=f"high_res_relu_{block_id}",
        )(x)

        return x

    def gauge_correct(x):
        mean = tf.reduce_mean(
            x,
            axis=(1, 2, 3, 4),
            keepdims=True,
        )
        return x - mean

    # ==================================================================
    # Branch 1: high-frequency / local branch
    # ==================================================================
    h = conv_norm_relu(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
    )

    high_dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 4),
        (1, 2, 6),
        (1, 2, 4),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for i, dilation in enumerate(high_dilations):
        h = residual_block(
            h,
            filters=base_filters,
            dilation_rate=dilation,
            block_id=i,
        )

    p_high = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_high_raw",
    )(h)

    # ==================================================================
    # Branch 2: smooth / low-frequency branch
    # ==================================================================
    smooth_pool = (1, 2, 5)

    s = layers.AveragePooling3D(
        pool_size=smooth_pool,
        strides=smooth_pool,
        padding="valid",
        name="smooth_downsample",
    )(inputs)

    s = conv_norm_relu(
        s,
        filters=smooth_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
    )

    s = conv_norm_relu(
        s,
        filters=smooth_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 2),
    )

    if dropout_rate is not None and dropout_rate > 0.0:
        s = layers.Dropout(
            dropout_rate,
            name="smooth_dropout",
        )(s)

    s = conv_norm_relu(
        s,
        filters=smooth_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
    )

    # Upsample smooth features, not scalar pressure
    s = layers.UpSampling3D(
        size=smooth_pool,
        name="smooth_feature_upsample",
    )(s)

    # Full-resolution smooth refinement
    s = conv_norm_relu(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
    )

    s = conv_norm_relu(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
    )

    p_smooth = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_smooth_raw",
    )(s)

    # ==================================================================
    # Combine branches
    # ==================================================================
    p_total_raw = layers.Add(name="p_total_raw")(
        [p_high, p_smooth]
    )

    p_total = layers.Lambda(
        gauge_correct,
        name="p_total_gauge_corrected",
    )(p_total_raw)

    p_total_out = layers.Lambda(
        lambda x: tf.squeeze(x, axis=-1),
        name="p_total",
    )(p_total)

    if return_heads:
        p_high_out = layers.Lambda(
            lambda x: tf.squeeze(x, axis=-1),
            name="p_high",
        )(p_high)

        p_smooth_out = layers.Lambda(
            lambda x: tf.squeeze(x, axis=-1),
            name="p_smooth",
        )(p_smooth)

        outputs = {
            "p_total": p_total_out,
            "p_high": p_high_out,
            "p_smooth": p_smooth_out,
        }
    else:
        outputs = p_total_out

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN3D_parallel_high_smooth",
    )

    model.summary()
    return model



def SimpleCNN3D_two_heads_(
    rank,
    in_channels=4,
    base_filters=16,
    dropout_rate=0.05,
    regularization=1e-5,
    return_heads=True,
    smooth_pool=2,          # downsample factor for smooth head (2 is a good start)
    smooth_filters=None,    # if None -> base_filters
    use_gating=True,       # optional gated mixing of local/smooth
    pad_mode="SYMMETRIC",   # forwarded to padded_conv3d
):
    """
    Two-head 3D CNN for delta_delta_p prediction.

    Head 1: p_local
      - Full-resolution, sharp details.

    Head 2: p_smooth
      - Band-limited low-frequency branch (pool->conv->upsample).

    Combination:
      p_total_raw = p_local + p_smooth   (or gated if use_gating=True)
      p_total = p_total_raw - mean(p_total_raw)  (gauge correction)

    Returns:
      if return_heads=True: dict with p_total, p_local, p_smooth (all squeezed to [B, Z, Y, X])
      else: p_total only (squeezed)
    """

    # ----------------------------
    # Parse spatial dims
    # ----------------------------
    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    if smooth_filters is None:
        smooth_filters = base_filters

    inputs = Input(shape=(sz, sy, sx, in_channels), name="inputs")

    # -------------------------------------------------------------------------
    # Shared trunk (your existing structure)
    # -------------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode=pad_mode,
    )
    x = layers.LayerNormalization(name="ln_in")(x)
    x = layers.Activation("relu", name="relu_in")(x)

    # Residual blocks (no dilation)
    for i in range(7):
        x_res = x

        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode=pad_mode,
        )
        y = layers.LayerNormalization(name=f"ln_{i}_a")(y)
        y = layers.Activation("relu", name=f"relu_{i}_a")(y)

        if dropout_rate and dropout_rate > 0:
            y = layers.Dropout(dropout_rate, name=f"drop_{i}")(y)

        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode=pad_mode,
        )
        y = layers.LayerNormalization(name=f"ln_{i}_b")(y)

        x = layers.Add(name=f"add_{i}")([x_res, y])
        x = layers.Activation("relu", name=f"relu_{i}_out")(x)

    # -------------------------------------------------------------------------
    # Head 1: LOCAL (full-resolution)
    # -------------------------------------------------------------------------
    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_local",
    )(x)

    # -------------------------------------------------------------------------
    # Head 2: SMOOTH (band-limited by pooling)
    # -------------------------------------------------------------------------
    s = x
    if smooth_pool and smooth_pool > 1:
        s = layers.AveragePooling3D(pool_size=smooth_pool, padding="same", name="smooth_pool")(s)

    s = layers.Conv3D(
        smooth_filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_conv1",
    )(s)
    s = layers.LayerNormalization(name="smooth_ln1")(s)
    s = layers.Activation("relu", name="smooth_relu1")(s)

    s = layers.Conv3D(
        smooth_filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_conv2",
    )(s)
    s = layers.LayerNormalization(name="smooth_ln2")(s)
    s = layers.Activation("relu", name="smooth_relu2")(s)

    p_smooth = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_smooth",
    )(s)

    # Upsample back to full resolution
    if smooth_pool and smooth_pool > 1:
        p_smooth = layers.UpSampling3D(size=smooth_pool, name="smooth_upsample")(p_smooth)

        # --- Static cropping to exactly (sz, sy, sx) to avoid Keras Lambda TypeSpec issues ---
        def ceil_div(a, b):
            return (a + b - 1) // b

        up_z = ceil_div(sz, smooth_pool) * smooth_pool
        up_y = ceil_div(sy, smooth_pool) * smooth_pool
        up_x = ceil_div(sx, smooth_pool) * smooth_pool

        cz = max(up_z - sz, 0)
        cy = max(up_y - sy, 0)
        cx = max(up_x - sx, 0)

        crop_z = (cz // 2, cz - cz // 2)
        crop_y = (cy // 2, cy - cy // 2)
        crop_x = (cx // 2, cx - cx // 2)

        if (cz or cy or cx):
            p_smooth = layers.Cropping3D(
                cropping=(crop_z, crop_y, crop_x),
                name="smooth_crop_to_input",
            )(p_smooth)

    # -------------------------------------------------------------------------
    # Combine heads (sum or gated)
    # -------------------------------------------------------------------------
    if use_gating:
        # alpha in [0,1], per-voxel
        alpha = layers.Conv3D(
            1, 1, padding="same", activation="sigmoid",
            name="alpha_gate",
        )(x)

        one_minus_alpha = layers.Lambda(lambda a: 1.0 - a, name="one_minus_alpha")(alpha)

        p_total_raw = layers.Add(name="p_total_raw")([
            layers.Multiply(name="local_weighted")([alpha, p_local]),
            layers.Multiply(name="smooth_weighted")([one_minus_alpha, p_smooth]),
        ])
    else:
        p_total_raw = layers.Add(name="p_total_raw")([p_local, p_smooth])

    # -------------------------------------------------------------------------
    # Gauge correction on the SUM (important!)
    # -------------------------------------------------------------------------
    p_mean = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=(1, 2, 3, 4), keepdims=True),
        name="p_total_mean",
    )(p_total_raw)

    p_total = layers.Subtract(name="p_total_gauge_corrected")([p_total_raw, p_mean])

    # -------------------------------------------------------------------------
    # Remove channel dimension: [B,Z,Y,X,1] -> [B,Z,Y,X]
    # (Use Lambda to keep Keras graph happy)
    # -------------------------------------------------------------------------
    squeeze = lambda t: tf.squeeze(t, axis=-1)

    p_total_s  = layers.Lambda(squeeze, name="p_total")(p_total)
    p_local_s  = layers.Lambda(squeeze, name="p_local_squeezed")(p_local)
    p_smooth_s = layers.Lambda(squeeze, name="p_smooth_squeezed")(p_smooth)

    if return_heads:
        outputs = {"p_total": p_total_s, "p_local": p_local_s, "p_smooth": p_smooth_s}
    else:
        outputs = p_total_s

    model = Model(inputs=inputs, outputs=outputs, name="CNN3D_two_heads")
    model.summary()
    return model



## test no smooth
def SimpleCNN3D_two_heads_no_smooth(
    rank,
    in_channels=4,
    base_filters=16,
    dropout_rate=0.05,
    regularization=1e-5,
    return_heads=True,
):
    """
    Two-head 3D CNN for delta_delta_p prediction.

    Head 1: local head
        - Same idea as your original final output layer.
        - Good for obstacle-local peaks and sharp structures.

    Head 2: smooth head
        - Coarse low-resolution branch.
        - Designed to learn smooth far-field pressure mode.

    If return_heads=True:
        returns dict with p_total, p_smooth, p_local.

    If return_heads=False:
        returns only p_total, compatible with your old training loop.
    """

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels))

    # -------------------------------------------------------------------------
    # Shared trunk: same as your current model
    # -------------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )

    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ]

    for dilation in dilations:
        x_res = x

        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )

        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)

        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)

        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )

        y = layers.LayerNormalization()(y)

        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_local",
    )(x)

    # Enforce zero-mean constraint directly on the total output
    # (p_smooth and p_local learn freely; only the sum is constrained)
    p_mean = tf.reduce_mean(
        p_local,
        axis=(1, 2, 3, 4),
        keepdims=True,
    )
    p_total = layers.Subtract(name="p_total_gauge_corrected")(
        [p_local, p_mean]
    )

    # Remove channel dimension
    p_total = tf.squeeze(p_total, axis=-1)
    #p_smooth = tf.squeeze(p_smooth, axis=-1)
    p_local = tf.squeeze(p_local, axis=-1)

    if return_heads:
        outputs = {
            "p_total": p_total,
            "p_local": p_local,
        }
    else:
        outputs = p_total

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN3D_two_heads",
    )

    model.summary()
    return model




def SimpleCNN3D_multi_out_v0(
    rank,
    in_channels=4,
    out_channels=4,
    base_filters=16,
    dropout_rate=0.05,
    regularization=1e-5,
):
    """
    Multi-output variant of SimpleCNN3D_two_heads.
    Shares the same residual trunk but the final 1x1 conv produces `out_channels`
    feature maps (e.g. 4 for [ddp, ddU_x, ddU_y, ddU_z]).
    Zero-mean gauge correction is applied only to channel 0 (pressure).
    Output shape: (batch, z, y, x, out_channels).
    """
    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels))

    # Shared trunk (identical to SimpleCNN3D_two_heads)
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [(1, 1, 1)] * 7

    for dilation in dilations:
        x_res = x
        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)
        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)
        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    # Final 1x1 conv — produces all output channels at once
    out = layers.Conv3D(
        filters=out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="output_raw",
    )(x)  # (batch, z, y, x, out_channels)

    # Zero-mean gauge correction on pressure channel (index 0) only
    p_raw = out[..., 0:1]  # (batch, z, y, x, 1)
    p_mean = tf.reduce_mean(p_raw, axis=(1, 2, 3, 4), keepdims=True)
    p_corrected = layers.Subtract(name="p_gauge_corrected")([p_raw, p_mean])  # (batch, z, y, x, 1)

    if out_channels > 1:
        other_channels = out[..., 1:]  # (batch, z, y, x, out_channels-1)
        outputs = tf.concat([p_corrected, other_channels], axis=-1, name="multi_out")  # (batch, z, y, x, out_channels)
    else:
        outputs = tf.squeeze(p_corrected, axis=-1)  # scalar, for out_channels=1

    model = Model(inputs=inputs, outputs=outputs, name="CNN3D_multi_out")
    model.summary()
    return model


def SimpleCNN3D_multi_out(
    rank,
    in_channels=4,
    out_channels=4,
    base_filters=16,
    dropout_rate=0.05,
    regularization=1e-5,
):
    """
    Multi-output 3D CNN predicting [ddp, ddUx, ddUy, ddUz].

    Shared trunk: dilated residual CNN (proven dilation schedule).

    Pressure head (2-head):
      - p_local: full-res 1x1 projection from trunk
      - p_smooth: pool -> conv -> upsample branch for far-field
      - p_total = gauge_correct(p_local + p_smooth)

    Velocity head:
      - Single 1x1 conv from trunk -> 3 channels (Ux, Uy, Uz)
      - No gauge correction (velocity is not gauge-ambiguous)

    Output shape: (batch, z, y, x, 4)  ->  [p, Ux, Uy, Uz]
    """
    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    reg = regularizers.l2(regularization) if regularization else None
    inputs = Input(shape=(sz, sy, sx, in_channels))

    # -------------------------------------------------------------------------
    # Shared trunk: proven dilated residual schedule
    # -------------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (1, 2, 3),
        (1, 2, 2),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for dilation in dilations:
        x_res = x
        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)
        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)
        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    # -------------------------------------------------------------------------
    # Pressure head — local branch
    # -------------------------------------------------------------------------
    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_local",
    )(x)

    # -------------------------------------------------------------------------
    # Pressure head — smooth branch (pool -> conv -> upsample)
    # -------------------------------------------------------------------------
    smooth_pool = (1, 2, 5)

    s = layers.AveragePooling3D(
        pool_size=smooth_pool,
        strides=smooth_pool,
        padding="same",   # ceil(dim/pool) -> upsampled = pool*ceil >= dim, crop excess
        name="smooth_downsample",
    )(x)

    s = padded_conv3d(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    s = layers.LayerNormalization()(s)
    s = layers.Activation("relu")(s)

    s = padded_conv3d(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 2),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    s = layers.LayerNormalization()(s)
    s = layers.Activation("relu")(s)

    if dropout_rate:
        s = layers.Dropout(dropout_rate, name="smooth_dropout")(s)

    p_smooth_coarse = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_smooth_coarse",
    )(s)

    # Upsample by the pool factors — pooled_dim = ceil(dim/pool) due to same-padding,
    # so upsampled_dim = pool * ceil(dim/pool) >= dim. Crop the small excess.
    p_smooth = layers.UpSampling3D(
        size=smooth_pool,
        name="p_smooth_upsample",
    )(p_smooth_coarse)

    cz = smooth_pool[0] * -(-sz // smooth_pool[0]) - sz
    cy = smooth_pool[1] * -(-sy // smooth_pool[1]) - sy
    cx = smooth_pool[2] * -(-sx // smooth_pool[2]) - sx
    if cz or cy or cx:
        p_smooth = layers.Cropping3D(
            cropping=(
                (cz // 2, cz - cz // 2),
                (cy // 2, cy - cy // 2),
                (cx // 2, cx - cx // 2),
            ),
            name="p_smooth_crop",
        )(p_smooth)

    # Anti-blocking blur on smooth head
    p_smooth = layers.AveragePooling3D(
        pool_size=(1, 3, 9),
        strides=(1, 1, 1),
        padding="same",
        name="p_smooth_antiblock",
    )(p_smooth)

    # -------------------------------------------------------------------------
    # Combine pressure heads + gauge correction
    # -------------------------------------------------------------------------
    p_total_raw = layers.Add(name="p_total_raw")([p_local, p_smooth])

    p_mean = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=(1, 2, 3, 4), keepdims=True),
        name="p_mean",
    )(p_total_raw)
    p_total = layers.Subtract(name="p_gauge_corrected")([p_total_raw, p_mean])
    # shape: (B, Z, Y, X, 1)

    # -------------------------------------------------------------------------
    # Velocity head — simple 1x1 projection, (out_channels - 1) velocity channels
    # -------------------------------------------------------------------------
    n_vel = out_channels - 1  # e.g. 3 for [Ux, Uy, Uz] when out_channels=4
    u_out = layers.Conv3D(
        filters=n_vel,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="u_out_raw",
    )(x)
    # shape: (B, Z, Y, X, n_vel)

    # -------------------------------------------------------------------------
    # Concatenate: [p, U...] -> (B, Z, Y, X, out_channels)
    # -------------------------------------------------------------------------
    outputs = layers.Concatenate(axis=-1, name="multi_out")([p_total, u_out])

    model = Model(inputs=inputs, outputs=outputs, name="CNN3D_multi_out_v2")
    model.summary()
    return model


def SimpleCNN3D_multi_out_divU(
    rank,
    in_channels=4,
    out_channels=4,
    base_filters=16,
    dropout_rate=0.05,
    regularization=1e-5,
    div_u_ch_idx=None,
    mask_dilation=(2, 2, 2),
    div_u_mean=0.0,
    div_u_std=1.0,
):
    """
    Masked multi-output 3D CNN predicting [ddp, ddUx, ddUy, ddUz].

    Identical to SimpleCNN3D_multi_out but the velocity head (ddU channels) is
    hard-zeroed outside the region where div(UStar) != 0.

    Physically: the velocity correction only needs to be non-zero in cells where
    div(U*) != 0 (and their immediate neighbours, to allow the divergence to be
    redistributed). A dilated binary mask derived from the divUStar input channel
    enforces this constraint exactly — the gradient is also zeroed outside the
    active region, giving the velocity head a much cleaner learning signal.

    IMPORTANT: only works when use_feature_decomposition=False, so that the raw
    spatial divUStar values are present as a dedicated input channel.

    Parameters
    ----------
    rank : tuple/list or int
        Spatial dimensions (sz, sy, sx) or single int.
    in_channels : int
        Total number of input channels (must include divUStar).
    out_channels : int
        4  =>  [ddp, ddUx, ddUy, ddUz]
    base_filters : int
    dropout_rate : float
    regularization : float
    div_u_ch_idx : int  (required)
        Index of the divUStar channel inside the input tensor.  Matches
        `div_u_idx` from train_init.py (the raw spatial block channel order).
    mask_dilation : tuple of 3 ints
        MaxPool3D kernel size used to dilate the divUStar binary mask so that
        the immediate neighbours of active cells are also included.
        Default (1, 5, 5) keeps the full z-extent but dilates ±2 cells in y/x.
    """
    if div_u_ch_idx is None:
        raise ValueError(
            "SimpleCNN3D_multi_out_divU requires div_u_ch_idx to be set "
            "to the channel index of divUStar in the raw spatial input."
        )

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    reg = regularizers.l2(regularization) if regularization else None
    inputs = Input(shape=(sz, sy, sx, in_channels))

    # -------------------------------------------------------------------------
    # Shared trunk: proven dilated residual schedule
    # -------------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (1, 2, 3),
        (1, 2, 2),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for dilation in dilations:
        x_res = x
        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)
        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)
        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )
        y = layers.LayerNormalization()(y)
        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    # -------------------------------------------------------------------------
    # Pressure head — local branch
    # -------------------------------------------------------------------------
    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_local",
    )(x)

    # -------------------------------------------------------------------------
    # Pressure head — smooth branch (pool -> conv -> upsample)
    # -------------------------------------------------------------------------
    smooth_pool = (1, 2, 5)

    s = layers.AveragePooling3D(
        pool_size=smooth_pool,
        strides=smooth_pool,
        padding="same",
        name="smooth_downsample",
    )(x)

    s = padded_conv3d(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    s = layers.LayerNormalization()(s)
    s = layers.Activation("relu")(s)

    s = padded_conv3d(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 2),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    s = layers.LayerNormalization()(s)
    s = layers.Activation("relu")(s)

    if dropout_rate:
        s = layers.Dropout(dropout_rate, name="smooth_dropout")(s)

    p_smooth_coarse = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="p_smooth_coarse",
    )(s)

    p_smooth = layers.UpSampling3D(
        size=smooth_pool,
        name="p_smooth_upsample",
    )(p_smooth_coarse)

    cz = smooth_pool[0] * -(-sz // smooth_pool[0]) - sz
    cy = smooth_pool[1] * -(-sy // smooth_pool[1]) - sy
    cx = smooth_pool[2] * -(-sx // smooth_pool[2]) - sx
    if cz or cy or cx:
        p_smooth = layers.Cropping3D(
            cropping=(
                (cz // 2, cz - cz // 2),
                (cy // 2, cy - cy // 2),
                (cx // 2, cx - cx // 2),
            ),
            name="p_smooth_crop",
        )(p_smooth)

    p_smooth = layers.AveragePooling3D(
        pool_size=(1, 3, 9),
        strides=(1, 1, 1),
        padding="same",
        name="p_smooth_antiblock",
    )(p_smooth)

    # -------------------------------------------------------------------------
    # Combine pressure heads + gauge correction
    # -------------------------------------------------------------------------
    p_total_raw = layers.Add(name="p_total_raw")([p_local, p_smooth])

    p_mean = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=(1, 2, 3, 4), keepdims=True),
        name="p_mean",
    )(p_total_raw)
    p_total = layers.Subtract(name="p_gauge_corrected")([p_total_raw, p_mean])
    # shape: (B, Z, Y, X, 1)

    # -------------------------------------------------------------------------
    # Velocity head — raw prediction from trunk
    # -------------------------------------------------------------------------
    n_vel = out_channels - 1  # 3 for [Ux, Uy, Uz]
    u_out_raw = layers.Conv3D(
        filters=n_vel,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=reg,
        name="u_out_raw",
    )(x)
    # shape: (B, Z, Y, X, n_vel)

    # -------------------------------------------------------------------------
    # Hard mask: zero ddU outside where divUStar != 0
    #
    # Extract the divUStar channel from the raw input, binarize, then dilate
    # with MaxPool3D so that the immediate spatial neighbours of active cells
    # are also included (the correction must spread to neighbours to drive
    # div(U) to zero).  The mask is then multiplied element-wise with u_out.
    # -------------------------------------------------------------------------
    div_u_raw = layers.Lambda(
        lambda t: t[:, :, :, :, div_u_ch_idx:div_u_ch_idx + 1],
        name="div_u_extract",
    )(inputs)  # (B, Z, Y, X, 1) — still in normalized space

    # Undo std normalization so that physical zeros map to zero before binarizing.
    # div_u_mean / div_u_std are the training-time statistics for the divU channel.
    _divu_std  = float(div_u_std)
    _divu_mean = float(div_u_mean)
    div_u_phys = layers.Lambda(
        lambda t: t * _divu_std + _divu_mean,
        name="div_u_denorm",
    )(div_u_raw)  # (B, Z, Y, X, 1) — adim physical values

    

    def percentile_mask(t):
        abs_t = tf.abs(t)

        # Flatten per sample
        flat = tf.reshape(abs_t, [tf.shape(t)[0], -1])

        # Sort values
        sorted_flat = tf.sort(flat, axis=-1)

        n = tf.shape(sorted_flat)[-1]

        # 90th percentile index
        p90_idx = tf.cast(0.90 * tf.cast(n, tf.float32), tf.int32)

        # Extract p90 value
        p90 = sorted_flat[:, p90_idx]

        # Broadcast back
        p90 = p90[:, None, None, None, None]

        # Keep only values above p90
        return tf.cast(abs_t > p90, tf.float32)

    bin_mask = layers.Lambda(
        percentile_mask,
        name="div_u_bin_mask",
    )(div_u_phys)


    dilated_mask = layers.MaxPool3D(
        pool_size=mask_dilation,
        strides=(1, 1, 1),
        padding="same",
        name="div_u_mask_dilated",
    )(bin_mask)  # dilated to include neighbours

    # Broadcast mask across all velocity channels
    if n_vel > 1:
        dilated_mask_broadcast = layers.Lambda(
            lambda t: tf.repeat(t, repeats=n_vel, axis=-1),
            name="div_u_mask_broadcast",
        )(dilated_mask)  # (B, Z, Y, X, n_vel)
    else:
        dilated_mask_broadcast = dilated_mask

    u_out = layers.Multiply(name="u_out_masked")([u_out_raw, dilated_mask_broadcast])
    # shape: (B, Z, Y, X, n_vel) — guaranteed zero outside active region

    # -------------------------------------------------------------------------
    # Concatenate: [p, U...] -> (B, Z, Y, X, out_channels)
    # -------------------------------------------------------------------------
    outputs = layers.Concatenate(axis=-1, name="multi_out")([p_total, u_out])

    model = Model(inputs=inputs, outputs=outputs, name="CNN3D_multi_out_divU")
    model.summary()
    return model



def debug_div_mask(model, x_sample, div_u_ch_idx, z_slice=None):
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # --- build small debug model (only what we need) ---
    debug_model = tf.keras.Model(
        inputs=model.input,
        outputs={
            "mask": model.get_layer("div_u_mask_dilated").output
        }
    )

    # forward pass
    pred = debug_model(x_sample, training=False)

    mask = pred["mask"].numpy()[..., 0]           # (B, Z, Y, X)
    div_u = x_sample[..., div_u_ch_idx]           # normalized input

    # pick slice
    b = 0
    nz = div_u.shape[1]
    z = z_slice if z_slice is not None else nz // 2

    # -------------------------
    # ✅ PRINT BASIC CHECKS
    # -------------------------
    print("\n=== SIMPLE MASK DEBUG ===")

    print("divU stats:")
    print("  min:", np.min(div_u))
    print("  max:", np.max(div_u))
    print("  mean:", np.mean(div_u))

    print("\nmask stats:")
    print("  active fraction:", np.mean(mask > 0.5))

    # alignment check (rough)
    div_binary = np.abs(div_u) > np.percentile(np.abs(div_u), 95)  # should match the percentile_mask logic
    mask_binary = mask > 0.5

    agreement = np.mean(div_binary == mask_binary)
    print("\nmask vs div(U*) agreement:", agreement)

    # -------------------------
    # ✅ PLOT
    # -------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("div(U*)")
    plt.imshow(div_u[b, z], cmap="RdBu")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("divU mask (dilated)")
    plt.imshow(mask[b, z], cmap="gray")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("debug_div_mask.png")



# BEST SO FAR!!! (alpha 0.25)
def SimpleCNN3D_two_heads(
    rank,
    in_channels=4,
    base_filters=24,
    dropout_rate=0.05,
    regularization=1e-5,
    return_heads=True,
):
    """
    Two-head 3D CNN for delta_delta_p prediction.

    Head 1: local head
        - Same idea as your original final output layer.
        - Good for obstacle-local peaks and sharp structures.

    Head 2: smooth head
        - Coarse low-resolution branch.
        - Designed to learn smooth far-field pressure mode.

    If return_heads=True:
        returns dict with p_total, p_smooth, p_local.

    If return_heads=False:
        returns only p_total, compatible with your old training loop.
    """

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels))

    # -------------------------------------------------------------------------
    # Shared trunk: same as your current model
    # -------------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )

    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 2, 5),
        (1, 2, 3),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for dilation in dilations:
        x_res = x

        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )

        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)

        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)

        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )

        y = layers.LayerNormalization()(y)

        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    # -------------------------------------------------------------------------
    # Head 1: local / peak head
    # -------------------------------------------------------------------------
    # This is intentionally very close to your original final output projection.
    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_local",
    )(x)

    # Head 1: option 2

    #p_local = padded_conv3d(
    #    x,
    #    filters=base_filters,
    #    kernel_size=3,
    #    dilation_rate=(1, 1, 1),
    #    regularization=regularization,
    #    pad_mode="SYMMETRIC",
    #)
    #p_local = layers.LayerNormalization()(p_local)
    #p_local = layers.Activation("relu")(p_local)

    #p_local = padded_conv3d(
    #    p_local,
    #    filters=base_filters,
    #    kernel_size=3,
    #    dilation_rate=(1, 1, 1),
    #    regularization=regularization,
    #    pad_mode="SYMMETRIC",
    #)
    #p_local = layers.LayerNormalization()(p_local)
    #p_local = layers.Activation("relu")(p_local)

    #p_local = layers.Conv3D(
    #    filters=1,
    #    kernel_size=1,
    #    padding="same",
    #    use_bias=False,
    #    kernel_regularizer=regularizers.l2(regularization),
    #    name="p_local",
    #)(p_local)

    # -------------------------------------------------------------------------
    # Head 2: smooth / far-field head
    # -------------------------------------------------------------------------
    smooth_pool = (2, 2, 5)

    s = layers.AveragePooling3D(
        pool_size=smooth_pool,
        strides=smooth_pool,
        padding="valid",
        name="smooth_downsample",
    )(x)

    s = padded_conv3d(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    s = layers.LayerNormalization()(s)
    s = layers.Activation("relu")(s)

    s = padded_conv3d(
        s,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 2),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    s = layers.LayerNormalization()(s)
    s = layers.Activation("relu")(s)

    if dropout_rate:
        s = layers.Dropout(dropout_rate)(s)

    # Predict smooth pressure directly at coarse resolution
    p_smooth = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_smooth_coarse",
    )(s)


    # Upsample smooth pressure back to full resolution using Upsampling3D
    # Calculate upsampling factors
    up_z = (sz + sz // 1 - 1) // (sz // 1)  # Ceiling division
    up_y = (sy + sy // 2 - 1) // (sy // 2)
    up_x = (sx + sx // 5 - 1) // (sx // 5)
    
    p_smooth = layers.UpSampling3D(
        size=smooth_pool,
        name="p_smooth_upsample",
    )(p_smooth)

    # Crop p_smooth to match p_local shape (handles size mismatches)
    def crop_to_match_shape(inputs):
        p_sm, p_loc = inputs
        target_shape = tf.shape(p_loc)
        current_shape = tf.shape(p_sm)
        z_start = (current_shape[1] - target_shape[1]) // 2
        y_start = (current_shape[2] - target_shape[2]) // 2
        x_start = (current_shape[3] - target_shape[3]) // 2
        return p_sm[
            :,
            z_start:z_start + target_shape[1],
            y_start:y_start + target_shape[2],
            x_start:x_start + target_shape[3],
            :
        ]
    
    p_smooth = layers.Lambda(
        crop_to_match_shape,
        name="p_smooth_crop_to_match",
    )([p_smooth, p_local])

    # Anti-blocking smoothing after upsampling
    p_smooth = layers.AveragePooling3D(
        pool_size=(1, 3, 9),
        strides=(1, 1, 1),
        padding="same",
        name="p_smooth_antiblock",
    )(p_smooth)


    # -------------------------------------------------------------------------
    # Combine heads: both heads learn independently
    # -------------------------------------------------------------------------
    p_total = layers.Add(name="p_total_raw")([p_smooth, p_local])

    # Remove channel dimension
    p_total = tf.squeeze(p_total, axis=-1)
    p_smooth = tf.squeeze(p_smooth, axis=-1)
    p_local = tf.squeeze(p_local, axis=-1)

    if return_heads:
        outputs = {
            "p_total": p_total,
            "p_smooth": p_smooth,
            "p_local": p_local,
        }
    else:
        outputs = p_total

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN3D_two_heads",
    )

    model.summary()
    return model


def SimpleCNN3D(
    rank,
    in_channels=4,
    out_channels=1,
    base_filters=8,
    dropout_rate=0.05,
    regularization=1e-5,
):
    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels))

    # Initial projection: symmetric padding
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )

    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)


    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 4),
        (1, 2, 6),
        (1, 2, 6),
        (1, 2, 4),
        (1, 1, 2),
        (1, 1, 1),
    ]


    for dilation in dilations:
        x_res = x

        # First conv: dilated, symmetric padded
        y = padded_conv3d(
            x,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=dilation,
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )

        y = layers.LayerNormalization()(y)
        y = layers.Activation("relu")(y)

        if dropout_rate:
            y = layers.Dropout(dropout_rate)(y)

        # Second conv: local, also symmetric padded
        y = padded_conv3d(
            y,
            filters=base_filters,
            kernel_size=3,
            dilation_rate=(1, 1, 1),
            regularization=regularization,
            pad_mode="SYMMETRIC",
        )

        y = layers.LayerNormalization()(y)

        x = layers.Add()([x_res, y])
        x = layers.Activation("relu")(x)

    # Final 1x1 projection does not need padding treatment
    x = layers.Conv3D(
        filters=out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
    )(x)

    if out_channels == 1:
        x = tf.squeeze(x, axis=-1)
        # Zero-gauge pressure constraint (only for scalar pressure output)
        x = x - tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        model = Model(inputs, x, name="SimpleDilatedCNN3D_SymmetricEverywhere")
    else:
        # Multi-channel output: shape (N, sz, sy, sx, out_channels)
        # Apply zero-gauge constraint to the first channel (ddp) only
        ddp = x[..., 0:1] - tf.reduce_mean(x[..., 0:1], axis=(1, 2, 3), keepdims=True)
        rest = x[..., 1:]
        x = tf.concat([ddp, rest], axis=-1)
        model = Model(inputs, x, name="SimpleDilatedCNN3D_MultiOutput")
    model.summary()

    return model



def SimpleCNN3D1(rank, in_channels=4, base_filters=8, n_layers=3,
                  use_residual=True, dropout_rate=0.1, regularization=1e-4):
    """
    Improved 3D CNN for compressed CFD blocks.
    
    Args:
        rank: spatial size of the block — either a scalar (cubic) or a (sz, sy, sx) tuple
        in_channels: number of input features per voxel
        base_filters: number of filters in the first conv layer
        n_layers: number of convolutional layers
        use_residual: whether to use residual connections
        dropout_rate: dropout rate for regularization
        regularization: L2 regularization factor
    Returns:
        A model mapping (sz, sy, sx, in_channels) -> (sz, sy, sx)
    """
    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels))
    x = inputs

    filters = base_filters
    for i in range(n_layers):
        x_res = x
        x = Conv3D(filters=filters, kernel_size=3, padding='same',
                   kernel_regularizer=regularizers.l2(regularization))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        if use_residual and i > 0 and x_res.shape[-1] == x.shape[-1]:
            x = Add()([x, x_res])
        x = LayerNormalization()(x)
        filters = min(filters * 2, 256)  # Increase filters, cap at 256

    # Final projection to 1 channel (e.g., pressure)
    x = Conv3D(filters=1, kernel_size=1, kernel_regularizer=regularizers.l2(regularization))(x)
    x = tf.squeeze(x, axis=-1)

    model = Model(inputs, x, name="ImprovedCNN3D")
    model.summary()
    return model

def ImprovedCNN3D_v2(rank, in_channels=4, base_filters=16, n_levels=4, 
                     dropout_rate=0.2, regularization=1e-5):
    """Enhanced CNN with SE blocks, dilations, and better residuals."""
    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)
    
    inputs = Input(shape=(sz, sy, sx, in_channels))
    reg = regularizers.l2(regularization) if regularization else None
    
    x = inputs
    filters = base_filters
    
    for level in range(n_levels):
        # Multi-kernel inception-style block
        paths = [
            Conv3D(filters//3, 1, padding='same', kernel_regularizer=reg)(x),
            Conv3D(filters//3, 3, padding='same', kernel_regularizer=reg)(x),
            Conv3D(filters//3, 3, dilation_rate=2, padding='same', kernel_regularizer=reg)(x),
        ]
        x = Concatenate()(paths)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        x = Dropout(dropout_rate)(x) if dropout_rate else x
        
        # SE block (channel attention)
        se = GlobalAveragePooling3D()(x)
        se = Dense(max(1, filters // 16), activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = Reshape((1, 1, 1, filters))(se)
        x = Multiply()([x, se])
        
        filters = min(filters * 2, 256)
    
    x = Conv3D(1, 1, kernel_regularizer=reg)(x)
    x = tf.squeeze(x, axis=-1)
    
    model = Model(inputs, x, name="ImprovedCNN3D_v2")
    return model

def UNet3D(rank, in_channels=4, base_filters=16, n_levels=2, 
           dropout_rate=0.1, regularization=1e-4):
    """
    3D U-Net architecture for CFD blocks.
    Encoder path downsamples (Conv3D + MaxPooling3D), decoder path upsamples with skip connections.
    
    Args:
        rank: spatial size of the block — either a scalar (cubic) or a (sz, sy, sx) tuple
        in_channels: number of input features per voxel (default: 4)
        base_filters: number of filters in the first conv layer (default: 16)
        n_levels: number of encoding/decoding levels (default: 2)
        dropout_rate: dropout rate for regularization (default: 0.1)
        regularization: L2 regularization factor (default: 1e-4)
    
    Returns:
        A Keras model mapping (sz, sy, sx, in_channels) -> (sz, sy, sx)
    """
    from tensorflow.keras.layers import (
        Conv3D, BatchNormalization, Activation, MaxPooling3D, 
        UpSampling3D, Concatenate, Dropout, LayerNormalization
    )
    
    # Handle rank as scalar or tuple
    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)
    
    inputs = Input(shape=(sz, sy, sx, in_channels), name='input')
    
    # Regularizer
    reg = regularizers.l2(regularization) if regularization else None
    
    # Encoder path (downsampling)
    encoder_outputs = []
    x = inputs
    current_filters = base_filters
    
    for level in range(n_levels):
        # Double convolution block
        x = Conv3D(current_filters, kernel_size=3, padding='same',
                   kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)  # Changed from 'gelu' to 'relu'
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        
        x = Conv3D(current_filters, kernel_size=3, padding='same',
                   kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)  # Changed from 'gelu' to 'relu'
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        
        # Save for skip connection
        encoder_outputs.append(x)
        
        # Downsample if not the last level
        if level < n_levels - 1:
            x = MaxPooling3D(pool_size=2, padding='same')(x)
            current_filters = min(current_filters * 2, 256)

    
    # Decoder path (upsampling)
    for level in range(n_levels - 1, 0, -1):
        current_filters = current_filters // 2
        
        # Upsample
        x = UpSampling3D(size=2)(x)
        
        # Crop upsampled features to match encoder output shape (handles non-divisible dims)
        encoder_feat = encoder_outputs[level - 1]
        # Use Lambda to crop x to match encoder spatial dimensions
        def crop_to_shape(upsampled_and_target):
            x_up, x_target = upsampled_and_target
            # Dynamically crop to match target shape
            z_start = (tf.shape(x_up)[1] - tf.shape(x_target)[1]) // 2
            y_start = (tf.shape(x_up)[2] - tf.shape(x_target)[2]) // 2
            x_start = (tf.shape(x_up)[3] - tf.shape(x_target)[3]) // 2
            z_end = z_start + tf.shape(x_target)[1]
            y_end = y_start + tf.shape(x_target)[2]
            x_end = x_start + tf.shape(x_target)[3]
            return x_up[:, z_start:z_end, y_start:y_end, x_start:x_end, :]
        x = Lambda(crop_to_shape)([x, encoder_feat])
        
        # Skip connection: concatenate with corresponding encoder output
        x = Concatenate()([x, encoder_feat])
        
        # Double convolution block
        x = Conv3D(current_filters, kernel_size=3, padding='same',
                   kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        
        x = Conv3D(current_filters, kernel_size=3, padding='same',
                   kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
    
    # Final output projection to 1 channel
    x = Conv3D(filters=1, kernel_size=1, kernel_regularizer=reg)(x)
    x = tf.squeeze(x, axis=-1)
    
    model = Model(inputs, x, name="UNet3D")
    model.summary()
    return model


def UNet3D_deep(rank, in_channels=4, base_filters=4, n_levels=4, 
                dropout_rate=0.1, regularization=1e-4):
    """
    Deeper 3D U-Net (4 levels) for CFD blocks.
    See UNet3D for argument details.
    """
    return UNet3D(rank, in_channels=in_channels, base_filters=base_filters, n_levels=n_levels, 
                  dropout_rate=dropout_rate, regularization=regularization)


def UNet3D_attention(rank, in_channels=4, base_filters=8, n_levels=2,
                     dropout_rate=0.1, regularization=1e-4):
    """
    3D U-Net with skip connections for multi-scale CFD fields.
    Designed for fields with sharp patterns near obstacles and smooth far-field.

    - Encoder & Decoder: ReLU (stable, smooth training)
    - Skip connections: preserve encoder detail in decoder
    - Simpler and more robust than attention-gating

    Args:
        rank: spatial size — scalar (cubic) or (sz, sy, sx) tuple
        in_channels: input features per voxel (default: 4)
        base_filters: filters in first conv layer (default: 8)
        n_levels: encoder/decoder levels (default: 2; increase for larger domains)
        dropout_rate: dropout rate (default: 0.1)
        regularization: L2 factor (default: 1e-4)

    Returns:
        Keras model mapping (sz, sy, sx, in_channels) -> (sz, sy, sx)
    """
    from tensorflow.keras.layers import (
        Conv3D, BatchNormalization, Activation, MaxPooling3D,
        UpSampling3D, Concatenate, Dropout, Add, Cropping3D
    )

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    reg = regularizers.l2(regularization) if regularization else None

    def conv_block(x, filters, activation='relu', use_dropout=True):
        """Double conv block with batch norm."""
        x = Conv3D(filters, kernel_size=3, padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        if dropout_rate and use_dropout:
            x = Dropout(dropout_rate)(x)
        x = Conv3D(filters, kernel_size=3, padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    inputs = Input(shape=(sz, sy, sx, in_channels), name='input')

    # --- Encoder ---
    encoder_outputs = []
    x = inputs
    current_filters = base_filters

    for level in range(n_levels):
        # Double convolution
        x = conv_block(x, current_filters, activation='relu')
        encoder_outputs.append(x)
        
        # Downsample if not last level
        if level < n_levels - 1:
            x = MaxPooling3D(pool_size=2, padding='same')(x)
            current_filters = min(current_filters * 2, 256)

    # --- Decoder with skip connections ---
    for level in range(n_levels - 1, 0, -1):
        current_filters = current_filters // 2
        
        # Upsample
        x = UpSampling3D(size=2)(x)
        
        # Get encoder skip connection
        encoder_feat = encoder_outputs[level - 1]
        
        # Crop upsampled x to match encoder skip connection spatial dimensions
        def crop_to_match(inp):
            x_up, x_ref = inp
            # Dynamically crop x_up to match x_ref spatial shape
            z_start = (tf.shape(x_up)[1] - tf.shape(x_ref)[1]) // 2
            y_start = (tf.shape(x_up)[2] - tf.shape(x_ref)[2]) // 2
            x_start = (tf.shape(x_up)[3] - tf.shape(x_ref)[3]) // 2
            z_end = z_start + tf.shape(x_ref)[1]
            y_end = y_start + tf.shape(x_ref)[2]
            x_end = x_start + tf.shape(x_ref)[3]
            return x_up[:, z_start:z_end, y_start:y_end, x_start:x_end, :]
        
        x = Lambda(crop_to_match)([x, encoder_feat])
        
        # Concatenate
        x = Concatenate()([x, encoder_feat])
        
        # Double convolution with ReLU for decoder
        x = conv_block(x, current_filters, activation='relu')

    # --- Output layer ---
    x = Conv3D(filters=1, kernel_size=1, kernel_regularizer=reg)(x)
    x = tf.squeeze(x, axis=-1)

    model = Model(inputs, x, name="UNet3D_attention")
    model.summary()
    return model


def MLP(n_layers, depth=512, PC_input=None, PC_p=None, dropout_rate=None, regularization=None):
    """
    Creates the MLP NN.
    """
    
    inputs = Input(int(PC_input))
    if len(depth) == 1:
        depth = [depth]*n_layers
    
    # Regularization parameter
    if regularization is not None:
        regularizer = regularizers.l2(regularization)
        print(f'\nUsing L2 regularization. Value: {regularization}\n')
    else:
        regularizer = None
    
    x = tf.keras.layers.Dense(depth[0], activation='relu', kernel_regularizer=regularizer)(inputs)
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    for i in range(n_layers - 1):
        x = tf.keras.layers.Dense(depth[i+1], activation='relu', kernel_regularizer=regularizer)(x)
        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(PC_p)(x)

    model = Model(inputs, outputs, name="MLP")
    print(model.summary())

    return model

def dense_attention(n_layers=3, depth=[512], PC_input=None, PC_p=None, dropout_rate=None, regularization=None):
    """
    Creates the MLP with an attention mechanism.
    """
    inputs = Input((int(PC_input),))
    if len(depth) == 1:
        depth = [depth[0]] * n_layers

    # Regularization parameter
    regularizer = regularizers.l2(regularization) if regularization else None

    x = tf.keras.layers.Dense(depth[0], activation='relu', kernel_regularizer=regularizer)(inputs)
    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Applying a multi-head attention layer
    x = tf.expand_dims(x, 1)  # Add a new dimension for the sequence length
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    attn_output = tf.keras.layers.LayerNormalization()(attn_output)
    attn_output = tf.squeeze(attn_output, 1)  # Remove the added dimension

    # Adding additional dense layers
    for i in range(1, n_layers):
        x = tf.keras.layers.Dense(depth[i], activation='relu', kernel_regularizer=regularizer)(attn_output)
        if dropout_rate is not None:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        attn_output = tf.keras.layers.LayerNormalization()(x + attn_output)  # Residual connection

    outputs = tf.keras.layers.Dense(PC_p)(attn_output)

    model = Model(inputs, outputs, name="MLP_with_Attention")
    print(model.summary())

    return model

def conv1D(n_layers=3, depth=[512], PC_input=None, PC_p=None, dropout_rate=None, regularization=None, kernel_size=3):
    """
    Creates a 1D ConvNet with regularization and dropout, similar to an MLP.
    """
    
    # Define input layer
    inputs = Input(shape=(PC_input, 1))  # 1D Conv input shape requires an extra dimension
    
    if len(depth) == 1:
        depth = [depth[0]] * n_layers
    
    # Regularization parameter
    regularizer = regularizers.l2(regularization) if regularization else None
    
    # First convolutional layer
    x = tf.keras.layers.Conv1D(
        filters=depth[0], 
        kernel_size=kernel_size, 
        activation='relu',
        padding='same',
        kernel_regularizer=regularizer
    )(inputs)

    # Optional dropout
    if dropout_rate:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Additional convolutional layers
    for i in range(1, n_layers):
        x = tf.keras.layers.Conv1D(
            filters=depth[i], 
            kernel_size=kernel_size,
            padding='same',
            activation='relu', 
            kernel_regularizer=regularizer
        )(x)
        
        if dropout_rate:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Flatten and final dense layer
    x = tf.keras.layers.Flatten()(x)  # Convert 1D convolution output to a 1D vector
    outputs = tf.keras.layers.Dense(PC_p)(x)

    # Create and compile the model
    model = Model(inputs, outputs, name="1D_ConvNet")

    print(model.summary())

    return model












#####################










def residual_block_3d(
    x,
    filters,
    regularization=1e-5,
    dropout_rate=0.0,
    dilation_rate=(1, 1, 1),
    name=None,
):
    """
    Residual block using padded_conv3d for stride=1 convolutions.
    """
    x_res = x

    y = padded_conv3d(
        x,
        filters=filters,
        kernel_size=3,
        dilation_rate=dilation_rate,
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    y = layers.LayerNormalization(name=None if name is None else f"{name}_ln1")(y)
    y = layers.Activation("relu", name=None if name is None else f"{name}_relu1")(y)

    if dropout_rate:
        y = layers.Dropout(dropout_rate, name=None if name is None else f"{name}_drop")(y)

    y = padded_conv3d(
        y,
        filters=filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    y = layers.LayerNormalization(name=None if name is None else f"{name}_ln2")(y)

    x = layers.Add(name=None if name is None else f"{name}_add")([x_res, y])
    x = layers.Activation("relu", name=None if name is None else f"{name}_relu2")(x)
    return x


def crop_to_match(source, target, name=None):
    """
    Center-crop source tensor spatially to match target tensor.
    Expects 5D tensors: [B, Z, Y, X, C]
    """
    def _crop(inputs):
        src, tgt = inputs
        src_shape = tf.shape(src)
        tgt_shape = tf.shape(tgt)

        z_start = tf.maximum((src_shape[1] - tgt_shape[1]) // 2, 0)
        y_start = tf.maximum((src_shape[2] - tgt_shape[2]) // 2, 0)
        x_start = tf.maximum((src_shape[3] - tgt_shape[3]) // 2, 0)

        return src[
            :,
            z_start:z_start + tgt_shape[1],
            y_start:y_start + tgt_shape[2],
            x_start:x_start + tgt_shape[3],
            :
        ]

    return layers.Lambda(_crop, name=name)([source, target])


def SimpleCNN3D_two_heads_smooth(
    rank,
    in_channels=4,
    base_filters=24,
    dropout_rate=0.1,
    regularization=1e-5,
    return_heads=True,
    smooth_stride=(2, 2, 3),
    use_second_smooth_scale=False,
):
    """
    Improved two-head 3D CNN for delta_delta_p prediction.

    Main changes vs previous version:
    - stronger local head
    - learned smooth head downsampling
    - residual processing in smooth branch
    - upsample features instead of final 1-channel field
    - refinement after upsampling
    - no output gauge correction
    """

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels), name="inputs")

    # -------------------------------------------------------------------------
    # Shared trunk
    # -------------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    x = layers.LayerNormalization(name="stem_ln")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 2, 3),
        (1, 2, 3),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for i, dilation in enumerate(dilations):
        x = residual_block_3d(
            x,
            filters=base_filters,
            regularization=regularization,
            dropout_rate=dropout_rate,
            dilation_rate=dilation,
            name=f"trunk_res{i+1}",
        )

    # Lightweight skip projection for smooth-head refinement later
    x_skip_smooth = layers.Conv3D(
        filters=base_filters // 2,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="x_skip_smooth_proj",
    )(x)
    x_skip_smooth = layers.LayerNormalization(name="x_skip_smooth_ln")(x_skip_smooth)
    x_skip_smooth = layers.Activation("relu", name="x_skip_smooth_relu")(x_skip_smooth)

    # -------------------------------------------------------------------------
    # Head 1: local / high-frequency head
    # -------------------------------------------------------------------------
    l = padded_conv3d(
        x,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    l = layers.LayerNormalization(name="local_ln1")(l)
    l = layers.Activation("relu", name="local_relu1")(l)

    l = padded_conv3d(
        l,
        filters=base_filters // 2,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    l = layers.LayerNormalization(name="local_ln2")(l)
    l = layers.Activation("relu", name="local_relu2")(l)

    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_local",
    )(l)

    # -------------------------------------------------------------------------
    # Head 2: smooth / low-frequency head
    # -------------------------------------------------------------------------
    # Learned downsampling instead of average pooling
    s = layers.Conv3D(
        filters=base_filters,
        kernel_size=3,
        strides=smooth_stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_down_conv1",
    )(x)
    s = layers.LayerNormalization(name="smooth_down_ln1")(s)
    s = layers.Activation("relu", name="smooth_down_relu1")(s)

    # Coarse residual processing
    s = residual_block_3d(
        s,
        filters=base_filters,
        regularization=regularization,
        dropout_rate=dropout_rate,
        dilation_rate=(1, 1, 1),
        name="smooth_res1",
    )
    s = residual_block_3d(
        s,
        filters=base_filters,
        regularization=regularization,
        dropout_rate=dropout_rate,
        dilation_rate=(1, 1, 2),
        name="smooth_res2",
    )

    if use_second_smooth_scale:
        # Optional second coarser scale for larger-range pressure organization
        s2 = layers.Conv3D(
            filters=base_filters * 2,
            kernel_size=3,
            strides=(1, 1, 2),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization),
            name="smooth_down_conv2",
        )(s)
        s2 = layers.LayerNormalization(name="smooth_down_ln2")(s2)
        s2 = layers.Activation("relu", name="smooth_down_relu2")(s2)

        s2 = residual_block_3d(
            s2,
            filters=base_filters * 2,
            regularization=regularization,
            dropout_rate=dropout_rate,
            dilation_rate=(1, 1, 1),
            name="smooth_res3",
        )
        s2 = residual_block_3d(
            s2,
            filters=base_filters * 2,
            regularization=regularization,
            dropout_rate=dropout_rate,
            dilation_rate=(1, 1, 2),
            name="smooth_res4",
        )

        # Upsample coarse features back to first smooth scale
        s2 = layers.UpSampling3D(size=(1, 1, 2), name="smooth_up_s2")(s2)
        s2 = layers.Conv3D(
            filters=base_filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization),
            name="smooth_up_s2_refine",
        )(s2)
        s2 = layers.LayerNormalization(name="smooth_up_s2_ln")(s2)
        s2 = layers.Activation("relu", name="smooth_up_s2_relu")(s2)

        s2 = crop_to_match(s2, s, name="smooth_crop_s2_to_s")
        s = layers.Add(name="smooth_merge_scales")([s, s2])

    # Upsample smooth features back to full resolution
    s = layers.UpSampling3D(size=smooth_stride, name="smooth_up_to_full")(s)
    s = crop_to_match(s, x, name="smooth_crop_to_full")

    # Fuse with a lightweight skip from the trunk for spatial alignment
    s = layers.Concatenate(name="smooth_concat_skip")([s, x_skip_smooth])

    # Refine at full resolution
    s = layers.Conv3D(
        filters=base_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_refine_conv1",
    )(s)
    s = layers.LayerNormalization(name="smooth_refine_ln1")(s)
    s = layers.Activation("relu", name="smooth_refine_relu1")(s)

    if dropout_rate:
        s = layers.Dropout(dropout_rate, name="smooth_refine_drop")(s)

    s = layers.Conv3D(
        filters=base_filters // 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_refine_conv2",
    )(s)
    s = layers.LayerNormalization(name="smooth_refine_ln2")(s)
    s = layers.Activation("relu", name="smooth_refine_relu2")(s)

    p_smooth = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_smooth",
    )(s)

    # -------------------------------------------------------------------------
    # Combine heads
    # -------------------------------------------------------------------------
    p_total = layers.Add(name="p_total")([p_smooth, p_local])

    # Remove channel dimension
    p_total = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_total_squeeze")(p_total)
    p_smooth = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_smooth_squeeze")(p_smooth)
    p_local = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_local_squeeze")(p_local)

    if return_heads:
        outputs = {
            "p_total": p_total,
            "p_smooth": p_smooth,
            "p_local": p_local,
        }
    else:
        outputs = p_total

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN3D_two_heads_v2",
    )

    model.summary()
    return model


def SimpleCNN3D_two_heads_smooth_old(
    rank,
    in_channels=4,
    base_filters=24,
    dropout_rate=0.1,
    regularization=1e-5,
    return_heads=True,
    smooth_stride=(2, 2, 3),
    use_second_smooth_scale=True,
):
    """
    Improved two-head 3D CNN for delta_delta_p prediction.

    Main changes vs previous version:
    - stronger local head
    - learned smooth head downsampling
    - residual processing in smooth branch
    - upsample features instead of final 1-channel field
    - refinement after upsampling
    - no output gauge correction
    """

    if isinstance(rank, (tuple, list)):
        sz, sy, sx = rank
    else:
        sz = sy = sx = int(rank)

    inputs = Input(shape=(sz, sy, sx, in_channels), name="inputs")

    # -------------------------------------------------------------------------
    # Shared trunk
    # -------------------------------------------------------------------------
    x = padded_conv3d(
        inputs,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    x = layers.LayerNormalization(name="stem_ln")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    dilations = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 3),
        (1, 2, 5),
        (1, 2, 3),
        (1, 1, 2),
        (1, 1, 1),
    ]

    for i, dilation in enumerate(dilations):
        x = residual_block_3d(
            x,
            filters=base_filters,
            regularization=regularization,
            dropout_rate=dropout_rate,
            dilation_rate=dilation,
            name=f"trunk_res{i+1}",
        )

    # Lightweight skip projection for smooth-head refinement later
    x_skip_smooth = layers.Conv3D(
        filters=base_filters // 2,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="x_skip_smooth_proj",
    )(x)
    x_skip_smooth = layers.LayerNormalization(name="x_skip_smooth_ln")(x_skip_smooth)
    x_skip_smooth = layers.Activation("relu", name="x_skip_smooth_relu")(x_skip_smooth)

    # -------------------------------------------------------------------------
    # Head 1: local / high-frequency head
    # -------------------------------------------------------------------------
    l = padded_conv3d(
        x,
        filters=base_filters,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    l = layers.LayerNormalization(name="local_ln1")(l)
    l = layers.Activation("relu", name="local_relu1")(l)

    l = padded_conv3d(
        l,
        filters=base_filters // 2,
        kernel_size=3,
        dilation_rate=(1, 1, 1),
        regularization=regularization,
        pad_mode="SYMMETRIC",
    )
    l = layers.LayerNormalization(name="local_ln2")(l)
    l = layers.Activation("relu", name="local_relu2")(l)

    p_local = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_local",
    )(l)

    # -------------------------------------------------------------------------
    # Head 2: smooth / low-frequency head
    # -------------------------------------------------------------------------
    # Learned downsampling instead of average pooling
    s = layers.Conv3D(
        filters=base_filters,
        kernel_size=3,
        strides=smooth_stride,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_down_conv1",
    )(x)
    s = layers.LayerNormalization(name="smooth_down_ln1")(s)
    s = layers.Activation("relu", name="smooth_down_relu1")(s)

    # Coarse residual processing
    s = residual_block_3d(
        s,
        filters=base_filters,
        regularization=regularization,
        dropout_rate=dropout_rate,
        dilation_rate=(1, 1, 1),
        name="smooth_res1",
    )
    s = residual_block_3d(
        s,
        filters=base_filters,
        regularization=regularization,
        dropout_rate=dropout_rate,
        dilation_rate=(1, 1, 2),
        name="smooth_res2",
    )

    if use_second_smooth_scale:
        # Optional second coarser scale for larger-range pressure organization
        s2 = layers.Conv3D(
            filters=base_filters * 2,
            kernel_size=3,
            strides=(1, 1, 2),
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization),
            name="smooth_down_conv2",
        )(s)
        s2 = layers.LayerNormalization(name="smooth_down_ln2")(s2)
        s2 = layers.Activation("relu", name="smooth_down_relu2")(s2)

        s2 = residual_block_3d(
            s2,
            filters=base_filters * 2,
            regularization=regularization,
            dropout_rate=dropout_rate,
            dilation_rate=(1, 1, 1),
            name="smooth_res3",
        )
        s2 = residual_block_3d(
            s2,
            filters=base_filters * 2,
            regularization=regularization,
            dropout_rate=dropout_rate,
            dilation_rate=(1, 1, 2),
            name="smooth_res4",
        )

        # Upsample coarse features back to first smooth scale
        s2 = layers.UpSampling3D(size=(1, 1, 2), name="smooth_up_s2")(s2)
        s2 = layers.Conv3D(
            filters=base_filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(regularization),
            name="smooth_up_s2_refine",
        )(s2)
        s2 = layers.LayerNormalization(name="smooth_up_s2_ln")(s2)
        s2 = layers.Activation("relu", name="smooth_up_s2_relu")(s2)

        s2 = crop_to_match(s2, s, name="smooth_crop_s2_to_s")
        s = layers.Add(name="smooth_merge_scales")([s, s2])

    # Upsample smooth features back to full resolution
    s = layers.UpSampling3D(size=smooth_stride, name="smooth_up_to_full")(s)
    s = crop_to_match(s, x, name="smooth_crop_to_full")

    # Fuse with a lightweight skip from the trunk for spatial alignment
    s = layers.Concatenate(name="smooth_concat_skip")([s, x_skip_smooth])

    # Refine at full resolution
    s = layers.Conv3D(
        filters=base_filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_refine_conv1",
    )(s)
    s = layers.LayerNormalization(name="smooth_refine_ln1")(s)
    s = layers.Activation("relu", name="smooth_refine_relu1")(s)

    if dropout_rate:
        s = layers.Dropout(dropout_rate, name="smooth_refine_drop")(s)

    s = layers.Conv3D(
        filters=base_filters // 2,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="smooth_refine_conv2",
    )(s)
    s = layers.LayerNormalization(name="smooth_refine_ln2")(s)
    s = layers.Activation("relu", name="smooth_refine_relu2")(s)

    p_smooth = layers.Conv3D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(regularization),
        name="p_smooth",
    )(s)

    # -------------------------------------------------------------------------
    # Combine heads
    # -------------------------------------------------------------------------
    p_total = layers.Add(name="p_total")([p_smooth, p_local])

    # Remove channel dimension
    p_total = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_total_squeeze")(p_total)
    p_smooth = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_smooth_squeeze")(p_smooth)
    p_local = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="p_local_squeeze")(p_local)

    if return_heads:
        outputs = {
            "p_total": p_total,
            "p_smooth": p_smooth,
            "p_local": p_local,
        }
    else:
        outputs = p_total

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN3D_two_heads_v2",
    )

    model.summary()
    return model
