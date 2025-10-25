import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras import layers
import spektral
from spektral.layers import GCNConv
from tensorflow.keras.layers import Layer

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

def SimpleCNN3D(rank, in_channels=4, base_filters=8, n_layers=3,
                  use_residual=True, dropout_rate=0.1, regularization=1e-4):
    """
    Improved 3D CNN for compressed CFD blocks.
    
    Args:
        rank: spatial size of the compressed block (e.g., 4 for a 4x4x4 block)
        in_channels: number of input features per voxel
        base_filters: number of filters in the first conv layer
        n_layers: number of convolutional layers
        use_residual: whether to use residual connections
        dropout_rate: dropout rate for regularization
        regularization: L2 regularization factor
    Returns:
        A model mapping (rank, rank, rank, in_channels) -> (rank, rank, rank)
    """

    inputs = Input(shape=(rank, rank, rank, in_channels))
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