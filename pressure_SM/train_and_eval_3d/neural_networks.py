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

    n_nodes = 4 * 4 * 4
    node_features = 4
    output_dim = 1

    # Input: (4,4,4,4)
    X_in = Input(shape=(4, 4, 4, 4), name='X_in')  # (4,4,4,4) input
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
    # Reshape back to (4,4,4)
    outputs = layers.Reshape((4, 4, 4))(x)

    model = Model(inputs=[X_in, A_in], outputs=outputs, name="GNN")
    print(model.summary())
    return model

class SpectralConv3d(Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2, modes3),
            initializer="random_normal",
            trainable=True,
            name="w_real"
        )
        self.weights_imag = self.add_weight(
            shape=(in_channels, out_channels, modes1, modes2, modes3),
            initializer="random_normal",
            trainable=True,
            name="w_imag"
        )

    def call(self, x):
        # x: (batch, 4, 4, 4, in_channels)
        x_ft = tf.signal.fft3d(tf.cast(x, tf.complex64))
        # Only keep the lower modes and apply learned weights
        x_ft_low = x_ft[:, :self.modes1, :self.modes2, :self.modes3, :]
        weights = tf.complex(self.weights_real, self.weights_imag)
        # (batch, modes1, modes2, modes3, in_channels) x (in_channels, out_channels, modes1, modes2, modes3)
        # -> (batch, modes1, modes2, modes3, out_channels)
        x_ft_low = tf.transpose(x_ft_low, [0, 4, 1, 2, 3])  # (batch, in_channels, modes1, modes2, modes3)
        out_ft_low = tf.einsum('bcmnk,coijk->bomnk', x_ft_low, weights)
        out_ft_low = tf.transpose(out_ft_low, [0, 2, 3, 4, 1])  # (batch, modes1, modes2, modes3, out_channels)
        # Zero pad to original size
        out_ft = tf.zeros_like(x_ft[..., :self.out_channels], dtype=tf.complex64)
        out_ft = tf.tensor_scatter_nd_update(
            out_ft,
            indices=tf.reshape(tf.stack(tf.meshgrid(
                tf.range(tf.shape(x)[0]),
                tf.range(self.modes1),
                tf.range(self.modes2),
                tf.range(self.modes3),
                tf.range(self.out_channels),
                indexing='ij'
            ), axis=-1), [-1, 5]),
            updates=tf.reshape(out_ft_low, [-1])
        )
        x = tf.signal.ifft3d(out_ft)
        return tf.math.real(x)

def FNO3d(
    modes1=4, modes2=4, modes3=4,
    width=32,
    n_layers=4,
    input_shape=(4, 4, 4, 4),
    output_shape=(4, 4, 4)
):
    """
    Simple 3D Fourier Neural Operator.
    Input:  (None, 4, 4, 4, 4)
    Output: (None, 4, 4, 4)
    """
    inp = Input(shape=input_shape, name="FNO_input")
    x = layers.Dense(width)(inp)

    for _ in range(n_layers):
        x1 = SpectralConv3d(width, width, modes1, modes2, modes3)(x)
        x2 = layers.Conv3D(width, 1, padding='same', activation=None)(x)
        x = layers.Add()([x1, x2])
        x = layers.Activation('gelu')(x)

    x = layers.Dense(1)(x)  # (None, 4, 4, 4, 1)
    x = layers.Reshape(output_shape)(x)  # (None, 4, 4, 4)
    model = Model(inputs=inp, outputs=x, name="FNO3d")
    print(model.summary())
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