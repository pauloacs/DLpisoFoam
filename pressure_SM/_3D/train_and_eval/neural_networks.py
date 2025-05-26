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

class SpectralConv3D(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # e.g., (4, 4, 4)

        # Complex weights: [in_channels, out_channels, *modes]
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
        x = tf.cast(x, tf.float32)  # Ensure input is float32
        B, X, Y, Z, C = tf.unstack(tf.shape(x))  # dynamic shapes

        # FFT
        x_ft = tf.signal.fft3d(tf.cast(x, tf.complex64))  # (B, X, Y, Z, C)
        x_ft = tf.transpose(x_ft, [0, 4, 1, 2, 3])  # (B, C, X, Y, Z)

        # Only keep the low-frequency modes
        modes_x, modes_y, modes_z = self.modes
        x_ft_crop = x_ft[:, :, :modes_x, :modes_y, :modes_z]  # (B, C, mx, my, mz)

        # Build complex weight tensor
        w_complex = tf.complex(self.weight_real, self.weight_imag)  # (Cin, Cout, mx, my, mz)

        # Multiply in Fourier space
        out_ft = tf.einsum("bixyz,ioxyz->boxyz", x_ft_crop, w_complex)  # (B, Cout, mx, my, mz)

        # Pad the result back to full size
        pad_dims = [[0, 0], [0, 0], [0, X - modes_x], [0, Y - modes_y], [0, Z - modes_z]]
        out_ft_padded = tf.pad(out_ft, pad_dims)

        # Transpose and inverse FFT
        out_ft_padded = tf.transpose(out_ft_padded, [0, 2, 3, 4, 1])  # (B, X, Y, Z, Cout)
        x_out = tf.signal.ifft3d(out_ft_padded)
        return tf.math.real(x_out)


from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv3D, LayerNormalization, Add, Lambda

def FNO3d(input_shape=(4, 4, 4, 4), out_channels=1, modes=(4, 4, 4), width=32, n_layers=4):
    """
    FNO 3D model.
    """
    inputs = Input(shape=input_shape, dtype=tf.float32)  # Ensure float32 input
    x = tf.keras.layers.Conv3D(width, kernel_size=1)(inputs)  # Project to width channels

    for _ in range(n_layers):
        x1 = SpectralConv3D(width, width, modes)(x)
        x2 = tf.keras.layers.Conv3D(width, kernel_size=1)(x)
        x = Add()([x1, x2])
        x = tf.keras.layers.Activation('gelu')(x)
        x = LayerNormalization()(x)

    x = tf.keras.layers.Conv3D(64, kernel_size=1, activation='gelu')(x)
    x = tf.keras.layers.Conv3D(out_channels, kernel_size=1)(x)

    x = tf.squeeze(x, axis=-1) if out_channels == 1 else x  # Output: (4, 4, 4)
    model = Model(inputs=inputs, outputs=x, name="FNO3D")
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