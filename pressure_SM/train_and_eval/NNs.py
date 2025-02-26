import tensorflow as tf
from tensorflow.keras import Input, Model, regularizers

################################################################################
## Neural Networks architectures
################################################################################

def densePCA(n_layers, depth=512, PC_input=None, PC_p=None, dropout_rate=None, regularization=None):
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

def densePCA_attention(n_layers=3, depth=[512], PC_input=None, PC_p=None, dropout_rate=None, regularization=None):
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

def conv1D_PCA(n_layers=3, depth=[512], PC_input=None, PC_p=None, dropout_rate=None, regularization=None, kernel_size=3):
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