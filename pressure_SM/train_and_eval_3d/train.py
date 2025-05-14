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
from .neural_networks import MLP, dense_attention, conv1D, FNO3d, GNN
import warnings

warnings.filterwarnings("ignore", message="Unmanaged memory use is high")
warnings.filterwarnings("ignore", message="Sending large graph of size")
warnings.filterwarnings("ignore", message="full garbage collections took")

class Training:

  def __init__(self, var_p, var_in, standardization_method):
    self.var_in = var_in
    self.var_p = var_p
    self.standardization_method = standardization_method
  
  @tf.function
  def train_step(self, inputs, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(inputs, training=True)
      loss=self.loss_object(labels, predictions)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  #@tf.function
  def perform_validation(self):

    losses = []

    for (x_val, y_val) in self.test_dataset:
      x_val = tf.cast(x_val[...,0,0], dtype='float32')
      y_val = tf.cast(y_val[...,0,0], dtype='float32')

      val_logits = self.model(x_val)
      val_loss = self.loss_object(y_true = y_val , y_pred = val_logits)
      losses.append(val_loss)

    return losses
  
  def my_mse_loss(self):
    def loss_f(y_true, y_pred):

      loss = tf.reduce_mean(tf.square(y_true - y_pred) )

      return 1e6 * loss
    return loss_f

  def prepare_data_to_tf(self, gridded_h5_fn: str = 'gridded_sim_data.h5', outarray_flat_fn: str= 'features_data.h5', flatten_data: bool = False):

    self.gridded_h5_fn = gridded_h5_fn
    filename_flat = outarray_flat_fn
     
    print('Loading Blocks data\n')
    f = tables.open_file(filename_flat, mode='r')
    input = f.root.inputs[...] 
    output = f.root.outputs[...] 
    f.close()

    standardization_method="std"
    print(f'Normalizing feature data based on standardization method: {standardization_method}')
    x, y = utils.normalize_feature_data(input, output, standardization_method)
    x, y = utils.unison_shuffled_copies(x, y)
    print('Data shuffled \n')
    if flatten_data:
      x = x.reshape((x.shape[0], x.shape[1], 1, 1))
      y = y.reshape((y.shape[0], y.shape[1], 1, 1))

    # Convert values to compatible tf Records - much faster
    split = 0.9
    if not (os.path.isfile('train_data.tfrecords') and os.path.isfile('test_data.tfrecords')):
      print("TFRecords train and test data is not available... writing it\n")
      count_train = utils.write_images_to_tfr_short(x[:int(split*x.shape[0]),...], y[:int(split*y.shape[0]),...], filename="train_data")
      count_test = utils.write_images_to_tfr_short(x[int(split*x.shape[0]):,...], y[int(split*y.shape[0]):,...], filename="test_data")
    else:
      print("TFRecords train and test data already available, using it... If you want to write new data, delete 'train_data.tfrecords' and 'test_data.tfrecords'!\n")
    self.len_train = int(split*x.shape[0])

    return 0 
   
  def load_data_and_train(self,
      lr,
      batch_size,
      model_name,
      beta_1,
      num_epoch,
      n_layers,
      width,
      dropout_rate,
      regularization,
      model_architecture,
      new_model,
      ranks,
      flatten_data):

    train_path = 'train_data.tfrecords'
    test_path = 'test_data.tfrecords'

    self.train_dataset = utils.load_dataset_tf(filename = train_path, batch_size = batch_size, buffer_size=1024)
    self.test_dataset = utils.load_dataset_tf(filename = test_path, batch_size = batch_size, buffer_size=1024)

    # Training 

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)#, decay=0.45*lr, amsgrad=True)
    self.loss_object = self.my_mse_loss()

    print(model_architecture)
    if new_model:
      model_architecture_norm = model_architecture.lower()
      match model_architecture_norm:
        case 'mlp_small' | 'mlp_big' | 'mlp_small_unet' | 'mlp_huge' | 'mlp_huger':
          self.model = MLP(
          n_layers, width,
          ranks * ranks * ranks * ranks,
          ranks * ranks * ranks,
          dropout_rate, regularization
          )
        case 'conv1d':
          self.model = conv1D(
          n_layers, width,
          ranks * ranks * ranks * ranks,
          ranks * ranks * ranks,
          dropout_rate, regularization
          )
        case 'mlp_attention':
          self.model = dense_attention(
          n_layers, width,
          ranks * ranks * ranks * ranks,
          ranks * ranks * ranks,
          dropout_rate, regularization
          )
        case 'gnn':
            self.model = GNN()
        case 'fno3d':
          self.model = FNO3d()
        case _:
          raise ValueError('Invalid NN model type')
    else:
      model_path = f"model_{model_name}.h5"
      print(f"Loading model: {model_path}")
      self.model = tf.keras.models.load_model(model_path)

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

      losses_val  = self.perform_validation()

      losses_train_mean = np.mean(losses_train)
      losses_val_mean = np.mean(losses_val)

      epochs_train_losses.append(losses_train_mean)
      epochs_val_losses.append(losses_val_mean)
      print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean)))

      progbar.update(step+1)

      # It was found that if the min_delta is too small, or patience is too high it can cause overfitting
      stopEarly = utils.Callback_EarlyStopping(epochs_val_losses, min_delta=0.1/100, patience=100)
      if stopEarly:
        print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,num_epoch))
        break

      if epoch > 20:
        mod = 'model_' + model_name + '.h5'
        if losses_val_mean < min_yet:
          print(f'saving model: {mod}', flush=True)
          self.model.save(mod)
          min_yet = losses_val_mean
    
    print("Terminating training")
    mod = 'model_' + model_name + '.h5'
    ## Plot loss vs epoch
    plt.plot(list(range(len(epochs_train_losses))), epochs_train_losses, label ='train')
    plt.plot(list(range(len(epochs_val_losses))), epochs_val_losses, label ='val')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'loss_vs_epoch_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.png')

    ## Save losses data
    np.savetxt(f'train_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_train_losses, fmt='%d')
    np.savetxt(f'test_loss_beta{beta_1}lr{lr}reg{regularization}drop{dropout_rate}.txt', epochs_val_losses, fmt='%d')