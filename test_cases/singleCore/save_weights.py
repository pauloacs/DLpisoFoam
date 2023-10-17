import tensorflow as tf

model = tf.keras.models.load_model('NN_model.h5')
model.save_weights('weights.h5')
