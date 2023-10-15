import tensorflow as tf

model = tf.keras.models.load_model('model_first_.h5')
model.save_weights('weights.h5')
