import tensorflow as tf

model = tf.keras.models.load_model('model_small-std-0.95-dropNone-lr0.0001-regNone.h5')
model.save_weights('weights.h5')
