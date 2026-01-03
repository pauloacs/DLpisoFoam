import tensorflow as tf

model = tf.keras.models.load_model('model_MLP_small-std-drop0.1-lr0.0005-regNone-batch1024.h5')
model.save_weights('weights.h5')
