import tensorflow as tf

# Convert the model
saved_model_dir = './models/my_model.h5'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('./models/my_model.tflite', 'wb') as f:
  f.write(tflite_model)