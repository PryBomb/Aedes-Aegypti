import pickle
import tensorflow as tf
from tensorflow.keras.models import model_from_json

# 1. Load your trained model from the pickle file
print("Loading model from pickle...")
with open('mosquito_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_from_json(model_data['architecture'])
model.set_weights(model_data['weights'])

# 2. Convert it to TensorFlow Lite
print("Converting to TensorFlow Lite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save the model
with open('mosquito_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Successfully converted and saved as 'mosquito_model.tflite'")
