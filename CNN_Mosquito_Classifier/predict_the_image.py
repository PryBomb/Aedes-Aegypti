import pickle
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load the pickled model
with open('mosquito_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_from_json(model_data['architecture'])
model.set_weights(model_data['weights'])

CLASS_NAMES = ['Aegypti', 'Albopictus']

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension

    pred = model.predict(img)
    class_idx = np.argmax(pred, axis=1)[0]
    confidence = pred[0][class_idx]

    print(f"{img_path} -> {CLASS_NAMES[class_idx]} ({confidence*100:.2f}% confidence)")
    return CLASS_NAMES[class_idx], confidence

# Example usage:
predict_image('Aedes_aegypti.jpg')