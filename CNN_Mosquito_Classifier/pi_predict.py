import cv2
import numpy as np
import argparse
import time
import os

# Try importing the lightweight tflite_runtime first (preferred for Raspberry Pi)
# Fallback to full tensorflow if running on a PC
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

CLASS_NAMES = ['Aegypti', 'Albopictus']

def main():
    parser = argparse.ArgumentParser(description="Mosquito Classifier for Raspberry Pi")
    parser.add_argument('--image', type=str, required=True, help="Path to the mosquito image")
    parser.add_argument('--model', type=str, default="mosquito_model.tflite", help="Path to TFLite model")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        return

    # 1. Load the TFLite model
    print(f"Loading model '{args.model}'...")
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Prepare the image (Resize to 64x64, normalize, add batch dimension)
    start_time = time.time()
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_data = np.expand_dims(img_normalized, axis=0)

    # 3. Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    inference_time = (time.time() - start_time) * 1000

    # 4. Process results
    class_idx = np.argmax(output_data[0])
    confidence = output_data[0][class_idx] * 100

    print("-" * 30)
    print(f"Prediction : {CLASS_NAMES[class_idx]}")
    print(f"Confidence : {confidence:.2f}%")
    print(f"Infer Time : {inference_time:.1f} ms")
    print("-" * 30)

if __name__ == "__main__":
    main()
