# Mosquito Species Classifier (CNN)

A Convolutional Neural Network (CNN) designed to classify mosquito species into two categories: **Aedes aegypti** and **Aedes albopictus**. Built with TensorFlow/Keras and OpenCV, the pipeline automates image loading, preprocessing, model training, and performance evaluation.

---

## Model Evaluation & Performance

Evaluated on a balanced test set of 720 images (360 samples per class), the model achieves an overall classification accuracy of **91%**.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Aedes aegypti** | 0.89 | 0.95 | 0.92 | 360 |
| **Aedes albopictus** | 0.95 | 0.88 | 0.91 | 360 |
| **Accuracy** | | | **0.91** | **720** |
| **Macro Average** | 0.92 | 0.91 | 0.91 | 720 |
| **Weighted Average** | 0.92 | 0.91 | 0.91 | 720 |

### Performance Summary
- **High Sensitivity on Aegypti**: Recall of 95% minimizes false negatives for *Aedes aegypti*.
- **High Specificity on Albopictus**: Precision of 95% provides high confidence when predicting *Aedes albopictus*.
- **Balanced Generalization**: F1-scores of 0.92 and 0.91 show consistent discriminative capability across both species.

---

## Neural Network Architecture

The model uses a sequential convolutional neural network tuned for 64x64 RGB images:

1. **Input Layer**: Shape `(64, 64, 3)` with pixel values normalized to the `[0, 1]` range.
2. **Feature Extraction Block 1**:
   - `Conv2D(32 filters, 3x3 kernel, ReLU activation)`
   - `MaxPooling2D(2x2 pool size)`
3. **Feature Extraction Block 2**:
   - `Conv2D(64 filters, 3x3 kernel, ReLU activation)`
   - `MaxPooling2D(2x2 pool size)`
4. **Feature Extraction Block 3**:
   - `Conv2D(64 filters, 3x3 kernel, ReLU activation)`
5. **Classification Head**:
   - `Flatten()`
   - `Dense(64 units, ReLU activation)`
   - `Dense(2 units, Softmax activation)`
6. **Training Configuration**:
   - Optimizer: `Adam`
   - Loss Function: `sparse_categorical_crossentropy`
   - Metric: `accuracy`
   - Epochs: 20

---

## Project Structure

```text
mosquito classifier/
|-- dataset2/
|   |-- aegypti/
|   |   `-- aegypti/
|   |       |-- aegypti_0/            (Training images)
|   |       |-- aegypti_0_testing/    (Testing images)
|   |       |-- ...
|   `-- albopictus/
|       `-- albopictus/
|           |-- albopictus_0/        (Training images)
|           |-- albopictus_0_testing/(Testing images)
|           `-- ...
|-- cnn_model_local.py                (Data loading, training, and evaluation script)
|-- requirements.txt                  (Python environment dependencies)
`-- README.md                         (Project documentation and run guide)
```

The data loader dynamically assigns:
- Label `0`: Folder names containing `aegypti`
- Label `1`: Folder names containing `albopictus`
- Test split: Folders containing the `_testing` suffix
- Train split: Folders without the `_testing` suffix

---

## Prerequisites

- Python 3.10 or 3.11 installed on your system.
- Hardware: Standard CPU or CUDA-compatible GPU.

---

## Step-by-Step Instructions to Run the Model

### Step 1: Navigate to the Project Root

Open your terminal (PowerShell, Command Prompt, or Bash) and navigate to the project directory:

```bash
cd "f:\codes\mosquito classifier"
```

### Step 2: Create and Activate a Virtual Environment

#### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts running scripts, execute `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.)*

#### On Windows (Command Prompt):
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies

Install the packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Alternatively, install only the core dependencies:
```bash
pip install tensorflow opencv-python numpy scikit-learn
```

### Step 4: Run the Training and Evaluation Script

Execute the training script:

```bash
python cnn_model_local.py
```

### Expected Output Flow:
1. **Scanning Dataset**: The script traverses `dataset2` and prints the total number of loaded training and testing images.
2. **Normalization**: Image pixel values are converted to float and scaled between `0.0` and `1.0`.
3. **Training Phase**: The CNN trains for 20 epochs, displaying loss and accuracy for each epoch along with validation metrics on the test split.
4. **Evaluation & Classification Report**: The script prints final test accuracy followed by precision, recall, and F1-score breakdown for both classes.

---

## Optional: Saving and Exporting the Model

To save the trained model for future inference without retraining, add the following line at the end of `cnn_model_local.py`:

```python
model.save("mosquito_cnn_model.keras")
print("Model saved to mosquito_cnn_model.keras")
```

To load and use the saved model on a new single image:

```python
import cv2
import numpy as np
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model("mosquito_cnn_model.keras")
class_names = ["Aedes aegypti", "Aedes albopictus"]

# Preprocess new image
img = cv2.imread("path_to_image.jpg")
img = cv2.resize(img, (64, 64))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
predictions = model.predict(img)
class_id = np.argmax(predictions[0])
confidence = predictions[0][class_id] * 100

print(f"Prediction: {class_names[class_id]} ({confidence:.2f}% confidence)")
```

---

## Troubleshooting

- **Image loading returns 0 images**: Verify that `dataset2` is located in the root directory alongside `cnn_model_local.py`.
- **OpenCV read error**: Ensure all image files inside the dataset folders have valid `.jpg`, `.jpeg`, or `.png` extensions.
- **Memory limit issues**: If running on lower-memory machines, reduce the image batch size or decrease the target image resolution from `(64, 64)` in `cnn_model_local.py`.
