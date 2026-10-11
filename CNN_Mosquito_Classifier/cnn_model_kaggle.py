import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

def load_dataset(dataset_path):
    # Standard YOLO format subdirectories
    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')
    
    # If the Kaggle dataset doesn't use subfolders, look in the root directory
    if not os.path.exists(images_path):
        images_path = dataset_path
        labels_path = dataset_path

    X = []
    Y = []
    
    if not os.path.exists(images_path):
        return np.array(X), np.array(Y)

    for img_name in os.listdir(images_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            
            # Find and check the label file FIRST
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_file = os.path.join(labels_path, label_name)
            
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    content = f.read().strip().split()
                    
                    if content:
                        label = int(content[0])
                        
                        # FILTER: Only keep the image if the label is 0 or 1
                        if label == 0 or label == 1:
                            img_path = os.path.join(images_path, img_name)
                            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                            
                            if img is not None:
                                img = cv2.resize(img, (64, 64))
                                X.append(img)
                                Y.append(label)
                                
    return np.array(X), np.array(Y)

# 1. Download dataset via KaggleHub
print("Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("pradeepisawasan/aedes-mosquitos")
print("Path to dataset files:", dataset_path)

print("Loading data...")

# 2. Check if the Kaggle dataset has pre-split train/test folders
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

if os.path.exists(train_dir) and os.path.exists(test_dir):
    print("Found pre-split train/test folders.")
    X_train, Y_train = load_dataset(train_dir)
    X_test, Y_test = load_dataset(test_dir)
else:
    # If the Kaggle dataset is just one large folder, load all and split manually
    print("No train/test folders found. Loading all images and splitting manually...")
    X_all, Y_all = load_dataset(dataset_path)
    
    if len(X_all) == 0:
        raise ValueError("CRITICAL: No valid image/label pairs found! This Kaggle dataset likely uses class folders (e.g., /Aedes/, /Culex/) instead of YOLO .txt label files.")
        
    # Split 80% for training, 20% for testing
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} images, testing on {len(X_test)} images.")

# 3. Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. Build the CNN Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax') 
])

# 5. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the CNN
print("Training CNN...")
history = model.fit(X_train, Y_train, 
                    epochs=20, 
                    validation_data=(X_test, Y_test))

# 7. Evaluate and Predict
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"CNN Accuracy: {test_acc * 100:.2f}%")

predictions = model.predict(X_test)
y_pred_classes = np.argmax(predictions, axis=1) 

print("\nClassification Report:")
print(classification_report(Y_test, y_pred_classes))