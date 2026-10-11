import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Dataset path ───────────────────────────────────────────────────────────────
DATASET_PATH = r'C:\Users\LAPTOP\Downloads\archive'
CACHE_FILE   = 'dataset_cache.npz'   # saved next to this script for fast reload
IMG_SIZE     = (64, 64)

# ── 1. Scan all image paths and labels (fast — no I/O per image yet) ───────────
def scan_paths(base_path):
    """Returns two lists: (img_paths, labels, is_test_flags)"""
    paths, labels, is_test = [], [], []
    for root, dirs, files in os.walk(base_path):
        folder_lower = root.lower()
        if 'aegypti' in folder_lower and 'albopictus' not in folder_lower:
            label = 0
        elif 'albopictus' in folder_lower:
            label = 1
        else:
            continue
        test_flag = '_testing' in folder_lower
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(root, f))
                labels.append(label)
                is_test.append(test_flag)
    return paths, labels, is_test

# ── 2. Load a single image (used by thread pool) ───────────────────────────────
def load_image(args):
    path, label, is_test = args
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    return img, label, is_test

# ── 3. Parallel dataset loader with .npz cache ────────────────────────────────
def load_dataset(base_path, cache_file=CACHE_FILE, workers=8):
    # --- Fast path: load from cache if available ---
    if os.path.exists(cache_file):
        print(f"Loading dataset from cache: {cache_file}")
        data = np.load(cache_file)
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test  = data['X_test']
        Y_test  = data['Y_test']
        print(f"  Loaded {len(X_train)} train / {len(X_test)} test images from cache.")
        return X_train, Y_train, X_test, Y_test

    # --- Slow path: scan + parallel read ---
    print(f"Scanning {base_path} for images...")
    paths, labels, is_test_flags = scan_paths(base_path)
    total = len(paths)
    print(f"Found {total} images. Loading in parallel with {workers} workers...")

    X_train, Y_train = [], []
    X_test,  Y_test  = [], []
    done = 0

    args_list = list(zip(paths, labels, is_test_flags))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(load_image, a): a for a in args_list}
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  {done}/{total} images loaded...", flush=True)
            if result is None:
                continue
            img, label, test_flag = result
            if test_flag:
                X_test.append(img);  Y_test.append(label)
            else:
                X_train.append(img); Y_train.append(label)

    # Build properly shaped arrays
    X_train_arr = np.array(X_train, dtype=np.float32).reshape(-1, 64, 64, 3) if X_train else np.empty((0, 64, 64, 3), dtype=np.float32)
    X_test_arr  = np.array(X_test,  dtype=np.float32).reshape(-1, 64, 64, 3) if X_test  else np.empty((0, 64, 64, 3), dtype=np.float32)
    Y_train_arr = np.array(Y_train, dtype=np.int32)
    Y_test_arr  = np.array(Y_test,  dtype=np.int32)

    # Save cache for next run
    print(f"Saving cache to {cache_file}...")
    np.savez_compressed(cache_file,
                        X_train=X_train_arr, Y_train=Y_train_arr,
                        X_test=X_test_arr,   Y_test=Y_test_arr)
    print("Cache saved.")

    return X_train_arr, Y_train_arr, X_test_arr, Y_test_arr

# ── 1. Load the data ───────────────────────────────────────────────────────────
X_train, Y_train, X_test, Y_test = load_dataset(DATASET_PATH)

print(f"Loaded {len(X_train)} training images.")
print(f"Loaded {len(X_test)} testing images.")

# Guard: stop early if no images were found
if len(X_train) == 0:
    raise ValueError(
        f"No training images found in '{DATASET_PATH}'.\n"
        "Make sure the folder contains 'aegypti' and 'albopictus' subfolders with .jpg/.png images."
    )

# ── 2. Normalize pixel values (0-1) ───────────────────────────────────────────
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# ── 3. Build the CNN Architecture ─────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),

    # 2 Output classes (Aegypti and Albopictus)
    layers.Dense(2, activation='softmax')
])

# ── 4. Compile the model ───────────────────────────────────────────────────────
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ── 5. Train the CNN ──────────────────────────────────────────────────────────
print("\nTraining CNN...")
history = model.fit(X_train, Y_train,
                    epochs=20,
                    validation_data=(X_test, Y_test))

# ── 6. Evaluate and Predict ───────────────────────────────────────────────────
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print(f"CNN Accuracy: {test_acc * 100:.2f}%")

predictions = model.predict(X_test)
y_pred_classes = np.argmax(predictions, axis=1)

print("\nClassification Report:")
print(classification_report(Y_test, y_pred_classes, target_names=['Aegypti', 'Albopictus']))

# ── 7. Save the model as a .pkl file (architecture + weights) ─────────────────
model_data = {
    'architecture': model.to_json(),
    'weights': model.get_weights()
}

with open('mosquito_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved to mosquito_model.pkl")