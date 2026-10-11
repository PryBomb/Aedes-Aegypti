"""
setup_dataset.py
----------------
Downloads the Aedes mosquito dataset from Kaggle via kagglehub and
copies images into:
    archive/aegypti/       <- Aedes aegypti images
    archive/albopictus/    <- Aedes albopictus images

Run once before training:
    python setup_dataset.py
"""

import os
import shutil
import kagglehub

# ── 1. Download dataset ────────────────────────────────────────────────────────
print("Downloading dataset from Kaggle (sajid123/aedes-mosquitos-dataset)...")
dataset_path = kagglehub.dataset_download("sajid123/aedes-mosquitos-dataset")
print(f"Dataset downloaded to: {dataset_path}\n")

# ── 2. Print folder tree so we can see what arrived ───────────────────────────
print("Dataset structure:")
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        sub = '  ' * (level + 1)
        for f in files[:5]:
            print(f"{sub}{f}")
        if len(files) > 5:
            print(f"{sub}... ({len(files) - 5} more files)")
print()

# ── 3. Destination folders ────────────────────────────────────────────────────
DEST = {
    'aegypti':    os.path.join('archive', 'aegypti'),
    'albopictus': os.path.join('archive', 'albopictus'),
}
for path in DEST.values():
    os.makedirs(path, exist_ok=True)

# ── 4. Walk the downloaded dataset and copy images to the right folder ─────────
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}
counts = {'aegypti': 0, 'albopictus': 0, 'skipped': 0}

for root, dirs, files in os.walk(dataset_path):
    folder_lower = root.lower()

    if 'aegypti' in folder_lower and 'albopictus' not in folder_lower:
        dest_key = 'aegypti'
    elif 'albopictus' in folder_lower:
        dest_key = 'albopictus'
    else:
        dest_key = None

    for fname in files:
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue

        if dest_key is None:
            fname_lower = fname.lower()
            if 'aegypti' in fname_lower and 'albopictus' not in fname_lower:
                dk = 'aegypti'
            elif 'albopictus' in fname_lower:
                dk = 'albopictus'
            else:
                counts['skipped'] += 1
                continue
        else:
            dk = dest_key

        src = os.path.join(root, fname)
        dst = os.path.join(DEST[dk], fname)

        # Avoid overwriting duplicate filenames
        if os.path.exists(dst):
            base, ext = os.path.splitext(fname)
            counter = 1
            while os.path.exists(dst):
                dst = os.path.join(DEST[dk], f"{base}_{counter}{ext}")
                counter += 1

        shutil.copy2(src, dst)
        counts[dk] += 1

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("=" * 50)
print("Dataset setup complete!")
print(f"  archive/aegypti/     : {counts['aegypti']} images")
print(f"  archive/albopictus/  : {counts['albopictus']} images")
print(f"  Skipped (no match)   : {counts['skipped']} files")
print("=" * 50)
print("\nYou can now run:  python cnn_model_local.py")
