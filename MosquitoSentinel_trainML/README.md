# 🦟 MosquitoSentinel — YOLOv11 Mosquito Detection

End-to-end mosquito detection pipeline:
**Train on Windows → Monitor via web UI → Deploy to Raspberry Pi 5 for real-time camera detection.**

Detects two mosquito species: **Aedes Aegypti** (Dengue / Zika vector) and **Aedes Albopictus** (Tiger mosquito).

---

## 📁 Project Structure

```
MosquitoSentinel_trainML/
├── dataset/                    # Roboflow-exported dataset
│   ├── data.yaml               # Dataset configuration
│   ├── train/                  # Training images & labels
│   ├── valid/                  # Validation images & labels
│   └── test/                   # Test images & labels
│
├── train.py                    # YOLOv11 training script
├── predict.py                  # Inference script (test images)
├── requirements.txt            # Windows dependencies
│
├── sentinel_ui/                # 🖥️  Windows training monitor UI
│   ├── app.py                  # FastAPI + WebSocket server
│   └── templates/index.html   # Dark-mode training dashboard
│
└── sentinel_pi/                # 🍓  Raspberry Pi 5 deployment
    ├── export_model.py         # Export best.pt → OpenVINO (run on PC)
    ├── detect.py               # Picamera2 + YOLO inference engine
    ├── app.py                  # FastAPI MJPEG stream server
    ├── templates/dashboard.html # Live detection dashboard
    ├── requirements_pi.txt     # Pi-specific dependencies
    ├── setup_pi.sh             # Automated Pi setup script
    ├── run.sh                  # Start the dashboard
    └── sentinel_pi.service     # systemd autostart service
```

---

## 🚀 Windows Setup

### 1. Activate Virtual Environment

```powershell
# PowerShell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training YOLOv11

```bash
# Train for 50 epochs and export to OpenVINO for Pi deployment
python train.py --epochs 50 --imgsz 640 --batch 8 --export
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 10 | Training epochs |
| `--imgsz` | 640 | Training image size |
| `--batch` | 8 | Batch size |
| `--model` | yolo11n.pt | YOLO model variant |
| `--export` | false | Export to OpenVINO after training |
| `--export-imgsz` | 320 | Export image size (smaller = faster on Pi) |

Best weights saved to: `runs/detect/mosquito_yolo11/weights/best.pt`
OpenVINO model saved to: `runs/detect/mosquito_yolo11/weights/best_openvino_model/`

---

## 🖥️ Training Monitor UI (Windows)

Launch the live training dashboard in your browser:

```bash
uvicorn sentinel_ui.app:app --reload --port 8000
```

Open **http://localhost:8000** — you'll see:
- Live mAP & loss charts updating each epoch
- Epoch history table
- WebSocket log feed
- Detection image gallery

---

## 🔍 Model Inference (Test Images)

```bash
python predict.py --weights runs/detect/mosquito_yolo11/weights/best.pt \
                  --source dataset/test/images --conf 0.25
```

Annotated output saved to `runs/detect/predict/`.

---

## 🍓 Raspberry Pi 5 Deployment

### Step 1 — Export Model (on your Windows PC)

```bash
python sentinel_pi/export_model.py
# OR use the --export flag when training (see above)
```

### Step 2 — Copy files to the Pi

```bash
# Replace <pi-ip> with your Pi's IP address
scp -r sentinel_pi/ pi@<pi-ip>:~/mosquito/
scp -r runs/detect/mosquito_yolo11/weights/best_openvino_model pi@<pi-ip>:~/mosquito/
```

### Step 3 — Set up the Pi (run on the Pi)

```bash
cd ~/mosquito
chmod +x sentinel_pi/setup_pi.sh
./sentinel_pi/setup_pi.sh
```

### Step 4 — Start the dashboard

```bash
./sentinel_pi/run.sh
```

Open **http://\<pi-ip\>:8000** from any browser on your network.

### Optional — Autostart on boot

```bash
sudo cp sentinel_pi/sentinel_pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sentinel_pi
sudo systemctl start sentinel_pi
```

---

## ⚡ Performance Notes

| Mode | Hardware | Expected FPS |
|------|----------|-------------|
| PyTorch (best.pt) | Pi 5 CPU | 2–5 FPS |
| **OpenVINO (best_openvino_model)** | **Pi 5 CPU** | **10–20 FPS** |
| PyTorch | Windows CPU | 5–15 FPS |

> **Always use OpenVINO format on the Pi** — it's ~3x faster than raw PyTorch on ARM CPUs.

---

## 📁 Repository Structure

```
MosquitoSentinel_trainML/
├── dataset/
│   ├── data.yaml           # Dataset configuration file
│   ├── train/              # Training images & labels
│   ├── valid/              # Validation images & labels
│   └── test/               # Test images & labels
├── .venv/                  # Python 3.11 virtual environment
├── requirements.txt        # Python package dependencies
├── train.py                # Model training script
├── predict.py              # Model inference/prediction script
└── README.md               # Project documentation
```

---

## 🚀 Environment Setup

### 1. Virtual Environment Activation

**PowerShell (Windows):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt (cmd):**
```cmd
.\.venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 🏋️ Training YOLOv11

To train the YOLOv11 nano model on the mosquito dataset:

```bash
python train.py --epochs 10 --imgsz 640 --batch 8
```

### Key Training Options

- `--epochs`: Number of training epochs (default: `10`).
- `--imgsz`: Input image size (default: `640`).
- `--batch`: Batch size (default: `8`).
- `--model`: Base YOLO model checkpoint, e.g., `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` (default: `yolo11n.pt`).
- `--data`: Path to `data.yaml` (default: `dataset/data.yaml`).

Trained weights and metrics plots (confusion matrix, PR curve, loss graphs) are automatically saved to:
`runs/detect/mosquito_yolo11/weights/best.pt`

---

## 🔍 Model Inference & Testing

To test your trained model on unseen test images:

```bash
python predict.py --weights runs/detect/mosquito_yolo11/weights/best.pt --source dataset/test/images --conf 0.25
```

Annotated output images with bounding boxes and confidence scores will be saved in `runs/detect/predict/`.
