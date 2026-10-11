#!/bin/bash
# ╔══════════════════════════════════════════════════════════╗
# ║   MosquitoSentinel — Raspberry Pi 5 Setup Script        ║
# ║   Run this script ONCE on the Pi after copying files.   ║
# ╚══════════════════════════════════════════════════════════╝
# Usage:
#   chmod +x setup_pi.sh
#   ./setup_pi.sh

set -e  # Exit on any error

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        MosquitoSentinel Pi 5 — Setup Script         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

INSTALL_DIR="$HOME/mosquito"
VENV_DIR="$HOME/mosquito_env"

# ── 1. System packages ────────────────────────────────────────────────────────
echo "[1/5] Installing system packages…"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    libcap-dev \
    libatlas-base-dev \
    libopenjp2-7 \
    libtiff5 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libhdf5-dev \
    git \
    wget

echo "    ✓ System packages installed"

# ── 2. Virtual environment ────────────────────────────────────────────────────
echo "[2/5] Creating Python virtual environment at $VENV_DIR…"
python3 -m venv --system-site-packages "$VENV_DIR"
# NOTE: --system-site-packages is needed to access picamera2 (system package)
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
echo "    ✓ Virtual environment ready"

# ── 3. PyTorch ARM64 ─────────────────────────────────────────────────────────
echo "[3/5] Installing PyTorch for ARM64 (aarch64)…"
echo "    This may take 5–15 minutes depending on your internet speed."
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    --quiet
echo "    ✓ PyTorch installed"

# ── 4. Project dependencies ───────────────────────────────────────────────────
echo "[4/5] Installing project dependencies…"
pip install -r "$INSTALL_DIR/sentinel_pi/requirements_pi.txt" --quiet
echo "    ✓ Project dependencies installed"

# ── 5. Verify installation ────────────────────────────────────────────────────
echo "[5/5] Verifying installation…"
python3 -c "import torch; print(f'    PyTorch: {torch.__version__}')"
python3 -c "import ultralytics; print(f'    Ultralytics: {ultralytics.__version__}')"
python3 -c "import cv2; print(f'    OpenCV: {cv2.__version__}')"
python3 -c "import fastapi; print(f'    FastAPI: {fastapi.__version__}')"

# Test picamera2 (system package via --system-site-packages)
python3 -c "from picamera2 import Picamera2; print('    Picamera2: OK')" 2>/dev/null || \
    echo "    Picamera2: Not found (check camera is connected and enabled via raspi-config)"

echo ""
echo "══════════════════════════════════════════════════════"
echo "✅  Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy your exported model:"
echo "     scp -r <your-pc>:<path>/best_openvino_model  ~/mosquito/"
echo ""
echo "  2. Start the dashboard:"
echo "     source ~/mosquito_env/bin/activate"
echo "     cd ~/mosquito"
echo "     ./sentinel_pi/run.sh"
echo ""
echo "  3. Open in browser (from any device on your network):"
echo "     http://$(hostname -I | awk '{print $1}'):8000"
echo "══════════════════════════════════════════════════════"
echo ""
