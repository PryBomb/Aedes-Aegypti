#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# MosquitoSentinel — Start Script (run on Raspberry Pi)
# Usage: ./sentinel_pi/run.sh
# ─────────────────────────────────────────────────────────────────────

VENV_DIR="$HOME/mosquito_env"
INSTALL_DIR="$HOME/mosquito"
PORT=8000

# Activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "   Run setup_pi.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Check model exists
if [ ! -d "$INSTALL_DIR/best_openvino_model" ]; then
    echo "⚠️  Warning: OpenVINO model not found at $INSTALL_DIR/best_openvino_model"
    echo "   Run export_model.py on your PC and copy the folder here."
fi

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        MosquitoSentinel — Live Detection             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  Dashboard : http://${LOCAL_IP}:${PORT}"
echo "  Video feed: http://${LOCAL_IP}:${PORT}/video"
echo "  Stats API : http://${LOCAL_IP}:${PORT}/api/stats"
echo ""
echo "  Open the dashboard URL from any device on your network."
echo "  Press Ctrl+C to stop."
echo ""

cd "$INSTALL_DIR"
uvicorn sentinel_pi.app:app --host 0.0.0.0 --port "$PORT" --workers 1
