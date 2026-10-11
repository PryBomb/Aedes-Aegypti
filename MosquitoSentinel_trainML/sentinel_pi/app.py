"""
sentinel_pi/app.py
──────────────────
FastAPI web server for the Raspberry Pi 5 live detection dashboard.

Endpoints:
  GET  /              → serves the live dashboard HTML
  GET  /video         → MJPEG stream (annotated camera feed)
  GET  /api/stats     → JSON detection statistics
  WS   /ws            → WebSocket push of detection events (new detections, FPS)

Usage (on the Raspberry Pi):
    source ~/mosquito_env/bin/activate
    cd ~/mosquito
    uvicorn sentinel_pi.app:app --host 0.0.0.0 --port 8000

Then open http://<pi-ip>:8000 from any browser on your network.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from sentinel_pi import detect as detector_module
from sentinel_pi.detect import start_detector, get_detector, DEFAULT_MODEL_PATH

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App setup ─────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="MosquitoSentinel — Live Detection", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─── WebSocket manager ─────────────────────────────────────────────────────────
class WsManager:
    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WsManager()


# ─── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("Starting MosquitoSentinel detection engine…")
    try:
        start_detector(model_path=DEFAULT_MODEL_PATH, use_usb_camera=False)
        logger.info("Detection engine started ✓")
    except FileNotFoundError as e:
        logger.error(str(e))

    # Background task: broadcast stats to WS clients every second
    asyncio.create_task(broadcast_stats_loop())


async def broadcast_stats_loop():
    last_count = 0
    while True:
        await asyncio.sleep(1)
        if not ws_manager.clients:
            continue
        det = get_detector()
        if det is None:
            continue
        stats = det.get_stats()
        payload = {
            "type": "stats",
            "fps": stats.fps,
            "total": stats.total_detections,
            "counts": stats.counts,
            "timestamp": stats.timestamp,
        }
        # Emit a separate "alert" event if new detections arrived
        if stats.total_detections > last_count and stats.last_detections:
            payload["new_detections"] = stats.last_detections
            payload["type"] = "detection"
        last_count = stats.total_detections
        await ws_manager.broadcast(payload)


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


async def _mjpeg_generator() -> AsyncGenerator[bytes, None]:
    """Generate MJPEG frames from the detector's frame queue."""
    det = get_detector()
    if det is None:
        # Send a placeholder frame if detector isn't ready
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n"
        return

    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        try:
            frame_bytes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: det.frame_queue.get(timeout=2.0)
            )
            yield boundary + frame_bytes + b"\r\n"
        except Exception:
            await asyncio.sleep(0.05)


@app.get("/video")
async def video_stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stats")
async def get_stats():
    det = get_detector()
    if det is None:
        return JSONResponse({"error": "Detector not running"}, status_code=503)
    stats = det.get_stats()
    return JSONResponse({
        "fps":   stats.fps,
        "total": stats.total_detections,
        "counts": stats.counts,
        "last":  stats.last_detections,
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Keep alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
