"""
sentinel_ui/app.py
──────────────────
Windows-side FastAPI server for monitoring YOLOv11 training progress.

Features:
  * WebSocket /ws         -- streams live training log lines as JSON events
  * GET /                 -- serves the dashboard HTML
  * GET /api/runs         -- lists available training runs
  * GET /api/metrics/{run} -- returns parsed metrics CSV for a run
  * GET /api/images/{run_name} -- returns base64 gallery of annotated images
  * GET /api/status       -- returns current training status

Usage (from MosquitoSentinel_trainML root):
  .venv\\Scripts\\uvicorn.exe sentinel_ui.app:app --reload --port 8000
"""

import asyncio
import json
import csv
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
RUNS_DIR      = Path(__file__).parent.parent / "runs" / "detect"

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def find_latest_log() -> Optional[Path]:
    """Return the results.csv of the most recently modified training run."""
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(
        (p for p in RUNS_DIR.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        log = c / "results.csv"
        if log.exists():
            return log
    return None


def parse_metrics_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Parse YOLO results.csv into a list of epoch dicts (keys stripped)."""
    rows: List[Dict[str, Any]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k.strip(): v.strip() for k, v in row.items()})
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# Background polling task (started in lifespan)
# ---------------------------------------------------------------------------

async def tail_metrics_loop():
    """Poll results.csv every 2 s and push new epoch rows via WebSocket."""
    last_row_count = 0
    last_run: Optional[str] = None

    while True:
        await asyncio.sleep(2)
        if not manager.active:
            continue

        log_path = find_latest_log()
        if log_path is None:
            continue

        current_run = log_path.parent.name
        if current_run != last_run:
            last_run = current_run
            last_row_count = 0
            await manager.broadcast({"type": "run_start", "run": current_run})

        rows = parse_metrics_csv(log_path)
        if len(rows) > last_row_count:
            for row in rows[last_row_count:]:
                await manager.broadcast({"type": "epoch", "data": row})
            last_row_count = len(rows)

            if rows:
                latest = rows[-1]
                await manager.broadcast({
                    "type": "metrics_update",
                    "epoch":    latest.get("epoch", "?"),
                    "box_loss": latest.get("train/box_loss", ""),
                    "cls_loss": latest.get("train/cls_loss", ""),
                    "mAP50":    latest.get("metrics/mAP50(B)", ""),
                    "mAP50_95": latest.get("metrics/mAP50-95(B)", ""),
                })


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- startup ----------
    asyncio.create_task(tail_metrics_loop())
    print("[OK] MosquitoSentinel Training Monitor started.")
    print("     Open -> http://localhost:8000")
    yield
    # ---------- shutdown ----------
    # nothing to clean up


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MosquitoSentinel Training Monitor",
    version="1.0.0",
    lifespan=lifespan,
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/runs")
async def list_runs():
    if not RUNS_DIR.exists():
        return JSONResponse({"runs": []})
    runs = [
        d.name
        for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if d.is_dir()
    ]
    return JSONResponse({"runs": runs})


@app.get("/api/metrics/{run_name}")
async def get_metrics(run_name: str):
    csv_path = RUNS_DIR / run_name / "results.csv"
    if not csv_path.exists():
        return JSONResponse({"rows": []})
    return JSONResponse({"rows": parse_metrics_csv(csv_path)})


@app.get("/api/images/{run_name}")
async def get_images(run_name: str):
    """Return up to 12 base64-encoded prediction images from a run directory."""
    pred_dir = RUNS_DIR / run_name
    if not pred_dir.exists():
        return JSONResponse({"images": []})

    exts = {".jpg", ".jpeg", ".png"}
    images = []
    for img in sorted(pred_dir.iterdir()):
        if img.suffix.lower() in exts and img.is_file():
            try:
                encoded = base64.b64encode(img.read_bytes()).decode()
                images.append({
                    "name": img.name,
                    "data": f"data:image/jpeg;base64,{encoded}",
                })
                if len(images) >= 12:
                    break
            except Exception:
                pass
    return JSONResponse({"images": images})


@app.get("/api/status")
async def get_status():
    lock       = RUNS_DIR.parent.parent / ".training_active"
    latest_log = find_latest_log()
    epochs_done = len(parse_metrics_csv(latest_log)) if latest_log else 0
    return JSONResponse({
        "training_active": lock.exists(),
        "epochs_done":     epochs_done,
        "latest_run":      latest_log.parent.name if latest_log else None,
    })


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Push full history on connect so the chart populates immediately
        log_path = find_latest_log()
        if log_path:
            await websocket.send_text(json.dumps({
                "type": "history",
                "run":  log_path.parent.name,
                "rows": parse_metrics_csv(log_path),
            }))
        # Keep connection alive until client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
