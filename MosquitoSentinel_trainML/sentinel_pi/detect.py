"""
sentinel_pi/detect.py
─────────────────────
Real-time mosquito detection engine for Raspberry Pi 5.

  • Captures frames from the official Camera Module via picamera2
  • Runs YOLOv11 inference using the OpenVINO-exported model
  • Annotates frames with bounding boxes and class labels
  • Shares annotated frames and detection stats via thread-safe queues

This module is imported by app.py; do not run it directly.
"""

import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = Path(__file__).parent / "best_openvino_model"
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
INFERENCE_SIZE = 320        # smaller = faster; increase for accuracy
CONF_THRESHOLD = 0.35
MAX_QUEUE_SIZE = 2          # keep only the freshest frames

# Class colour palette (BGR for OpenCV)
CLASS_COLORS = {
    "Aedes Aegypti":     (0, 220, 255),   # yellow-cyan
    "Aedes Albopictus":  (180, 60, 255),  # purple
}
DEFAULT_COLOR = (0, 200, 100)


# ─── Detection result ──────────────────────────────────────────────────────────
@dataclass
class DetectionStats:
    fps: float = 0.0
    total_detections: int = 0
    counts: dict = field(default_factory=dict)
    last_detections: list = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ─── Detector class ────────────────────────────────────────────────────────────
class MosquitoDetector:
    """
    Runs continuously in a background thread.
    Consumers read from `frame_queue` and `stats_queue`.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, use_usb_camera: bool = False):
        self.model_path = model_path
        self.use_usb_camera = use_usb_camera
        self.running = False

        # Shared state
        self.frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.stats_queue: queue.Queue[DetectionStats] = queue.Queue(maxsize=4)
        self._stats = DetectionStats()
        self._stats_lock = threading.Lock()

        # Load model
        logger.info(f"Loading model from: {model_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                "Run export_model.py on your PC and copy the folder to the Pi."
            )
        self.model = YOLO(str(model_path))
        logger.info("Model loaded ✓")

    # ── Camera initialisation ─────────────────────────────────────────────────
    def _init_picamera2(self):
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(0.5)  # warm-up
            logger.info(f"Picamera2 started ({FRAME_WIDTH}×{FRAME_HEIGHT})")
            return picam2, "picam"
        except Exception as e:
            logger.warning(f"Picamera2 failed ({e}), falling back to USB camera")
            return None, "fail"

    def _init_usb_camera(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise RuntimeError("Cannot open USB camera (device 0)")
        logger.info("USB camera opened")
        return cap, "usb"

    # ── Annotation helpers ────────────────────────────────────────────────────
    def _annotate(self, frame: np.ndarray, results) -> tuple[np.ndarray, list]:
        """Draw bounding boxes and labels onto a frame, return annotated frame + detections."""
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = self.model.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS.get(label, DEFAULT_COLOR)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                text = f"{label}  {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                detections.append({"label": label, "conf": round(conf, 3),
                                   "box": [x1, y1, x2, y2]})
        return frame, detections

    def _overlay_info(self, frame: np.ndarray, fps: float, count: int) -> np.ndarray:
        """Draw FPS / count overlay on frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (220, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.putText(frame, f"Detections: {count}", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return frame

    # ── Main inference loop ───────────────────────────────────────────────────
    def run(self):
        self.running = True

        # Initialise camera
        if self.use_usb_camera:
            cam, cam_type = self._init_usb_camera()
        else:
            cam, cam_type = self._init_picamera2()
            if cam_type == "fail":
                cam, cam_type = self._init_usb_camera()

        fps_counter = 0
        fps_start   = time.time()
        current_fps = 0.0
        counts: dict[str, int] = {}

        try:
            while self.running:
                t0 = time.time()

                # Capture frame
                if cam_type == "picam":
                    frame = cam.capture_array()
                    # Picamera2 returns RGB, convert to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ok, frame = cam.read()
                    if not ok:
                        logger.warning("Frame read failed, skipping")
                        time.sleep(0.05)
                        continue

                # Resize for inference
                inf_frame = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))

                # Run inference
                results = self.model.predict(
                    source=inf_frame,
                    conf=CONF_THRESHOLD,
                    verbose=False,
                    imgsz=INFERENCE_SIZE,
                )

                # Annotate on full-resolution frame
                annotated, detections = self._annotate(frame.copy(), results)

                # Update counts
                for d in detections:
                    counts[d["label"]] = counts.get(d["label"], 0) + 1

                # FPS
                fps_counter += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start   = time.time()

                # Overlay
                self._overlay_info(annotated, current_fps, len(detections))

                # Encode to JPEG
                ok, jpeg = cv2.imencode(".jpg", annotated,
                                        [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    continue

                # Push to queues (drop oldest if full)
                if self.frame_queue.full():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: pass
                self.frame_queue.put_nowait(jpeg.tobytes())

                # Update stats
                with self._stats_lock:
                    self._stats = DetectionStats(
                        fps=round(current_fps, 1),
                        total_detections=sum(counts.values()),
                        counts=dict(counts),
                        last_detections=detections[-5:],   # last 5
                        timestamp=time.time(),
                    )

        finally:
            self.running = False
            if cam_type == "picam":
                cam.close()
            else:
                cam.release()
            logger.info("Detector stopped")

    def get_stats(self) -> DetectionStats:
        with self._stats_lock:
            return self._stats

    def stop(self):
        self.running = False


# ─── Module-level singleton ────────────────────────────────────────────────────
_detector: Optional[MosquitoDetector] = None
_thread:   Optional[threading.Thread] = None


def start_detector(model_path: Path = DEFAULT_MODEL_PATH,
                   use_usb_camera: bool = False) -> MosquitoDetector:
    global _detector, _thread
    _detector = MosquitoDetector(model_path=model_path, use_usb_camera=use_usb_camera)
    _thread = threading.Thread(target=_detector.run, daemon=True, name="detector")
    _thread.start()
    logger.info("Detector thread started")
    return _detector


def get_detector() -> Optional[MosquitoDetector]:
    return _detector
