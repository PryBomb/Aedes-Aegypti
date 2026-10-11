"""Backend-agnostic YOLO detector wrapper.

Supports pt / ncnn / onnx / openvino backends via Ultralytics.
detect() accepts a BGR numpy array, measures inference time, and returns
a list of Detection objects. No colour conversion is applied — Ultralytics
handles BGR→RGB internally when it needs to.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import of ultralytics — allows tests with a fake backend to run without
# ultralytics installed, and avoids importing torch at module level.
_ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO as _YOLO  # type: ignore[import]
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _YOLO = None  # type: ignore[assignment]


@dataclass
class Detection:
    """A single bounding-box detection from YOLO.

    Attributes:
        xyxy: (x1, y1, x2, y2) in pixel coordinates.
        conf: Detection confidence in [0, 1].
        class_id: Integer class index from the model.
        class_name: String class name from the model's names dict.
        is_target: True if class_name matches target_class_names, False if non-target.
        infer_ms: Inference time for this frame in milliseconds.
    """
    xyxy: tuple[float, float, float, float]
    conf: float
    class_id: int
    class_name: str
    is_target: bool = True
    infer_ms: float = 0.0


class Detector:
    """Backend-agnostic YOLO11 inference wrapper.

    Args:
        weights_path: Path to model weights (.pt, .onnx, _openvino_model/, etc.).
        backend: One of 'pt', 'ncnn', 'onnx', 'openvino'. Used for logging/docs;
                 Ultralytics auto-detects from the file extension.
        imgsz: Inference image size.
        conf: Confidence threshold.
        iou: NMS IoU threshold.
        target_class_names: List of class names considered target (default: ["Aedes Aegypti"]).
        non_target_class_names: List of class names considered non-target (default: ["Aedes Albopictus"]).
        warmup_runs: Number of dummy inferences to warm up the model before timing.
        device: Inference device ('' = auto, 'cpu', '0' for CUDA).
    """

    def __init__(
        self,
        weights_path: str,
        backend: str = "pt",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.5,
        target_class_names: Optional[List[str]] = None,
        non_target_class_names: Optional[List[str]] = None,
        warmup_runs: int = 2,
        device: str = "",
    ) -> None:
        self._weights_path = weights_path
        self._backend = backend
        self._imgsz = imgsz
        self._conf = conf
        self._iou = iou
        self._target_class_names = [
            n.lower() for n in (target_class_names if target_class_names is not None else ["Aedes Aegypti"])
        ]
        self._non_target_class_names = [
            n.lower() for n in (non_target_class_names if non_target_class_names is not None else ["Aedes Albopictus"])
        ]
        self._warmup_runs = warmup_runs
        self._device = device
        self._model = None
        self._class_names: dict[int, str] = {}

    def load(self) -> None:
        """Load the model and run warm-up inferences to prime the runtime."""
        if not _ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        logger.info("Loading YOLO model: %s (backend=%s, imgsz=%d)",
                    self._weights_path, self._backend, self._imgsz)
        self._model = _YOLO(self._weights_path)

        # Read class names from the model itself — never from hard-coded strings
        self._class_names = self._model.names  # {int: str}

        # Validate configured names case-insensitively against model names
        model_names_lower = {v.lower(): v for v in self._class_names.values()}
        for configured_name in self._target_class_names:
            if configured_name not in model_names_lower:
                raise ValueError(
                    f"Configured target class '{configured_name}' not found in model's class names: "
                    f"{list(self._class_names.values())}"
                )
        for configured_name in self._non_target_class_names:
            if configured_name not in model_names_lower:
                raise ValueError(
                    f"Configured non-target class '{configured_name}' not found in model's class names: "
                    f"{list(self._class_names.values())}"
                )

        logger.info(
            "YOLO classes validated: target=%s, non_target=%s",
            self._target_class_names, self._non_target_class_names,
        )

        # Warm-up: send dummy frames so JIT / OpenVINO / NCNN runtimes are ready
        if self._warmup_runs > 0:
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            logger.debug("Running %d warm-up inference(s)…", self._warmup_runs)
            for _ in range(self._warmup_runs):
                self._model.predict(
                    dummy,
                    imgsz=self._imgsz,
                    conf=self._conf,
                    iou=self._iou,
                    device=self._device,
                    verbose=False,
                )
            logger.info("YOLO warm-up complete.")

    def detect(self, frame_bgr: np.ndarray) -> tuple[List[Detection], float]:
        """Run inference on a single BGR frame.

        No colour conversion is applied here. Ultralytics internally converts
        BGR→RGB as part of its preprocessing pipeline.

        Returns all detected classes with their is_target flag set.
        """
        if self._model is None:
            raise RuntimeError("Detector.load() must be called before detect().")

        t0 = time.perf_counter()
        results = self._model.predict(
            frame_bgr,
            imgsz=self._imgsz,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self._class_names.get(cls_id, str(cls_id))
                conf = float(boxes.conf[i].item())

                is_target = cls_name.lower() in self._target_class_names
                xyxy = tuple(float(v) for v in boxes.xyxy[i].tolist())
                detections.append(Detection(
                    xyxy=xyxy,           # type: ignore[arg-type]
                    conf=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    is_target=is_target,
                    infer_ms=infer_ms,
                ))

        return detections, infer_ms

    @property
    def class_names(self) -> dict[int, str]:
        """Class names loaded from the model."""
        return self._class_names
