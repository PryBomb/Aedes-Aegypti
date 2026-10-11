"""Picamera2 CSI frame source — Pi-only.

Picamera2 configured with format='RGB888'.
Per the Picamera2 manual, RGB888 stores bytes in B, G, R order in memory,
so capture_array() already returns a BGR-ordered array.
No cv2.cvtColor is applied — the array passes downstream as-is.

Import fails gracefully on non-Pi hardware; the error is only raised
at runtime if mode == 'pi'.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from sentinel.capture.base import Frame, FrameSource

logger = logging.getLogger(__name__)

_PICAMERA2_AVAILABLE = False
try:
    from picamera2 import Picamera2  # type: ignore[import]
    _PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None  # type: ignore[assignment,misc]


class PiCamera(FrameSource):
    """Frame source using the Raspberry Pi CSI camera via Picamera2.

    The camera is configured with format='RGB888'. Per the Picamera2 manual,
    bytes are stored as B, G, R — i.e. the array is already BGR. No colour
    conversion is performed anywhere in this class.

    Args:
        width: Capture width in pixels.
        height: Capture height in pixels.
        fps: Target frames per second.
        max_retries: How many times to retry opening the camera before giving up.
        retry_delay_s: Initial delay between retries (doubles on each attempt).
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        max_retries: int = 5,
        retry_delay_s: float = 1.0,
    ) -> None:
        if not _PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "picamera2 is not installed. It is only available on Raspberry Pi OS. "
                "Set mode: usb or mode: file in config.yaml to develop on Windows."
            )
        self._width = width
        self._height = height
        self._fps = fps
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._cam: Optional["Picamera2"] = None
        self._frame_id: int = 0
        self._camera_ok: bool = False

    # ── FrameSource interface ──────────────────────────────────────────────

    def start(self) -> None:
        """Open the camera with exponential-backoff retry."""
        delay = self._retry_delay_s
        for attempt in range(1, self._max_retries + 1):
            try:
                self._cam = Picamera2()
                config = self._cam.create_preview_configuration(
                    main={"format": "RGB888", "size": (self._width, self._height)},
                    controls={"FrameRate": float(self._fps)},
                )
                self._cam.configure(config)
                self._cam.start()
                # Discard first few frames while the sensor stabilises
                for _ in range(3):
                    self._cam.capture_array()
                self._camera_ok = True
                logger.info(
                    "PiCamera started: %dx%d @ %d FPS (format=RGB888, memory=BGR)",
                    self._width, self._height, self._fps,
                )
                return
            except Exception as exc:
                logger.warning(
                    "PiCamera open attempt %d/%d failed: %s — retrying in %.1f s",
                    attempt, self._max_retries, exc, delay,
                )
                self._camera_ok = False
                if attempt < self._max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)

        raise RuntimeError(
            f"PiCamera failed to open after {self._max_retries} attempts."
        )

    def read(self) -> Optional[Frame]:
        """Capture and return a frame. Returns None on error (sets camera_ok=False)."""
        if self._cam is None:
            return None
        try:
            # capture_array() returns H×W×3 uint8 with bytes in B,G,R order
            bgr = self._cam.capture_array()
            self._frame_id += 1
            self._camera_ok = True
            return Frame(
                bgr=bgr,
                frame_id=self._frame_id,
                timestamp_utc=time.time(),
                timestamp_mono=time.monotonic(),
            )
        except Exception as exc:
            self._camera_ok = False
            logger.error("PiCamera read error: %s", exc)
            return None

    def stop(self) -> None:
        """Stop and close the camera."""
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            except Exception as exc:
                logger.warning("PiCamera stop error: %s", exc)
            finally:
                self._cam = None
        self._camera_ok = False
        logger.info("PiCamera stopped.")

    @property
    def camera_ok(self) -> bool:
        return self._camera_ok
