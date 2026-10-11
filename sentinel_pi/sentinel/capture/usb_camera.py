"""USB webcam frame source using cv2.VideoCapture."""
from __future__ import annotations

import logging
import time
from typing import Optional

import cv2

from sentinel.capture.base import Frame, FrameSource

logger = logging.getLogger(__name__)


class UsbCamera(FrameSource):
    """Frame source for USB/built-in cameras via cv2.VideoCapture.

    OpenCV VideoCapture returns BGR frames natively — no colour conversion needed.

    Args:
        device_index: OpenCV device index (from config.camera.usb_index).
        width: Requested capture width (best-effort; camera may override).
        height: Requested capture height.
        fps: Requested frame rate (best-effort).
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._camera_ok: bool = False

    def start(self) -> None:
        """Open the USB camera."""
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            self._camera_ok = False
            raise RuntimeError(
                f"Failed to open USB camera at device index {self._device_index}. "
                "Check config.camera.usb_index and that the camera is connected."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        self._camera_ok = True
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "UsbCamera opened (index=%d): actual=%dx%d @ %.1f FPS",
            self._device_index, actual_w, actual_h, actual_fps,
        )

    def read(self) -> Optional[Frame]:
        """Read one BGR frame. Returns None on failure."""
        if self._cap is None or not self._cap.isOpened():
            self._camera_ok = False
            return None
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            self._camera_ok = False
            logger.warning("UsbCamera: failed to read frame.")
            return None
        self._frame_id += 1
        self._camera_ok = True
        return Frame(
            bgr=bgr,
            frame_id=self._frame_id,
            timestamp_utc=time.time(),
            timestamp_mono=time.monotonic(),
        )

    def stop(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._camera_ok = False
        logger.info("UsbCamera stopped.")

    @property
    def camera_ok(self) -> bool:
        return self._camera_ok
