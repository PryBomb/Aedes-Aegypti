"""Video file frame source using cv2.VideoCapture.

Supports two pacing modes:
- real_time=True (default): throttles read() to match the source FPS,
  suitable for integration tests and development.
- real_time=False: reads as fast as possible — used by benchmark.py to
  measure pure inference throughput without being bottlenecked by I/O waits.

Also supports optional looping for long-running soak tests.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2

from sentinel.capture.base import Frame, FrameSource

logger = logging.getLogger(__name__)


class FileSource(FrameSource):
    """Frame source that reads from a video file.

    Args:
        path: Path to the video file.
        real_time: If True, pace reads to match the video's native FPS.
                   If False, read as fast as possible (for benchmarking).
        loop: If True, restart from the beginning when the video ends.
    """

    def __init__(
        self,
        path: str | Path,
        real_time: bool = True,
        loop: bool = False,
    ) -> None:
        self._path = Path(path)
        self._real_time = real_time
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._source_fps: float = 30.0
        self._frame_interval_s: float = 1.0 / 30.0
        self._last_read_mono: float = 0.0
        self._camera_ok: bool = False

    def start(self) -> None:
        """Open the video file."""
        if not self._path.exists():
            self._camera_ok = False
            raise FileNotFoundError(
                f"Video file not found: {self._path}. "
                "Set file_source.path in config.yaml."
            )
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            self._camera_ok = False
            raise RuntimeError(f"OpenCV could not open video file: {self._path}")

        self._camera_ok = True
        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_interval_s = 1.0 / self._source_fps
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            "FileSource opened: %s (%d frames @ %.2f FPS, real_time=%s, loop=%s)",
            self._path.name, total_frames, self._source_fps,
            self._real_time, self._loop,
        )

    def read(self) -> Optional[Frame]:
        """Read next frame, throttling to source FPS if real_time=True.

        Returns None when the video ends (and loop=False).
        """
        if self._cap is None:
            self._camera_ok = False
            return None

        # Real-time pacing: sleep if we're ahead of the source frame rate
        if self._real_time and self._last_read_mono > 0:
            elapsed = time.monotonic() - self._last_read_mono
            sleep_for = self._frame_interval_s - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        ret, bgr = self._cap.read()
        if not ret:
            if self._loop:
                logger.debug("FileSource: end of file, looping.")
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, bgr = self._cap.read()
                if not ret:
                    self._camera_ok = False
                    return None
            else:
                self._camera_ok = False
                return None

        self._frame_id += 1
        self._camera_ok = True
        now_mono = time.monotonic()
        self._last_read_mono = now_mono
        return Frame(
            bgr=bgr,
            frame_id=self._frame_id,
            timestamp_utc=time.time(),
            timestamp_mono=now_mono,
        )

    def stop(self) -> None:
        """Release the video file handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._camera_ok = False
        logger.info("FileSource stopped.")

    @property
    def camera_ok(self) -> bool:
        """True if the video file is open and healthy."""
        return self._camera_ok

    @property
    def source_fps(self) -> float:
        """The FPS as reported by the video file."""
        return self._source_fps
