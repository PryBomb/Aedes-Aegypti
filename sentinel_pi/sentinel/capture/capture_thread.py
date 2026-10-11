"""Capture thread with drop-oldest queue and a thread-safe ring buffer feeder.

Architecture:
  FrameSource (hardware thread)
      |
      v  queue maxsize=1, drop-OLDEST
  CaptureThread
      |-- puts frames into detect_queue (size=1, drop-oldest)
      |-- also writes JPEG-compressed thumbnails to RingBuffer

Drop-oldest semantics: if the inference worker is behind, the stale frame is
discarded and a fresh frame takes its place. This ensures inference always sees
the most recent view, and capture is never blocked by inference speed.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from sentinel.capture.base import Frame, FrameSource
from sentinel.capture.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class CaptureThread:
    """Runs a FrameSource in a daemon thread, feeds a single-slot detect queue.

    The detect queue uses drop-oldest semantics: if the consumer (inference) is
    behind, the stale frame is evicted and the fresh frame enqueued, so inference
    is never ahead by more than one frame.

    Args:
        source: A started FrameSource implementation.
        ring_buffer: Optional RingBuffer to receive every captured frame.
    """

    def __init__(
        self,
        source: FrameSource,
        ring_buffer: Optional[RingBuffer] = None,
    ) -> None:
        self._source = source
        self._ring = ring_buffer
        self._detect_queue: queue.Queue[Frame] = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.frames_captured: int = 0
        self.frames_dropped: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="CaptureThread", daemon=True
        )
        self._thread.start()
        logger.info("CaptureThread started.")

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the capture thread to stop and join it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info(
            "CaptureThread stopped. captured=%d dropped=%d",
            self.frames_captured, self.frames_dropped,
        )

    def get_frame(self, timeout: float = 0.1) -> Optional[Frame]:
        """Retrieve the latest frame from the detect queue (blocking, with timeout)."""
        try:
            return self._detect_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def camera_ok(self) -> bool:
        return self._source.camera_ok

    # ── Internal ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main capture loop — runs in a daemon thread."""
        while not self._stop_event.is_set():
            frame = self._source.read()
            if frame is None:
                if not self._stop_event.is_set():
                    time.sleep(0.005)
                continue

            self.frames_captured += 1

            # Feed ring buffer (stores JPEG thumbnails — see RingBuffer)
            if self._ring is not None:
                self._ring.push(frame)

            # Drop-oldest enqueue: evict stale frame if queue is full
            if self._detect_queue.full():
                try:
                    self._detect_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass
            try:
                self._detect_queue.put_nowait(frame)
            except queue.Full:
                # Race: another producer filled it between our check and put
                self.frames_dropped += 1
