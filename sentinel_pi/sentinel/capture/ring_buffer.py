"""Thread-safe ring buffer — stores the last N seconds of frames.

Memory design decision
======================
One 1280×720 BGR frame = 1280 × 720 × 3 ≈ 2.76 MB.
At 30 FPS and 3 s of pre-roll that is 90 frames × 2.76 MB ≈ 248 MB.
Storing raw BGR at full rate would require ~250 MB per monitoring window
and would be impractical on a Pi 5 with 8 GB RAM once other components
are running.

Resolution: frames are stored as **JPEG-compressed thumbnails at 50% scale**
(640×360). At JPEG quality=80:
  - Typical compressed size: ~15–25 KB per frame
  - 90 frames of pre-roll: ~1.4–2.3 MB total

Stored data per slot:
  - jpeg_bytes: compressed frame (decompressed by EvidenceSaver when writing clip)
  - frame_id, timestamp_utc, timestamp_mono

The full-resolution BGR frame is NOT stored in the ring buffer. The EvidenceSaver
will use the ring buffer thumbnails to write a clip at reduced resolution (or
optionally re-capture at full resolution from the detect queue if still fresh).
The design decision is documented in DECISIONS.md.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import cv2
import numpy as np

from sentinel.capture.base import Frame

logger = logging.getLogger(__name__)

# Thumbnail scale factor applied before JPEG compression
_THUMB_SCALE = 0.5
# JPEG encode quality
_JPEG_QUALITY = 80
# Approximate overhead per slot (bytes) beyond the compressed image
_SLOT_OVERHEAD = 128


@dataclass
class RingSlot:
    """One compressed entry in the ring buffer."""
    jpeg_bytes: bytes
    frame_id: int
    timestamp_utc: float
    timestamp_mono: float

    @property
    def size_bytes(self) -> int:
        return len(self.jpeg_bytes) + _SLOT_OVERHEAD


class RingBuffer:
    """Thread-safe time-bounded ring buffer with a hard memory cap.

    Frames older than `duration_s` seconds are evicted on each push.
    If the buffer exceeds `max_memory_mb` MB, the oldest frames are
    evicted until it fits.

    Args:
        duration_s: Maximum age of frames to retain (seconds).
        max_memory_mb: Hard cap on total memory usage (MB).
    """

    def __init__(self, duration_s: float = 5.0, max_memory_mb: float = 50.0) -> None:
        self._duration_s = duration_s
        self._max_bytes = int(max_memory_mb * 1024 * 1024)
        self._slots: Deque[RingSlot] = deque()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

    def push(self, frame: Frame) -> None:
        """Compress and store a frame. Evicts stale / oversized entries."""
        # Compress to full-resolution JPEG (no scaling)
        ok, buf = cv2.imencode(".jpg", frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            logger.warning("RingBuffer: JPEG encode failed for frame %d", frame.frame_id)
            return

        slot = RingSlot(
            jpeg_bytes=buf.tobytes(),
            frame_id=frame.frame_id,
            timestamp_utc=frame.timestamp_utc,
            timestamp_mono=frame.timestamp_mono,
        )

        with self._lock:
            self._slots.append(slot)
            self._total_bytes += slot.size_bytes
            self._evict_stale(frame.timestamp_mono)
            self._evict_oversized()

    def get_since(self, since_mono: float) -> List[RingSlot]:
        """Return all slots with timestamp_mono >= since_mono."""
        with self._lock:
            return [s for s in self._slots if s.timestamp_mono >= since_mono]

    def get_last_n_seconds(self, seconds: float) -> List[RingSlot]:
        """Return all slots within the last `seconds` seconds."""
        cutoff = time.monotonic() - seconds
        return self.get_since(cutoff)

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()
            self._total_bytes = 0

    @property
    def size(self) -> int:
        """Number of slots currently in the buffer."""
        with self._lock:
            return len(self._slots)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory used by stored frames."""
        with self._lock:
            return self._total_bytes

    # ── Private ───────────────────────────────────────────────────────────

    def _evict_stale(self, now_mono: float) -> None:
        """Remove frames older than duration_s. Must be called under lock."""
        cutoff = now_mono - self._duration_s
        while self._slots and self._slots[0].timestamp_mono < cutoff:
            evicted = self._slots.popleft()
            self._total_bytes -= evicted.size_bytes

    def _evict_oversized(self) -> None:
        """Remove oldest frames until under memory cap. Must be called under lock."""
        while self._slots and self._total_bytes > self._max_bytes:
            evicted = self._slots.popleft()
            self._total_bytes -= evicted.size_bytes
            logger.debug(
                "RingBuffer memory cap hit — evicted frame %d (buffer now %d bytes)",
                evicted.frame_id, self._total_bytes,
            )
