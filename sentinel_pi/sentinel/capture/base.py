"""Base interface for all frame sources (FrameSource ABC)."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """A single captured video frame with metadata.

    Attributes:
        bgr: BGR uint8 numpy array of shape (H, W, 3).
        frame_id: Monotonically increasing integer counter from the source.
        timestamp_utc: Wall-clock UTC time of capture (seconds since epoch).
        timestamp_mono: Monotonic time of capture (seconds, from time.monotonic()).
    """
    bgr: np.ndarray
    frame_id: int
    timestamp_utc: float
    timestamp_mono: float


class FrameSource(ABC):
    """Abstract base class for all frame sources.

    Subclasses must implement start(), read(), and stop().
    All returned frames must carry BGR arrays — no cv2.cvtColor conversions
    should occur downstream for colour normalisation purposes.
    """

    @abstractmethod
    def start(self) -> None:
        """Initialise hardware / open file. Called once before read()."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next Frame, or None if no frame is available.

        Implementations must be non-blocking or only block for a short time
        (≤ 1 capture interval). Returning None allows the caller to decide
        whether to retry or stop.
        """

    _camera_ok: bool = False

    @abstractmethod
    def stop(self) -> None:
        """Release hardware / close file handles. Safe to call multiple times."""

    @property
    def camera_ok(self) -> bool:
        """True if the underlying capture device is healthy and active.

        Defaults to False and becomes True only after a successful start/read.
        """
        return getattr(self, "_camera_ok", False)

    # ── Context-manager support ────────────────────────────────────────────
    def __enter__(self) -> "FrameSource":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
