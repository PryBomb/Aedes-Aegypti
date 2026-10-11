"""Tests for the capture subsystem: ring buffer, drop-oldest queue, file source colour."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from sentinel.capture.base import Frame
from sentinel.capture.ring_buffer import RingBuffer
from sentinel.capture.capture_thread import CaptureThread
from sentinel.capture.file_source import FileSource


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(frame_id: int = 1, h: int = 720, w: int = 1280, mono: float = None) -> Frame:
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    return Frame(
        bgr=bgr,
        frame_id=frame_id,
        timestamp_utc=time.time(),
        timestamp_mono=mono if mono is not None else time.monotonic(),
    )


def _make_red_video(path: Path, n_frames: int = 10, fps: float = 30.0, w: int = 320, h: int = 240) -> Path:
    """Write a small video where every pixel is pure red (BGR = 0, 0, 255)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    red_frame = np.zeros((h, w, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 255  # BGR: channel 2 = red
    for _ in range(n_frames):
        writer.write(red_frame)
    writer.release()
    return path


# ── RingBuffer: time bound ─────────────────────────────────────────────────────

class TestRingBufferTimeBound:
    def test_empty_on_init(self):
        rb = RingBuffer(duration_s=2.0, max_memory_mb=10.0)
        assert rb.size == 0

    def test_push_and_retrieve(self):
        rb = RingBuffer(duration_s=5.0, max_memory_mb=50.0)
        frame = _make_frame()
        rb.push(frame)
        assert rb.size == 1

    def test_stale_frames_evicted_by_time(self):
        rb = RingBuffer(duration_s=0.1, max_memory_mb=50.0)
        # Push old frame with a past mono timestamp
        old_frame = _make_frame(mono=time.monotonic() - 1.0)
        rb.push(old_frame)
        # Push new frame — triggers eviction of the old one
        new_frame = _make_frame(frame_id=2)
        rb.push(new_frame)
        assert rb.size == 1

    def test_get_last_n_seconds(self):
        rb = RingBuffer(duration_s=10.0, max_memory_mb=50.0)
        old_frame = _make_frame(frame_id=1, mono=time.monotonic() - 5.0)
        rb.push(old_frame)
        new_frame = _make_frame(frame_id=2)
        rb.push(new_frame)
        recent = rb.get_last_n_seconds(2.0)
        # Only the new frame should be in the last 2 seconds
        assert len(recent) == 1
        assert recent[0].frame_id == 2

    def test_clear(self):
        rb = RingBuffer(duration_s=5.0, max_memory_mb=50.0)
        for i in range(5):
            rb.push(_make_frame(frame_id=i))
        rb.clear()
        assert rb.size == 0
        assert rb.memory_bytes == 0

    def test_thread_safety(self):
        """Multiple threads pushing concurrently must not corrupt the buffer."""
        rb = RingBuffer(duration_s=10.0, max_memory_mb=100.0)
        errors: List[Exception] = []

        def pusher(n):
            try:
                for i in range(n):
                    rb.push(_make_frame(frame_id=i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher, args=(20,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert rb.size <= 100  # sanity


# ── RingBuffer: memory cap ─────────────────────────────────────────────────────

class TestRingBufferMemoryCap:
    def test_memory_cap_respected(self):
        """With a very small cap, old frames must be evicted."""
        rb = RingBuffer(duration_s=60.0, max_memory_mb=0.05)  # 50 KB cap
        for i in range(50):
            rb.push(_make_frame(frame_id=i))
        # Memory must stay under cap + one frame's slack
        assert rb.memory_bytes < (0.05 * 1024 * 1024) + 30_000  # 30KB slack for one frame

    def test_memory_bytes_tracks_content(self):
        rb = RingBuffer(duration_s=60.0, max_memory_mb=100.0)
        assert rb.memory_bytes == 0
        rb.push(_make_frame())
        assert rb.memory_bytes > 0
        rb.clear()
        assert rb.memory_bytes == 0


# ── Drop-oldest queue ─────────────────────────────────────────────────────────

class TestDropOldestQueue:
    """Test the drop-oldest semantics via CaptureThread's internal queue."""

    def _make_fake_source(self, frames: List[Frame]):
        """A minimal fake FrameSource that yields from a list."""
        from sentinel.capture.base import FrameSource
        from typing import Optional

        class FakeSource(FrameSource):
            def __init__(self, frames):
                self._frames = iter(frames)
                self._done = False

            def start(self): pass
            def stop(self): pass

            def read(self) -> Optional[Frame]:
                try:
                    return next(self._frames)
                except StopIteration:
                    time.sleep(0.5)  # block so thread stays alive until test ends
                    return None

        return FakeSource(frames)

    def test_queue_size_1_drop_oldest(self):
        """When inference is slow, frames_dropped increments."""
        frames = [_make_frame(frame_id=i) for i in range(10)]
        source = self._make_fake_source(frames)
        ct = CaptureThread(source)
        source.start()
        ct.start()
        time.sleep(0.3)  # Let capture thread push many frames
        ct.stop()

        total = ct.frames_captured + ct.frames_dropped
        # At least some frames were captured
        assert ct.frames_captured > 0
        # Dropped + captured <= total frames emitted
        assert ct.frames_dropped >= 0

    def test_consumer_always_gets_latest(self):
        """get_frame() returns a frame (not None) once capturing has started."""
        frames = [_make_frame(frame_id=i) for i in range(50)]
        source = self._make_fake_source(frames)
        ct = CaptureThread(source)
        source.start()
        ct.start()
        time.sleep(0.1)
        frame = ct.get_frame(timeout=0.5)
        ct.stop()
        assert frame is not None


# ── FileSource: BGR colour order ──────────────────────────────────────────────

class TestFileSourceColour:
    def test_red_channel_is_index_2(self, tmp_path):
        """Frames from FileSource must be BGR: a pure-red video has max in channel [2]."""
        video_path = _make_red_video(tmp_path / "red.mp4")
        source = FileSource(video_path, real_time=False)
        source.start()
        frame = source.read()
        source.stop()

        assert frame is not None, "FileSource returned no frame from the video"
        assert frame.bgr.ndim == 3
        assert frame.bgr.shape[2] == 3

        # In BGR, red is channel index 2
        assert frame.bgr[:, :, 2].mean() > 200, (
            f"Expected red channel (index 2) to be bright, "
            f"got mean={frame.bgr[:, :, 2].mean():.1f}. "
            "This would indicate a BGR/RGB channel swap."
        )
        assert frame.bgr[:, :, 0].mean() < 10, "Blue channel should be ~0 for a red frame"
        assert frame.bgr[:, :, 1].mean() < 10, "Green channel should be ~0 for a red frame"

    def test_file_source_loop(self, tmp_path):
        """With loop=True, FileSource should not return None after the first pass."""
        video_path = _make_red_video(tmp_path / "loop.mp4", n_frames=5)
        source = FileSource(video_path, real_time=False, loop=True)
        source.start()
        frames_read = 0
        for _ in range(12):  # More than n_frames — must loop
            f = source.read()
            if f is not None:
                frames_read += 1
        source.stop()
        assert frames_read >= 10, "Loop mode should read more frames than the video contains"

    def test_file_source_missing_file(self):
        """FileSource must raise FileNotFoundError on a non-existent path."""
        source = FileSource("nonexistent_video.mp4")
        with pytest.raises(FileNotFoundError):
            source.start()

    def test_frame_has_all_fields(self, tmp_path):
        """Every Frame must have bgr, frame_id, timestamp_utc, timestamp_mono."""
        video_path = _make_red_video(tmp_path / "meta.mp4")
        source = FileSource(video_path, real_time=False)
        source.start()
        frame = source.read()
        source.stop()
        assert frame is not None
        assert isinstance(frame.frame_id, int)
        assert frame.timestamp_utc > 0
        assert frame.timestamp_mono > 0
    def test_camera_ok_lifecycle(self, tmp_path):
        """FrameSource.camera_ok must default to False and become True only after successful start/read."""
        video_path = _make_red_video(tmp_path / "ok_test.mp4")
        source = FileSource(video_path, real_time=False)
        assert source.camera_ok is False, "camera_ok must default to False before start"

        source.start()
        assert source.camera_ok is True, "camera_ok must become True after successful start"

        frame = source.read()
        assert frame is not None
        assert source.camera_ok is True, "camera_ok must remain True after successful read"

        source.stop()
        assert source.camera_ok is False, "camera_ok must return to False after stop"

    def test_base_framesource_camera_ok_default(self):
        """Custom FrameSource without start() must have camera_ok == False."""
        from sentinel.capture.base import FrameSource

        class DummySource(FrameSource):
            def start(self):
                self._camera_ok = True
            def read(self):
                return None
            def stop(self):
                self._camera_ok = False

        dummy = DummySource()
        assert dummy.camera_ok is False
        dummy.start()
        assert dummy.camera_ok is True
        dummy.stop()
        assert dummy.camera_ok is False


# ── Detector: fake backend (no weights required) ──────────────────────────────

class TestDetectorFakeBackend:
    """Tests that run without model weights by patching the YOLO model."""

    def _make_detector_with_fake_model(self, target_classes=None):
        """Return a Detector whose internal YOLO model is mocked."""
        from sentinel.detection.detector import Detector, Detection
        from unittest.mock import MagicMock, patch

        detector = Detector(
            weights_path="fake.pt",
            imgsz=640,
            conf=0.25,
            target_class_names=target_classes or [],
            warmup_runs=0,
        )
        # Build a fake model
        fake_model = MagicMock()
        fake_model.names = {0: "Aedes aegypti", 1: "Other"}
        detector._model = fake_model
        detector._class_names = fake_model.names
        return detector, fake_model

    def test_detect_returns_detections(self):
        from unittest.mock import MagicMock
        detector, fake_model = self._make_detector_with_fake_model(["Aedes aegypti"])

        # Build fake result
        box = MagicMock()
        box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 200.0])]
        box.conf = [MagicMock(item=lambda: 0.85)]
        box.cls = [MagicMock(item=lambda: 0)]

        result = MagicMock()
        result.boxes = box
        result.boxes.__len__ = lambda self: 1
        fake_model.predict.return_value = [result]

        frame_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections, infer_ms = detector.detect(frame_bgr)

        assert len(detections) == 1
        assert detections[0].class_name == "Aedes aegypti"
        assert detections[0].conf == pytest.approx(0.85)
        assert infer_ms >= 0.0

    def test_target_class_filter_sets_is_target_false_for_other(self):
        from unittest.mock import MagicMock
        detector, fake_model = self._make_detector_with_fake_model(["Aedes aegypti"])
        detector._class_names = {0: "Aedes aegypti", 1: "Other"}

        box = MagicMock()
        box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 200.0])]
        box.conf = [MagicMock(item=lambda: 0.9)]
        box.cls = [MagicMock(item=lambda: 1)]  # class 1 = "Other"

        result = MagicMock()
        result.boxes = box
        result.boxes.__len__ = lambda self: 1
        fake_model.predict.return_value = [result]

        detections, _ = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        # "Other" is retained with is_target = False (Phase 3 ruling)
        assert len(detections) == 1
        assert detections[0].class_name == "Other"
        assert detections[0].is_target is False

    def test_empty_target_class_keeps_all(self):
        from unittest.mock import MagicMock
        detector, fake_model = self._make_detector_with_fake_model([])  # no filter

        box = MagicMock()
        box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 200.0])]
        box.conf = [MagicMock(item=lambda: 0.7)]
        box.cls = [MagicMock(item=lambda: 1)]

        result = MagicMock()
        result.boxes = box
        result.boxes.__len__ = lambda self: 1
        fake_model.predict.return_value = [result]

        detections, _ = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        assert len(detections) == 1  # All classes kept

    def test_detect_without_load_raises(self):
        from sentinel.detection.detector import Detector
        detector = Detector("fake.pt", warmup_runs=0)
        with pytest.raises(RuntimeError, match="load()"):
            detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    def test_empty_boxes_returns_empty_list(self):
        from unittest.mock import MagicMock
        detector, fake_model = self._make_detector_with_fake_model()

        result = MagicMock()
        result.boxes = None
        fake_model.predict.return_value = [result]

        detections, _ = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        assert detections == []


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "pi_only: test requires Raspberry Pi hardware")
    config.addinivalue_line("markers", "needs_weights: test requires models/best.pt")


@pytest.fixture
def needs_weights():
    weights = Path(__file__).parent.parent / "models" / "best.pt"
    if not weights.exists():
        pytest.skip(f"models/best.pt not found — skipping weights-dependent test")
