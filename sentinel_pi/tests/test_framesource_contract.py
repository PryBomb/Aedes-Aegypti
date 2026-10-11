"""Strict contract test suite for all concrete FrameSource implementations.

Ensures that every FrameSource:
- is instantiable
- implements start(), read(), stop(), camera_ok
- starts with camera_ok == False
- sets camera_ok == True upon successful start()
- yields valid Frame objects with BGR arrays, frame_id, timestamps
- resets camera_ok == False after stop()
- supports context manager semantics (with source:)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sentinel.capture.base import Frame, FrameSource
from sentinel.capture.file_source import FileSource
from sentinel.capture.usb_camera import UsbCamera


def _make_dummy_video(path: Path) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (160, 120))
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(5):
        writer.write(frame)
    writer.release()
    return path


class TestFrameSourceContract:
    """Verifies that every concrete FrameSource strictly conforms to the FrameSource contract."""

    def _verify_source_contract(self, source: FrameSource):
        assert isinstance(source, FrameSource)
        assert hasattr(source, "start")
        assert hasattr(source, "read")
        assert hasattr(source, "stop")
        assert hasattr(source, "camera_ok")

        # 1. Before start: camera_ok must be False
        assert source.camera_ok is False, "camera_ok must be False before start() is called"

        # 2. After start: camera_ok must be True
        source.start()
        assert source.camera_ok is True, "camera_ok must be True after successful start()"

        # 3. Read frame and verify structure
        frame = source.read()
        if frame is not None:
            assert isinstance(frame, Frame)
            assert isinstance(frame.bgr, np.ndarray)
            assert frame.bgr.ndim == 3
            assert frame.bgr.shape[2] == 3
            assert isinstance(frame.frame_id, int)
            assert frame.frame_id > 0
            assert frame.timestamp_utc > 0.0
            assert frame.timestamp_mono > 0.0
            assert source.camera_ok is True

        # 4. After stop: camera_ok must be False
        source.stop()
        assert source.camera_ok is False, "camera_ok must be False after stop() is called"

    def test_file_source_contract(self, tmp_path):
        video_path = _make_dummy_video(tmp_path / "contract.mp4")
        source = FileSource(video_path, real_time=False)
        self._verify_source_contract(source)

    def test_file_source_context_manager(self, tmp_path):
        video_path = _make_dummy_video(tmp_path / "ctx.mp4")
        source = FileSource(video_path, real_time=False)
        assert source.camera_ok is False
        with source:
            assert source.camera_ok is True
            frame = source.read()
            assert frame is not None
        assert source.camera_ok is False

    @patch("cv2.VideoCapture")
    def test_usb_camera_contract(self, mock_vidcap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_cap.get.return_value = 30.0
        mock_vidcap_cls.return_value = mock_cap

        source = UsbCamera(device_index=0)
        self._verify_source_contract(source)

    @patch("cv2.VideoCapture")
    def test_usb_camera_context_manager(self, mock_vidcap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_vidcap_cls.return_value = mock_cap

        source = UsbCamera(device_index=0)
        assert source.camera_ok is False
        with source:
            assert source.camera_ok is True
            f = source.read()
            assert f is not None
        assert source.camera_ok is False

    @pytest.mark.pi_only
    def test_pi_camera_contract(self):
        try:
            from sentinel.capture.pi_camera import PiCamera
            source = PiCamera()
        except RuntimeError as e:
            pytest.skip(f"PiCamera requires Raspberry Pi hardware: {e}")
        self._verify_source_contract(source)
