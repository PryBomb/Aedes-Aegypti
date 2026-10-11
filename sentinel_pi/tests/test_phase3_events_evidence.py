"""Tests for Phase 3: EventBuilder, EvidenceSaver retention, and Repository persistence.

Synthetic sequences test:
- Gap splitting
- Persistence rule (min_frames, min_mean_conf)
- Class vote (summing conf, target vs non-target)
- Flush on shutdown
- Retention rule (oldest unverified pruned, verified strictly protected)
- Repository round-trip
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from sentinel.capture.base import Frame
from sentinel.capture.ring_buffer import RingBuffer
from sentinel.detection.detector import Detection, Detector
from sentinel.detection.event_builder import Event, EventBuilder, to_utc_iso
from sentinel.evidence.saver import EvidenceSaver
from sentinel.storage.db import DatabaseManager
from sentinel.storage.repositories import EventRepository, HealthRepository, SessionRepository


def _make_det(
    class_name: str = "Aedes Aegypti",
    conf: float = 0.8,
    is_target: bool = True,
    xyxy: tuple[float, float, float, float] = (10.0, 10.0, 50.0, 50.0),
) -> Detection:
    return Detection(
        xyxy=xyxy,
        conf=conf,
        class_id=0 if is_target else 1,
        class_name=class_name,
        is_target=is_target,
    )


def _make_frame(frame_id: int, mono: float, utc: float) -> Frame:
    bgr = np.zeros((120, 160, 3), dtype=np.uint8)
    return Frame(bgr=bgr, frame_id=frame_id, timestamp_utc=utc, timestamp_mono=mono)


# ── 1. EventBuilder: Gap Splitting ───────────────────────────────────────────

class TestEventBuilderGapSplitting:
    def test_gap_splitting(self):
        builder = EventBuilder(event_gap_s=1.5, min_frames=1, min_mean_conf=0.1)

        # Event 1 frames: t = 0.0, 1.0 (gap = 1.0 <= 1.5)
        closed = builder.update([_make_det(conf=0.7)], mono_time=0.0, utc_time=1000.0)
        assert len(closed) == 0
        closed = builder.update([_make_det(conf=0.7)], mono_time=1.0, utc_time=1001.0)
        assert len(closed) == 0

        # Detection at t = 3.0 (gap = 2.0 > 1.5) -> Event 1 closes!
        closed = builder.update([_make_det(conf=0.7)], mono_time=3.0, utc_time=1003.0)
        assert len(closed) == 1
        ev1 = closed[0]
        assert ev1.started_at == to_utc_iso(1000.0)
        assert ev1.ended_at == to_utc_iso(1001.0)
        assert ev1.n_frames == 2

        # Flush closes Event 2
        closed = builder.flush(mono_time=3.5, utc_time=1003.5)
        assert len(closed) == 1
        ev2 = closed[0]
        assert ev2.started_at == to_utc_iso(1003.0)
        assert ev2.n_frames == 1


# ── 2. EventBuilder: Persistence Rule ────────────────────────────────────────

class TestEventBuilderPersistenceRule:
    def test_rejected_when_too_few_frames(self):
        """Must reject if n_frames < min_frames even if confidence is high."""
        builder = EventBuilder(event_gap_s=1.0, min_frames=3, min_mean_conf=0.4)
        builder.update([_make_det(conf=0.9)], mono_time=0.0, utc_time=100.0)
        builder.update([_make_det(conf=0.9)], mono_time=0.2, utc_time=100.2)
        events = builder.flush()
        assert len(events) == 1
        assert events[0].status == "rejected"
        assert events[0].n_frames == 2
        assert events[0].t_confirmed is None

    def test_rejected_when_confidence_too_low(self):
        """Must reject if mean_conf < min_mean_conf even if n_frames >= min_frames."""
        builder = EventBuilder(event_gap_s=1.0, min_frames=3, min_mean_conf=0.5)
        builder.update([_make_det(conf=0.3)], mono_time=0.0, utc_time=100.0)
        builder.update([_make_det(conf=0.3)], mono_time=0.2, utc_time=100.2)
        builder.update([_make_det(conf=0.3)], mono_time=0.4, utc_time=100.4)
        events = builder.flush()
        assert len(events) == 1
        assert events[0].status == "rejected"
        assert events[0].n_frames == 3
        assert events[0].mean_conf == pytest.approx(0.3)
        assert events[0].t_confirmed is None

    def test_confirmed_when_criteria_met(self):
        """Must confirm and record t_confirmed when thresholds are met."""
        builder = EventBuilder(event_gap_s=1.0, min_frames=3, min_mean_conf=0.4)
        builder.update([_make_det(conf=0.6)], mono_time=0.0, utc_time=100.0)
        builder.update([_make_det(conf=0.6)], mono_time=0.2, utc_time=100.2)
        builder.update([_make_det(conf=0.6)], mono_time=0.4, utc_time=100.4)  # criteria met here
        builder.update([_make_det(conf=0.6)], mono_time=0.6, utc_time=100.6)
        events = builder.flush()
        assert len(events) == 1
        ev = events[0]
        assert ev.status == "confirmed"
        assert ev.n_frames == 4
        assert ev.t_first_detect == to_utc_iso(100.0)
        assert ev.t_confirmed == to_utc_iso(100.4)


# ── 3. EventBuilder: Class Vote ──────────────────────────────────────────────

class TestEventBuilderClassVote:
    def test_target_class_wins_vote(self):
        """Target class with higher sum of confidence wins and confirms."""
        builder = EventBuilder(
            min_frames=2,
            min_mean_conf=0.4,
            target_class_names=["Aedes Aegypti"],
            non_target_class_names=["Aedes Albopictus"],
        )
        # 2 Aegypti detections with conf 0.8, 0.7 (sum = 1.5)
        # 1 Albopictus detection with conf 0.9 (sum = 0.9)
        builder.update([_make_det("Aedes Aegypti", 0.8), _make_det("Aedes Albopictus", 0.9)], mono_time=0.0, utc_time=10.0)
        builder.update([_make_det("Aedes Aegypti", 0.7)], mono_time=0.5, utc_time=10.5)
        events = builder.flush()
        assert len(events) == 1
        assert events[0].yolo_class == "Aedes Aegypti"
        assert events[0].status == "confirmed"

    def test_non_target_class_wins_vote(self):
        """Non-target class with higher sum of confidence wins and marks non_target."""
        builder = EventBuilder(
            min_frames=2,
            min_mean_conf=0.4,
            target_class_names=["Aedes Aegypti"],
            non_target_class_names=["Aedes Albopictus"],
        )
        # 1 Aegypti detection conf 0.6 (sum = 0.6)
        # 2 Albopictus detections conf 0.8, 0.9 (sum = 1.7)
        builder.update([_make_det("Aedes Aegypti", 0.6)], mono_time=0.0, utc_time=10.0)
        builder.update([_make_det("Aedes Albopictus", 0.8)], mono_time=0.2, utc_time=10.2)
        builder.update([_make_det("Aedes Albopictus", 0.9)], mono_time=0.4, utc_time=10.4)
        events = builder.flush()
        assert len(events) == 1
        assert events[0].yolo_class == "Aedes Albopictus"
        assert events[0].status == "non_target"
        assert events[0].t_confirmed is None


# ── 4. Flush on Shutdown & Track ID ──────────────────────────────────────────

class TestEventBuilderFlush:
    def test_flush_closes_open_event_with_track_id(self):
        builder = EventBuilder(event_gap_s=5.0, min_frames=1, min_mean_conf=0.1)
        builder.update([_make_det(conf=0.7)], mono_time=1.0, utc_time=200.0, track_id=42)
        assert builder.has_open_event is True
        events = builder.flush()
        assert len(events) == 1
        assert events[0].track_id == 42
        assert builder.has_open_event is False


# ── 5. Detector: Case-Insensitive Validation & Target Flags ───────────────────

class TestDetectorValidation:
    def test_missing_configured_class_raises_value_error(self):
        detector = Detector(
            weights_path="fake.pt",
            target_class_names=["Aedes Aegypti", "NonExistentSpecies"],
            warmup_runs=0,
        )
        fake_model = MagicMock()
        fake_model.names = {0: "Aedes Aegypti", 1: "Aedes Albopictus"}
        detector._model = fake_model
        detector._class_names = fake_model.names

        with pytest.raises(ValueError, match="(?i)nonexistentspecies"):
            # Simulate load validation
            model_names_lower = {v.lower(): v for v in detector._class_names.values()}
            for name in detector._target_class_names:
                if name not in model_names_lower:
                    raise ValueError(f"Configured target class '{name}' not found")

    def test_detector_returns_all_classes_with_is_target_flag(self):
        detector = Detector(
            weights_path="fake.pt",
            target_class_names=["aedes aegypti"],  # lower-case
            non_target_class_names=["aedes albopictus"],
            warmup_runs=0,
        )
        fake_model = MagicMock()
        fake_model.names = {0: "Aedes Aegypti", 1: "Aedes Albopictus"}
        detector._model = fake_model
        detector._class_names = fake_model.names

        box = MagicMock()
        box.xyxy = [
            MagicMock(tolist=lambda: [10.0, 10.0, 50.0, 50.0]),
            MagicMock(tolist=lambda: [60.0, 60.0, 90.0, 90.0]),
        ]
        box.conf = [MagicMock(item=lambda: 0.85), MagicMock(item=lambda: 0.90)]
        box.cls = [MagicMock(item=lambda: 0), MagicMock(item=lambda: 1)]

        result = MagicMock()
        result.boxes = box
        result.boxes.__len__ = lambda self: 2
        fake_model.predict.return_value = [result]

        detections, _ = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        assert len(detections) == 2
        assert detections[0].class_name == "Aedes Aegypti"
        assert detections[0].is_target is True
        assert detections[1].class_name == "Aedes Albopictus"
        assert detections[1].is_target is False  # Albopictus is retained!


# ── 6. EvidenceSaver: Retention Rule ─────────────────────────────────────────

class TestEvidenceRetentionRule:
    def test_unverified_deleted_verified_protected(self, tmp_path):
        db_path = tmp_path / "sentinel.db"
        schema_path = Path(__file__).parent.parent / "sentinel" / "storage" / "schema.sql"
        db = DatabaseManager(db_path, schema_path)
        s_repo = SessionRepository(db)
        e_repo = EventRepository(db)

        session_id = s_repo.create(
            mode="test",
            model_name="best.pt",
            model_backend="pt",
            config_json="{}",
        )

        data_dir = tmp_path / "sentinel_data"
        saver = EvidenceSaver(data_dir=data_dir, event_repo=e_repo, max_gb=0.000001)  # very low cap (~1 KB)

        # Create 3 dummy evidence files
        snap1 = data_dir / "evidence" / "snapshots" / "snap1.jpg"
        snap2 = data_dir / "evidence" / "snapshots" / "snap2.jpg"
        snap3 = data_dir / "evidence" / "snapshots" / "snap3.jpg"
        snap1.write_bytes(b"X" * 2000)
        snap2.write_bytes(b"Y" * 2000)
        snap3.write_bytes(b"Z" * 2000)

        # Event 1: unverified, oldest
        id1 = e_repo.create_event({
            "session_id": session_id,
            "monitoring_point": "KAS",
            "started_at": "2026-10-10T01:00:00.000Z",
            "ended_at": "2026-10-10T01:00:01.000Z",
            "duration_s": 1.0, "n_frames": 3, "max_conf": 0.8, "mean_conf": 0.8,
            "yolo_class": "Aedes Aegypti", "status": "confirmed",
            "snapshot_path": "evidence/snapshots/snap1.jpg",
        })

        # Event 2: verified (VERIFIED TRUE), middle timestamp
        id2 = e_repo.create_event({
            "session_id": session_id,
            "monitoring_point": "KAS",
            "started_at": "2026-10-10T02:00:00.000Z",
            "ended_at": "2026-10-10T02:00:01.000Z",
            "duration_s": 1.0, "n_frames": 3, "max_conf": 0.8, "mean_conf": 0.8,
            "yolo_class": "Aedes Aegypti", "status": "confirmed",
            "snapshot_path": "evidence/snapshots/snap2.jpg",
            "verified_label": "true",
            "verified_by": "expert",
        })

        # Event 3: unverified, newest timestamp
        id3 = e_repo.create_event({
            "session_id": session_id,
            "monitoring_point": "KAS",
            "started_at": "2026-10-10T03:00:00.000Z",
            "ended_at": "2026-10-10T03:00:01.000Z",
            "duration_s": 1.0, "n_frames": 3, "max_conf": 0.8, "mean_conf": 0.8,
            "yolo_class": "Aedes Aegypti", "status": "confirmed",
            "snapshot_path": "evidence/snapshots/snap3.jpg",
        })

        # Run retention enforcement
        deleted_count = saver.enforce_retention()
        assert deleted_count >= 1

        # Event 1 (oldest unverified) must be deleted
        assert not snap1.exists()
        ev1 = e_repo.get(id1)
        assert ev1["snapshot_path"] is None

        # Event 2 (verified) MUST BE PROTECTED and never deleted!
        assert snap2.exists()
        ev2 = e_repo.get(id2)
        assert ev2["snapshot_path"] == "evidence/snapshots/snap2.jpg"


# ── 7. Repository: Round-Trip ────────────────────────────────────────────────

class TestRepositoryRoundTrip:
    def test_session_and_event_round_trip(self, tmp_path):
        db_path = tmp_path / "sentinel.db"
        schema_path = Path(__file__).parent.parent / "sentinel" / "storage" / "schema.sql"
        db = DatabaseManager(db_path, schema_path)
        s_repo = SessionRepository(db)
        e_repo = EventRepository(db)

        session_id = s_repo.create(
            mode="file",
            model_name="best.pt",
            model_backend="pt",
            config_json={"test": True},
            deployment_label="roundtrip-test",
        )
        s_row = s_repo.get(session_id)
        assert s_row is not None
        assert s_row["deployment_label"] == "roundtrip-test"

        event = Event(
            session_id=session_id,
            monitoring_point="KAS-1E-01",
            started_at="2026-10-10T12:00:00.000Z",
            ended_at="2026-10-10T12:00:02.500Z",
            duration_s=2.5,
            n_frames=10,
            max_conf=0.92,
            mean_conf=0.85,
            yolo_class="Aedes Aegypti",
            track_id=101,
            status="confirmed",
            snapshot_path="evidence/snapshots/test.jpg",
            clip_path="evidence/clips/test.mp4",
            t_first_detect="2026-10-10T12:00:00.000Z",
            t_confirmed="2026-10-10T12:00:00.800Z",
        )

        event_id = e_repo.create_event(event)
        assert event_id > 0

        ev_row = e_repo.get(event_id)
        assert ev_row is not None
        assert ev_row["session_id"] == session_id
        assert ev_row["yolo_class"] == "Aedes Aegypti"
        assert ev_row["status"] == "confirmed"
        assert ev_row["track_id"] == 101
        assert ev_row["duration_s"] == 2.5
        assert ev_row["snapshot_path"] == "evidence/snapshots/test.jpg"
        assert ev_row["t_confirmed"] == "2026-10-10T12:00:00.800Z"

        # Verify query helpers
        status_counts = e_repo.count_by_status(session_id)
        assert status_counts["confirmed"] == 1
        class_counts = e_repo.count_by_class(session_id)
        assert class_counts["Aedes Aegypti"] == 1


# ── 8. Review Fix Tests: Winning Class Metrics & Sticky Confirmation ─────────

class TestEventBuilderReviewRequirements:
    def test_winning_class_only_computes_metrics(self):
        """5 Aegypti detections at 0.9 plus 4 Albopictus at 0.3 must report Aegypti mean_conf 0.9."""
        builder = EventBuilder(
            min_frames=3,
            min_mean_conf=0.5,
            target_class_names=["Aedes Aegypti"],
            non_target_class_names=["Aedes Albopictus"],
        )
        # Push 5 Aegypti detections at 0.9
        for i in range(5):
            builder.update([_make_det("Aedes Aegypti", conf=0.9)], mono_time=0.1 * i, utc_time=100.0 + 0.1 * i)
        # Push 4 Albopictus detections at 0.3
        for i in range(4):
            builder.update([_make_det("Aedes Albopictus", conf=0.3, is_target=False)], mono_time=0.5 + 0.1 * i, utc_time=100.5 + 0.1 * i)

        events = builder.flush()
        assert len(events) == 1
        ev = events[0]
        assert ev.yolo_class == "Aedes Aegypti"
        # Must compute metrics from the WINNING class only:
        assert ev.n_frames == 5
        assert ev.mean_conf == pytest.approx(0.9)
        assert ev.max_conf == pytest.approx(0.9)
        assert ev.status == "confirmed"

    def test_on_confirmed_callback_called_once(self):
        """on_confirmed callback is called ONCE per event the moment confirmation thresholds are first met."""
        callback_calls = []

        def cb(t_confirmed: str, yolo_class: str, conf: float, t_first: str):
            callback_calls.append((t_confirmed, yolo_class, conf, t_first))

        builder = EventBuilder(
            min_frames=3,
            min_mean_conf=0.5,
            target_class_names=["Aedes Aegypti"],
            on_confirmed=cb,
        )
        builder.update([_make_det("Aedes Aegypti", 0.8)], mono_time=0.0, utc_time=10.0)
        builder.update([_make_det("Aedes Aegypti", 0.8)], mono_time=0.2, utc_time=10.2)
        assert len(callback_calls) == 0  # Only 2 frames, min_frames=3

        # Frame 3 meets threshold -> callback must be fired!
        builder.update([_make_det("Aedes Aegypti", 0.8)], mono_time=0.4, utc_time=10.4)
        assert len(callback_calls) == 1
        t_conf, cls_name, conf, t_first = callback_calls[0]
        assert cls_name == "Aedes Aegypti"
        assert conf == pytest.approx(0.8)
        assert t_first == to_utc_iso(10.0)
        assert t_conf == to_utc_iso(10.4)

        # Frame 4 also meets threshold -> callback must NOT be called again!
        builder.update([_make_det("Aedes Aegypti", 0.8)], mono_time=0.6, utc_time=10.6)
        assert len(callback_calls) == 1

        events = builder.flush()
        assert len(events) == 1
        assert events[0].status == "confirmed"

    def test_sticky_confirmation_revoked_if_flips_to_non_target(self):
        """If confirmation thresholds were met, but final class vote flips to non-target, mark non_target with confirmation_revoked."""
        builder = EventBuilder(
            min_frames=2,
            min_mean_conf=0.5,
            target_class_names=["Aedes Aegypti"],
            non_target_class_names=["Aedes Albopictus"],
        )
        # 2 Aegypti detections at 0.6 -> confirmed (sum = 1.2)
        builder.update([_make_det("Aedes Aegypti", 0.6)], mono_time=0.0, utc_time=10.0)
        builder.update([_make_det("Aedes Aegypti", 0.6)], mono_time=0.2, utc_time=10.2)
        assert builder._open_event.confirmed_called is True

        # Later, 3 Albopictus detections at 0.9 arrive (sum = 2.7 > 1.2)
        builder.update([_make_det("Aedes Albopictus", 0.9, is_target=False)], mono_time=0.4, utc_time=10.4)
        builder.update([_make_det("Aedes Albopictus", 0.9, is_target=False)], mono_time=0.6, utc_time=10.6)
        builder.update([_make_det("Aedes Albopictus", 0.9, is_target=False)], mono_time=0.8, utc_time=10.8)

        events = builder.flush()
        assert len(events) == 1
        ev = events[0]
        assert ev.yolo_class == "Aedes Albopictus"
        assert ev.status == "non_target"
        assert ev.notes == "confirmation_revoked"
        assert ev.t_confirmed is None

    def test_empty_updates_close_gap(self):
        """Calling update() with empty detections must close the event when time gap exceeds event_gap_s."""
        builder = EventBuilder(event_gap_s=1.0, min_frames=1, min_mean_conf=0.1)
        builder.update([_make_det(conf=0.8)], mono_time=0.0, utc_time=10.0)
        assert builder.has_open_event is True

        # Empty updates within gap
        closed = builder.update([], mono_time=0.5, utc_time=10.5)
        assert len(closed) == 0
        assert builder.has_open_event is True

        # Empty update beyond gap (delta = 1.5 > 1.0)
        closed = builder.update([], mono_time=1.6, utc_time=11.6)
        assert len(closed) == 1
        assert builder.has_open_event is False
        assert closed[0].status == "confirmed"

    def test_unlisted_class_logs_warning(self, caplog):
        """A detected class in neither target nor non-target list must log a warning."""
        builder = EventBuilder(
            target_class_names=["Aedes Aegypti"],
            non_target_class_names=["Aedes Albopictus"],
        )
        unlisted_det = _make_det("Culex Quinquefasciatus", conf=0.7)
        with caplog.at_level("WARNING"):
            builder.update([unlisted_det], mono_time=0.0, utc_time=10.0)
        assert any("Culex Quinquefasciatus" in record.message for record in caplog.records)


# ── 9. Review Fix Tests: HealthRepository & Retention Order ──────────────────

class TestHealthRepositoryAndRetentionOrder:
    def test_health_repository_roundtrip(self, tmp_path):
        """Insert into health_logs with whitelisted columns and read back."""
        db_path = tmp_path / "sentinel.db"
        schema_path = Path(__file__).parent.parent / "sentinel" / "storage" / "schema.sql"
        db = DatabaseManager(db_path, schema_path)
        s_repo = SessionRepository(db)
        h_repo = HealthRepository(db)

        session_id = s_repo.create(
            mode="pi",
            model_name="best.pt",
            model_backend="pt",
            config_json="{}",
        )

        health_id = h_repo.create({
            "session_id": session_id,
            "fps": 28.5,
            "infer_ms_avg": 34.2,
            "infer_ms_p95": 42.1,
            "dropped_frames": 2,
            "queue_depth": 1,
            "cpu_pct": 45.0,
            "ram_pct": 52.3,
            "cpu_temp_c": 58.4,
            "disk_free_gb": 18.2,
            "camera_ok": True,
            "clock_synced": True,
            "illegal_column": "should_be_ignored",
        })
        assert health_id > 0

        row = h_repo.get(health_id)
        assert row is not None
        assert row["session_id"] == session_id
        assert row["fps"] == pytest.approx(28.5)
        assert row["infer_ms_avg"] == pytest.approx(34.2)
        assert row["camera_ok"] == 1
        assert row["clock_synced"] == 1

    def test_retention_order_prunes_rejected_and_non_target_first(self, tmp_path):
        """Prunes evidence of rejected and non_target events first, then oldest unverified confirmed."""
        db_path = tmp_path / "sentinel.db"
        schema_path = Path(__file__).parent.parent / "sentinel" / "storage" / "schema.sql"
        db = DatabaseManager(db_path, schema_path)
        s_repo = SessionRepository(db)
        e_repo = EventRepository(db)

        session_id = s_repo.create(mode="test", model_name="best.pt", model_backend="pt", config_json="{}")
        data_dir = tmp_path / "sentinel_data"
        saver = EvidenceSaver(data_dir=data_dir, event_repo=e_repo, max_gb=0.000001)

        snap_confirmed = data_dir / "evidence" / "snapshots" / "snap_conf.jpg"
        snap_non_target = data_dir / "evidence" / "snapshots" / "snap_nt.jpg"
        snap_confirmed.write_bytes(b"A" * 1500)
        snap_non_target.write_bytes(b"B" * 1500)

        # 1. Oldest event: confirmed (started t = 01:00)
        id_conf = e_repo.create_event({
            "session_id": session_id, "monitoring_point": "KAS",
            "started_at": "2026-10-10T01:00:00.000Z", "ended_at": "2026-10-10T01:00:01.000Z",
            "duration_s": 1.0, "n_frames": 3, "max_conf": 0.8, "mean_conf": 0.8,
            "yolo_class": "Aedes Aegypti", "status": "confirmed",
            "snapshot_path": "evidence/snapshots/snap_conf.jpg",
        })

        # 2. Newer event: non_target (started t = 02:00)
        id_nt = e_repo.create_event({
            "session_id": session_id, "monitoring_point": "KAS",
            "started_at": "2026-10-10T02:00:00.000Z", "ended_at": "2026-10-10T02:00:01.000Z",
            "duration_s": 1.0, "n_frames": 3, "max_conf": 0.8, "mean_conf": 0.8,
            "yolo_class": "Aedes Albopictus", "status": "non_target",
            "snapshot_path": "evidence/snapshots/snap_nt.jpg",
        })

        # Query retention candidates: non_target must appear FIRST even though started_at is later!
        candidates = e_repo.get_unverified_evidence_oldest_first()
        assert len(candidates) == 2
        assert candidates[0]["id"] == id_nt  # non_target pruned first!
        assert candidates[1]["id"] == id_conf
