"""Pure-logic event builder for MosquitoSentinel.

No I/O or database access — injectable clock and pure state accumulation.

Rules:
- update() MUST be called for every processed frame, including frames with an empty
  detection list (detections=[]). A time gap where only empty updates occur exceeding
  event_gap_s will close the ongoing event.
- A detection joins an open event if within events.event_gap_s of its last detection.
- Class vote: sum confidence per class across all detections; label = highest sum.
  If the winning class is non-target -> status is 'non_target'.
- All event statistics (n_frames, mean_conf, max_conf, best_detection, best_frame)
  are computed from the WINNING class only.
- An optional on_confirmed callback is called ONCE per event the moment the confirmation
  thresholds are first met (passing t_confirmed, class, conf, t_first_detect).
- Confirmation is sticky: if confirmed thresholds are met, the event is confirmed unless
  the final class vote flips to non-target at close. If it flips to non-target, status is
  set to 'non_target' and notes is set to 'confirmation_revoked'.
- Target event is confirmed when n_frames >= min_frames and mean_conf >= min_mean_conf
  for the winning target class; otherwise 'rejected' (still stored).
- A warning is logged when a detected class is in neither target nor non-target list.
- flush() closes any open event (shutdown/EOF). Optional track_id stored if given.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from sentinel.capture.base import Frame
from sentinel.detection.detector import Detection

logger = logging.getLogger(__name__)


def to_utc_iso(ts_seconds: float) -> str:
    """Format epoch seconds as UTC ISO-8601 string with milliseconds and 'Z' suffix."""
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class Event:
    """Represents a closed or finalized detection event ready for storage."""
    session_id: int = 1
    monitoring_point: str = "KAS-1E-01"
    started_at: str = ""
    ended_at: str = ""
    duration_s: float = 0.0
    n_frames: int = 0
    max_conf: float = 0.0
    mean_conf: float = 0.0
    yolo_class: str = ""
    cnn_label: Optional[str] = None
    cnn_conf: Optional[float] = None
    track_id: Optional[int] = None
    status: str = "rejected"  # 'confirmed' | 'rejected' | 'non_target'
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    t_first_detect: Optional[str] = None
    t_confirmed: Optional[str] = None
    t_saved: Optional[str] = None
    verified_label: Optional[str] = None
    verified_by: Optional[str] = None
    verified_at: Optional[str] = None
    notes: Optional[str] = None
    id: Optional[int] = None

    # In-memory evidence context (not stored in DB columns)
    best_detection: Optional[Detection] = None
    best_frame: Optional[Frame] = None
    first_detect_mono: float = 0.0
    last_detect_mono: float = 0.0

    def to_db_dict(self) -> Dict[str, object]:
        """Convert fields to a dictionary matching the SQLite 'events' schema."""
        return {
            "session_id": self.session_id,
            "monitoring_point": self.monitoring_point,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": round(self.duration_s, 4),
            "n_frames": self.n_frames,
            "max_conf": round(self.max_conf, 4),
            "mean_conf": round(self.mean_conf, 4),
            "yolo_class": self.yolo_class,
            "cnn_label": self.cnn_label,
            "cnn_conf": self.cnn_conf,
            "track_id": self.track_id,
            "status": self.status,
            "snapshot_path": self.snapshot_path,
            "clip_path": self.clip_path,
            "t_first_detect": self.t_first_detect,
            "t_confirmed": self.t_confirmed,
            "t_saved": self.t_saved,
            "verified_label": self.verified_label,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at,
            "notes": self.notes,
        }


class _ClassDetections:
    """Accumulator for detections of a single class within an event."""

    def __init__(self, class_name: str) -> None:
        self.class_name = class_name
        self.confs: List[float] = []
        self.frame_ids: Set[int] = set()
        self.best_conf: float = -1.0
        self.best_detection: Optional[Detection] = None
        self.best_frame: Optional[Frame] = None

    def add(self, det: Detection, frame: Optional[Frame], synthetic_frame_id: int) -> None:
        self.confs.append(det.conf)
        f_id = frame.frame_id if frame is not None else synthetic_frame_id
        self.frame_ids.add(f_id)
        if det.conf > self.best_conf:
            self.best_conf = det.conf
            self.best_detection = det
            self.best_frame = frame

    @property
    def conf_sum(self) -> float:
        return sum(self.confs)

    @property
    def n_frames(self) -> int:
        return len(self.frame_ids)

    @property
    def mean_conf(self) -> float:
        return (sum(self.confs) / len(self.confs)) if self.confs else 0.0

    @property
    def max_conf(self) -> float:
        return max(self.confs) if self.confs else 0.0


class _OpenEventState:
    """Internal mutable accumulator for an ongoing event."""

    def __init__(
        self,
        session_id: int,
        monitoring_point: str,
        track_id: Optional[int],
        mono_now: float,
        utc_now: float,
    ) -> None:
        self.session_id = session_id
        self.monitoring_point = monitoring_point
        self.track_id = track_id
        self.first_detect_mono = mono_now
        self.last_detect_mono = mono_now
        self.first_detect_utc = utc_now
        self.last_detect_utc = utc_now

        self.t_first_detect: str = to_utc_iso(utc_now)
        self.t_confirmed: Optional[str] = None
        self.confirmed_called: bool = False

        self.classes: Dict[str, _ClassDetections] = {}
        self._synthetic_frame_counter: int = 0

    def add_frame_detections(
        self,
        detections: List[Detection],
        frame: Optional[Frame],
        mono_now: float,
        utc_now: float,
        track_id: Optional[int] = None,
    ) -> None:
        self.last_detect_mono = mono_now
        self.last_detect_utc = utc_now
        if track_id is not None:
            self.track_id = track_id

        self._synthetic_frame_counter += 1
        for det in detections:
            if det.class_name not in self.classes:
                self.classes[det.class_name] = _ClassDetections(det.class_name)
            self.classes[det.class_name].add(det, frame, self._synthetic_frame_counter)

    def winning_class(self) -> str:
        if not self.classes:
            return ""
        return max(self.classes.items(), key=lambda item: item[1].conf_sum)[0]

    def winning_stats(self) -> Optional[_ClassDetections]:
        winner = self.winning_class()
        return self.classes.get(winner)


class EventBuilder:
    """Stateful event aggregator with injectable clock and pure logic."""

    def __init__(
        self,
        event_gap_s: float = 1.5,
        min_frames: int = 3,
        min_mean_conf: float = 0.4,
        target_class_names: Optional[List[str]] = None,
        non_target_class_names: Optional[List[str]] = None,
        session_id: int = 1,
        monitoring_point: str = "KAS-1E-01",
        on_confirmed: Optional[Callable[[str, str, float, str], None]] = None,
        mono_clock: Optional[Callable[[], float]] = None,
        utc_clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.event_gap_s = event_gap_s
        self.min_frames = min_frames
        self.min_mean_conf = min_mean_conf
        self.target_class_names = [
            n.lower() for n in (target_class_names if target_class_names is not None else ["Aedes Aegypti"])
        ]
        self.non_target_class_names = [
            n.lower() for n in (non_target_class_names if non_target_class_names is not None else ["Aedes Albopictus"])
        ]
        self.session_id = session_id
        self.monitoring_point = monitoring_point
        self.on_confirmed = on_confirmed

        # Injectable clocks
        import time
        self._mono_clock = mono_clock if mono_clock is not None else time.monotonic
        self._utc_clock = utc_clock if utc_clock is not None else time.time

        self._open_event: Optional[_OpenEventState] = None

    @property
    def has_open_event(self) -> bool:
        """True if an event is currently accumulating."""
        return self._open_event is not None

    def update(
        self,
        detections: List[Detection],
        frame: Optional[Frame] = None,
        mono_time: Optional[float] = None,
        utc_time: Optional[float] = None,
        track_id: Optional[int] = None,
    ) -> List[Event]:
        """Process incoming detections and return any finalized events.

        CRITICAL USAGE REQUIREMENT:
        update() MUST be called for every processed video frame, including when
        detections is empty (detections=[]). When no detections occur across frames
        and the elapsed monotonic time exceeds event_gap_s, the ongoing open event
        times out and is closed and returned by this call.

        Args:
            detections: List of Detection instances for this frame (can be empty).
            frame: Optional Frame object carrying bgr, frame_id, and timestamps.
            mono_time: Optional monotonic timestamp override.
            utc_time: Optional UTC timestamp override (seconds).
            track_id: Optional tracker ID associated with detections.

        Returns:
            List of closed Event instances (if any event timed out during this update).
        """
        now_mono = (
            mono_time
            if mono_time is not None
            else (frame.timestamp_mono if frame is not None else self._mono_clock())
        )
        now_utc = (
            utc_time
            if utc_time is not None
            else (frame.timestamp_utc if frame is not None else self._utc_clock())
        )

        closed_events: List[Event] = []

        # Check if active event timed out due to gap
        if self._open_event is not None:
            if (now_mono - self._open_event.last_detect_mono) > self.event_gap_s:
                closed_events.append(self._finalize_event(self._open_event))
                self._open_event = None

        if detections:
            # Check for classes not in target or non-target lists
            for det in detections:
                cls_lower = det.class_name.lower()
                if (cls_lower not in self.target_class_names) and (cls_lower not in self.non_target_class_names):
                    logger.warning(
                        "Detected class '%s' is in neither target nor non-target list",
                        det.class_name,
                    )

            if self._open_event is None:
                self._open_event = _OpenEventState(
                    session_id=self.session_id,
                    monitoring_point=self.monitoring_point,
                    track_id=track_id,
                    mono_now=now_mono,
                    utc_now=now_utc,
                )

            self._open_event.add_frame_detections(
                detections=detections,
                frame=frame,
                mono_now=now_mono,
                utc_now=now_utc,
                track_id=track_id,
            )

            # Check confirmation thresholds for winning class
            winner_stats = self._open_event.winning_stats()
            if winner_stats is not None:
                winner = winner_stats.class_name
                if winner.lower() in self.target_class_names:
                    if (
                        winner_stats.n_frames >= self.min_frames
                        and winner_stats.mean_conf >= self.min_mean_conf
                    ):
                        if not self._open_event.confirmed_called:
                            self._open_event.confirmed_called = True
                            self._open_event.t_confirmed = to_utc_iso(now_utc)
                            if self.on_confirmed is not None:
                                self.on_confirmed(
                                    self._open_event.t_confirmed,
                                    winner,
                                    winner_stats.mean_conf,
                                    self._open_event.t_first_detect,
                                )

        return closed_events

    def flush(
        self,
        mono_time: Optional[float] = None,
        utc_time: Optional[float] = None,
    ) -> List[Event]:
        """Close and return any open event immediately (e.g. on shutdown/EOF)."""
        if self._open_event is not None:
            event = self._finalize_event(self._open_event)
            self._open_event = None
            return [event]
        return []

    def _finalize_event(self, state: _OpenEventState) -> Event:
        """Resolve final status, winner class, and compute metrics from WINNING class only."""
        winner_stats = state.winning_stats()
        if winner_stats is None:
            winner = ""
            n_frames = 0
            mean_conf = 0.0
            max_conf = 0.0
            best_detection = None
            best_frame = None
        else:
            winner = winner_stats.class_name
            # Compute stats from the WINNING class only
            n_frames = winner_stats.n_frames
            mean_conf = winner_stats.mean_conf
            max_conf = winner_stats.max_conf
            best_detection = winner_stats.best_detection
            best_frame = winner_stats.best_frame

        winner_lower = winner.lower()
        is_target = winner_lower in self.target_class_names
        is_non_target = winner_lower in self.non_target_class_names

        notes: Optional[str] = None

        if is_non_target or (not is_target):
            status = "non_target"
            t_confirmed = None
            if state.confirmed_called:
                # Sticky confirmation flipped to non-target at close!
                notes = "confirmation_revoked"
        else:
            # Target class
            if state.confirmed_called:
                # Sticky confirmation: once thresholds were met for target, stays confirmed
                status = "confirmed"
                t_confirmed = state.t_confirmed or to_utc_iso(state.last_detect_utc)
            elif n_frames >= self.min_frames and mean_conf >= self.min_mean_conf:
                status = "confirmed"
                t_confirmed = state.t_confirmed or to_utc_iso(state.last_detect_utc)
            else:
                status = "rejected"
                t_confirmed = None

        duration_s = max(0.0, state.last_detect_mono - state.first_detect_mono)

        return Event(
            session_id=state.session_id,
            monitoring_point=state.monitoring_point,
            started_at=to_utc_iso(state.first_detect_utc),
            ended_at=to_utc_iso(state.last_detect_utc),
            duration_s=duration_s,
            n_frames=n_frames,
            max_conf=max_conf,
            mean_conf=mean_conf,
            yolo_class=winner,
            track_id=state.track_id,
            status=status,
            t_first_detect=state.t_first_detect,
            t_confirmed=t_confirmed,
            notes=notes,
            best_detection=best_detection,
            best_frame=best_frame,
            first_detect_mono=state.first_detect_mono,
            last_detect_mono=state.last_detect_mono,
        )
