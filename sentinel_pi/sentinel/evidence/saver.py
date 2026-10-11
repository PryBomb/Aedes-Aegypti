"""Evidence saver and retention manager for MosquitoSentinel.

Saves:
- Annotated JPEG snapshots of the best-confidence detection frame.
- MP4 video clips combining pre-roll from the RingBuffer and post-roll frames.

Retention:
- Enforces evidence.max_gb cap.
- Deletes oldest UNVERIFIED evidence first.
- Evidence tied to verified events (verified_label IS NOT NULL) is strictly protected and never deleted.
- Paths stored in database and returned are RELATIVE to data_dir.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sentinel.capture.base import Frame
from sentinel.capture.ring_buffer import RingBuffer, RingSlot
from sentinel.detection.event_builder import Event
from sentinel.storage.repositories import EventRepository

logger = logging.getLogger(__name__)


class EvidenceSaver:
    """Manages saving annotated snapshots, clips, and retention cleanup."""

    def __init__(
        self,
        data_dir: str | Path,
        event_repo: Optional[EventRepository] = None,
        max_gb: float = 20.0,
        save_clips: bool = True,
        jpeg_quality: int = 90,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.event_repo = event_repo
        self.max_gb = max_gb
        self.save_clips = save_clips
        self.jpeg_quality = jpeg_quality

        self.snapshots_dir = self.data_dir / "evidence" / "snapshots"
        self.clips_dir = self.data_dir / "evidence" / "clips"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save_evidence(
        self,
        event: Event,
        ring_buffer: Optional[RingBuffer] = None,
        postroll_frames: Optional[List[Frame]] = None,
        preroll_s: float = 3.0,
        fps: float = 30.0,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Save snapshot and clip for confirmed or non_target events.

        Returns:
            (rel_snapshot_path, rel_clip_path): relative paths to data_dir.
        """
        # Only save evidence for confirmed or non_target events
        if event.status not in ("confirmed", "non_target"):
            return None, None

        ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = f"event_{event.id or 'pending'}_{ts_slug}"

        # 1. Snapshot
        rel_snapshot = self._save_snapshot(event, prefix)
        event.snapshot_path = rel_snapshot

        # 2. Clip
        rel_clip = None
        if self.save_clips:
            rel_clip = self._save_clip(
                event=event,
                prefix=prefix,
                ring_buffer=ring_buffer,
                postroll_frames=postroll_frames,
                preroll_s=preroll_s,
                fps=fps,
            )
            event.clip_path = rel_clip

        # 3. Check retention
        self.enforce_retention()

        return rel_snapshot, rel_clip

    def _save_snapshot(self, event: Event, prefix: str) -> Optional[str]:
        """Save best-confidence frame as an annotated JPEG snapshot."""
        if event.best_frame is None or event.best_frame.bgr is None:
            return None

        img = event.best_frame.bgr.copy()
        det = event.best_detection

        if det is not None:
            x1, y1, x2, y2 = [int(v) for v in det.xyxy]
            color = (0, 255, 0) if event.status == "confirmed" else (0, 165, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.conf:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        filename = f"{prefix}_snap.jpg"
        abs_path = self.snapshots_dir / filename
        ok = cv2.imwrite(str(abs_path), img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            logger.error("Failed to save snapshot to %s", abs_path)
            return None

        rel_path = f"evidence/snapshots/{filename}"
        logger.info("Saved snapshot: %s", rel_path)
        return rel_path

    def _save_clip(
        self,
        event: Event,
        prefix: str,
        ring_buffer: Optional[RingBuffer],
        postroll_frames: Optional[List[Frame]],
        preroll_s: float,
        fps: float,
    ) -> Optional[str]:
        """Decode pre-roll from ring buffer and append post-roll frames into an MP4 clip."""
        frames_to_write: List[np.ndarray] = []

        # Collect pre-roll from ring buffer
        if ring_buffer is not None:
            cutoff = event.first_detect_mono - preroll_s
            slots: List[RingSlot] = ring_buffer.get_since(cutoff)
            for slot in slots:
                # Only include frames captured up to the event end (or current time)
                bgr = cv2.imdecode(np.frombuffer(slot.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    frames_to_write.append(bgr)

        # Collect post-roll frames if passed
        if postroll_frames:
            for f in postroll_frames:
                if f.bgr is not None:
                    frames_to_write.append(f.bgr)

        if not frames_to_write:
            # If no buffer available, use the best frame as a 1-frame clip fallback
            if event.best_frame is not None and event.best_frame.bgr is not None:
                frames_to_write.append(event.best_frame.bgr)
            else:
                return None

        filename = f"{prefix}_clip.mp4"
        abs_path = self.clips_dir / filename

        h, w = frames_to_write[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(abs_path), fourcc, fps, (w, h))

        if not writer.isOpened():
            logger.warning("Could not open VideoWriter for clip: %s", abs_path)
            return None

        for bgr in frames_to_write:
            # Resize if dimensions differ
            if bgr.shape[:2] != (h, w):
                bgr = cv2.resize(bgr, (w, h))
            writer.write(bgr)
        writer.release()

        rel_path = f"evidence/clips/{filename}"
        logger.info("Saved clip: %s (%d frames)", rel_path, len(frames_to_write))
        return rel_path

    def get_total_evidence_bytes(self) -> int:
        """Calculate total disk usage of the evidence directory."""
        evidence_root = self.data_dir / "evidence"
        if not evidence_root.exists():
            return 0
        total = 0
        for f in evidence_root.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    def enforce_retention(self) -> int:
        """Prune oldest UNVERIFIED evidence files if exceeding max_gb.

        Never touches verified events (verified_label IS NOT NULL).
        Returns the number of files deleted.
        """
        max_bytes = int(self.max_gb * 1024 * 1024 * 1024)
        current_bytes = self.get_total_evidence_bytes()
        if current_bytes <= max_bytes:
            return 0

        logger.info(
            "Evidence size (%d bytes) exceeds cap (%d bytes) — running retention cleaner",
            current_bytes,
            max_bytes,
        )

        files_deleted = 0

        # Query unverified evidence rows from DB, ordered oldest first
        if self.event_repo is not None:
            unverified_rows = self.event_repo.get_unverified_evidence_oldest_first()
            for row in unverified_rows:
                event_id = row["id"]
                snap_rel = row["snapshot_path"]
                clip_rel = row["clip_path"]

                # Delete snapshot
                if snap_rel:
                    snap_abs = self.data_dir / snap_rel
                    if snap_abs.exists():
                        try:
                            snap_size = snap_abs.stat().st_size
                            snap_abs.unlink()
                            current_bytes -= snap_size
                            files_deleted += 1
                        except OSError as e:
                            logger.warning("Failed to delete snapshot %s: %s", snap_abs, e)

                # Delete clip
                if clip_rel:
                    clip_abs = self.data_dir / clip_rel
                    if clip_abs.exists():
                        try:
                            clip_size = clip_abs.stat().st_size
                            clip_abs.unlink()
                            current_bytes -= clip_size
                            files_deleted += 1
                        except OSError as e:
                            logger.warning("Failed to delete clip %s: %s", clip_abs, e)

                # Clear evidence paths in DB for this unverified event
                self.event_repo.update_evidence_paths(event_id, snapshot_path=None, clip_path=None)

                if current_bytes <= max_bytes:
                    break

        return files_deleted
