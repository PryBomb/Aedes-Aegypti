"""Offline video replay script for MosquitoSentinel.

Features:
- --virtual-start (ISO-8601 with UTC offset, e.g. 2026-10-12T06:00:00+08:00).
  Event timestamps = virtual_start + (frame_index / video_fps). NEVER the wall clock.
- Evaluates actual detections against expected_events.json if present.
- Prints PASS/FAIL per episode and total accuracy.
- Prints events table (id, started_at, ended_at, duration_s, n_frames, mean_conf, status).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel.capture.file_source import FileSource
from sentinel.capture.ring_buffer import RingBuffer
from sentinel.config import load_config
from sentinel.detection.detector import Detector
from sentinel.detection.event_builder import Event, EventBuilder
from sentinel.evidence.saver import EvidenceSaver
from sentinel.storage.db import DatabaseManager
from sentinel.storage.repositories import EventRepository, SessionRepository

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("replay_video")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Default epoch used when --virtual-start is NOT specified (for backward compat)
_DEFAULT_BASE_EPOCH_UTC = 1700000000.0


def parse_virtual_start(s: str) -> float:
    """Parse an ISO-8601 datetime string (with offset) and return as UTC epoch seconds."""
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --virtual-start {s!r}: expected ISO-8601 with offset "
            f"(e.g. 2026-10-12T06:00:00+08:00). Error: {exc}"
        )
    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"--virtual-start {s!r} has no UTC offset. "
            "Please include an offset, e.g. 2026-10-12T06:00:00+08:00"
        )
    return dt.astimezone(timezone.utc).timestamp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline replay of a video through MosquitoSentinel pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to video file to replay")
    parser.add_argument("--weights", default="models/best.pt", help="Path to model weights")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data-dir", default="", help="Override base data_dir outside OneDrive")
    parser.add_argument("--expected", default="", help="Path to expected_events.json ground truth")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--virtual-start",
        default="",
        help=(
            "Virtual start time for event timestamps (ISO-8601 with UTC offset). "
            "Event time = virtual_start + frame_index / fps. "
            "Example: 2026-10-12T06:00:00+08:00"
        ),
    )
    return parser.parse_args()


def compare_with_expected(
    expected_path: Path,
    session_events: List[Event],
    base_epoch_utc: float,
) -> None:
    """Compare recorded events against ground truth expected_events.json."""
    if not expected_path.exists():
        logger.info("No ground truth file found at %s — skipping comparison.", expected_path)
        return

    with open(expected_path, "r", encoding="utf-8") as f:
        expected_episodes: List[Dict[str, Any]] = json.load(f)

    print("\n+------------------------------------------------------------------------------------------------+")
    print("| GROUND TRUTH VALIDATION (Actual vs Expected)")
    print("+---------+-------------------+-----------------+-------------------+---------------+------------+")
    print("| Episode | Expected Class    | Expected Status | Actual Class      | Actual Status | Result     |")
    print("+---------+-------------------+-----------------+-------------------+---------------+------------+")

    passed_count = 0
    total_episodes = len(expected_episodes)

    for ep in expected_episodes:
        ep_num = ep.get("episode", 0)
        exp_cls = ep.get("ground_truth_class", "None")
        exp_status = ep.get("expected_status", "none")
        start_s = ep.get("start_time_s", 0.0)
        end_s = ep.get("end_time_s", 0.0)

        # Find matching events overlapping with this episode's time window
        matching_events = [
            ev for ev in session_events
            if (start_s - 1.0) <= (ev.first_detect_mono) <= (end_s + 1.0)
        ]

        if exp_status == "none":
            if not matching_events:
                passed = True
                act_cls = "None"
                act_status = "none"
            else:
                passed = False
                act_cls = matching_events[0].yolo_class
                act_status = matching_events[0].status
        else:
            if matching_events:
                best_ev = matching_events[0]
                act_cls = best_ev.yolo_class
                act_status = best_ev.status
                passed = (act_cls.lower() == exp_cls.lower()) and (act_status == exp_status)
            else:
                passed = False
                act_cls = "None"
                act_status = "none"

        if passed:
            passed_count += 1
            result_str = "[PASS]"
        else:
            result_str = "[FAIL]"

        print(
            f"| {ep_num:<7} | {exp_cls:<17} | {exp_status:<15} | {act_cls:<17} | {act_status:<13} | {result_str:<10} |"
        )

    print("+---------+-------------------+-----------------+-------------------+---------------+------------+")
    pct = (passed_count / total_episodes * 100.0) if total_episodes > 0 else 0.0
    print(f"| OVERALL: {passed_count}/{total_episodes} episodes passed ({pct:.1f}% accuracy)")
    print("+------------------------------------------------------------------------------------------------+\n")


def print_events_table(event_repo: EventRepository, session_id: int) -> None:
    """Print stored events as a formatted table."""
    rows = []
    with event_repo.db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, ended_at, duration_s, n_frames, mean_conf, status
            FROM events WHERE session_id = ?
            ORDER BY started_at ASC
            """,
            (session_id,),
        ).fetchall()

    print("\n+------+------------------------+------------------------+------------+---------+-----------+-----------+")
    print("|  id  | started_at             | ended_at               | duration_s | n_frames | mean_conf | status    |")
    print("+------+------------------------+------------------------+------------+---------+-----------+-----------+")
    for r in rows:
        print(
            f"| {r['id']:<4} | {r['started_at']:<22} | {r['ended_at']:<22} "
            f"| {r['duration_s']:<10.3f} | {r['n_frames']:<7} | {r['mean_conf']:<9.4f} | {r['status']:<9} |"
        )
    print("+------+------------------------+------------------------+------------+---------+-----------+-----------+")
    print(f"  Total: {len(rows)} events\n")


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Video file does not exist: %s", video_path)
        sys.exit(1)

    # 1. Load config
    config_path = _PROJECT_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path, project_root=_PROJECT_ROOT)

    # 2. Resolve virtual start epoch
    if args.virtual_start:
        base_epoch_utc = parse_virtual_start(args.virtual_start)
        logger.info(
            "Virtual start: %s  (epoch %.3f UTC)",
            args.virtual_start, base_epoch_utc,
        )
    else:
        base_epoch_utc = _DEFAULT_BASE_EPOCH_UTC
        logger.info("No --virtual-start given; using default epoch %.3f", base_epoch_utc)

    # 3. Setup isolated data_dir
    data_dir = Path(args.data_dir) if args.data_dir else config.get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using data directory: %s", data_dir)

    # 4. Setup SQLite database
    db_path = data_dir / "sentinel.db"
    schema_path = _PROJECT_ROOT / "sentinel" / "storage" / "schema.sql"
    db_manager = DatabaseManager(db_path, schema_path)
    session_repo = SessionRepository(db_manager)
    event_repo = EventRepository(db_manager)

    weights_path = _PROJECT_ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    session_id = session_repo.create(
        mode="replay",
        model_name=weights_path.name,
        model_backend=config.yolo.backend,
        config_json=config.to_dict(),
        deployment_label=f"offline-replay-{video_path.name}",
    )
    logger.info("Created replay session ID: %d", session_id)

    # 5. Detector
    detector = Detector(
        weights_path=str(weights_path),
        backend=config.yolo.backend,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=config.yolo.iou,
        target_class_names=config.yolo.target_class_names,
        non_target_class_names=config.yolo.non_target_class_names,
        warmup_runs=1,
    )
    detector.load()

    # 6. Ring buffer and Evidence saver
    ring_buffer = RingBuffer(duration_s=config.capture.preroll_s + 2.0, max_memory_mb=50.0)
    evidence_saver = EvidenceSaver(
        data_dir=data_dir,
        event_repo=event_repo,
        max_gb=float(config.evidence.max_gb),
        save_clips=config.evidence.save_clips,
    )

    # 7. Event builder
    event_builder = EventBuilder(
        event_gap_s=config.events.event_gap_s,
        min_frames=config.events.min_frames,
        min_mean_conf=config.events.min_mean_conf,
        target_class_names=config.yolo.target_class_names,
        non_target_class_names=config.yolo.non_target_class_names,
        session_id=session_id,
        monitoring_point=config.site.monitoring_point,
    )

    # 8. Replay video frames using pure VIDEO timestamps
    source = FileSource(video_path, real_time=False, loop=False)
    frame_count = 0
    total_detections = 0
    recorded_events: List[Event] = []

    with source:
        video_fps = source.source_fps or 30.0
        frame_interval_s = 1.0 / video_fps

        while True:
            frame = source.read()
            if frame is None:
                break
            frame_count += 1

            # Deterministic VIDEO timestamps: virtual_start + frame_index / fps
            video_time_s = (frame_count - 1) * frame_interval_s
            frame.timestamp_mono = video_time_s
            frame.timestamp_utc = base_epoch_utc + video_time_s

            ring_buffer.push(frame)

            detections, _ = detector.detect(frame.bgr)
            total_detections += len(detections)

            # update must be called on every frame, even with empty detections
            closed_events = event_builder.update(
                detections=detections,
                frame=frame,
                mono_time=video_time_s,
                utc_time=base_epoch_utc + video_time_s,
            )
            for ev in closed_events:
                if ev.status in ("confirmed", "non_target"):
                    evidence_saver.save_evidence(ev, ring_buffer=ring_buffer)
                event_repo.create_event(ev)
                recorded_events.append(ev)

        # 9. Flush open events at end of video
        end_video_time_s = frame_count * frame_interval_s
        flushed_events = event_builder.flush(
            mono_time=end_video_time_s,
            utc_time=base_epoch_utc + end_video_time_s,
        )
        for ev in flushed_events:
            if ev.status in ("confirmed", "non_target"):
                evidence_saver.save_evidence(ev, ring_buffer=ring_buffer)
            event_repo.create_event(ev)
            recorded_events.append(ev)

    session_repo.end_session(session_id)

    # 10. Print summary table
    status_counts = event_repo.count_by_status(session_id)
    class_counts = event_repo.count_by_class(session_id)
    total_events = sum(status_counts.values())

    print("\n+----------------------------------------------------+")
    print(f"| REPLAY SUMMARY: {video_path.name}")
    print("+----------------------------------------------------+")
    print(f"|  Frames processed   : {frame_count}")
    print(f"|  Raw detections     : {total_detections}")
    print(f"|  Total events       : {total_events}")
    if args.virtual_start:
        print(f"|  Virtual start      : {args.virtual_start}")
    print("+----------------------------------------------------+")
    print("|  Events by Status:")
    for status, cnt in sorted(status_counts.items()):
        print(f"|    - {status:<15}: {cnt}")
    print("+----------------------------------------------------+")
    print("|  Events by YOLO Class:")
    for cls_name, cnt in sorted(class_counts.items()):
        print(f"|    - {cls_name:<15}: {cnt}")
    print("+----------------------------------------------------+")

    # 11. Events detail table
    print_events_table(event_repo, session_id)

    # 12. Compare against expected_events.json if available
    expected_file = Path(args.expected) if args.expected else video_path.with_name("expected_events.json")
    if expected_file.exists():
        compare_with_expected(expected_file, recorded_events, base_epoch_utc)


if __name__ == "__main__":
    main()
