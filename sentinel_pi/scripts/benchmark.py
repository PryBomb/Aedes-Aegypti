"""Benchmark script: measure capture-only and end-to-end FPS per backend/imgsz.

Usage examples
--------------
# Benchmark file source on laptop (no Pi required):
  python scripts/benchmark.py --source file --video path/to/video.mp4 \
      --weights models/best.pt --imgsz 640 320 --duration 30

# Multiple weight files and sizes:
  python scripts/benchmark.py --source file --video path/to/video.mp4 \
      --weights models/best.pt models/best_ncnn_model \
      --imgsz 640 320 --duration 30 --warmup 5

Output: prints a summary table, saves CSV to data/benchmarks/<timestamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Optional

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure project root is on sys.path ──────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel.capture.file_source import FileSource
from sentinel.capture.usb_camera import UsbCamera
from sentinel.capture.base import Frame

logger = logging.getLogger("benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cpu_temp() -> Optional[float]:
    """Return CPU temperature in °C if readable (Pi / Linux only)."""
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal.exists():
        try:
            return int(thermal.read_text().strip()) / 1000.0
        except Exception:
            pass
    return None


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _platform_info() -> dict:
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


# ── Capture-only benchmark ────────────────────────────────────────────────────

def benchmark_capture_only(source, duration_s: float) -> dict:
    """Measure raw capture FPS without inference."""
    count = 0
    t_start = time.monotonic()
    t_end = t_start + duration_s

    while time.monotonic() < t_end:
        frame = source.read()
        if frame is None:
            break
        count += 1

    elapsed = time.monotonic() - t_start
    fps = count / elapsed if elapsed > 0 else 0.0
    return {"capture_frames": count, "capture_elapsed_s": round(elapsed, 3), "capture_fps": round(fps, 2)}


# ── End-to-end benchmark ──────────────────────────────────────────────────────

def benchmark_e2e(source, weights: str, imgsz: int, duration_s: float, warmup_runs: int) -> dict:
    """Measure end-to-end FPS and inference latency statistics."""
    try:
        from sentinel.detection.detector import Detector
    except ImportError as e:
        logger.error("Could not import Detector: %s", e)
        return {"error": str(e)}

    detector = Detector(
        weights_path=weights,
        imgsz=imgsz,
        warmup_runs=warmup_runs,
    )
    try:
        detector.load()
    except Exception as e:
        logger.error("Model load failed: %s", e)
        return {"error": str(e)}

    infer_times: List[float] = []
    frames_processed = 0
    frames_dropped = 0

    t_start = time.monotonic()
    t_end = t_start + duration_s

    while time.monotonic() < t_end:
        frame = source.read()
        if frame is None:
            break
        frames_processed += 1

        try:
            _, infer_ms = detector.detect(frame.bgr)
            infer_times.append(infer_ms)
        except Exception as e:
            logger.warning("Inference error on frame %d: %s", frame.frame_id, e)

    elapsed = time.monotonic() - t_start
    e2e_fps = frames_processed / elapsed if elapsed > 0 else 0.0

    result = {
        "weights": str(Path(weights).name),
        "imgsz": imgsz,
        "e2e_frames": frames_processed,
        "e2e_elapsed_s": round(elapsed, 3),
        "e2e_fps": round(e2e_fps, 2),
        "infer_mean_ms": round(mean(infer_times), 2) if infer_times else None,
        "infer_p50_ms":  round(_percentile(infer_times, 50), 2) if infer_times else None,
        "infer_p95_ms":  round(_percentile(infer_times, 95), 2) if infer_times else None,
        "infer_p99_ms":  round(_percentile(infer_times, 99), 2) if infer_times else None,
        "infer_min_ms":  round(min(infer_times), 2) if infer_times else None,
        "infer_max_ms":  round(max(infer_times), 2) if infer_times else None,
        "cpu_temp_c":    _cpu_temp(),
    }
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MosquitoSentinel benchmark — capture and inference throughput",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", choices=["file", "usb", "pi"], default="file",
                   help="Frame source type")
    p.add_argument("--video", default="",
                   help="Path to video file (required when --source=file)")
    p.add_argument("--usb-index", type=int, default=0,
                   help="USB camera device index (when --source=usb)")
    p.add_argument("--weights", nargs="+", default=[],
                   help="One or more model weight paths to benchmark. "
                        "Omit to run capture-only.")
    p.add_argument("--imgsz", nargs="+", type=int, default=[640],
                   help="One or more inference image sizes to test")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Benchmark duration per combination (seconds)")
    p.add_argument("--warmup", type=int, default=3,
                   help="Number of warm-up inferences before timing")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--fps", type=int, default=30, help="Capture FPS (file/usb)")
    return p.parse_args()


def make_source(args: argparse.Namespace):
    """Instantiate the appropriate FrameSource."""
    if args.source == "file":
        if not args.video:
            logger.error("--video is required when --source=file")
            sys.exit(1)
        return FileSource(args.video, real_time=False, loop=True)
    elif args.source == "usb":
        return UsbCamera(device_index=args.usb_index, width=args.width,
                         height=args.height, fps=args.fps)
    elif args.source == "pi":
        try:
            from sentinel.capture.pi_camera import PiCamera
        except RuntimeError as e:
            logger.error(str(e))
            sys.exit(1)
        return PiCamera(width=args.width, height=args.height, fps=args.fps)
    raise ValueError(f"Unknown source: {args.source}")


def main() -> None:
    args = parse_args()
    pinfo = _platform_info()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    out_dir = _PROJECT_ROOT / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"benchmark_{ts}.csv"

    all_rows: List[dict] = []

    # ── Capture-only pass ──────────────────────────────────────────────────
    logger.info("=== Capture-only benchmark (source=%s, %.0f s) ===", args.source, args.duration)
    source = make_source(args)
    with source:
        cap_result = benchmark_capture_only(source, args.duration)

    cap_row = {
        "timestamp": ts,
        "source": args.source,
        "test": "capture_only",
        **cap_result,
        **{f"platform_{k}": v for k, v in pinfo.items()},
        "cpu_temp_c": _cpu_temp(),
    }
    all_rows.append(cap_row)

    _print_capture_summary(cap_result)

    # ── End-to-end passes ──────────────────────────────────────────────────
    for weights in args.weights:
        for imgsz in args.imgsz:
            logger.info(
                "=== E2E benchmark: %s  imgsz=%d  source=%s  (%.0f s) ===",
                Path(weights).name, imgsz, args.source, args.duration,
            )
            source = make_source(args)
            with source:
                e2e_result = benchmark_e2e(source, weights, imgsz, args.duration, args.warmup)

            row = {
                "timestamp": ts,
                "source": args.source,
                "test": "e2e",
                **e2e_result,
                **{f"platform_{k}": v for k, v in pinfo.items()},
            }
            all_rows.append(row)
            _print_e2e_summary(e2e_result)

    # ── Write CSV ──────────────────────────────────────────────────────────
    all_keys: list[str] = []
    for row in all_rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Results saved to: %s", csv_path)


def _print_capture_summary(r: dict) -> None:
    print("\n+--- Capture-only --------------------------+")
    print(f"|  Frames captured : {r.get('capture_frames', '?')}")
    print(f"|  Duration        : {r.get('capture_elapsed_s', '?')} s")
    print(f"|  FPS             : {r.get('capture_fps', '?')}")
    print("+--------------------------------------------+")


def _print_e2e_summary(r: dict) -> None:
    if "error" in r:
        print(f"\n  ERROR: {r['error']}")
        return
    print(f"\n+--- E2E: {r.get('weights','?')}  imgsz={r.get('imgsz','?')} --+")
    print(f"|  Frames processed : {r.get('e2e_frames', '?')}")
    print(f"|  Duration         : {r.get('e2e_elapsed_s', '?')} s")
    print(f"|  E2E FPS          : {r.get('e2e_fps', '?')}")
    print(f"|  Infer mean ms    : {r.get('infer_mean_ms', '?')}")
    print(f"|  Infer p50  ms    : {r.get('infer_p50_ms', '?')}")
    print(f"|  Infer p95  ms    : {r.get('infer_p95_ms', '?')}")
    print(f"|  Infer p99  ms    : {r.get('infer_p99_ms', '?')}")
    print(f"|  CPU temp         : {r.get('cpu_temp_c', 'N/A')} deg C")
    print("+----------------------------------------------------+")


if __name__ == "__main__":
    main()
