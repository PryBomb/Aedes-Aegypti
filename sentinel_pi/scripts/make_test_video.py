"""Build a synthetic test video from YOLO valid/ dataset.

Picks:
- 5 Aedes Aegypti images (class 0)
- 3 Aedes Albopictus images (class 1)
- 2 no-mosquito background images

Each image is displayed for 2.0 s, separated by 4.0 s of plain background.
Writes video to --out, and writes expected_events.json with ground-truth sequence.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("make_test_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic test video from dataset.")
    parser.add_argument(
        "--dataset-dir",
        default="../MosquitoSentinel_trainML/dataset",
        help="Path to YOLO dataset root or valid/ folder",
    )
    parser.add_argument(
        "--out",
        default="data/test_ground_truth.mp4",
        help="Path to output mp4 video file",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    return parser.parse_args()


def find_dataset_folders(dataset_dir: Path) -> Tuple[Path, Path]:
    """Find images and labels folders under dataset_dir or dataset_dir/valid."""
    if (dataset_dir / "valid" / "images").exists() and (dataset_dir / "valid" / "labels").exists():
        return dataset_dir / "valid" / "images", dataset_dir / "valid" / "labels"
    if (dataset_dir / "images").exists() and (dataset_dir / "labels").exists():
        return dataset_dir / "images", dataset_dir / "labels"
    raise FileNotFoundError(f"Could not locate images/ and labels/ in {dataset_dir}")


def pick_images_by_class(
    img_dir: Path,
    lbl_dir: Path,
    n_aegypti: int = 5,
    n_albopictus: int = 3,
) -> Tuple[List[Path], List[Path]]:
    """Select pure class 0 and pure class 1 images from label files."""
    aegypti_imgs: List[Path] = []
    albopictus_imgs: List[Path] = []

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            continue
        classes = {l.split()[0] for l in lines}

        # Look for corresponding image
        base_name = lbl_path.stem
        # In roboflow, image might be .jpg or .png
        img_candidates = list(img_dir.glob(f"{base_name}.*"))
        if not img_candidates:
            continue
        img_path = img_candidates[0]

        if classes == {"0"} and len(aegypti_imgs) < n_aegypti:
            aegypti_imgs.append(img_path)
        elif classes == {"1"} and len(albopictus_imgs) < n_albopictus:
            albopictus_imgs.append(img_path)

        if len(aegypti_imgs) >= n_aegypti and len(albopictus_imgs) >= n_albopictus:
            break

    if len(aegypti_imgs) < n_aegypti or len(albopictus_imgs) < n_albopictus:
        raise RuntimeError(
            f"Could not find enough samples: {len(aegypti_imgs)}/{n_aegypti} Aegypti, "
            f"{len(albopictus_imgs)}/{n_albopictus} Albopictus"
        )

    return aegypti_imgs, albopictus_imgs


def create_letterboxed_frame(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to fit within target dimensions with neutral gray padding."""
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_dir, lbl_dir = find_dataset_folders(dataset_dir)
    logger.info("Using dataset images: %s", img_dir)
    logger.info("Using dataset labels: %s", lbl_dir)

    aegypti_imgs, albopictus_imgs = pick_images_by_class(img_dir, lbl_dir, n_aegypti=5, n_albopictus=3)
    logger.info("Selected %d Aedes Aegypti images", len(aegypti_imgs))
    logger.info("Selected %d Aedes Albopictus images", len(albopictus_imgs))

    fps = args.fps
    width = args.width
    height = args.height
    image_duration_s = 2.0
    gap_duration_s = 4.0
    frames_per_image = int(round(image_duration_s * fps))
    frames_per_gap = int(round(gap_duration_s * fps))

    # Plain background frame
    plain_bg = np.full((height, width, 3), 128, dtype=np.uint8)

    # Prepare episodes
    episodes_data = []
    # 5 Aedes Aegypti
    for p in aegypti_imgs:
        episodes_data.append(("Aedes Aegypti", "confirmed", p))
    # 3 Aedes Albopictus
    for p in albopictus_imgs:
        episodes_data.append(("Aedes Albopictus", "non_target", p))
    # 2 No mosquito
    episodes_data.append(("None", "none", None))
    episodes_data.append(("None", "none", None))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        logger.error("Failed to open VideoWriter for %s", out_path)
        sys.exit(1)

    expected_events = []
    current_time_s = 0.0

    # Initial 4s gap
    for _ in range(frames_per_gap):
        writer.write(plain_bg)
    current_time_s += gap_duration_s

    for idx, (cls_name, expected_status, img_path) in enumerate(episodes_data, start=1):
        start_s = current_time_s
        end_s = current_time_s + image_duration_s

        if img_path is not None:
            raw_img = cv2.imread(str(img_path))
            if raw_img is None:
                raise RuntimeError(f"Could not read image: {img_path}")
            frame = create_letterboxed_frame(raw_img, width, height)
            img_file_str = img_path.name
        else:
            # Synthetic no-mosquito texture (plain subtle gradient/solid)
            frame = np.full((height, width, 3), 110 if idx % 2 == 0 else 140, dtype=np.uint8)
            img_file_str = f"no_mosquito_{idx}.png"

        # Write image frames for 2.0 s
        for _ in range(frames_per_image):
            writer.write(frame)
        current_time_s += image_duration_s

        expected_events.append({
            "episode": idx,
            "ground_truth_class": cls_name,
            "expected_status": expected_status,
            "image_file": img_file_str,
            "start_time_s": round(start_s, 3),
            "end_time_s": round(end_s, 3),
            "duration_s": round(image_duration_s, 3),
        })

        # Gap frames for 4.0 s
        for _ in range(frames_per_gap):
            writer.write(plain_bg)
        current_time_s += gap_duration_s

    writer.release()
    logger.info("Video successfully written to %s (%.1f s)", out_path, current_time_s)

    # Write expected_events.json alongside the video
    json_path = out_path.with_name("expected_events.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(expected_events, f, indent=2)
    logger.info("Expected events ground truth written to %s", json_path)


if __name__ == "__main__":
    main()
