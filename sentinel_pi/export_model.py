"""
sentinel_pi/export_model.py
───────────────────────────
Run this script ONCE on your Windows PC after training completes.
It exports best.pt → OpenVINO format for significantly faster CPU inference
on the Raspberry Pi 5 (~3x faster than raw PyTorch).

Usage:
    python sentinel_pi/export_model.py
    python sentinel_pi/export_model.py --weights runs/detect/mosquito_yolo11/weights/best.pt

Then copy the generated folder to the Pi:
    scp -r runs/detect/mosquito_yolo11/weights/best_openvino_model  pi@<pi-ip>:~/mosquito/
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11 best.pt to OpenVINO format")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/mosquito_yolo11/weights/best.pt",
        help="Path to trained best.pt weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size for export (320 gives best speed/accuracy balance on Pi)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"❌  Weights not found at: {weights_path}")
        print("    Run train.py first to generate best.pt")
        return

    print("=" * 50)
    print("  MosquitoSentinel — OpenVINO Export")
    print("=" * 50)
    print(f"  Source : {weights_path.resolve()}")
    print(f"  Format : OpenVINO (FP32, INT8 quantization available)")
    print(f"  imgsz  : {args.imgsz}x{args.imgsz}")
    print()

    model = YOLO(str(weights_path))

    print("Exporting model to OpenVINO format…")
    export_path = model.export(
        format="openvino",
        imgsz=args.imgsz,
        half=False,      # FP32 — most stable on Pi 5
        dynamic=False,
    )

    out = Path(str(export_path))
    print()
    print(f"✅  Export complete!")
    print(f"   Output folder: {out.resolve()}")
    print()
    print("Next steps — transfer to Raspberry Pi:")
    print(f"   scp -r \"{out.resolve()}\" pi@<pi-ip>:~/mosquito/")
    print("   scp -r sentinel_pi/ pi@<pi-ip>:~/mosquito/")


if __name__ == "__main__":
    main()
