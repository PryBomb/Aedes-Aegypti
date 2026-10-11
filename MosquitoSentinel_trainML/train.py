import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# ─── Dataset registry ─────────────────────────────────────────────────────────
# Maps short aliases to their data.yaml paths (relative to this file).
DATASET_REGISTRY: dict[str, Path] = {
    "v1": Path(__file__).parent / "dataset"            / "data.yaml",   # original 2-class dataset
    "v2": Path(__file__).parent / "mosquito_dataset_v2" / "data.yaml",  # MosquitoDataset v2 (Aedes Aegypti)
}

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 model on Mosquito Dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model checkpoint (e.g., yolo11n.pt, yolo11s.pt)")
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset alias: 'v1' = original 2-class dataset, 'v2' = MosquitoDataset v2 (Aedes Aegypti only). "
             "Overrides --data when specified."
    )
    parser.add_argument(
        "--data", type=str,
        default=str(Path(__file__).parent / "dataset" / "data.yaml"),
        help="Explicit path to a dataset yaml. Ignored when --dataset is used."
    )
    parser.add_argument("--project", type=str, default="runs/detect", help="Project save directory")
    parser.add_argument("--name", type=str, default="mosquito_yolo11", help="Experiment name")
    parser.add_argument("--export", action="store_true", help="Export best.pt to OpenVINO format after training (recommended for Raspberry Pi 5)")
    parser.add_argument("--export-imgsz", type=int, default=320, dest="export_imgsz", help="Image size for OpenVINO export (default 320 for Pi speed)")
    args = parser.parse_args()

    # Resolve dataset path: alias takes priority over explicit --data
    if args.dataset is not None:
        args.data = str(DATASET_REGISTRY[args.dataset])

    data_path = Path(args.data).resolve()
    dataset_label = args.dataset if args.dataset else "custom"
    print(f"=== Mosquito Sentinel YOLOv11 Training ===")
    print(f"Dataset        : [{dataset_label}] {data_path}")
    print(f"Model          : {args.model}")
    print(f"Epochs: {args.epochs} | Image Size: {args.imgsz} | Batch Size: {args.batch}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    # Initialize YOLOv11 model with pretrained weights
    model = YOLO(args.model)

    # Train model
    print("\nStarting training...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True
    )

    print("\nEvaluating best model on validation set...")
    metrics = model.val()
    
    print("\n=== Training Completed Successfully ===")
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Best model saved to: {best_weights.resolve()}")
    if hasattr(metrics, 'results_dict'):
        print(f"mAP50: {metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95: {metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # Optional: export to OpenVINO format for Raspberry Pi 5 deployment
    if args.export:
        print("\nExporting model to OpenVINO format for Raspberry Pi 5…")
        best_model = YOLO(str(best_weights))
        export_path = best_model.export(
            format="openvino",
            imgsz=args.export_imgsz,
            half=False,
        )
        print(f"OpenVINO model saved to: {Path(str(export_path)).resolve()}")
        print("Transfer this folder to ~/mosquito/ on your Raspberry Pi 5.")

if __name__ == "__main__":
    main()
