import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained YOLOv11 Mosquito model")
    parser.add_argument("--weights", type=str, default="runs/detect/mosquito_yolo11/weights/best.pt", help="Path to model weights")
    parser.add_argument("--source", type=str, default="dataset/test/images", help="Image file, folder, or video source")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--project", type=str, default="runs/detect", help="Save directory project")
    parser.add_argument("--name", type=str, default="predict", help="Save directory name")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Warning: Trained weights not found at '{weights_path}'. Falling back to default 'yolo11n.pt'")
        weights_path = "yolo11n.pt"

    print(f"=== Mosquito Sentinel YOLOv11 Inference ===")
    print(f"Loading weights: {weights_path}")
    print(f"Source input: {args.source}")
    print(f"Confidence threshold: {args.conf}")

    model = YOLO(str(weights_path))

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    save_dir = Path(args.project) / args.name
    print(f"\nInference completed! Predictions saved to: {save_dir.resolve()}")

if __name__ == "__main__":
    main()
