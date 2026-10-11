# Phase 0 Audit Report

## 1. File Audit

| File | Status | Reasons & Findings |
|---|---|---|
| `sentinel_pi/detect.py` | **Rewrite** | **Reasons:** Standalone script that tightly couples capture and inference in a single blocking loop. Does not match the pipeline architecture (FrameSource → RingBuffer → EventBuilder).<br>**Bugs/Risks:** Inference blocking capture. If YOLO takes 100ms, the camera is bottlenecked to 10 FPS (**estimated**). Frame buffering and dropping (Section 4) are not implemented properly. Hardcoded config values (confidence, model path). |
| `sentinel_pi/app.py` | **Rewrite** | **Reasons:** Built with FastAPI + WebSockets instead of the mandated Flask stack. Lacks the required endpoints (Events, Trends, Alerts, Health, Manual, Feedback).<br>**Bugs/Risks:** Does not fulfill SOP 2.2.1, 2.3.1, or 2.3.3 requirements (verification workflows, forms, db). |
| `sentinel_pi/export_model.py` | **Keep / Adapt** | **Reasons:** Valid script for exporting to OpenVINO. Good to keep as a utility for the developer/user to run on their PC.<br>**Bugs/Risks:** Hardcoded input path `runs/detect/.../best.pt` could be brittle. Should rely on argparse completely. |
| `sentinel_pi/setup_pi.sh` | **Adapt** | **Reasons:** Sets up `--system-site-packages` correctly for `picamera2`.<br>**Bugs/Risks:** Installs full PyTorch (`torch`, `torchvision`). While Ultralytics often requires PyTorch to import, installing it on a Pi takes a lot of space and time. Missing SQLite and `tflite_runtime` (or `ai-edge-litert`) dependencies. |
| `sentinel_pi/requirements_pi.txt` | **Rewrite** | **Reasons:** Contains `fastapi` and `uvicorn`, missing `Flask`, `gpiozero`, and TFLite runtimes.<br>**Bugs/Risks:** Out of sync with the architectural constraints. |
| `sentinel_pi/run.sh` | **Rewrite** | **Reasons:** Currently boots `uvicorn sentinel_pi.app:app`. Needs to be changed to run the new orchestrator `main.py` which will spawn the Flask app and the pipeline threads. |
| `sentinel_pi/sentinel_pi.service`| **Adapt** | **Reasons:** Standard systemd unit, but `ExecStart` needs to point to the new entry point (`main.py`) instead of `uvicorn`. |
| `CNN_Mosquito_Classifier/cnn_model_local.py` | **Extract Logic** | **Reasons:** The training script itself isn't deployed to the Pi, but its preprocessing logic must be extracted for `cnn_confirmer.py`.<br>**Bugs/Risks:** Training uses `cv2.imread(..., cv2.IMREAD_COLOR)` which yields BGR format, resized to 64x64, normalized `/ 255.0`. The Pi pipeline must strictly replicate this (handling the fact that `picamera2` outputs RGB natively). |
| `CNN_Mosquito_Classifier/pi_predict.py` | **Extract Logic** | **Reasons:** Correctly replicates the CNN preprocessing (BGR, 64x64, /255.0) and uses `tflite_runtime`. This logic will be ported into `sentinel/detection/cnn_confirmer.py`. |

## 2. General Bugs and Risks Discovered

1. **Inference Blocking Capture:** The existing `detect.py` captures a frame and runs YOLO sequentially. The architecture correctly states inference must never block capture. The new threaded pipeline with a drop-oldest queue is mandatory.
2. **Color Space Mismatch:** `picamera2` natively captures in RGB. The CNN model was trained using `cv2.imread` (BGR). The YOLO model might also be trained on BGR or RGB depending on Ultralytics defaults. Explicit color space management is required to avoid degrading accuracy.
3. **Heavy PyTorch Dependency on Pi:** `setup_pi.sh` explicitly installs heavy PyTorch wheels. We should evaluate if we can run OpenVINO or NCNN via lightweight bindings without the full PyTorch installation, though Ultralytics pip package usually forces PyTorch as a dependency.

## 3. Addendum

### Skipped Files Audit
* `CNN_Mosquito_Classifier/convert_to_tflite.py`: Standard script converting Keras `.pkl` to TFLite format via `tf.lite.TFLiteConverter`. Kept as utility.
* `CNN_Mosquito_Classifier/cnn_model_local.py` (Train/Test Split & Leakage Risk): The train/test split is made based solely on the presence of `_testing` in the folder path. It is not randomized per image. **Data Leakage Risk:** If images from the same source, video, or burst capture were manually separated into both training and testing directories, highly correlated frames will exist in both sets. This would artificially inflate the reported accuracy.
* `MosquitoSentinel_trainML/train.py`: A clean YOLOv11 training script using `argparse`. No issues.
* `MosquitoSentinel_trainML/predict.py`: Standard YOLOv11 inference script. 
* `sentinel_pi/templates/dashboard.html`: The current HTML dashboard uses WebSockets for real-time updates and native DOM manipulation without a bundled Chart.js. It does not fulfill the requirement for a multi-page Flask app.
* `MosquitoSentinel_trainML/sentinel_ui/`: Contains a Windows-side FastAPI UI for monitoring YOLO training progress by parsing `results.csv`. Useful utility, but not deployed to the Pi.

### Model Export on ARM
* `export_model.py` currently targets OpenVINO, which is heavily optimized for Intel hardware. While OpenVINO supports ARM, Ultralytics natively supports inference on ARM using **NCNN**, **TFLite**, and **ONNX** (via ONNXRuntime). 
* **Proposal:** Add a `--format` argument to `export_model.py` via `argparse` allowing the user to select between `openvino`, `ncnn`, `tflite`, and `onnx` to properly benchmark the best ARM backend.

### Colour Order Management
* **Picamera2 memory format:** When Picamera2 is configured with `format="RGB888"`, the bytes are actually written into memory in **B, G, R** order (to align with OpenCV's default format).
* **Configured Format:** We will configure the camera for `RGB888`.
* **Channel Order:** The array returned by `cam.capture_array()` will therefore natively be **BGR**.
* **Ultralytics Expectation:** When passed a numpy array, Ultralytics YOLO implicitly expects a **BGR** array (assuming it came from `cv2.imread`), which it converts to RGB internally during pre-processing.
* **Current Bug in `detect.py`:** It calls `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)` on an array that is *already BGR*, effectively flipping it to RGB. When passed to YOLO, YOLO expects BGR and flips it *again*, resulting in inverted colour channels during inference.
* **Proposed Test:** Generate a pure RED image array (which is `[0, 0, 255]` in BGR). Pass it through the capture pipeline. Assert that `frame[0, 0, 2] == 255` (confirming it is BGR) before it hits the YOLO or CNN models.

### CSI Capture Path
* **Yes**, `detect.py` has a working CSI capture path. It calls `_init_picamera2`, configures the preview format to `RGB888`, and captures arrays using `cam.capture_array()`. If it fails, it gracefully falls back to USB capture (`cv2.VideoCapture`).

### Performance Numbers Note
* YOLO inference at 10 FPS on Pi 5 (blocking issue): **Estimated** (extrapolated from typical 100ms inference times).
* OpenVINO Pi 5 speedup (~3x faster than PyTorch): **Estimated** / Quoted from literature.
