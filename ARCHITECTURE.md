# MosquitoSentinel — Architecture & Build Specification

**Audience:** the coding agent (Antigravity), acting as senior developer. **Architect/researcher:** Claude (design decisions below are binding unless the user changes them).
**Read `PAPER_BRIEF.md` first.** You have not seen the thesis manuscript; that file is the digest of what the thesis promises (SOP, objectives, scope, delimitations, gaps). Implementation pushback is welcome via `DECISIONS.md`; changing thesis requirements silently is not.
**Project:** *MosquitoSentinel: An AI-Powered Real-Time Vision-Based Detection and Alert System for Aedes aegypti Surveillance* (CpE Practice and Design 1, Colegio de Montalban).

---

## 0. How you must work

1. **Build in phases (Section 12). Stop at the end of each phase** and report: what was built, how to run it, what is untested. Do not start the next phase until the user says so.
2. **Phase 0 is an audit only.** Read the existing code first and produce a reuse/rewrite report. Do not modify or delete anything in Phase 0.
3. **Never delete datasets, trained weights, or `runs/`.** Datasets live outside the repo (e.g. `C:\datasets\aedes\`).
4. If a decision is not specified here, **put it in `config.yaml` with a sensible default and a `# TODO(confirm)` comment**. Do not hard-code. Do not invent hardware behavior; if unsure, ask.
5. Code must run in **three modes** so it can be developed on Windows without the Pi:
   - `pi` (real camera + GPIO), `usb` (webcam), `file` (video file replay). GPIO must have a mock fallback.
6. Python 3.11+, type hints, docstrings, `logging` (no bare `print`), `pytest` tests for all pure-logic modules.

## 1. Thesis requirements the system must satisfy

Every feature below exists to produce evidence for a Statement of the Problem (SOP) item. Do not build features that serve none of them.

| SOP | Needs | Provided by |
|---|---|---|
| 2.1 Operational consistency/stability | uptime, FPS, dropped frames, latency, temperature, restarts, capture failures | `monitoring/health.py`, `health_logs` table, dashboard Health page |
| 2.2.1 Detection accuracy of *Aedes aegypti* | event-level precision/recall/F1 against human verification; offline model metrics | `events.verified_label`, Verify UI, `scripts/evaluate_events.py` |
| 2.2.2 Detection Index accuracy over time | system index vs index from verified events | `detection_index` stores `raw_value` and `verified_value`; `scripts/evaluate_index.py` (MAE, MAPE) |
| 2.3.1 Acceptability and usability | user questionnaire results | `/feedback` page (SUS-style Likert form) stored in `survey_responses` |
| 2.3.2 Timeliness and accuracy of alerts | alert latency, true/false alerts | timestamps in `events`/`alerts`, alert verification in UI, `scripts/evaluate_alerts.py` |
| 2.3.3 Effectiveness vs manual inspection | manual counts side by side with system output | `manual_counts` table + entry form + CSV export |

## 2. Hard constraints from the manuscript (Scope & Delimitations)

- Target: **Aedes aegypti only**. Other species are logged as `non_target`, never counted in the Detection Index.
- **No acoustic / wing-beat analysis. No LSTM. No Streamlit. No cloud dependency** (must run fully offline on the Pi).
- Monitoring windows: **06:00–09:00 and 16:00–19:00 local time (Asia/Manila)**, configurable. A `test_mode: always_on` override is allowed for development only.
- Record per event: image, date/time, species, confidence, monitoring location.
- Alerts: LED + buzzer + dashboard (Flask).
- Surveillance and early-warning only; never display or imply disease prediction.
- Hardware: Raspberry Pi 5, Global Shutter Camera RAW10 Zoom (CSI, via `picamera2`).
- **Known deviation:** the paper currently says captured images only; the system now processes **video frames**. The authors will revise the manuscript. Keep the code honest: frame-based analysis within the camera field of view, limited by processing speed.

## 3. Technology stack

- Capture: `picamera2` (CSI), OpenCV (`cv2.VideoCapture`) for USB/file.
- Detector: **Ultralytics YOLO11** (trained model, user-provided weights). Model backend must be swappable via config: `pt | ncnn | onnx | openvino`. **Do not assume OpenVINO runs well on the Pi 5 (ARM)**; `scripts/benchmark.py` decides the default.
- Species confirmer: **CNN, TFLite** (`mosquito_model.tflite`), runtime `ai-edge-litert` or `tflite_runtime` (detect which is installed). **Do not install full TensorFlow on the Pi.**
- Storage: SQLite (WAL mode). Dashboard: Flask + vanilla JS + **locally bundled** Chart.js (no CDN).
- GPIO: `gpiozero` with mock pin factory fallback. Service: `systemd`.

## 4. Pipeline architecture

```
[Scheduler: monitoring windows]
        |
[FrameSource: pi | usb | file]  --(capture thread; queue size 1, drop-oldest)-->  [RingBuffer: last N sec]
        |
[Detector worker: YOLO11 (+ optional tracker)]  --> per-frame detections
        |
[EventBuilder]  groups detections into EVENTS (persistence + time-gap rule)
        |
[CNN Confirmer (optional, config flag)]  runs on a few best crops per event only
        |
[EvidenceSaver]  best snapshot + short clip (pre-roll from ring buffer)
        |
[SQLite]  events / sessions / index / alerts / health / manual / survey
        |
[IndexCalculator]  -> [AlertEngine: CUSUM + EARS-C1]  -> [GPIO LED/buzzer] + [Flask dashboard]
        ^
[HealthMonitor] samples FPS, latency, drops, temp, CPU, RAM, disk, clock-sync
```

Threads: capture, inference, writer/analytics, health, Flask (separate process or thread via `run.sh`). Use bounded queues. **Inference must never block capture; if it falls behind, drop stale frames and count them.**

## 5. Repository layout

```
sentinel_pi/
├── config.yaml                 # all tunables (Section 11)
├── main.py                     # orchestrator, graceful shutdown
├── sentinel/
│   ├── config.py               # load/validate config.yaml (dataclasses)
│   ├── scheduler.py            # monitoring windows, test_mode
│   ├── capture/                # base.py, pi_camera.py, usb_camera.py, file_source.py, ring_buffer.py
│   ├── detection/              # detector.py (backend-agnostic), cnn_confirmer.py, event_builder.py
│   ├── evidence/               # saver.py (snapshot, clip, retention/disk cap)
│   ├── storage/                # schema.sql, db.py, repositories.py
│   ├── analytics/              # detection_index.py, cusum.py, ears.py, baseline.py
│   ├── alerts/                 # alert_engine.py, gpio_output.py (with mock)
│   ├── monitoring/             # health.py
│   └── dashboard/              # app.py, api.py, templates/, static/ (chart.js bundled)
├── scripts/                    # benchmark.py, evaluate_events.py, evaluate_index.py,
│                               # evaluate_alerts.py, export_csv.py, simulate_alerts.py
├── tests/
├── deploy/                     # setup_pi.sh, sentinel.service, run.sh
├── models/                     # (gitignored) best weights + mosquito_model.tflite
├── data/                       # (gitignored) sentinel.db, snapshots/, clips/
└── README.md
```

Reuse from the existing project **only after the Phase 0 audit approves it** (threaded queues, Flask routes, systemd/setup scripts are candidates).

## 6. Detection & event logic (core of accuracy)

**Unit of counting is an EVENT, never a frame.**

- **EventBuilder rule (config):** a detection joins an open event if it occurs within `event_gap_s` of the event's last detection. An event is `confirmed` when it has ≥ `min_frames` detections with mean YOLO confidence ≥ `min_mean_conf`. Events that never reach this are `rejected` (kept in DB, excluded from the index).
- **Tracker:** optional (`use_tracker`, ByteTrack via Ultralytics). Mosquitoes are small and fast, so **do not depend on track IDs for correctness**; the time-gap rule is the source of truth. Track IDs are stored only as extra info.
- **CNN confirmer (`cnn.enabled`):** classes `['Aegypti','Albopictus']`, input 64×64. It has **no background class**, so it must run **only on YOLO crops**, on at most `cnn.max_crops_per_event` best crops per event, never on every frame. Event label = mean-probability vote. If the CNN says Albopictus with prob ≥ `cnn.reject_threshold`, event status = `non_target`.
  - **Preprocessing must exactly match training.** Read `cnn_model_local.py` and `pi_predict.py` and replicate resize, color order (training used `cv2.imread` → BGR), and scaling. `picamera2` frames may be RGB: convert explicitly and add a unit test with a known image.
  - Provide an **ablation switch** (`cnn.enabled: false`) so the thesis can report YOLO-only vs YOLO+CNN.
- Read YOLO class names from the model, not from hard-coded strings. Map whichever class represents Aedes aegypti via config (`yolo.target_class_names`).
- Store timestamps per event: `t_first_detect`, `t_confirmed`, `t_saved`. These feed alert-latency metrics.

## 7. Detection Index and alerts

**Detection Index (default, configurable):** per monitoring point and per time bin (`bin_minutes`, default 30):
`DI = confirmed_events_in_bin / active_monitoring_hours_in_bin` (events per hour).
Active hours exclude downtime (derived from health/heartbeat logs) so outages do not look like "no mosquitoes". Store `raw_value` (system) and `verified_value` (computed only from events a human marked true). Also store `bin_coverage` (fraction of the bin the system was actually running).

**Baseline mode:** the first `baseline.days` of data (or an imported baseline) define μ₀ and σ per time-of-day group (morning window / afternoon window). Alerts are **suppressed during baseline** and the dashboard shows "Baseline in progress".

**Alert engine (both methods computed and logged; `alert.method` selects which can fire):**
- **CUSUM:** `S_t = max(0, S_{t-1} + (x_t − μ₀ − k·σ))`, alert when `S_t > h·σ`. Defaults `k=0.5`, `h=4` (starting values, user will tune).
- **EARS-C1:** compare `x_t` to mean/SD of the previous `ears.lag_bins` (default 7) bins; alert when `(x_t − mean)/max(SD, min_sd) > ears.z` (default 3).
- Levels: `NORMAL`, `WATCH` (≥ watch fraction of threshold), `ALERT`. Add `cooldown_min` and hysteresis to prevent buzzer flapping. Resetting CUSUM after an alert is configurable.
- Outputs: LED (steady for WATCH, blinking for ALERT), buzzer (short pattern, `buzzer.enabled` and a dashboard "mute/acknowledge"), dashboard banner. Log `t_alert` and when the dashboard first rendered it (`t_displayed`).
- **Must not claim dengue prediction anywhere in UI text.** Use "increased Aedes aegypti activity detected".

## 8. Database schema (SQLite, WAL)

Create `schema.sql` with at least:

- `sessions(id, started_at, ended_at, mode, model_name, model_backend, config_json, git_hash)` — one per run.
- `events(id, session_id, monitoring_point, started_at, ended_at, duration_s, n_frames, max_conf, mean_conf, yolo_class, cnn_label, cnn_conf, status CHECK IN ('candidate','confirmed','rejected','non_target'), snapshot_path, clip_path, t_first_detect, t_confirmed, t_saved, verified_label CHECK IN ('true','false') NULL, verified_by, verified_at, notes)`
- `detection_index(id, monitoring_point, bin_start, bin_end, raw_value, verified_value, n_events, n_verified_true, active_hours, bin_coverage, cusum_value, ears_value, level)`
- `alerts(id, monitoring_point, level, method, t_alert, t_displayed, cusum_value, ears_value, bin_start, trigger_event_id, acknowledged_at, verified_outcome CHECK IN ('true_alert','false_alert') NULL, notes)`
- `health_logs(id, ts, session_id, fps, infer_ms_avg, infer_ms_p95, dropped_frames, queue_depth, cpu_pct, ram_pct, cpu_temp_c, disk_free_gb, camera_ok, clock_synced)`
- `manual_counts(id, ts, monitoring_point, method, count, inspector, notes)` — for SOP 2.3.3.
- `survey_responses(id, ts, respondent_role, answers_json, comments)` — for SOP 2.3.1.
- Indexes on `events(started_at)`, `events(status)`, `detection_index(bin_start)`, `health_logs(ts)`.

## 9. Dashboard (Flask, offline-first)

Pages and JSON API (`/api/...`):
1. **Live** — MJPEG stream with boxes, FPS, mode, current level (NORMAL/WATCH/ALERT), latest events.
2. **Events** — filter by date/status; thumbnail + clip player; **Verify buttons (True / False)** (this produces thesis evidence).
3. **Trends** — Detection Index over time, CUSUM statistic vs threshold, baseline band, alert markers.
4. **Alerts** — log with acknowledge and **true/false alert verification**.
5. **Health** — uptime, FPS, latency, drops, temperature, disk, restarts, clock-sync warning.
6. **Manual Inspection** — form for entering field counts.
7. **Feedback** — usability questionnaire.
8. **Export** — CSV of events, index, alerts, health, manual counts.
Dashboard must work with no internet, be readable on a phone, and survive the detector being down (shows "detector offline").

## 10. Reliability requirements (SOP 2.1)

- `systemd` unit with `Restart=always`; log restart counts. Camera failure → retry with backoff and log `camera_ok=0`, never crash-loop silently.
- The Pi 5 has **no RTC**: record `clock_synced` and warn on the dashboard if the clock is unsynced, because timestamps drive everything.
- Disk guard: cap clip/snapshot storage (`evidence.max_gb`), delete oldest unverified evidence first, never delete evidence tied to verified events.
- Thermal guard: log temperature; optionally reduce inference rate above `thermal.throttle_c`.
- Graceful shutdown flushes the open event and closes the session.

## 11. `config.yaml` (all tunables, with defaults)

```yaml
site: {monitoring_point: "KAS-1E-01", timezone: "Asia/Manila"}
mode: pi                 # pi | usb | file
file_source: {path: ""}
schedule: {windows: [["06:00","09:00"],["16:00","19:00"]], test_mode: false}
capture: {width: 1280, height: 720, fps: 30, preroll_s: 3, postroll_s: 3}
yolo: {weights: models/best.pt, backend: pt, imgsz: 640, conf: 0.25, iou: 0.5, target_class_names: [], use_tracker: false}
events: {event_gap_s: 1.5, min_frames: 3, min_mean_conf: 0.4}
cnn: {enabled: true, model: models/mosquito_model.tflite, max_crops_per_event: 3, reject_threshold: 0.8}
index: {bin_minutes: 30, method: events_per_hour}
baseline: {days: 7}
alert: {method: both, watch_fraction: 0.6, cooldown_min: 30, cusum: {k: 0.5, h: 4.0}, ears: {lag_bins: 7, z: 3.0, min_sd: 0.5}}
gpio: {led_pin: 17, buzzer_pin: 27, buzzer_enabled: true}
evidence: {max_gb: 20, save_clips: true}
thermal: {throttle_c: 80}
dashboard: {host: 0.0.0.0, port: 5000}
```
Pins and values are placeholders: confirm with the actual wiring.

## 12. Build phases and acceptance criteria

**Phase 0 — Audit (no code changes).** Read existing `detect.py`, `app.py`, `export_model.py`, `setup_pi.sh`, `requirements_pi.txt`, `run.sh`, `sentinel_pi.service`, and the CNN scripts. Output `AUDIT.md`: for each file, **keep / adapt / rewrite** with reasons; list bugs and risks (model format on ARM, camera path, color order, blocking calls, hard-coded paths). *Accept:* user approves the report.

**Phase 1 — Skeleton.** Folder layout, `config.py` validation, `schema.sql` + `db.py` + repositories, logging, pytest setup. *Accept:* tests pass; DB created from schema; config errors are readable.

**Phase 2 — Capture + detector + benchmark.** FrameSource (pi/usb/file), ring buffer, backend-agnostic detector, `scripts/benchmark.py` (reports FPS, avg/p95 latency per backend and imgsz, on a video file and live camera). *Accept:* file mode runs on Windows; benchmark output saved as CSV.

**Phase 3 — Events + CNN + evidence.** EventBuilder, CNN confirmer with preprocessing parity test, snapshot/clip saver, retention. *Accept:* replaying a test video yields stored events with clips; unit tests for gap/persistence rules; CNN gives identical output on PC vs Pi for the same image.

**Phase 4 — Analytics.** DetectionIndex, baseline, CUSUM, EARS, `simulate_alerts.py` (synthetic series to demo alerts). *Accept:* unit tests with known sequences; simulated surge triggers ALERT and flat series does not.

**Phase 5 — Alerts, health, orchestrator.** GPIO (with mock), AlertEngine, scheduler, HealthMonitor, `main.py` wiring, graceful shutdown. *Accept:* full pipeline runs end-to-end in file mode; alert rows contain all latency timestamps.

**Phase 6 — Dashboard.** All pages in Section 9, verification workflow, manual counts, survey, CSV export. *Accept:* works offline; verifying events changes `verified_value` in the index.

**Phase 7 — Deployment & evaluation.** `setup_pi.sh`, `sentinel.service`, README (setup, run, troubleshooting, how each SOP is evidenced), `evaluate_*.py` scripts. *Accept:* clean install on a fresh Pi OS Bookworm 64-bit, auto-start on boot, 1-hour soak test with no crash.

## 13. Do-not list

No acoustic features, no LSTM, no Streamlit, no cloud services, no CDN assets, no full TensorFlow on the Pi, no per-frame DB rows, no counting frames as detections, no disease-prediction claims, no deleting datasets/weights/runs, no hard-coded paths, no silent exception swallowing.

## 14. Development and deployment workflow (laptop to Pi)

**All coding happens on the user's Windows laptop. The Pi is a deployment target only**, reached over the network by IP (SSH/SCP). Never require editing files on the Pi.

- **Laptop-testable:** everything except `picamera2`, real GPIO, and true FPS/thermal numbers. Use `mode: file` / `usb` and the GPIO mock. Mark Pi-only tests with a `pi_only` pytest marker so they are skipped on the laptop and run on the Pi after deploy.
- **`deploy/push_to_pi.ps1`** (PowerShell, uses built-in `ssh`/`scp`; do not assume `rsync` exists on Windows). Reads `deploy/deploy.env` (gitignored): `PI_HOST`, `PI_USER`, `PI_PATH`. It copies code + `config.yaml` + `deploy/` scripts, **excluding** `.venv`, `__pycache__`, `.git`, `data/`, datasets and `runs/`. Models in `models/` are copied only when their hash changed. It then runs `deploy/remote_install.sh` over SSH.
- **`deploy/remote_install.sh`** (runs on the Pi): strips CRLF from itself and the other scripts, creates the venv **with `--system-site-packages`** (so apt-installed `picamera2` is visible), installs `requirements_pi.txt`, installs/refreshes the systemd unit, restarts the service, and prints the service status. Must be idempotent (safe to run repeatedly).
- **`deploy/pull_from_pi.ps1`:** copies `data/sentinel.db`, clips, snapshots and logs from the Pi back to the laptop for evaluation scripts.
- **Line endings:** add `.gitattributes` forcing `eol=lf` for `*.sh`, `*.service`, `*.py`, `*.yaml`. CRLF in a shell script breaks it on the Pi ("bad interpreter").
- **Never copy the Windows `.venv`** to the Pi. Keep two dependency files: `requirements_dev.txt` (laptop, training/tools) and `requirements_pi.txt` (minimal; no full TensorFlow; avoid heavy dependencies when the chosen backend is NCNN/ONNX).
- **Paths:** `pathlib`, relative to the project root. The project folder lives under a OneDrive path containing non-ASCII characters; nothing may break because of that, and nothing may contain absolute Windows paths.
- **`scripts/smoke_test_pi.py`** (run on the Pi right after deploy): checks OS/Python versions, camera detected, one frame captured, GPIO toggle, model files load, one inference timing, DB write, dashboard port reachable, clock sync, free disk, CPU temperature. Prints a PASS/FAIL table. This replaces ad-hoc manual checks.
- **Phase 7 acceptance addition:** one command from the laptop deploys and restarts the system on the Pi, and `smoke_test_pi.py` passes.
