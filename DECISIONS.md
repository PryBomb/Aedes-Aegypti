# Technical Decisions & Proposals (Resolved)

## 1. Flask vs. FastAPI
**Context:** `ARCHITECTURE.md` explicitly mandates a **Flask dashboard**. However, the existing `app.py` is built with **FastAPI and WebSockets**. 
**Resolution:** We will strictly comply with the thesis paper and rewrite the dashboard using **Flask**.
- We will use standard generator streaming for the live MJPEG feed.
- For real-time stats (FPS, detections, alerts), we will use simple JS polling every 1-2 seconds instead of WebSockets.
- `Flask-SocketIO` will **not** be used to avoid Eventlet/Gevent complexities on the Pi.
- We will run the Flask app with `threaded=True` to allow concurrent streaming and API requests without blocking.

## 2. Heavy PyTorch Dependency on Edge
**Context:** `setup_pi.sh` installs the full PyTorch package (`torch`, `torchvision`), which are massive and slow to install on a Pi 5.
**Resolution:** The `ultralytics` package strictly requires `torch`. We will keep the installation of PyTorch as specified to guarantee `ultralytics` imports work flawlessly without hacks. No architectural change needed. 

## 3. Color Space Consistency
**Context:** Ensuring consistent color representation between the Pi camera, the YOLO detector, and the CNN confirmer is critical.
**Resolution:** We will standardize on **BGR** arrays everywhere (which exactly matches OpenCV's `cv2.imread` format used during CNN training). 
- We will configure `picamera2` with `format="RGB888"`. 
- According to the Picamera2 manual, the `RGB888` format actually stores bytes in **B, G, R** order in memory.
- Therefore, the array returned by `cam.capture_array()` is natively BGR.
- We will **NOT** add any `cv2.cvtColor` swaps after capture.
- *Note:* The existing `detect.py` contains a bug where it explicitly calls `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)`. Since the array is already BGR, this erroneously flips it to RGB before inference. We will strictly avoid reusing that line.

## 4. Ring Buffer Memory Strategy
**Context:** One raw 1280×720 BGR frame ≈ 2.76 MiB. At 30 FPS and 3 s pre‑roll this is 90 frames → ~248 MiB raw data, which is unsuitable for the Pi.
**Decision:** Store each frame as a **full‑resolution JPEG (quality≈85)**. JPEG compression reduces a 1280×720 BGR frame to roughly 30‑50 KB, preserving mosquito detail while keeping memory low.
- Typical compressed size: ~30‑50 KB per frame.
- 90‑frame pre‑roll: ~3‑4.5 MiB total, well under the default `max_memory_mb=50`.
- If the memory cap is exceeded, the oldest frames are evicted until the buffer fits.
- Evidence clips will decode these JPEGs, ensuring sufficient visual quality.
- No down‑scaling is performed, avoiding loss of small features.
## 5. Phase 3 Rulings: Events, Evidence, and Data Isolation
**Context:** Phase 3 introduces event aggregation, evidence snapshot/clip generation, database persistence, and offline video replay.
**Rulings & Design Resolutions:**
- **Target & Non-Target Class Configuration:**
  - Model `best.pt` outputs `{0: 'Aedes Aegypti', 1: 'Aedes Albopictus'}`.
  - `yolo.target_class_names` defaults to `["Aedes Aegypti"]`.
  - Added `yolo.non_target_class_names` defaulting to `["Aedes Albopictus"]`.
  - Class name matching is strictly case-insensitive.
  - If a configured target or non-target class name is missing from the model's classes, a `ValueError` is raised immediately on model load.
  - The `Detector` returns all detected objects with an `is_target` boolean flag; non-target detections (e.g. Albopictus) are retained rather than discarded.
- **CNN Deferred to Phase 3b:**
  - CNN confirmation is excluded from Phase 3 and deferred to Phase 3b.
  - `cnn.enabled` defaults to `false`.
- **OneDrive Isolation & Relative Evidence Paths:**
  - Running SQLite with WAL mode on OneDrive leads to lock contention and sync corruption.
  - `paths.data_dir` is supported in config (defaults to `C:\sentinel_data` for local dev on Windows, or `data` relative / `/var/lib/sentinel` on Linux).
  - All file paths saved to SQLite (`events.snapshot_path`, `events.clip_path`) are stored **relative to `data_dir`** to ensure full portability between development laptops and Pi installations.
- **Event Aggregation & Class Voting:**
  - `EventBuilder` is pure logic with an injectable monotonic/UTC clock.
  - Detections within `events.event_gap_s` belong to the active event.
  - Multi-frame class vote sums confidence per class across all detections; the highest sum wins the label. If the winner is non-target, event status is `non_target`.
  - Target events are `confirmed` once `n_frames >= min_frames` and `mean_conf >= min_mean_conf`; otherwise `rejected`.
  - Tracks `t_first_detect` and `t_confirmed`.
  - `flush()` closes open events upon shutdown.
- **Evidence Retention Protection:**
  - Under `evidence.max_gb`, evidence cleanup prunes the oldest unverified snapshots and clips first.
  - Evidence associated with verified events (`verified_label IS NOT NULL`) is protected and never deleted by the retention manager.

## 6. Phase 4 Rulings: Analytics, Detection Index, and Alerting
**Context:** Phase 4 introduces the Detection Index, baseline statistics, CUSUM/EARS-C1 alerting, and the schema v2 migration.

**Schema v2 Migration:**
- `PRAGMA user_version` bumped from 1 to 2.
- `detection_index` gains two columns: `valid INTEGER NOT NULL DEFAULT 1` (0 = excluded from baseline/alerting due to low coverage) and `n_unverified INTEGER NOT NULL DEFAULT 0` (confirmed events not yet verified).
- New table: `baselines(id, monitoring_point, window_group, mu, sigma, n_bins, computed_at, source CHECK IN ('computed','manual'))` with `UNIQUE(monitoring_point, window_group)`.
- Migration in `db.py` is idempotent (safe to run repeatedly): uses `ALTER TABLE ADD COLUMN` with try/except on duplicate-column errors and `CREATE TABLE IF NOT EXISTS`.
- A test proves v1→v2 migration by creating a v1 database and opening it with `DatabaseManager`.

**Detection Index Bin Design:**
- `index.bin_minutes` default changed from 30 to **15**. **Alert delay is bounded by bin length** — shorter bins = lower alert latency but noisier index. Documented in config.yaml and config.py.
- Bins are aligned to **Asia/Manila (UTC+8) clock boundaries** (midnight-aligned multiples of `bin_minutes`), generated only within configured monitoring windows (`schedule.windows`).
- Bin timestamps are stored in **UTC** in the database.
- A bin with `bin_coverage < index.min_coverage` (default 0.5) is marked `valid=0` and excluded from baseline computation and alerting, preventing low-activity bins from distorting statistics.
- `raw_value = confirmed Aedes Aegypti events / active_hours`. Non-target and rejected events are NEVER counted.
- `verified_value = events with verified_label='true' / active_hours`. `n_unverified` counts confirmed events without a verified_label.
- Recompute is idempotent via UPSERT on `(monitoring_point, bin_start)`.

**Baseline:**
- `window_group`: the first configured window → `'morning'`; the second → `'afternoon'`.
- `mu` = mean of valid raw_values; `sigma` = population std-dev floored at `baseline.min_sigma` (default 0.5) to prevent division-by-zero in CUSUM/EARS.
- Baseline is "ready" only when `n_bins >= baseline.min_bins` (default 12). Until ready, alerting is suppressed (returns NORMAL).

**CUSUM (Page's one-sided upper):**
- Computed in **sigma units**:
  - `z_t = (x_t - mu) / sigma`
  - `S_t = max(0, S_{t-1} + z_t - k)`
  - Alert condition: `S_t > h + 1e-9` (**strict inequality** with epsilon to prevent boundary float noise; landing exactly on `h` does NOT alert).
  - This is mathematically equivalent to the natural-unit formulation:
    `S_t = max(0, S_{t-1} + x_t - mu - k*sigma)` vs `h*sigma`.
  - Default parameters: `k=0.5`, `h=4.0`.
- **Pre-reset vs Post-reset state:** `cusum_step` returns `(s_carry, s_display, alerted)`:
  - `s_display` is the pre-reset cumulative sum; stored in `detection_index.cusum_value` and used to determine the bin `level`.
  - `s_carry` is the post-reset value (reset to 0 if `reset_after_alert=true` and alerted); passed as `prev_s` into subsequent steps.
  - When CUSUM fires, the bin's level is **`ALERT`** in `detection_index`, eliminating the bug where post-reset 0 caused level to read `NORMAL`.

**EARS-C1 (Hutwagner et al. 2003):**
- Uses the last `ears.lag_bins` (default 7) valid observations as a rolling baseline window.
- Uses **population standard deviation (`ddof=0`)** floored at `ears.min_sd` (default 0.5):
  `sd_lag = sqrt(sum((y - mu)^2) / N)`
- `C1_t = (y_t - mu_lag) / max(sd_lag, ears.min_sd)`.
- Alert when `C1_t >= ears.z` (default 3.0).
- If fewer than `lag_bins` historical observations are available, EARS is not computed (suppressed).

**AlertEngine:**
- Pure decision logic; no GPIO. Per closed valid bin, returns NORMAL / WATCH / ALERT.
- `method` = `cusum | ears | both`; `both` means either firing triggers an alert.
- WATCH fires at `>= watch_fraction * threshold` (default 0.6 × h or 0.6 × z).
- **`alert.min_events` (default 3, validated >= 1):**
  - Minimum number of confirmed target events in the bin required before the level can become `ALERT`.
  - When `n_events < min_events`, the level is capped at `WATCH` and no alert row is written, preventing isolated single-event bins from generating false public health alerts.
- **Cooldown & Bin Timestamps:**
  - `alert.cooldown_min` (default 30) suppresses repeated alert rows within the cooldown window.
  - Cooldown is computed using **bin timestamps (`bin_start`)**, NEVER the wall clock (`datetime.now()`), allowing offline simulation, historic replay, and batch processing to enforce cooldown deterministically.
  - **During cooldown, the level stays `ALERT`**, but no new `alerts` row is created.
- Suppressed while baseline is not ready.

**Replay `--virtual-start`:**
- Added `--virtual-start` (ISO-8601 with UTC offset, e.g. `2026-10-12T06:00:00+08:00`).
- Event timestamps = `virtual_start + frame_index / video_fps`. NEVER the wall clock.
- Replay now prints an events table with `id, started_at, ended_at, duration_s, n_frames, mean_conf, status`.

### Phase 4 Review (Round 2) Decisions

1. **Verified Index (`verified_value` SOP 2.2.2 Evidence):**
   - `verified_value` must be `NULL` (`None`) unless the bin is **fully verified** (`n_unverified == 0`).
   - When fully verified, `verified_value = n_verified_true / active_hours`, including `0.0` when all events were marked false (`n_verified_true == 0`) or zero events occurred.
   - Partial verification no longer produces a biased intermediate rate; it remains `NULL` until all human verification reviews in the bin are complete.

2. **Alert State Partitioning & Window Reset:**
   - State is partitioned per `(monitoring_point, window_group)` (e.g., `(TEST-POINT-1, morning)` vs `(TEST-POINT-1, afternoon)`).
   - CUSUM carry `s_t` and EARS history never cross between morning and afternoon windows.
   - State resets at the start of each monitoring window (detected when `bin_start - last_bin_end >= 6 hours`).
   - EARS requires `ears.lag_bins` valid bins inside the current window before firing.
   - Default `ears.lag_bins = 4` (reduced from 7 to allow firing within typical 3-hour windows). *Tuning note: to be tuned on real Pi field data.*

3. **Restart Safety (`rebuild_state`):**
   - Added `AlertEngine.rebuild_state(bins, alerts, window_group_now, baseline=None)`.
   - Reconstructs `s_t`, `ears_history`, `last_alert_time`, and episode state from stored `detection_index` and `alerts` rows.
   - Live streaming evaluation and state reconstructed after process restart yield identical internal state and identical subsequent decisions.

4. **Alert Episodes & Cooldown Lifecycle:**
   - An `alerts` row is written **only when an episode STARTS** (`fired=True`).
   - While the episode is open, the bin `level` stays `ALERT` across consecutive bins (no ALERT/WATCH flapping).
   - The episode closes only after `alert.clear_bins` (default 2) consecutive valid bins where neither CUSUM nor EARS is at or above its WATCH threshold.
   - After closing, a new surge starts a new episode and writes a new row (provided `cooldown_min` has elapsed).
   - `cooldown_min` (default 30 min) governs the minimum gap between episode starts.

5. **CUSUM Carry Preservation on Suppression:**
   - `cusum_step` refactored into a pure stateless accumulator calculation returning `(s_new, alerted)`.
   - Suppression due to `n_events < min_events` does **not** reset the carry `s_t`.
   - The engine resets `s_t = 0.0` only when an `ALERT` is actually declared (and `reset_after_alert=True`).

6. **Event Thresholds & Evaluation:**
   - `evaluate(bin_result, baseline, n_events)`: `n_events` is a required argument.
   - New config `alert.watch_min_events` (default 2, validated `>= 1`): WATCH requires `n_events >= watch_min_events`.
   - EARS strict inequality: alerts only when `C1 > z + 1e-9` (avoiding floating-point boundary noise, identical to CUSUM).

7. **Surge Tuning: Low-Rate Baseline Comparison (Series f):**
   - Evaluated a low-rate baseline ($\mu=1.0$, $\sigma=0.5$) with a deterministic 3-hour surge (1 event in hr 1, 5 events in hr 2, 6 events in hr 3):

| Configuration | Bins | Min Ev | Fired? | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **15 min / min_events 3** | 12 | 3 | NO | Suppressed (max count/bin = 2 < 3) |
| **30 min / min_events 3** | 6 | 3 | YES (Bin 4, 120m) | Fired via CUSUM |
| **60 min / min_events 3** | 3 | 3 | YES (Bin 2, 120m) | Fired via CUSUM |
| **60 min / min_events 2** | 3 | 2 | YES (Bin 2, 120m) | Fired via CUSUM |

   - *Tuning note:* 15-minute bins with `min_events=3` suppress low-rate surges because events are diluted across bins (≤ 2 per bin). Aggregating to 30 or 60 minutes allows the counts to meet the significance threshold without losing timely detection. Defaults remain 15 min / 3 events until calibrated against real Pi field data.

### Phase 4 Review (Round 3, Final) Decisions

1. **Alerts Table Column `t_alert` & Repository Rebuild:**
   - `AlertEngine.rebuild_state` now reads `t_alert` (primary schema column) with fallback to `created_at`.
   - Access helper `_get` handles `sqlite3.Row` mappings seamlessly without raising `IndexError`.
   - `AlertsRepository.get_by_monitoring_point` added to retrieve typed rows directly by monitoring point.

2. **Window Boundaries via Manila Date + Window Group:**
   - Replaced the arbitrary 6-hour time gap heuristic with exact Manila-local calendar date (`bin_start.astimezone(UTC+8).date()`) plus `window_group`.
   - A change in date or window group resets running window state (`s_t`, `ears_history`, episode tracking).
   - Sensor or power outages inside the same window and date do **not** reset the window state.

3. **Episode Cooldown Continuity & Invariant:**
   - Invariant: A bin **never** reports `ALERT` unless it is inside an active episode (`episode_active == True`).
   - If an alert trigger recurs within `cooldown_min` (30 min) after an episode closes (measured from the last bin where the episode was active), it **continues** that episode: `level ALERT`, `episode_active = True`, `fired = False` (no duplicate alert row).
   - Only when a trigger occurs after `cooldown_min` has fully elapsed does a new episode start and a new alert row write (`fired = True`).

4. **Detection Index Coverage Validation:**
   - `verified_value` is `NULL` whenever `valid = 0` (zero or low coverage, $< 50\%$).
   - A partially or fully verified bin with invalid coverage never produces a non-null rate.

5. **Provisional Defaults:**
   - `index.bin_minutes = 30` (was 15).
   - `alert.ears.lag_bins = 3` (was 4).
   - Kept `alert.min_events = 3`.
   - *Note:* These values will be re-tuned once real Pi baseline data is collected from field deployments.

6. **Alert Parameter Tuning Simulation Evidence (`tune_alerts.py`):**
   - Evaluated 500 days (1000 monitoring windows) per configuration under Poisson baseline and surge conditions:

| Baseline ($\mu$) | Bin ($m$) | Min Ev | False Ep/Win | 2x Det Prob | 2x Med Delay | 3x Det Prob | 3x Med Delay | 5x Det Prob | 5x Med Delay |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.0** | 15m | 2 | 0.2970 | 52.0% | 51.3m | 76.7% | 47.9m | 93.4% | 30.5m |
| **1.0** | 15m | 3 | 0.0290 | 8.7% | 66.5m | 27.1% | 67.4m | 64.2% | 49.5m |
| **1.0** | **30m** | **3** | **0.0810** | **28.6%** | **72.5m** | **56.1%** | **60.0m** | **86.3%** | **46.9m** |
| **1.0** | 60m | 3 | 0.0060 | 8.0% | 125.6m | 33.3% | 111.4m | 75.6% | 90.6m |
| **2.0** | 15m | 2 | 0.7030 | 85.2% | 39.5m | 92.9% | 29.0m | 91.9% | 20.3m |
| **2.0** | 15m | 3 | 0.1430 | 48.7% | 57.0m | 75.1% | 45.2m | 96.0% | 26.0m |
| **2.0** | **30m** | **3** | **0.1970** | **64.2%** | **69.6m** | **90.8%** | **59.1m** | **98.2%** | **37.3m** |
| **2.0** | 60m | 3 | 0.0040 | 15.2% | 119.7m | 56.1% | 106.1m | 94.9% | 78.2m |

   - *Tuning evidence findings:*
     - `30m / min_events 3` offers the optimal trade-off: false alarm rate is well bounded ($\approx 0.08$ to $0.20$ false episodes per 3-hour window), while achieving $86\%-98\%$ detection on 5x surges with median detection delays under $47$ minutes.
     - 15-minute bins severely under-detect at low baseline rates ($27\%$ at 3x surge for $\mu=1.0$) because counts are diluted across bins.
     - 60-minute bins incur excessive detection latency ($> 90-110$ minutes), missing early intervention opportunities within a 3-hour window.


