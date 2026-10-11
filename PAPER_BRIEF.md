# MosquitoSentinel — Paper Brief (for the developer agent)

You have not read the thesis manuscript. This file is the digest. Treat it as the **source of truth for what the thesis promises**. `ARCHITECTURE.md` says how to build it.

## 1. Roles and decision rights

- **Claude = researcher + system architect.** Owns the thesis requirements, the architecture, and the data/evaluation design.
- **Antigravity (you) = senior developer.** Owns implementation quality: code structure, performance, testing, deployment, debugging.
- You **may and should push back** on technical feasibility (for example "this FPS is unreachable on a Pi 5", "this library is unstable on ARM"). Do it by writing the problem and a proposed alternative in `DECISIONS.md`, then continue with the safest option.
- You **may not silently change** thesis requirements: the SOP items, objectives, scope, delimitations, or the evaluation evidence listed below. If a change seems necessary, flag it in `DECISIONS.md` and ask the user.

## 2. Project identity

- **Title:** MosquitoSentinel: An AI-Powered Real-Time Vision-Based Detection and Alert System for *Aedes aegypti* Surveillance
- **Course:** CpE Practice and Design 1, BSCPE 4A, Institute of Computing Studies, Colegio de Montalban, 2026.
- **Deployment site:** Phase 1E Extension, Kasiglahan Village, Montalban (Rodriguez), Rizal, Philippines. This is a high-risk residential community. Local needs analysis with the Municipal Health Office and MESU found mosquito-borne cases in several barangays (Burgos, San Jose, San Isidro highest), with rising cases and recorded deaths from 2024.

## 3. The problem being solved

*Aedes aegypti* transmits dengue, chikungunya, Zika and yellow fever and breeds in stagnant water near homes. Existing surveillance has two gaps: (1) the national PIDSR system tracks **reported human cases**, not mosquito activity; (2) manual field surveillance (inspections, larval surveys, ovitraps) is **labor-intensive, slow and gives limited information over time**. MosquitoSentinel is meant to **complement** existing surveillance by directly and automatically monitoring adult *Aedes aegypti* activity and warning early.

## 4. The proposed system (as written in the paper)

Raspberry Pi 5 + Global Shutter Camera RAW10 Zoom + YOLOv11 detect and classify *Aedes aegypti*. Confirmed detections are recorded (date, time, species, confidence, location, image) and aggregated into a **Detection Index**. A **CUSUM / EARS-style** statistical method flags unusual increases in activity. Alerts go through **LED + buzzer** and a **Flask web dashboard**. Runs locally (edge AI), without depending on the cloud.

**Update since the paper was written (the paper will be revised to match):** the system processes **video frames**, not periodic still images. A second-stage **CNN species classifier** (Aegypti vs Albopictus, 64x64 input) may confirm YOLO detections.

## 5. Statement of the Problem (research questions) — the code must produce evidence for each

- **2.1** What is the performance of the system in terms of **operational consistency and stability** during the monitoring period?
- **2.2** What is the performance of the **YOLOv11-based vision detection** in terms of:
  - **2.2.1** detection accuracy of *Aedes aegypti*; and
  - **2.2.2** accuracy of calculating the **Detection Index over time**?
- **2.3** What is the **overall system performance** in terms of:
  - **2.3.1** user acceptability and usability;
  - **2.3.2** timeliness and accuracy of generated alerts; and
  - **2.3.3** effectiveness compared with traditional manual mosquito inspection methods?

## 6. Objectives

- **Main:** develop and evaluate MosquitoSentinel for automated *Aedes aegypti* detection and monitoring to support timely identification of mosquito activity and proactive public-health response.
- **3.1** Evaluate operational consistency and stability during the monitoring period.
- **3.2** Determine YOLOv11 detection accuracy and the accuracy of Detection Index calculation over time.
- **3.3** Evaluate user acceptability and usability, timeliness and accuracy of alerts, and practical effectiveness through comparative feedback against traditional manual inspection.

## 7. Scope and delimitations (hard rules)

- Surveillance and early-warning tool only. **It does not diagnose disease, predict dengue cases, or detect pathogens in mosquitoes.** Never write UI text implying otherwise.
- Species: *Aedes aegypti* only. Other insects/species are out of scope (log as non-target).
- Image-based only: **no acoustic, wing-beat, genetic, microscopic or molecular methods.**
- Field of view limits: activity outside the camera view (or between processed frames) may be missed. Document this honestly.
- Detection quality depends on dataset quality, quantity, diversity and representation.
- Part of testing is in a **controlled setup**; results may not represent complex outdoor conditions (multiple insects, shadows, backgrounds).
- Tested under selected light conditions only; not extreme weather (heavy rain, wind, flooding, severe glare).
- **Fixed hardware** (Pi 5 + Global Shutter Camera RAW10 Zoom). Other hardware may change results.
- **Monitoring windows: 06:00–09:00 and 16:00–19:00.** Not evaluated for 24-hour operation.
- Short-range deployment near breeding-prone areas; not municipal or city scale.
- Validated against manual inspection and available surveillance methods; differences in methods may limit the comparison.
- Academic prototype: no commercial production, regulatory approval, or long-term sustainability claims.

## 8. Design implications drawn from the literature (why the architecture looks the way it does)

- Mosquitoes are **small and hard to capture clearly**; a YOLOv8s mosquito dataset study reported only mAP@50 57.1%. Therefore: resolution, camera placement/zoom, dataset diversity and augmentation matter, and the system must be tested on real Pi-camera footage, not only the training dataset.
- High accuracies in related work (97–99%) came from **laboratory or close-up conditions** (intelligent traps, egg images). Do not assume field performance matches; the thesis must report field validation honestly.
- Performance degrades with illumination, pose, occlusion and non-diverse data, so log confidence and keep evidence (snapshot/clip) for human verification.
- Edge deployment needs model optimization for the Pi (quantization/export), so FPS and latency are measured, not assumed.
- Statistical early warning (CUSUM, EARS, adaptive thresholds) is established for human-case data; **applying it to automatically collected mosquito detections is the paper's contribution.** Therefore the baseline, parameters and alert logic must be explicit, configurable and reproducible.
- Real-world surveillance systems need **reliability and timely transmission** (pilot early-warning studies reported reporting within 24 h and problems with connectivity/workforce). Hence offline-first operation and health logging.
- A local drainage-monitoring system used Raspberry Pi + computer vision + a **Flask dashboard** with alerts and historical records; this is the precedent for the dashboard design.

## 9. Gaps in the paper that the code must make explicit (do not hide them)

1. **Detection Index has no formula in the paper.** `ARCHITECTURE.md` section 7 proposes one (confirmed events per active hour per time bin). It must be configurable and documented in the README so the authors can state it in the methodology.
2. **"Real-time" vs interval capture:** define real-time as "processed immediately after capture/reception of each frame".
3. **"Confirmed detection" is undefined:** implemented as the event rule in `ARCHITECTURE.md` section 6; thresholds are config values.
4. **CUSUM/EARS parameters and baseline are undefined:** implemented as configurable with a baseline mode.
5. Section 6.8 mentions a "monitoring chamber" while the Scope describes field deployment. Keep the camera placement configurable and log which setup produced each session.

## 10. Evidence the authors must be able to show the panel

- Model metrics (precision, recall, F1, mAP50, mAP50-95, confusion matrix) and **event-level** field accuracy from human verification.
- Detection Index: system value vs verified value over time, with error (MAE/MAPE).
- Alerts: latency from detection to alert to dashboard display, and true vs false alerts.
- Stability: uptime, FPS, dropped frames, inference latency, temperature, restarts, over the monitoring period.
- Usability: questionnaire results from users.
- Comparison with manual inspection: manual counts side by side with system output.
- Everything exportable as CSV so the authors can build tables and charts for the manuscript.

## 11. Terminology (keep consistent in code, UI and docs)

Use **MosquitoSentinel** (never "DengueSentinel"), ***Aedes aegypti***, **Raspberry Pi 5**, **Global Shutter Camera RAW10 Zoom**, **YOLOv11**, **Detection Index**, **Flask dashboard**. UI wording for alerts: "increased *Aedes aegypti* activity detected".
