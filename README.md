# AquaGuard
Real-time drowning detection using computer vision and deep learning.

---

## What This Is

I'm building a system that monitors a pool camera feed, detects every swimmer simultaneously, and alerts a lifeguard within seconds if someone shows signs of drowning.

I've been a competitive swimmer and water polo player for 10+ years and certified lifeguard for 4 years. I know what drowning actually looks like, and how easy it is to miss in a crowded pool. At my internship at Guident, I work on real-time AI monitoring systems for autonomous vehicles where AI flags anomalies and a human intervenes, keeping the human-in-the-loop. The same idea applies directly to pool safety, so combining my experiences, I built it.

---

## How It Works

The pipeline runs in stages:

1. **YOLO26** detects every person in the frame, draws a bounding box around each one, and estimates their pose
2. **ByteTrack** assigns a persistent ID to each swimmer across frames so the same person stays Person_1 throughout the video
3. A **sliding window** collects 90 frames (~3 seconds) of keypoints per person
4. **Feature engineering** converts raw keypoints into meaningful signals — body angle, arm symmetry, head position over time, wrist height relative to shoulders
5. An **LSTM classifier** analyzes the 3-second window and outputs a drowning confidence score
6. If confidence stays above 85% for 3 consecutive seconds, the system triggers a **Twilio SMS alert** to the lifeguard and logs the incident to **Firebase**

---

## Stack

- YOLO26 — person detection and pose estimation
- PyTorch — LSTM model
- scikit-learn — Random Forest baseline
- FastAPI — ML model serving
- Twilio — SMS alerts
- Firebase — incident logging
- Streamlit — live dashboard
- OpenCV — video processing

---

## Progress

- [x] YOLOv8 detection pipeline working
- [x] Multi-person tracking
- [x] Dataset collection and labeling
- [x] Feature engineering
- [x] Model training
- [x] FastAPI backend
- [ ] Alerts and logging
- [ ] Dashboard
- [ ] Deployment

---

*Ethan Kim — MIT EECS '29*
