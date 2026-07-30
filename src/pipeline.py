import cv2
import sys
import time
import datetime
import requests
import numpy as np
from collections import defaultdict, deque

sys.path.append('.')
from src.detection.yolo_pose_estimator import PoseEstimator
from src.features.feature_engineer import FeatureEngineer

API = "http://localhost:8000"
FPS = 30
SUSTAIN = {'distresss': 3 * FPS,
           'drowning': 2 * FPS}
ALERT_THRESHOLD = 0.85
WINDOW = 60

class AquaGuard:
    def __init__(self, source=0, model_name='yolo26s-pose.pt'):
        self.det = PoseEstimator(model_name=model_name)
        self.fe = FeatureEngineer()
        self.source = source
        self.buffers = defaultdict(lambda: deque(maxlen=WINDOW))
        self.counters = defaultdict(int)
        self.risk = {}
        self.alerted = set()
        self._track = getattr(self.det, 'track_all')

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Could not open source: {self.source}")
            return

        t0, frames = time.time(), 0
        print("Running. Press q to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1

            people = self._track(frame)

            for d in people:
                tid = d.get('track_id')
                if tid is None:
                    continue
                key = f"P{tid}"
                self.buffers[key].append(d['kp'])

                if len(self.buffers[key]) == WINDOW:
                    self._predict(key)
                    self._check_alert(key)

                self._draw(frame, d['bbox'], key)

            fps_now = frames / max(time.time() - t0, 1e-6)
            cv2.putText(frame, f"AquaGuard | {fps_now:.1f} FPS | {len(people)} tracked",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('AquaGuard', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {frames} frames at {frames/max(time.time()-t0, 1e-6):.1f} FPS avg")

    def _predict(self, key):
        feats = self.fe.extract_features(list(self.buffers[key]))
        vec = [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in feats.values()]
        try:
            r = requests.post(f"{API}/predict", timeout=0.15, json={
                "person_id": key,
                "features": vec,
                "timestamp": datetime.datetime.now().isoformat()})
            if r.ok:
                res = r.json()
                self.risk[key] = (res['label'], res['confidence'])
            elif r.status_code == 400:
                print(f"Feature mismatch: {r.json().get('detail')}")
        except requests.RequestException:
            pass
