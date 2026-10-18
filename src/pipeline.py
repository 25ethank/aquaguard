"""
Real-time AquaGuard pipeline.

  frame -> YOLO-pose + ByteTrack -> plausibility filter -> per-track keypoint buffer
        -> features -> POST /predict -> AlertPolicy -> POST /incidents (-> SMS)

Usage:
  python src/pipeline.py data/raw_videos/clip.mp4
  python src/pipeline.py 0                       # webcam
  python src/pipeline.py clip.mp4 --out demo.mp4 --headless
"""
import argparse
import datetime
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import requests

sys.path.append('.')
from src import config
from src.alerts.policy import AlertPolicy
from src.detection.yolo_pose_estimator import PoseEstimator, is_plausible_person
from src.features.feature_engineer import FeatureEngineer

ALERT_TEXT = {
    'drowning': "MOTIONLESS SWIMMER - possible unconscious victim",
    'distress': "ACTIVE STRUGGLE - swimmer in distress",
}


@dataclass
class Track:
    buffer: deque = field(default_factory=lambda: deque(maxlen=config.WINDOW))
    age: int = 0                 # frames this ID has been seen
    last_seen: int = 0           # frame index
    last_pred: int = -10**9      # frame index of last prediction
    label: str = '...'
    conf: float = 0.0
    danger: float = 0.0          # P(distress) + P(drowning)
    proba: dict = field(default_factory=dict)


class AquaGuard:
    def __init__(self, source=0, model_name=config.POSE_MODEL, api=config.API_URL):
        self.det = PoseEstimator(model_name=model_name, conf_threshold=config.CONF_THRESHOLD)
        self.source = source
        self.api = api
        self.http = requests.Session()
        self.tracks = {}
        self.policy = AlertPolicy()
        self.fe = None   # created once we know the source fps

    # ---------------------------------------------------------------- main loop
    def run(self, out_path=None, headless=False):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Could not open source: {self.source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS
        if fps < 1 or fps > 240:
            fps = config.FPS
        self.fe = FeatureEngineer(fps=fps)

        writer = None
        if out_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        self._check_api()
        t0, n = time.time(), 0
        print("Running. Press q to quit." if not headless else "Running headless.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_video = n / fps

            people = [p for p in self.det.track_all(frame)
                      if p.get('track_id') is not None
                      and is_plausible_person(p['bbox'], frame.shape)]

            for p in people:
                key = f"P{p['track_id']}"
                tr = self.tracks.setdefault(key, Track())
                tr.age += 1
                tr.last_seen = n
                if tr.age < config.MIN_TRACK_AGE:
                    continue          # junk tracks (splashes, reflections) die before here
                tr.buffer.append(p['kp'])

                if (len(tr.buffer) >= config.MIN_FILL
                        and n - tr.last_pred >= config.PREDICT_EVERY):
                    tr.last_pred = n
                    self._predict(key, tr)
                    alert = self.policy.update(key, tr.proba, t_video)
                    if alert:
                        self._alert(alert)

                self._draw(frame, p['bbox'], key, tr)

            self._purge(n)
            n += 1
            live_fps = n / max(time.time() - t0, 1e-6)
            cv2.putText(frame, f"AquaGuard | {live_fps:.1f} FPS | {len(people)} tracked",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer:
                writer.write(frame)
            if not headless:
                cv2.imshow('AquaGuard', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {n} frames at {n / max(time.time() - t0, 1e-6):.1f} FPS avg")

    # ---------------------------------------------------------------- helpers
    def _check_api(self):
        try:
            h = self.http.get(f"{self.api}/health", timeout=2).json()
            print(f"API ok: {h['n_features']} features, classes {h['classes']}")
        except Exception as e:
            print(f"WARNING: API not reachable at {self.api} ({e}). "
                  f"Start it with: uvicorn src.api.main:app")

    def _predict(self, key, tr):
        feats = self.fe.extract_features(list(tr.buffer))
        try:
            r = self.http.post(f"{self.api}/predict", timeout=0.3, json={
                "person_id": key,
                "features": FeatureEngineer.to_vector(feats),
                "timestamp": datetime.datetime.now().isoformat()})
            if r.ok:
                res = r.json()
                tr.label, tr.conf, tr.proba = res['label'], res['confidence'], res['proba']
                tr.danger = self.policy.danger(tr.proba)[0]
            elif r.status_code in (400, 422):
                print(f"Feature mismatch: {r.json().get('detail')}")
        except requests.RequestException:
            pass

    def _alert(self, a):
        kind = ALERT_TEXT[a['label']]
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        print(f"\n*** ALERT {ts} | {kind} | {a['person_id']} | "
              f"{a['confidence']:.0%} for {a['sustained_s']}s ***\n")
        try:
            self.http.post(f"{self.api}/incidents", timeout=1.0, json={
                **a, "alert_type": kind, "timestamp": ts,
                "source": str(self.source)})
        except requests.RequestException:
            pass

    def _purge(self, n):
        """Forget tracks that haven't been seen recently so state doesn't grow forever."""
        for key in [k for k, t in self.tracks.items() if n - t.last_seen > config.STALE_AFTER]:
            del self.tracks[key]
            self.policy.forget(key)

    def _draw(self, frame, bbox, key, tr):
        if tr.danger >= config.ALERT_THRESHOLD:
            color = (0, 0, 255)
        elif tr.danger >= 0.5:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{key} {tr.label} {tr.conf:.0%}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('source', nargs='?', default='0')
    ap.add_argument('--out', help='save annotated video (for your demo GIF)')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--model', default=config.POSE_MODEL)
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    AquaGuard(source=src, model_name=args.model).run(out_path=args.out, headless=args.headless)
