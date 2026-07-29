import os
import cv2
import numpy as np
from ultralytics import YOLO

# COCO-17 keypoint indices
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

SKELETON = [
    (0, 3), (0, 4), (3, 5), (4, 6),     # head to shoulders
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 6),                             # shoulders
    (11, 12),                           # hips
    (5, 11), (6, 12),                   # torso
    (11, 13), (13, 15),                 # left leg
    (12, 14), (14, 16),                 # right leg
]

class PoseEstimator:
    """
    Unified detection + pose via YOLO-pose. One model, one pass.
    Keypoints are (17, 4): x, y (normalized to frame), z placeholder, confidence.
    """

    def __init__(self, model_name='yolo26m-pose.pt', conf_threshold=0.3):
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, 'data', 'models', model_name)
        self.model = YOLO(path if os.path.exists(path) else model_name)
        self.conf_threshold = conf_threshold

    def _parse(self, result, with_ids=False):
        """Turn a YOLO result into a list of person dicts."""
        people = []
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return people

        ids = None
        if with_ids and result.boxes.id is not None:
            ids = result.boxes.id.int().tolist()

        for i in range(len(result.boxes)):
            conf = float(result.boxes.conf[i].item())
            if conf < self.conf_threshold:
                continue
            bbox = [int(v) for v in result.boxes.xyxy[i].tolist()]

            xy = result.keypoints.xyn[i].cpu().numpy()        # (17, 2)
            kconf = result.keypoints.conf[i].cpu().numpy()    # (17,)
            z = np.zeros((len(xy), 1))
            kp = np.concatenate([xy, z, kconf[:, None]], axis=1)   # (17, 4)

            person = {'bbox': bbox, 'conf': conf, 'kp': kp}
            if ids is not None:
                person['track_id'] = ids[i]
            people.append(person)

        return people

    def detect_all(self, frame):
        """All people in frame, with keypoints. Used by extraction."""
        r = self.model(frame, classes=[0], verbose=True)[0]
        return self._parse(r)

    def track(self, frame):
        """Detection + pose + ByteTrack. Returns dicts with persistent track_id."""
        r = self.model.track(frame, classes=[0], persist=True,
                             tracker='bytetrack.yaml', verbose=False)[0]

        out = []
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return out

        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else None

        for i in range(len(r.boxes)):
            conf = float(r.boxes.conf[i].item())
            if conf < self.conf_threshold:
                continue
            bbox = [int(v) for v in r.boxes.xyxy[i].tolist()]
            xy = r.keypoints.xyn[i].cpu().numpy()
            kconf = r.keypoints.conf[i].cpu().numpy()
            z = np.zeros((len(xy), 1))
            kp = np.concatenate([xy, z, kconf[:, None]], axis=1)

            entry = {'bbox': bbox, 'conf': conf, 'kp': kp}
            if ids is not None:
                entry['track_id'] = ids[i]
            out.append(entry)
        return out

    def extract_keypoints(self, frame, bbox=None):
        """
        Backward-compatible: returns keypoints for the largest person, or None.
        bbox is accepted but ignored (detection is built in now).
        """
        people = self.detect_all(frame)
        if not people:
            return None
        largest = max(people, key=lambda p: (p['bbox'][2] - p['bbox'][0]) *
                                            (p['bbox'][3] - p['bbox'][1]))
        return largest['kp']

    def draw(self, frame, person, color=None, label=None):
        """Draw one person's box and skeleton. person = dict from detect_all"""
        x1, y1, x2, y2 = person['bbox']
        color = color or (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        h, w = frame.shape[:2]
        kp = person['kp']
        for a, b in SKELETON:
            if kp[a][3] < 0.4 or kp[b][3] < 0.4:
                continue
            pa = (int(kp[a][0] * w), int(kp[a][1] * h))
            pb = (int(kp[b][0] * w), int(kp[b][1] * h))
            cv2.line(frame, pa, pb, (0, 0, 255), 2)

        if label:
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    pose = PoseEstimator()
    cap = cv2.VideoCapture(src)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        people = pose.detect_all(frame)
        for i, p in enumerate(people):
            pose.draw(frame, p, label=f"P{i} {p['conf']:.2f}")
        cv2.putText(frame, f"detections: {len(people)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if not people else (255, 255, 255), 2)
        cv2.imshow('AquaGuard pose', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
