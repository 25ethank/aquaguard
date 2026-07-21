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

    def _parse(self, result, frame_shape, with_ids=False):
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
