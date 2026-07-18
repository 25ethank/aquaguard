import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import urllib.request
import os

class PoseEstimator:
    _MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    _MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'data', 'models', 'pose_landmarker.task')

    def __init__(self):
        if not os.path.exists(self._MODEL_PATH):
            urllib.request.urlretrieve(self._MODEL_URL, self._MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=self._MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract_keypoints(self, frame, bbox=None):
        """
        Extracts 33 body keypoints from a frame.
        If bbox provided, crop to that region first (for multi-person).
        Returns array of shape (33, 3): x, y, z coords, or None if no pose detected.
        """
        if bbox:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            pad = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        keypoints = []
        for landmark in result.pose_landmarks[0]:
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        return np.array(keypoints)

    def draw_skeleton(self, frame, keypoints, bbox=None):
        """
        Draws skeleton of main body keypoints for visualization.
        """
        connections = [
            (0, 7), (0, 8), (7, 11), (8, 12),   # head
            (11, 13), (13, 15),                 # left arm
            (12, 14), (14, 16),                 # right arm
            (11, 12),                           # shoulders
            (23, 24),                           # hips
            (11, 23), (12, 24),                 # torso
            (23, 25), (25, 27),                 # left leg
            (24, 26), (26, 28),                 # right leg
            (27, 31), (28, 32)                  # feet
        ]

        if bbox:
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = x2 - x1
        else:
            h, w = frame.shape[:2]
            x1, y1 = 0, 0

        for start_idx, end_idx in connections:
            start_key = keypoints[start_idx]
            end_key = keypoints[end_idx]

            sx = int(start_key[0] * w) + x1
            sy = int(start_key[1] * h) + y1
            ex = int(end_key[0] * w) + x1
            ey = int(end_key[1] * h) + y1
            cv2.line(frame, (sx, sy), (ex, ey), (0, 0, 255), 2)

        return frame
