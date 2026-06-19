import mediapipe as mp
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, frame, bbox=None):
        """
        Extracts 33 body keypoints from a frame.
        If bbox provided, crop to that region first (for multi-person)
        Returns array in shape (33, 3): x, y, z coords for all keypoints, none if no pose detected
        """
        if bbox:
            x1, y1, x2, y2 = bbox
            pad = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)          # Convert to RGB from BGR for mp
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])

        return np.array(keypoints)

    def draw_skeleton(self, frame, keypoints, bbox=None):
        """
        Draws skeleton of main body keypoints for visualization
        """
        connections = [
            (11, 13), (13, 15),         # left arm
            (12, 14), (14, 16),         # right arm
            (11, 12),                   # shoulders
            (23, 24),                   # hips
            (11, 23), (12, 24),         # torso
            (23, 25), (25, 27),         # left leg
            (24, 26), (26, 28),         # right leg
            (27, 31), (28, 32)          # feet
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
