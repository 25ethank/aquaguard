import numpy as np
import pandas as pd

# MediaPipe indices
NOSE = 0
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

class FeatureEngineer:
    """
    Converts a sequence of pose keypoints into biomechanical features that distinguish normal swimming, distress, and drowning
    """
    def __init__(self, window_size=90, fps=30):
        self.window_size = window_size
        self.fps = fps

    def extract_features(self, keypoint_sequence):
        """
        Takes a sequence of pose keypoints arrays (window_size x 33 x 3)
        Returns a feature vector for ML classification
        """
        features = {}
        kps = np.array(keypoint_sequence)

        # Head Position - below water line?
        head_y_positions = kps[:, NOSE, 1]
        features["head_y_mean"] = np.mean(head_y_positions)
        features["head_y_std"] = np.std(head_y_positions)
        features["head_y_trend"] = np.polyfit(range(len(head_y_positions)), head_y_positions, 1)[0]

        # Arm Movement Frequency - thrashing = high frewuency
        left_wrist_y = kps[:, LEFT_WRIST, 1]
        right_wrist_y = kps[:, RIGHT_WRIST, 1]
        # Count direction changes
        features['left_wrist_freq'] = self._count_direction_changes(left_wrist_y)
        features['right_wrist_freq'] = self._count_direction_changes(right_wrist_y)

        # Arm Symmetry - drowning = asymmetric arm movement
        left_wrist_positions = kps[:, LEFT_WRIST, :2]
        right_wrist_positions = kps[:, RIGHT_WRIST, :2]
        features['arm_symmetry'] = self._compute_symmetry(left_wrist_positions, right_wrist_positions)

        # Vertical Displacement - sinking?
        center_y = [(kp[LEFT_HIP][1] + kp[RIGHT_HIP][1]) / 2 for kp in keypoint_sequence]
        features['vertical_displacement'] = center_y[-1] - center_y[0]
        features['vertical_velocity_mean'] = np.mean(np.diff(center_y))

        # Body Angle - drowning = vertical, swimming = horizontal
        shoulder_mid = (kps[:, LEFT_SHOULDER, :2] + kps[:, RIGHT_SHOULDER, :2]) / 2
        hip_mid = (kps[:, LEFT_HIP, :2] + kps[:, RIGHT_HIP, :2]) / 2
        d = hip_mid - shoulder_mid
        angles = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]) + 1e-6))
        features['body_angle_mean'] = np.mean(angles)
        features['body_angle_std'] = np.std(angles)
        features['body_vertical_ratio'] = np.mean(angles > 60)

        # Leg movement - drowning = minimal kick
        la_y = kps[:, LEFT_ANKLE, 1]
        ra_y = kps[:, RIGHT_ANKLE, 1]
        features['leg_activity'] = (np.std(la_y) + np.std(ra_y)) / 2

    def _count_direction_changes(self, sequence):
        """
        Count how many times movement direction reverses
        """
        changes = 0
        for i in range(1, len(sequence) - 1):
            if (sequence[i] - sequence[i-1]) * (sequence[i+1] - sequence[i]) < 0:
                changes += 1
        return changes

    def _compute_symmetry(self, left_positions, right_positions):
        """
        Returns 0-1 score where 1 = perfectly symmetric, 0 = completely asymmetric
        """
        left_arr = np.array(left_positions)
        right_arr = np.array(right_positions)
        diff = np.abs(left_arr - right_arr)
        return 1 - np.clip(np.mean(diff), 0, 1)
