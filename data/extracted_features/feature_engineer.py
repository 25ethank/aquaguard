import numpy as np
import pandas as pd

# MediaPipe indices
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26

class FeatureEngineer:
    def __init__(self, window_size=90):
        """
        window_size: number of frames to analyze at once
        """
        self.window_size = window_size

    def extract_features(self, keypoint_sequence):
        """
        Takes a sequence of keypoints arrays (window_size x 33 x 3)
        Returns a feature vector for ML classification
        """
        features = {}

        # Head Position - below water line?
        head_y_positions = [kp[NOSE][1] for kp in keypoint_sequence]
        features["head_y_mean"] = np.mean(head_y_positions)
        features["head_y_std"] = np.std(head_y_positions)
        features["head_y_trend"] = np.polyfit(range(len(head_y_positions)), head_y_positions, 1)[0]

        # Arm Movement Frequency - thrashing = high frewuency
        left_wrist_y = [kp[LEFT_WRIST][1] for kp in keypoint_sequence]
        right_wrist_y = [kp[RIGHT_WRIST][1] for kp in keypoint_sequence]
        # Count direction changes
        features['left_wrist_freq'] = self._count_direction_changes(left_wrist_y)
        features['right_wrist_freq'] = self._count_direction_changes(right_wrist_y)

    def _count_direction_changes(self, sequence):
        """
        Count how many times movement direction reverses
        """
        changes = 0
        for i in range(1, len(sequence) - 1):
            if (sequence[i] - sequence[i-1]) * (sequence[i+1] - sequence[i]) < 0:
                changes += 1
        return changes
