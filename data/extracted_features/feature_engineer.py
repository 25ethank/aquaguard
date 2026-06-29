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
