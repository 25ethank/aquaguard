import numpy as np

# COCO-17 indices
NOSE = 0
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

class FeatureEngineer:
    """
    Converts a sequence of pose keypoints into biomechanical features that distinguish normal swimming, distress, and drowning
    """
    def __init__(self, window_size=90, fps=30):
        self.window_size = window_size
        self.fps = fps

    def extract_features(self, keypoint_sequence):
        """
        Takes a sequence of pose keypoints arrays (window_size x 17 x 4)
        Returns a feature vector for ML classification
        """
        features = {}
        kps = np.array(keypoint_sequence)

        # Head Position - below water line?
        head_y = kps[:, NOSE, 1]
        features["head_y_mean"] = np.mean(head_y)
        features["head_y_std"] = np.std(head_y)
        features["head_y_trend"] = np.polyfit(np.arange(len(head_y)), head_y, 1)[0] * 1000

        # Arm Movement Frequency - thrashing = high frewuency
        lw_y = kps[:, LEFT_WRIST, 1]
        rw_y = kps[:, RIGHT_WRIST, 1]
        # Count direction changes
        features['left_wrist_freq'] = self._direction_changes(lw_y)
        features['right_wrist_freq'] = self._direction_changes(rw_y)

        # Arm Symmetry - drowning = asymmetric arm movement
        lw = kps[:, LEFT_WRIST, :2]
        rw = kps[:, RIGHT_WRIST, :2]
        mid_x = (kps[:, LEFT_SHOULDER, 0] + kps[:, RIGHT_SHOULDER, 0]) / 2
        rw_mirrored = rw.copy()
        rw_mirrored[:, 0] = 2 * mid_x - rw[:, 0]
        features['arm_symmetry'] = 1 - np.clip(np.mean(np.abs(lw - rw_mirrored)), 0, 1)

        # Wrists above shoulders
        shoulder_y = (kps[:, LEFT_SHOULDER, 1] + kps[:, RIGHT_SHOULDER, 1]) / 2
        wrist_y = (lw_y + rw_y) / 2
        features['wrist_above_shoulder_ratio'] = np.mean(wrist_y < shoulder_y)

        # Lateral arm spread (drowning = arms pressing out/down, not forward)
        wrist_spread = np.abs(kps[:, LEFT_WRIST, 0] - kps[:, RIGHT_WRIST, 0])
        shoulder_spread = np.abs(kps[:, LEFT_SHOULDER, 0] - kps[:, RIGHT_SHOULDER, 0]) + 1e-6
        features['arm_spread_ratio'] = np.mean(wrist_spread / shoulder_spread)

        # Vertical Displacement - sinking?
        center_y = (kps[:, LEFT_HIP, 1] + kps[:, RIGHT_HIP, 1]) / 2
        features['vertical_displacement'] = center_y[-1] - center_y[0]
        features['vertical_velocity_mean'] = np.mean(np.diff(center_y))
        features['vertical_bobbing'] = self._direction_changes(center_y)

        # Body Angle - drowning = vertical, swimming = horizontal
        shoulder_mid = (kps[:, LEFT_SHOULDER, :2] + kps[:, RIGHT_SHOULDER, :2]) / 2
        hip_mid = (kps[:, LEFT_HIP, :2] + kps[:, RIGHT_HIP, :2]) / 2
        d = hip_mid - shoulder_mid
        angles = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]) + 1e-6))
        features['body_angle_mean'] = np.mean(angles)
        features['body_angle_std'] = np.std(angles)
        features['body_vertical_ratio'] = np.mean(angles > 60)

        # Leg movement - drowning = minimal kick
        VIS_THRESH = 0.5
        la_vis = kps[:, LEFT_ANKLE, 3] > VIS_THRESH
        ra_vis = kps[:, RIGHT_ANKLE, 3] > VIS_THRESH

        if la_vis.sum() > 10 and ra_vis.sum() > 10:
            features['leg_activity'] = (np.std(kps[la_vis, LEFT_ANKLE, 1]) +
                                        np.std(kps[ra_vis, RIGHT_ANKLE, 1])) / 2
        else:
            features['leg_activity'] = np.nan

        return features

    def _direction_changes(self, seq):
        diffs = np.diff(seq)
        signs = np.sign(diffs)
        return int(np.sum(signs[1:] != signs[:-1]))
