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

VIS_THRESH = 0.5

# Fixed feature order. Training columns and the live /predict vector are both
# built from this list, so they can never drift apart.
FEATURE_NAMES = [
    "head_y_mean",
    "head_y_std",
    "head_y_trend",
    "left_wrist_freq_hz",
    "right_wrist_freq_hz",
    "arm_symmetry",
    "wrist_above_shoulder_ratio",
    "arm_spread_ratio",
    "vertical_velocity",
    "vertical_bobbing_hz",
    "body_angle_mean",
    "body_angle_std",
    "body_vertical_ratio",
    "leg_activity",
]


class FeatureEngineer:
    """
    Converts a sequence of pose keypoints into biomechanical features that
    distinguish normal swimming, distress, and drowning.

    Every feature is a mean, a ratio, or a per-second rate, so a 63-frame
    buffer and a 90-frame buffer of the same behavior produce the same values.
    (Raw counts like "direction changes in the window" scale with window length,
    which made short live buffers look like motionless swimmers.)
    """

    def __init__(self, fps=30):
        self.fps = fps

    def extract_features(self, keypoint_sequence):
        """
        keypoint_sequence: (T, 17, 4) array-like of x, y (normalized), z, conf.
        Returns a dict keyed by FEATURE_NAMES.
        """
        kps = np.asarray(keypoint_sequence, dtype=float)
        T = len(kps)
        if T < 3:
            raise ValueError(f"need at least 3 frames, got {T}")
        duration_s = T / self.fps
        f = {}

        # Movement must exceed ~15% of torso length to count as a stroke/bob.
        # Scales with the swimmer's size in frame, so near and far swimmers match.
        torso = np.linalg.norm(
            (kps[:, LEFT_SHOULDER, :2] + kps[:, RIGHT_SHOULDER, :2]) / 2 -
            (kps[:, LEFT_HIP, :2] + kps[:, RIGHT_HIP, :2]) / 2, axis=1)
        eps = max(0.15 * float(np.median(torso)), 0.003)

        # Head position: is the head low in frame / sinking?
        head_y = kps[:, NOSE, 1]
        f["head_y_mean"] = np.mean(head_y)
        f["head_y_std"] = np.std(head_y)
        # slope per frame, scaled for readability (already length-invariant)
        f["head_y_trend"] = np.polyfit(np.arange(T), head_y, 1)[0] * 1000

        # Arm movement frequency: thrashing = many direction changes per second
        lw_y = kps[:, LEFT_WRIST, 1]
        rw_y = kps[:, RIGHT_WRIST, 1]
        f["left_wrist_freq_hz"] = self._direction_changes(lw_y, eps) / duration_s
        f["right_wrist_freq_hz"] = self._direction_changes(rw_y, eps) / duration_s

        # Arm symmetry: mirror right wrist across the shoulder midline
        lw = kps[:, LEFT_WRIST, :2]
        rw = kps[:, RIGHT_WRIST, :2]
        mid_x = (kps[:, LEFT_SHOULDER, 0] + kps[:, RIGHT_SHOULDER, 0]) / 2
        rw_m = rw.copy()
        rw_m[:, 0] = 2 * mid_x - rw[:, 0]
        f["arm_symmetry"] = 1 - np.clip(np.mean(np.abs(lw - rw_m)), 0, 1)

        # Wrists above shoulders (instinctive drowning response: arms pressing down at surface)
        shoulder_y = (kps[:, LEFT_SHOULDER, 1] + kps[:, RIGHT_SHOULDER, 1]) / 2
        wrist_y = (lw_y + rw_y) / 2
        f["wrist_above_shoulder_ratio"] = np.mean(wrist_y < shoulder_y)

        # Lateral arm spread relative to shoulder width
        wrist_spread = np.abs(kps[:, LEFT_WRIST, 0] - kps[:, RIGHT_WRIST, 0])
        shoulder_spread = np.abs(kps[:, LEFT_SHOULDER, 0] - kps[:, RIGHT_SHOULDER, 0]) + 1e-6
        f["arm_spread_ratio"] = np.mean(wrist_spread / shoulder_spread)

        # Vertical motion of hip center: sinking speed and bobbing rate
        center_y = (kps[:, LEFT_HIP, 1] + kps[:, RIGHT_HIP, 1]) / 2
        f["vertical_velocity"] = (center_y[-1] - center_y[0]) / duration_s
        f["vertical_bobbing_hz"] = self._direction_changes(center_y, eps) / duration_s

        # Body angle: drowning = vertical, swimming = horizontal
        shoulder_mid = (kps[:, LEFT_SHOULDER, :2] + kps[:, RIGHT_SHOULDER, :2]) / 2
        hip_mid = (kps[:, LEFT_HIP, :2] + kps[:, RIGHT_HIP, :2]) / 2
        d = hip_mid - shoulder_mid
        angles = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0]) + 1e-6))
        f["body_angle_mean"] = np.mean(angles)
        f["body_angle_std"] = np.std(angles)
        f["body_vertical_ratio"] = np.mean(angles > 60)

        # Leg activity, gated on keypoint confidence: submerged ankles jitter
        # and would otherwise make passive victims look like the most active kickers.
        la_vis = kps[:, LEFT_ANKLE, 3] > VIS_THRESH
        ra_vis = kps[:, RIGHT_ANKLE, 3] > VIS_THRESH
        min_vis = max(5, int(0.15 * T))
        if la_vis.sum() >= min_vis and ra_vis.sum() >= min_vis:
            f["leg_activity"] = (np.std(kps[la_vis, LEFT_ANKLE, 1]) +
                                 np.std(kps[ra_vis, RIGHT_ANKLE, 1])) / 2
        else:
            f["leg_activity"] = np.nan   # imputed by the model pipeline

        return {k: float(f[k]) for k in FEATURE_NAMES}

    @staticmethod
    def to_vector(features):
        """Dict -> list in FEATURE_NAMES order, NaN -> None (JSON-safe)."""
        return [None if np.isnan(features[k]) else features[k] for k in FEATURE_NAMES]

    @staticmethod
    def _direction_changes(seq, eps):
        """
        Count reversals that move at least eps from the last extreme (zigzag with
        hysteresis). Counting raw sign flips makes a *motionless* swimmer look
        frantic, because pose jitter flips direction nearly every frame.
        """
        changes, direction, ext = 0, 0, seq[0]
        for v in seq[1:]:
            if direction == 0:
                if abs(v - ext) > eps:
                    direction, ext = (1 if v > ext else -1), v
            elif direction == 1:
                if v > ext:
                    ext = v
                elif ext - v > eps:
                    direction, ext, changes = -1, v, changes + 1
            else:
                if v < ext:
                    ext = v
                elif v - ext > eps:
                    direction, ext, changes = 1, v, changes + 1
        return changes
