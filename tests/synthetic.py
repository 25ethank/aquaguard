"""Synthetic COCO-17 keypoint sequences for tests (no video or GPU needed)."""
import numpy as np

FPS = 30


def swimmer(behavior, n_frames, seed=0, fps=FPS):
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    kp = np.zeros((n_frames, 17, 4))
    kp[:, :, 3] = 0.9                                   # keypoint confidence
    cx, cy = 0.5 + rng.uniform(-0.1, 0.1), 0.5 + rng.uniform(-0.1, 0.1)

    if behavior == 'normal':                          # horizontal, ~1 Hz stroke
        sh = np.stack([np.full(n_frames, cx - 0.08), np.full(n_frames, cy)], 1)
        hp = np.stack([np.full(n_frames, cx + 0.08), np.full(n_frames, cy + 0.01)], 1)
        wy = cy + 0.05 * np.sin(2 * np.pi * 1.0 * t)
        wr_y = (wy, cy - 0.05 * np.sin(2 * np.pi * 1.0 * t))
        wr_x = (cx - 0.15, cx - 0.12)
    elif behavior == 'distress':                      # upright, fast arm beats above shoulders
        sh = np.stack([np.full(n_frames, cx), np.full(n_frames, cy - 0.08)], 1)
        bob = 0.02 * np.sin(2 * np.pi * 1.5 * t)
        hp = np.stack([np.full(n_frames, cx), cy + 0.08 + bob], 1)
        wy = cy - 0.12 + 0.04 * np.sin(2 * np.pi * 3.0 * t)
        wr_y = (wy, wy)
        wr_x = (cx - 0.12, cx + 0.12)
    elif behavior == 'drowning':                      # upright, still, slowly sinking
        sink = 0.01 * t
        sh = np.stack([np.full(n_frames, cx), cy - 0.08 + sink], 1)
        hp = np.stack([np.full(n_frames, cx), cy + 0.08 + sink], 1)
        wr_y = (cy + 0.02 + sink, cy + 0.02 + sink)
        wr_x = (cx - 0.05, cx + 0.05)
        kp[:, 15:17, 3] = 0.1                         # ankles submerged
    else:
        raise ValueError(behavior)

    kp[:, 5, :2] = sh + [-0.02, 0]
    kp[:, 6, :2] = sh + [0.02, 0]
    kp[:, 11, :2] = hp + [-0.02, 0]
    kp[:, 12, :2] = hp + [0.02, 0]
    kp[:, 0, 0], kp[:, 0, 1] = sh[:, 0], sh[:, 1] - 0.04
    kp[:, 9, 0], kp[:, 9, 1] = wr_x[0], wr_y[0]
    kp[:, 10, 0], kp[:, 10, 1] = wr_x[1], wr_y[1]
    kp[:, 15, :2] = hp + [-0.02, 0.15]
    kp[:, 16, :2] = hp + [0.02, 0.15]
    kp[:, :, :2] += rng.normal(0, 0.002, (n_frames, 17, 2))
    return kp
