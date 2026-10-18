"""
Builds the training set from labeled video segments.

labels.csv columns: video_file, start_frame, end_frame, label
Output: data/extracted_features/features.csv (one row per window)
"""
import sys
import cv2
import pandas as pd

sys.path.append('.')
from src import config
from src.features.feature_engineer import FeatureEngineer, FEATURE_NAMES


def windows_from_frames(frames, label, video_file, fe,
                        window=config.WINDOW, stride=config.STRIDE,
                        min_coverage=config.MIN_COVERAGE):
    """
    frames: list of (frame_idx, keypoints-or-None), one entry per video frame.

    Slides a fixed-length window over real frame time (not over detections),
    keeping windows where at least min_coverage of frames had a detection.
    This mirrors the live pipeline, which also predicts from a partially
    filled buffer when detection drops out on a chaotic frame.
    """
    rows = []
    for s in range(0, len(frames) - window + 1, stride):
        chunk = frames[s:s + window]
        kps = [kp for _, kp in chunk if kp is not None]
        if len(kps) < min_coverage * window:
            continue
        feats = fe.extract_features(kps)
        feats.update({
            'label': label,
            'video': video_file,
            'start_frame': chunk[0][0],
            'coverage': round(len(kps) / window, 3),
        })
        rows.append(feats)
    return rows


def read_segment(video_path, start_frame, end_frame, pose):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append((idx, pose.extract_keypoints(frame)))
    cap.release()
    return frames


if __name__ == "__main__":
    from src.detection.yolo_pose_estimator import PoseEstimator   # heavy import, only needed here

    labels = pd.read_csv('data/extracted_features/labels.csv')
    pose = PoseEstimator(config.POSE_MODEL, config.CONF_THRESHOLD)
    fe = FeatureEngineer(fps=config.FPS)

    all_rows = []
    for _, r in labels.iterrows():
        path = f"data/raw_videos/{r['video_file']}"
        frames = read_segment(path, int(r['start_frame']), int(r['end_frame']), pose)
        hit = sum(kp is not None for _, kp in frames)
        rows = windows_from_frames(frames, r['label'], r['video_file'], fe)
        print(f"{r['video_file']} [{r['start_frame']}-{r['end_frame']}] {r['label']:9s} "
              f"detected {hit}/{len(frames)} frames -> {len(rows)} windows")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=FEATURE_NAMES + ['label', 'video', 'start_frame', 'coverage'])
    df.to_csv('data/extracted_features/features.csv', index=False)
    print(f"\nTotal: {len(df)} windows from {df['video'].nunique()} videos")
    print(df.groupby('label').agg(windows=('video', 'size'), videos=('video', 'nunique')))
