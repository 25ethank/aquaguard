import cv2
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.detection.yolo_detector import PersonDetector
from src.detection.pose_estimator import PoseEstimator
from src.features.feature_engineer import FeatureEngineer

WINDOW = 90
STRIDE = 15

def largest_box(boxes):
    """
    Pick most prominent person (largest area) - assumes subject is main figure
    """
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

def process_segment(video_path, start_frame, end_frame, label, detector, pose, fe):
    """
    Processes one labeled video segment into training samples.

    Runs detection + pose estimation frame by frame, then slides a 90-frame windo (stride 15)
    across buffered keypoints to extract overlapping feature vectors

    Args:
        video_path: path to video file
        start_frame, end_frame: labeled segment range (inclusive)
        label: ground-truth class ('normal', 'distress', 'drowning')
        detector, pose, fe: pipeline instances (reused across calls)

    Returns:
        List of feature dicts, each tagged with 'label' and 'source'
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    buffer, rows = [], []
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)
        if boxes:
            box = largest_box(boxes)
            kp = pose.extract_keypoints(frame, bbox=box[:4])
            if kp is not None:
                buffer.append(kp)

        if len(buffer) >= WINDOW:
            feats = fe.extract_features(buffer[:WINDOW])
            feats['label'] = label
            feats['source'] = f"{video_path}:{frame_idx}"
            buffer = buffer[STRIDE:]

        frame_idx += 1

    cap.release()
    return rows

if __name__ == "__main__":
    labels = pd.read_csv('data/extracted_features/labels.csv')
    detector, pose, fe = PersonDetector(), PoseEstimator(), FeatureEngineer()

    all_rows = []
    for _, r in labels.iterrows():
        path = f"data/raw_videos/{r['video_file']}"
        print(f"Processing {r['video_file']} [{r['start_frame']} - {r['end_frame']}] as {r['label']}")
        rows = process_segment(path, int(r['start_frame']), int(r['end_frame']), r['label'], detector, pose, fe)
        print(f"{len(rows)} samples")
        all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        df.to_csv('data/extracted_features/features.csv', index=False)
        print(f"\nTotal: {len(df)} samples")
        print(df['label'].value_counts)
