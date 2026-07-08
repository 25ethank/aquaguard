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
