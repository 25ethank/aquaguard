import cv2
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.yolo_detector import PersonDetector
from src.detection.pose_estimator import PoseEstimator

detector = PersonDetector()
pose_estimator = PoseEstimator()

cap = cv2.VideoCapture('data/raw_videos/film9.mp4')           # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect all people
    boxes = detector.detect(frame)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box

        # estimate pose for each person
        keypoints = pose_estimator.extract_keypoints(frame, bbox=(x1, y1, x2, y2))

        if keypoints is not None:
            # draw skeleton
            frame = pose_estimator.draw_skeleton(frame, keypoints, bbox=(x1, y1, x2, y2))
            print(f"Person {i}: {keypoints.shape} keypoints extracted")


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Aquaguard Pipeline Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
