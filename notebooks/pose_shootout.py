import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo26m-pose.pt')

VIDEO = 'data/raw_videos/flim20.mp4'
START_FRAME = 0

# COCO-17 skeleton connections
EDGES = [(5,7),(7,9),(6,8),(8,10),(5,6),(11,12),(5,11),(6,12),
         (11,13),(13,15),(12,14),(14,16),(0,3),(0,4),(3,5),(4,6)]

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
detected, total = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    total += 1
    r = model(frame, classes=[0], verbose=False)[0]

    n = 0 if r.keypoints is None else len(r.boxes)
    if n > 0:
        detected += 1
        h, w = frame.shape[:2]
        for i in range(n):
            kpts = r.keypoints.xyn[i].cpu().numpy()
            conf = r.keypoints.conf[i].cpu().numpy()
            for a, b in EDGES:
                if conf[a] > 0.5 and conf[b] > 0.5:
                    pa = (int(kpts[a,0]*w), int(kpts[a,1]*h))
                    pb = (int(kpts[b,0]*w), int(kpts[b,1]*h))
                    cv2.line(frame, pa, pb, (0,255,255), 2)
            x1,y1,x2,y2 = [int(v) for v in r.boxes.xyxy[i].tolist()]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.putText(frame, f"det rate: {detected}/{total}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow('YOLO26-pose shootout', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
print(f"Detection rate: {detected}/{total} = {100*detected/max(total,1):.1f}%")
