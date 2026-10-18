"""
Fine-tunes YOLO on aquatic footage to close the domain gap between
COCO-trained detection (land humans) and partially submerged swimmers.

IMPORTANT: the live pipeline needs a *pose* model (boxes + 17 keypoints).
Roboflow swimmer sets are boxes only, so this script cannot produce a drop-in
replacement for yolo26*-pose.pt. Two ways to use its output:

  A) Two-stage: fine-tuned detector finds swimmers -> stock pose model runs on
     each crop. Better recall, slower (two models).
  B) Pose fine-tune: label keypoints on ~200-500 of your own pool frames
     (CVAT / Roboflow keypoint project, COCO-17 layout) and train
     yolo26s-pose.pt with task='pose'. Single model, the right long-term fix.

Until one of those is wired in, keep config.POSE_MODEL pointed at a pose model.

Dataset: Roboflow Universe swimmer/drowning detection sets, collapsed to a
single 'person' class — the goal is reliable detection, not behavior
classification (the Random Forest handles that).
"""
from ultralytics import YOLO

DATA_YAML = 'data/training/swimmer/data.yaml'
BASE_MODEL = 'yolo26s.pt'
EPOCHS = 60
IMG_SIZE = 640


def main():
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=8,
        patience=15,          # early stop if val stops improving
        project='runs',
        name='aquaguard_detector',
    )
    print("Weights saved to runs/aquaguard_detector/weights/best.pt")
    print("Compare recall vs. the stock model on held-out pool clips before using it.")


if __name__ == "__main__":
    main()
