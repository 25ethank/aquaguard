"""
Fine-tunes YOLO on aquatic footage to close the domain gap between
COCO-trained detection (land humans) and partially submerged swimmers.

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


if __name__ == "__main__":
    main()
