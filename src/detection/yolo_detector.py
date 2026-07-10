from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self):
        # Use nano model for fast detection
        self.model = YOLO('yolov8s.pt')

    def detect(self, frame):
        """
        Takes a video frame, returns bounding boxes for people only
        Boxes are in the following format: [x1, y1, x2, y2, confidence]
        """
        results = self.model(frame, classes=[0])
        boxes = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                if conf > 0.3:
                    boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
        return boxes

# Test
if __name__ == "__main__":
    detector = PersonDetector()
    cap = cv2.VideoCapture(0)   # webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)

        # Create green boxes around detected people
        for box in boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
