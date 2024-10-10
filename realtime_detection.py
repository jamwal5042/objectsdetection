import torch
import cv2
from ultralytics import YOLO

detector = YOLO('yolov8x.pt')
video_stream = cv2.VideoCapture(0)

while True:
    is_frame_captured, frame = video_stream.read()
    if not is_frame_captured:
        print("Error: Frame capture failed")
        break

    detection_results = detector(frame)

    for detection in detection_results:
        classes = detection.boxes.cls
        conf_scores = detection.boxes.conf
        bounding_rects = detection.boxes.xyxy

        for idx, (coords, score, obj_class) in enumerate(zip(bounding_rects, conf_scores, classes)):
            x_min, y_min, x_max, y_max = map(int, coords)
            label_text = f'{detector.names[int(obj_class)]} {score:.2f}'

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('YOLO Detection Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()