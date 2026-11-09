
from ultralytics import YOLO
model = YOLO("yolo26s.pt")
model.predict(source=1, show=True)  # 0 = default webcam; press q to quit
