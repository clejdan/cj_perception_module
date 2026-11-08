from ultralytics import YOLO

model = YOLO("yolo26s.pt")  # load a pretrained YOLO26n model
results = model("/home/clejah/cj_perception_module/2026-11-07-192204.jpg")  # run inference