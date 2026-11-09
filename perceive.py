"""Minimal perception module: camera -> YOLO -> live JSON scene description.

Run:   python perceive.py [camera_index]
Output: scene.json is rewritten every frame (atomic), and the same JSON is
        printed to stdout. An agent reads scene.json (or calls get_scene()).
"""
import json
import os
import sys
import time

import cv2
from ultralytics import YOLO

MODEL = "yolo26n.pt"
OUT = "scene.json"
MIN_CONF = 0.4  # ignore weak detections


def describe(result, frame_w, frame_h):
    """Turn one YOLO result into a small, text-friendly dict."""
    objects = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2 / frame_w, (y1 + y2) / 2 / frame_h  # 0..1
        objects.append({
            "id": int(box.id) if box.id is not None else None,  # persistent track ID
            "label": result.names[int(box.cls)],
            "conf": round(float(box.conf), 2),
            "center": [round(cx, 2), round(cy, 2)],  # normalized, (0,0)=top-left
            "size": [round((x2 - x1) / frame_w, 2), round((y2 - y1) / frame_h, 2)],
        })
    return {"time": round(time.time(), 2), "objects": objects}


def get_scene(path=OUT):
    """What an agent calls to read the latest scene."""
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    source = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Error opening camera {source}"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        result = model.track(frame, conf=MIN_CONF, persist=True, verbose=False)[0]
        scene = describe(result, w, h)

        # write to temp file then rename, so readers never see a half-written file
        with open(OUT + ".tmp", "w") as f:
            json.dump(scene, f)
        os.replace(OUT + ".tmp", OUT)
        print(json.dumps(scene), flush=True)
