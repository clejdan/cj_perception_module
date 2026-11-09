import sys

import cv2

from ultralytics import solutions

# Camera index from the command line, e.g. `python distance.py 2` (default 0)
source = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(source)
assert cap.isOpened(), f"Error opening camera {source}"

# Initialize distance calculation object
distancecalculator = solutions.DistanceCalculation(
    model="yolo26n.pt",  # path to the YOLO26 model file.
    show=True,  # display the output window
)

# Process live frames
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Failed to read frame from camera.")
        break

    results = distancecalculator(im0)

    # Press q in the window to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
