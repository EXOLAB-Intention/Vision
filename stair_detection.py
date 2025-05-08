import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os
from ultralytics.utils.plotting import Annotator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(WEIGHTS_PATH)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

while True:

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())

    # results = model(color_image)
    results = model(color_image, verbose=False)

    for result in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = result.tolist()

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        print("-------------------------------")
        print(f"Class: {cls}, Confidence: {conf}")
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

        if int(cls) == 0:  # downstairs
            color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(color_image, "Downstairs", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        elif int(cls) == 1:  # upstairs
            color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(color_image, "Upstairs", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('YOLOv8 Stair Detection', color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()







# while True:

#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#     depth_frame = frames.get_depth_frame()

#     if not color_frame or not depth_frame:
#         continue

#     color_image = np.asanyarray(color_frame.get_data())

#     # results = model(color_image)
#     results = model(color_image, verbose=False)

#     for result in results:

#         annotator = Annotator(color_image)

#         boxes = result.boxes
#         for box in boxes:
            
#             b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
#             c = box.cls
#             annotator.box_label(b, model.names[int(c)])

#     img = annotator.result()
#     cv2.imshow('YOLOv8 Stair Detection', img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# pipeline.stop()
# cv2.destroyAllWindows()