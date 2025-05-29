from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(data="./stair_case_dataset2/stair_case.yaml", epochs=300, batch=32, imgsz=640, lrf=0.001, cache=False)
