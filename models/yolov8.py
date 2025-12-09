import os
from ultralytics import YOLO

COCO128_PATH = "datasets/coco128/images/train/"
COCO_VAL_PATH = "datasets/coco128/images/val/"


# load YOLOv8 model
def load_yolov8(model_name="yolov8n.pt"):
    """Load YOLOv8 model"""
    model = YOLO(model_name)
    return model


# open YOLOv8 folder
def run_yolo_folder(model, folder):
    for img in os.listdir(folder):
        if img.endswith((".jpg", ".png")):
            path = os.path.join(folder, img)
            model(path, save=True)
