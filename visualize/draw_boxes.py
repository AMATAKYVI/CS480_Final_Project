import cv2
import torch
from PIL import Image
import torchvision.transforms as T


# Draw bounding boxes for YOLO output
def draw_yolo(image_path, results, out_path="yolo_output.jpg"):
    img = cv2.imread(image_path)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")


# Draw bounding boxes for DETR output
def draw_detr(image_path, outputs, out_path="detr_output.jpg"):
    img = cv2.imread(image_path)
    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]

    probs = logits.softmax(-1)
    keep = probs.max(-1).values > 0.7

    img_h, img_w = img.shape[:2]

    for box in boxes[keep]:
        cx, cy, w, h = box.tolist()
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
