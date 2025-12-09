import torch
import time
from PIL import Image
import torchvision.transforms as T


# Run inference
def run_inference(model, image_path, is_yolo=False):
    image = Image.open(image_path).convert("RGB")

    if is_yolo:
        results = model(image)
        return results

    # DETR preprocessing
    transform = T.Compose([T.Resize(800), T.ToTensor()])
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
    return outputs
