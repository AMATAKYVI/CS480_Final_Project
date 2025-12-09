import os
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import DetrImageProcessor, DetrForObjectDetection

transform = T.Compose([T.Resize(800), T.ToTensor()])


# load DETR model
def load_detr():
    """Load actual DETR model from Hugging Face"""
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    return model


# Function to run DETR on a folder of images
def run_detr_folder(model, folder):
    for img in os.listdir(folder):
        if img.endswith((".jpg", ".png")):
            path = os.path.join(folder, img)
            image = Image.open(path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0)
            outputs = model(input_tensor)
            print(outputs)
