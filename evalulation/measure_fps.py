import time
import torch


# Function for measuring FPS
def measure_fps(model, input_tensor, iterations=50, is_yolo=False):
    start = time.time()

    for _ in range(iterations):
        if is_yolo:
            model(input_tensor)
        else:
            with torch.no_grad():
                model(input_tensor)

    end = time.time()
    fps = iterations / (end - start)
    return fps
