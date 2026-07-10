import time
import sys
import cv2
import os
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tensorrt as trt
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms as transforms
from cuda.bindings import runtime as cudart
import shutil
from zipfile import ZipFile





MODEL_NAME = "yolo26n.pt"
INPUT_SIZE = (640, 640)

PROJECT_DIR = Path.cwd()

MODEL_DIR = PROJECT_DIR / "Detection_Models"
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "PT_Output"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / MODEL_NAME


def setup_model():

    source_model = PROJECT_DIR / MODEL_NAME

    if source_model.exists() and not MODEL_PATH.exists():
        shutil.move(str(source_model), str(MODEL_PATH))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"Using Model : {MODEL_PATH}")


# ==========================================================
# Load Model
# ==========================================================

def load_model(model_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(model_path))
    model.to(device)

    print(f"Running on device : {device}")

    return model


# Preprocess

def preprocess(folder_path):

    image_paths = sorted([
        folder_path / file
        for file in os.listdir(folder_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ])

    print("\n" + "=" * 60)
    print("PREPROCESS")
    print("=" * 60)

    print(f"Images Found : {len(image_paths)}")

    for img in image_paths:
        print(img.name)

    return image_paths



# Inference

def inference(model, folder_path):

    torch.cuda.synchronize()

    start = time.perf_counter()

    results = model.predict(
        source=folder_path,
        imgsz=640,
        device=0,
        verbose=False
    )

    torch.cuda.synchronize()

    end = time.perf_counter()

    inference_time = (end - start) * 1000

    batch_size = len(results)

    print("\n" + "=" * 60)
    print("PYTORCH DETECTION PERFORMANCE")
    print("=" * 60)

    print(f"Batch Size           : {batch_size}")
    print(f"Batch Inference Time : {inference_time:.3f} ms")
    print(f"Per Image Latency    : {inference_time/batch_size:.3f} ms")
    print(f"Throughput           : {batch_size/(end-start):.2f} images/sec")

    return results



# Save Results

def postprocess(results, image_paths):

    print("\n" + "=" * 60)
    print("POSTPROCESS")
    print("=" * 60)

    for result, image_path in zip(results, image_paths):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(result.boxes) == 0:

            print(f"\nImage : {os.path.basename(image_path)}")
            print("No Detection")

            plt.figure(figsize=(8,6))
            plt.imshow(image)
            plt.axis("off")
            plt.show()

            continue

        confidences = result.boxes.conf

        top_idx = confidences.argmax()

        box = result.boxes[top_idx]

        class_id = int(box.cls.item())

        class_name = result.names[class_id]

        confidence = float(box.conf.item())

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            image,
            f"{class_name} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        print(f"\nImage      : {os.path.basename(image_path)}")
        print(f"Prediction : {class_name}")
        print(f"Confidence : {confidence:.4f}")

        filename = Path(result.path).name

        save_path = OUTPUT_DIR / filename

        cv2.imwrite(str(save_path), image)

        print(f"Saved : {save_path}")

    print("\nAll output images saved successfully.")


# Main


# def main():

#     setup_model()

#     model = load_model(MODEL_PATH)

#     preprocess(IMAGE_DIR)

#     results = inference(model, IMAGE_DIR)

#     save_results(results)

def main():
        
    setup_model()

    model = load_model(MODEL_PATH)

    image_paths = preprocess(IMAGE_DIR)
    results = inference(model,IMAGE_DIR)
    postprocess(results,image_paths)


if __name__ == "__main__":
    main()