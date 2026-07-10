import os
import time
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path



MODEL_NAME = "yolo26n-cls.pt"
INPUT_SIZE = (224, 224)

PROJECT_DIR = Path.cwd()

MODEL_DIR = PROJECT_DIR / "Classification _Models"
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


# Load Model


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

    results = model.predict(source=str(folder_path),imgsz=INPUT_SIZE,device=0,verbose=False)

    torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = (end - start) * 1000

    batch_size = len(results)

    print("\n" + "=" * 60)
    print("INFERENCE PERFORMANCE")
    print("=" * 60)

    print(f"Batch Size           : {batch_size}")
    print(f"Batch Time           : {total_time:.2f} ms")
    print(f"Latency / Image      : {total_time/batch_size:.2f} ms")
    print(f"Throughput           : {batch_size/(end-start):.2f} Images/sec")

    return results



# PostProcess

def postprocess(results, image_paths):

    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)


    for image_path, result in zip(image_paths, results):
        probs = result.probs
        class_id = probs.top1
        confidence = probs.top1conf.item()
        class_name = result.names[class_id]
        image = cv2.imread(image_path)
        confidence=confidence*100
        
        file_name=Path(image_path).name

        text = f"{class_name}: {confidence:.1f}%"
        print(f"Image: {file_name} -> {text}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x, text_y = 20, 40
  
        cv2.rectangle(image, 
                      (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + baseline), 
                      (0, 0, 0), 
                      cv2.FILLED)

        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        # 6. Save target image directly into your configured OUTPUT_DIR
        save_path = OUTPUT_DIR /file_name
        cv2.imwrite(str(save_path), image)
        print(f"Saved: {save_path}")





# Main

def main():

    setup_model()

    model = load_model(MODEL_PATH)

    image_paths=preprocess(IMAGE_DIR)

    results = inference(model, IMAGE_DIR)

    postprocess(results,image_paths)


if __name__ == "__main__":
    main()