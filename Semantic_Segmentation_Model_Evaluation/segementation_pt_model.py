import os
import time
import cv2
import torch
from pathlib import Path
from configparser import ConfigParser
from ultralytics import YOLO



PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")

MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["PT_OUTPUT_DIR"]

MODEL_NAME = config["PATHS"]["PT_MODEL_NAME"]
MODEL_PATH = MODEL_DIR / MODEL_NAME

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
print(f"Output Dir : {OUTPUT_DIR}")
print(f"Input Size : {INPUT_SIZE}")
print("=" * 60)




def setup_model():

    if MODEL_PATH.exists():
        print(f"Using existing model : {MODEL_PATH}")
        return

    print(f"Model not found. Downloading {MODEL_NAME}...")

    # Download model
    YOLO(MODEL_NAME)

    downloaded_model = PROJECT_DIR / MODEL_NAME

    if not downloaded_model.exists():
        raise FileNotFoundError(f"Failed to download {MODEL_NAME}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_model.replace(MODEL_PATH)

    print(f"Model saved to : {MODEL_PATH}") 


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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    results = model.predict(
        source=str(folder_path),
        imgsz=INPUT_SIZE,
        device = 0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    if torch.cuda.is_available():
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



# Save Results

def postprocess(results):

    print("=" * 60)
    print("POSTPROCESS")
    print("=" * 60)

    for result in results:

        annotated = result.plot()

        filename = Path(result.path).name

        save_path = OUTPUT_DIR / filename

        cv2.imwrite(str(save_path), annotated)

        print(f"Saved : {save_path}")

    print("\nAll output images saved successfully.")


# Main


def main():

    setup_model()

    model = load_model(MODEL_PATH)

    preprocess(IMAGE_DIR)

    results = inference(model, IMAGE_DIR)

    postprocess(results)   

if __name__ == "__main__":
    main()