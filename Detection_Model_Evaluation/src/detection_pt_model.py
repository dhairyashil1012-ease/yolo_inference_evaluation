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

# ==========================================================
# PATHS
# ==========================================================

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




# ==========================================================
# Setup Model
# ==========================================================

def setup_model():

    if MODEL_PATH.exists():
        print(f"Using existing model : {MODEL_PATH}")
        return

    print(f"Model not found. Downloading {MODEL_NAME}...")

    # Download model from Ultralytics
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

    print(f"Running on : {device}")

    model = YOLO(model_path)
    model.to(device)

    return model

model_name=load_model(MODEL_PATH)
print(f"IMAGE_PATH: {IMAGE_DIR}")

# Preprocess

def preprocess(folder_path):

    image_paths = sorted([
        folder_path / file
        for file in os.listdir(folder_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    ])

    if len(image_paths) == 0:
        raise ValueError("No images found.")    

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

    device = 0 if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=str(folder_path),
        imgsz=INPUT_SIZE,
        device=device,
        verbose=False,
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

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Unable to read {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(result.boxes) == 0:

            print(f"\nImage : {os.path.basename(image_path)}")
            print("No Detection")

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




def main():

    setup_model()

    model = load_model(MODEL_PATH)

    image_paths = preprocess(IMAGE_DIR)

    results = inference(model, IMAGE_DIR)

    postprocess(results, image_paths)


if __name__ == "__main__":
    main()


