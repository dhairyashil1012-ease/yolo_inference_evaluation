# import os
# import time
# import shutil
# from pathlib import Path

# import cv2
# import torch
# import matplotlib.pyplot as plt
# from ultralytics import YOLO


# # ==========================================================
# # Configuration
# # ==========================================================

# MODEL_NAME = "yolo26n-sem.pt"
# INPUT_SIZE = (1024, 2048)

# PROJECT_DIR = Path.cwd()

# MODEL_DIR = PROJECT_DIR / "Sem_Models"
# IMAGE_DIR = PROJECT_DIR / "Images"

# MODEL_DIR.mkdir(exist_ok=True)

# MODEL_PATH = MODEL_DIR / MODEL_NAME


# # ==========================================================
# # Utility Functions
# # ==========================================================

# def setup_model():
#     """
#     Move the model into Sem_Models folder if required.
#     """
#     source = PROJECT_DIR / MODEL_NAME

#     if source.exists() and not MODEL_PATH.exists():
#         shutil.move(str(source), str(MODEL_PATH))

#     print(f"Model Path : {MODEL_PATH}")


# # ==========================================================
# # Load Model
# # ==========================================================

# def load_model(model_path):
#     """
#     Load YOLO model onto GPU.
#     """

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = YOLO(str(model_path))
#     model.to(device)

#     print(f"Running on : {device}")

#     return model


# # ==========================================================
# # Preprocess
# # ==========================================================

# def preprocess(image_folder):

#     image_paths = sorted([
#         image_folder / file
#         for file in os.listdir(image_folder)
#         if file.lower().endswith(
#             (".jpg", ".jpeg", ".png", ".bmp", ".webp")
#         )
#     ])

#     print("=" * 60)
#     print("PREPROCESS")
#     print("=" * 60)

#     print(f"Number of Images : {len(image_paths)}")

#     for img in image_paths:
#         print(img.name)

#     return image_paths


# # ==========================================================
# # Inference
# # ==========================================================

# def inference(model, image_folder):

#     torch.cuda.synchronize()

#     start = time.perf_counter()

#     results = model.predict(
#         source=str(image_folder),
#         imgsz=INPUT_SIZE,
#         device=0,
#         verbose=False
#     )

#     torch.cuda.synchronize()

#     end = time.perf_counter()

#     total_time = (end - start) * 1000

#     batch_size = len(results)

#     print("\n" + "=" * 60)
#     print("PYTORCH PERFORMANCE")
#     print("=" * 60)

#     print(f"Batch Size           : {batch_size}")
#     print(f"Batch Time           : {total_time:.2f} ms")
#     print(f"Latency/Image        : {total_time / batch_size:.2f} ms")
#     print(f"Throughput           : {batch_size / (end-start):.2f} FPS")

#     return results


# # ==========================================================
# # Postprocess
# # ==========================================================

# def postprocess(results):

#     print("=" * 60)
#     print("POSTPROCESS")
#     print("=" * 60)

#     for result in results:

#         output = result.plot()

#         plt.figure(figsize=(14, 6))
#         plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
#         plt.title(Path(result.path).name)
#         plt.axis("off")
#         plt.show()


# # ==========================================================
# # Main
# # ==========================================================

# def main():

#     setup_model()

#     model = load_model(MODEL_PATH)

#     preprocess(IMAGE_DIR)

#     results = inference(model, IMAGE_DIR)

#     postprocess(results)


# # ==========================================================
# # Entry Point
# # ==========================================================

# if __name__ == "__main__":
#     main()



import os
import time
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


MODEL_NAME = "yolo26n-sem.pt"
INPUT_SIZE = (1024, 2048)

PROJECT_DIR = Path.cwd()

MODEL_DIR = PROJECT_DIR / "Sem_Models"
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
        source=str(folder_path),
        imgsz=INPUT_SIZE,
        device=0,
        verbose=False
    )

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

def save_results(results):

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
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

    save_results(results)


if __name__ == "__main__":
    main()