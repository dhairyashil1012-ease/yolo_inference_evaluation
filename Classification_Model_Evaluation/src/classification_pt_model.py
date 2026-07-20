import os
import time
import shutil
import cv2
import torch
import platform
import sys
import ultralytics
import psutil
import contextlib
from ultralytics import YOLO
from pathlib import Path
from configparser import ConfigParser
from src.report_generator import generate_pdf_report 
import os
import time
import psutil
import torch
import numpy as np

PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")

# ==========================================================
# PATHS
# ==========================================================

MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["PT_OUTPUT_DIR"]
REPORT_DIR=PROJECT_DIR/config['PATHS']["REPORT_OUTPUT_DIR"]
MODEL_NAME = config["PATHS"]["PT_MODEL_NAME"]


MODEL_PATH = MODEL_DIR / MODEL_NAME
REPORT_PDF_PATH = REPORT_DIR / "classification_pt_inference_report.pdf"


INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class Profile(contextlib.ContextDecorator):
 
    def __init__(self, t=0.0, sync_cuda=False):
        self.t = t
        self.sync_cuda = sync_cuda
        self.dt = 0.0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt

    def time(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()




def get_model_details(model, model_path):
    try:
        params = f"{sum(p.numel() for p in model.model.parameters()):,}"
    except Exception:
        params = "N/A"

    file_size_mb = f"{model_path.stat().st_size / (1024 * 1024):.2f} MB" if model_path.exists() else "Unknown"
    class_names = list(model.names.values())
    num_classes = len(model.names)
    precision = "FP16" if next(model.model.parameters()).dtype == torch.float16 else "FP32"

    return {
        "Total Parameters": params,
        "File Size": file_size_mb,
        "Number of Classes": num_classes,
        "Class Names": ", ".join(class_names),
        "Precision Mode": precision
    }




def setup_model():
    if MODEL_PATH.exists():
        print(f"Using existing model : {MODEL_PATH}")
        return

    print(f"Model not found. Downloading {MODEL_NAME}...")
    YOLO(MODEL_NAME)
    downloaded_model = PROJECT_DIR / MODEL_NAME

    if not downloaded_model.exists():
        raise FileNotFoundError(f"Failed to download {MODEL_NAME}")

    downloaded_model.replace(MODEL_PATH)
    print(f"Model saved to : {MODEL_PATH}")


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on : {device}")
    model = YOLO(model_path)
    model.to(device)
    return model


def preprocess(folder_path, sync_cuda=False):
    prof = Profile(sync_cuda=sync_cuda)
    
    with prof:
        image_paths = sorted([
            folder_path / file
            for file in os.listdir(folder_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])

        if len(image_paths) == 0:
            raise ValueError("No images found.")    

    print("\n" + "=" * 60)
    print("PREPROCESS")
    print("=" * 60)
    print(f"Images Found : {len(image_paths)}")
    
    preprocess_time_ms = prof.t * 1000
    return image_paths, preprocess_time_ms



def get_system_env_info():
    device_model = "CPU"
    cuda_version = "N/A"
    
    if torch.cuda.is_available():
        device_model = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        
    # Standard fallback
    os_name = f"{platform.system()} {platform.release()}"
    
    # Accurate Linux distribution detection using built-in tools
    if platform.system() == "Linux":
        try:
            # Works natively in Python 3.10+
            os_info = platform.freedesktop_os_release()
            os_name = os_info.get("PRETTY_NAME", os_name)
        except (AttributeError, OSError):
            # Safe manual fallback for older Python versions (< 3.10)
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME="):
                            # Extract the string inside the quotes
                            os_name = line.split("=")[1].strip().strip('"')
                            break
    
    return {
        "OS": os_name,
        "Python Version": sys.version.split()[0],
        "PyTorch Version": torch.__version__,
        "Ultralytics Version": ultralytics.__version__,
        "Inference Device": device_model,
        "CUDA Version": cuda_version
    }



# def preprocess(folder_path, sync_cuda=False):
#     prof = Profile(sync_cuda=sync_cuda)
    
#     with prof:
#         image_paths = sorted([
#             folder_path / file
#             for file in os.listdir(folder_path)
#             if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
#         ])

#         if len(image_paths) == 0:
#             raise ValueError("No images found.")    

#     print("\n" + "=" * 60)
#     print("PREPROCESS")
#     print("=" * 60)
#     print(f"Images Found : {len(image_paths)}")
    
#     preprocess_time_ms = prof.t * 1000
#     return image_paths, preprocess_time_ms


def preprocess_onnx(folder_path, sync_cuda=False):
    prof = Profile(sync_cuda=sync_cuda)
    
    with prof:
        image_files = sorted([
            file for file in os.listdir(folder_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
        
        if len(image_files) == 0:
            raise ValueError("No images found.")

        image_list = []

        print("=" * 60)
        print("PREPROCESS")
        print("=" * 60)

        for file in image_files:
            image_path = os.path.join(folder_path, file)
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (INPUT_SIZE[1], INPUT_SIZE[0]))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            image_list.append(img)

        batch_numpy = np.stack(image_list, axis=0)

    print("Number of Images :", len(image_files))
    print("Batch Shape      :", batch_numpy.shape)
    print("Input dtype      :", batch_numpy.dtype)
    print()
    
    preprocess_time_ms = prof.t * 1000
    return batch_numpy, image_files, preprocess_time_ms


def inference(model, folder_path,sync_cuda=False):
    prof = Profile(sync_cuda=sync_cuda)
    # 1. Reset GPU tracking and clear cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 2. Track baseline CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem_start = process.memory_info().rss

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 3. Define execution device
    device = 0 if torch.cuda.is_available() else "cpu"

    # 4. Execute inference and time it
    # start = time.perf_counter()
    with prof:

        results = model.predict(
            source=str(folder_path),
            imgsz=INPUT_SIZE,
            device=device,
            verbose=False
        )

    inference_time_ms = prof.t * 1000
    batch_size = len(results)

    if torch.cuda.is_available():
        peak_gpu_bytes = torch.cuda.max_memory_allocated(0)
        memory_usage = f"{peak_gpu_bytes / (1024 * 1024):.2f} MB"
    else:
        cpu_mem_end = process.memory_info().rss
        peak_cpu_diff = max(0, cpu_mem_end - cpu_mem_start)
        memory_usage = f"Peak Delta CPU: {peak_cpu_diff / (1024 * 1024):.2f} MB"

    throughput = batch_size / prof.t if prof.t > 0 else 0.0

    perf_metrics = {
        "Total Images Processed": batch_size,
        "Total Inference Time": f"{inference_time_ms:.2f} ms",
        "Throughput": f"{throughput:.2f} Images/sec",
        "Peak Memory Usage": memory_usage
    }

    return results, perf_metrics, inference_time_ms



def postprocess(results, image_paths,sync_cuda=False):

    prof = Profile(sync_cuda=sync_cuda)
    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    prediction_metadata = []
    with prof:
        for image_path, result in zip(image_paths, results):
            probs = result.probs
            if probs is None:
                continue
                
            class_id = probs.top1
            confidence = probs.top1conf.item()
            class_name = result.names[class_id]
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Unable to read {image_path}")
                continue
            
            confidence = confidence * 100
            file_name = Path(image_path).name
            text = f"{class_name}: {confidence:.1f}%"
            print(f"Image: {file_name} -> {text}")

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x, text_y = 20, 40
            cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

            save_path = OUTPUT_DIR / file_name
            cv2.imwrite(str(save_path), image)
            
            prediction_metadata.append({
                "file_name": file_name,
                "class_name": class_name,
                "confidence": confidence
            })

    postprocess_time_ms = prof.t * 1000
    return prediction_metadata, postprocess_time_ms


def main():
    setup_model()
    model = load_model(MODEL_PATH)
    # image_paths, preprocess_ms= preprocess(IMAGE_DIR,sync_cuda=sync_cuda)
    sync_cuda = torch.cuda.is_available()
    env_info = get_system_env_info()
    model_details = get_model_details(model, MODEL_PATH)
    image_paths, preprocess_ms = preprocess(IMAGE_DIR, sync_cuda=sync_cuda)
    results, perf_metrics, inference_ms= inference(model, IMAGE_DIR,sync_cuda=sync_cuda)
    prediction_metadata, postprocess_ms = postprocess(results, image_paths,sync_cuda=sync_cuda)
    
    batch_size = len(image_paths)
    avg_preprocess = preprocess_ms / batch_size
    avg_inference = inference_ms / batch_size
    avg_postprocess = postprocess_ms / batch_size
    
    perf_metrics.update({
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Inference Latency": f"{avg_inference:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
    })
    config_dict = {
        "Model Name": str(MODEL_NAME),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **env_info,       
        **model_details   
    }
    

    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="pt",
        config_data=config_dict,
        performance_data=perf_metrics,
        predictions=prediction_metadata
    )

if __name__ == "__main__":
    main()

