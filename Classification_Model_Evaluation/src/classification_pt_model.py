import os
import time
import cv2
import torch
import platform
import sys
import ultralytics
import contextlib
import numpy as np
import threading
import pynvml
from configparser import ConfigParser
from ultralytics import YOLO
from pathlib import Path
from src.report_generator import generate_pdf_report 
import os
import sys
import time
import cv2
import platform
import subprocess
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
from configparser import ConfigParser
import re
import json
import threading
import pynvml
import psutil
import tensorrt as trt
from cuda.bindings import runtime as cudart
from src.report_generator import generate_pdf_report

PROJECT_DIR = Path.cwd()
config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")

MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["PT_OUTPUT_DIR"]
REPORT_DIR = PROJECT_DIR / config['PATHS']["REPORT_OUTPUT_DIR"]
MODEL_NAME = config["PATHS"]["PT_MODEL_NAME"]

MODEL_PATH = MODEL_DIR / MODEL_NAME
REPORT_PDF_PATH = REPORT_DIR / "classification_pt_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

for d in [MODEL_DIR, IMAGE_DIR, OUTPUT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class Profile(contextlib.ContextDecorator):
    def __init__(self, t=0.0, sync_cuda=False):
        self.t = t
        self.sync_cuda = sync_cuda

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.start
        self.t += self.dt

def setup_model():
    if MODEL_PATH.exists():
        return
    YOLO(MODEL_NAME)
    downloaded_model = PROJECT_DIR / MODEL_NAME
    if downloaded_model.exists():
        downloaded_model.replace(MODEL_PATH)

def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    return model



def preprocess(folder_path,sync_cuda=False):
    prof = Profile(sync_cuda=sync_cuda)
    with prof:
        transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
        ])
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
        if not image_files:
            raise ValueError("No images found.")
            
        image_list, original_images = [], []
        for file in image_files:
            img = cv2.imread(str(folder_path / file))
            if img is None:
                continue
            original_images.append(img.copy())
            image_path = os.path.join(folder_path, file)
            image = Image.open(image_path).convert("RGB")
            image_list.append(transform(image))

        batch_numpy = torch.stack(image_list).numpy().astype(np.float32)
    return batch_numpy, image_files, original_images, prof.t * 1000




def get_system_env_info():
    device_model = "CPU"
    cuda_version = "N/A"
    if torch.cuda.is_available():
        device_model = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    os_name = f"{platform.system()} {platform.release()}"
    if platform.system() == "Linux":
        try:
            os_name = platform.freedesktop_os_release().get("PRETTY_NAME", os_name)
        except Exception:
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release") as f:
                    for line in f:
                        if line.startswith("PRETTY_NAME="):
                            os_name = line.split("=")[1].strip().strip('"')
                            break
    try:
        _, device = cudart.cudaGetDevice()
        _, prop = cudart.cudaGetDeviceProperties(device)
        device_model = prop.name.decode("utf-8")
        err, cuda_version_int = cudart.cudaRuntimeGetVersion()
        cuda_version = f"{cuda_version_int // 1000}.{(cuda_version_int % 1000) // 10}" if err == cudart.cudaError_t.cudaSuccess else "Unknown"
    except Exception:
        pass
    return {
        "OS": os_name,
        "Python Version": sys.version.split()[0],
        "TensorRT Version": trt.__version__,
        "Inference Device": device_model,
        "CUDA Version": cuda_version
    }


def get_model_details(model, model_path):
    try:
        params = f"{sum(p.numel() for p in model.model.parameters()):,}"
    except Exception:
        params = "N/A"
    precision = "FP16" if next(model.model.parameters()).dtype == torch.float16 else "FP32"
    return {
        "Total Parameters": params,
        "File Size": f"{model_path.stat().st_size / (1024 * 1024):.2f} MB" if model_path.exists() else "Unknown",
        "Number of Classes": len(model.names),
        "Class Names": ", ".join(list(model.names.values())),
        "Precision Mode": precision
    }




def inference(model, batch_numpy, sync_cuda=False):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = 0 if torch.cuda.is_available() else "cpu"

    prof = Profile(sync_cuda=sync_cuda)
    with prof:
            # Measures Host-to-Device cast + Execution to achieve parity with TRT evaluation
            tensor_input = torch.from_numpy(batch_numpy).to(device)
            results = model.predict(source=tensor_input, imgsz=INPUT_SIZE, verbose=False)    


    # 5. Calculate Timings
    total_time_ms =prof.t * 1000
    

    if torch.cuda.is_available():
        peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
        gpu_memory_str = f"{peak_gpu_reserved_mb:.2f} MB"
    else:
        gpu_memory_str = "N/A (Running on CPU)"

    return results,total_time_ms, gpu_memory_str


def postprocess(results, image_files, original_images, sync_cuda=False):
    prof = Profile(sync_cuda=sync_cuda)
    prediction_metadata = []
    with prof:
        for file_name, result, image in zip(image_files, results, original_images):
            probs = result.probs
            if probs is None:
                continue
            class_id = probs.top1
            confidence = probs.top1conf.item() * 100
            class_name = result.names[class_id]
            
            cv2.putText(image, f"{class_name}: {confidence:.1f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(str(OUTPUT_DIR / file_name), image)
            
            prediction_metadata.append({"file_name": file_name, "class_name": class_name, "confidence": confidence})
    return prediction_metadata, prof.t * 1000



def main():
    setup_model()
    model = load_model(MODEL_PATH)
    sync_cuda = torch.cuda.is_available()

    batch_numpy, image_files, original_images, preprocess_ms = preprocess(IMAGE_DIR, sync_cuda=sync_cuda)
    results, inference_ms,gpu_memory_str = inference(model, batch_numpy, sync_cuda=sync_cuda)
    prediction_metadata, postprocess_ms = postprocess(results, image_files, original_images, sync_cuda=sync_cuda)



    batch_size = len(image_files)
    total_pipeline_ms = preprocess_ms + inference_ms + postprocess_ms

    print("=" * 60)
    print("PYTORCH PERFORMANCE")
    print("=" * 60)
    print(f"Total Images Processed : {batch_size}")
    print(f"Total Pipeline Time    : {total_pipeline_ms:.2f} ms")
    print(f"Preprocess Latency     : {preprocess_ms / batch_size:.2f} ms/img")
    print(f"Inference Latency      : {inference_ms / batch_size:.2f} ms/img")
    print(f"Postprocess Latency    : {postprocess_ms / batch_size:.2f} ms/img")
    print(f"Peak VRAM Usage        : {gpu_memory_str} MB")

    performance_metadata = {
        "Total Images Processed": str(batch_size),
        "Total Run Time": f"{total_pipeline_ms:.2f} ms",
        "Preprocess Latency": f"{preprocess_ms / batch_size:.2f} ms per image",
        "Inference Latency": f"{inference_ms / batch_size:.2f} ms per image",
        "Postprocess Latency": f"{postprocess_ms / batch_size:.2f} ms per image",
        "Throughput": f"{(batch_size / (total_pipeline_ms / 1000.0)):.2f} Images/sec",
        "Peak Memory Usage": f"{gpu_memory_str} "
    }
    
    config_dict = {
        "Model Name": str(MODEL_NAME),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **get_system_env_info(),       
        **get_model_details(model, MODEL_PATH)   
    }
    
    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="pt",
        config_data=config_dict,
        performance_data=performance_metadata,
        predictions=prediction_metadata
    )

if __name__ == "__main__":
    main()




