import os
import time
import shutil
import cv2
import torch
import platform
import sys
import ultralytics
import psutil
from ultralytics import YOLO
from pathlib import Path
from configparser import ConfigParser
from src.report_generator import generate_pdf_report 
import os
import time
import psutil
import torch

PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")


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



def get_system_env_info():
    device_model = "CPU"
    cuda_version = "N/A"
    
    if torch.cuda.is_available():
        device_model = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        
    os_name = f"{platform.system()} {platform.release()}"
    
    if platform.system() == "Linux":
        try:
            os_info = platform.freedesktop_os_release()
            os_name = os_info.get("PRETTY_NAME", os_name)
        except (AttributeError, OSError):
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



def preprocess(folder_path):

    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    print("\n" + "=" * 60)
    print("PREPROCESS")
    print("=" * 60)
    print(f"Processing source directory: {folder.resolve()}")
    
    return folder




def inference(model, folder_path):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    process = psutil.Process(os.getpid())
    cpu_mem_start = process.memory_info().rss

    device = 0 if torch.cuda.is_available() else "cpu"

    start_wall = time.perf_counter()
    results = model.predict(
        source=str(folder_path),
        imgsz=INPUT_SIZE,
        device=device,
        verbose=False,
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_wall = time.perf_counter()

    batch_size = len(results)
    total_wall_ms = (end_wall - start_wall) * 1000
    total_wall_sec = end_wall - start_wall
    

    avg_preprocess_ms = sum(r.speed.get('preprocess', 0.0) for r in results) / batch_size
    avg_inference_ms = sum(r.speed.get('inference', 0.0) for r in results) / batch_size
    avg_postprocess_ms = sum(r.speed.get('postprocess', 0.0) for r in results) / batch_size
    avg_total_time_ms = avg_preprocess_ms + avg_postprocess_ms + avg_inference_ms

    throughput = batch_size / ((avg_total_time_ms*batch_size)/1000) 
    
    if torch.cuda.is_available():
        peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
        gpu_memory_str = f"{peak_gpu_reserved_mb:.2f} MB"
    else:
        gpu_memory_str = "N/A (Running on CPU)"

    perf_metrics = {
        "Total Images Processed": batch_size,
        "Total Run Time": f"{avg_total_time_ms*batch_size:.2f} ms",
        "Preprocess Latency": f"{avg_preprocess_ms:.2f} ms",
        "Inference Latency": f"{avg_inference_ms:.2f} ms",
        "Postprocess Latency": f"{avg_postprocess_ms:.2f} ms",
        "Peak Memory Usage": gpu_memory_str,
        "Throughput": f"{throughput:.2f} Images/sec"
    }

    return results, perf_metrics


def postprocess(results):
    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    prediction_metadata = []

    for result in results:
        probs = result.probs
        if probs is None:
            continue
            
        image_path = Path(result.path)
        
        class_id = probs.top1
        confidence = probs.top1conf.item() * 100
        class_name = result.names[class_id]
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Unable to read {image_path}")
            continue
        
        file_name = image_path.name
        text = f"{class_name}: {confidence:.1f}%"
        print(f"Image: {file_name} -> {text}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x, text_y = 20, 40
        
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        save_path = OUTPUT_DIR / file_name
        cv2.imwrite(str(save_path), image)
        
        prediction_metadata.append({
            "file_name": file_name,
            "class_name": class_name,
            "confidence": confidence
        })

    return prediction_metadata


def main():
    setup_model()
    model = load_model(MODEL_PATH)
    image_paths = preprocess(IMAGE_DIR)
    
    env_info = get_system_env_info()
    model_details = get_model_details(model, MODEL_PATH)
    
    results, perf_metrics = inference(model, IMAGE_DIR)
    predictions = postprocess(results)
    
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
        predictions=predictions
    )

if __name__ == "__main__":
    main()
