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


def get_system_env_info():
    
    device_model = "CPU"
    cuda_version = "N/A"
    
    if torch.cuda.is_available():
        device_model = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    
    return {
        "OS": f"{platform.system()} {platform.release()}",
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
        "Model Architecture": getattr(model.model, 'yaml', {}).get('type', 'YOLO-cls'),
        "Total Parameters": params,
        "File Size": file_size_mb,
        "Number of Classes": num_classes,
        "Class Names": ", ".join(class_names),
        "Precision Mode": precision
    }


def inference(model, folder_path):
    # Reset tracking for memory before execution starts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Track baseline CPU memory
    process = psutil.Process(os.getpid())
    cpu_mem_start = process.memory_info().rss

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    device = 0 if torch.cuda.is_available() else "cpu"

    # Run predictions across the whole source directory
    results = model.predict(
        source=str(folder_path),
        imgsz=INPUT_SIZE,
        device=device,
        verbose=False,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    total_time_ms = (end - start) * 1000
    batch_size = len(results)
    
    # Extract internal speed breakdown from YOLO results dictionary (averaging per image)
    avg_preprocess_ms = sum(r.speed.get('preprocess', 0.0) for r in results) / batch_size if batch_size > 0 else 0
    avg_inference_ms = sum(r.speed.get('inference', 0.0) for r in results) / batch_size if batch_size > 0 else 0
    avg_postprocess_ms = sum(r.speed.get('postprocess', 0.0) for r in results) / batch_size if batch_size > 0 else 0
    
    # Measure Memory Usage
    if torch.cuda.is_available():
        peak_gpu_bytes = torch.cuda.max_memory_allocated(0)
        memory_usage = f"{peak_gpu_bytes / (1024 * 1024):.2f} MB"
    else:
        cpu_mem_end = process.memory_info().rss
        peak_cpu_diff = max(0, cpu_mem_end - cpu_mem_start)
        memory_usage = f"Peak Delta CPU: {peak_cpu_diff / (1024 * 1024):.2f} MB"

    perf_metrics = {
        "Total Images Processed": batch_size,
        "Total Run Time": f"{total_time_ms:.2f} ms",
        "Preprocess Latency": f"{avg_preprocess_ms:.2f} ms per image",
        "Inference Latency": f"{avg_inference_ms:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess_ms:.2f} ms per image",
        "Throughput": f"{batch_size/(end-start):.2f} Images/sec",
        "Peak Memory Usage": memory_usage
    }

    print("\n" + "=" * 60)
    print("INFERENCE PERFORMANCE")
    print("=" * 60)
    for k, v in perf_metrics.items():
        print(f"{k:<25}: {v}")
    
    # Print the requested summary metric log structure[cite: 1]
    print(f"\nSummary -> preprocess: {avg_preprocess_ms:.1f}ms, inference: {avg_inference_ms:.1f}ms, postprocess: {avg_postprocess_ms:.1f}ms per image\n")

    return results, perf_metrics


def postprocess(results, image_paths):
    print("=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    prediction_metadata = []

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

    return prediction_metadata


def main():
    setup_model()
    model = load_model(MODEL_PATH)
    image_paths = preprocess(IMAGE_DIR)
    
    env_info = get_system_env_info()
    model_details = get_model_details(model, MODEL_PATH)
    
    results, perf_metrics = inference(model, IMAGE_DIR)
    predictions = postprocess(results, image_paths)
    
    config_dict = {
        "Model Path": str(MODEL_PATH),
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

