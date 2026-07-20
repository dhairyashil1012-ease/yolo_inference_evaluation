import os
import time
import sys
import platform
import cv2
import torch
import psutil
import ultralytics
import numpy as np
from pathlib import Path
from configparser import ConfigParser
from ultralytics import YOLO

from src.report_generation import generate_pdf_report

PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")

MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["PT_OUTPUT_DIR"]
REPORT_DIR = PROJECT_DIR/config['PATHS']['REPORT_OUTPUT_DIR']
MODEL_NAME = config["PATHS"]["PT_MODEL_NAME"]

MODEL_PATH = MODEL_DIR / MODEL_NAME
REPORT_PDF_PATH = REPORT_DIR / "semantic_segmentation_pt_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
print(f"Output Dir : {OUTPUT_DIR}")
print(f"Input Size : {INPUT_SIZE}")
print("=" * 60)



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

def get_model_details(model, model_path):
    model_path = Path(model_path)
    file_size_mb = f"{model_path.stat().st_size / (1024 * 1024):.2f} MB" if model_path.exists() else "Unknown"
    
    total_params = sum(p.numel() for p in model.model.parameters()) if hasattr(model, 'model') else 0
    
 
    precision = "FP16" if next(model.model.parameters()).dtype == torch.float16 else "FP32"

    class_dict = model.names if hasattr(model, 'names') else {}
    num_classes = len(class_dict)
    class_names_str = ", ".join(list(class_dict.values()))
    
    return {
        "Model Architecture": "YOLO-semantic",
        "Total Parameters": f"{total_params:,}" if total_params > 0 else "Unknown",
        "File Size": file_size_mb,
        "Number Of Classes": num_classes,
        "Class Names": class_names_str,
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_model.replace(MODEL_PATH)
    print(f"Model saved to : {MODEL_PATH}") 

def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)
    print(f"Running on device : {device}")
    return model


def preprocess(folder_path):
    image_paths = sorted([
        folder_path / file
        for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    print("\n" + "=" * 60)
    print("PREPROCESS")
    print("=" * 60)
    print(f"Images Found : {len(image_paths)}")
    return image_paths


def inference(model, folder_path):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_host_memory_stats()
        

    process = psutil.Process(os.getpid())
    cpu_mem_start = process.memory_info().rss

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    device = 0 if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=str(folder_path),
        imgsz=INPUT_SIZE,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    total_time_ms = (end - start)*1000
    batch_size = len(results)
    inference_time_from_results = sum(r.speed.get('inference', 0.0) for r in results)
    avg_preprocess_ms = sum(r.speed.get('preprocess',0.0) for r in results)/batch_size if batch_size > 0 else 0
    avg_inference_ms = sum(r.speed.get('inference',0.0) for r in results) / batch_size if batch_size > 0 else 0
    avg_postprocess_ms = sum(r.speed.get('postprocess', 0.0) for r in results) / batch_size if batch_size > 0 else 0

    if torch.cuda.is_available():
        peak_gpu_bytes = torch.cuda.max_memory_allocated(0)
        memory_usage = f"{peak_gpu_bytes / (1024 * 1024):.2f} MB"

    else:
        cpu_mem_end = process.memory_info().rss
        peak_cpu_diff = max(0, cpu_mem_end - cpu_mem_start)
        memory_usage = f"Peak Delta CPU: {peak_cpu_diff / (1024 * 1024):.2f} MB"

    perf_metrics = {
        "Total Images Processed": batch_size,
        "Inference Time (time.perf_counter)": f"{total_time_ms:.2f} ms",
        "Inference Time (YOLO Results)": f"{inference_time_from_results:.2f} ms",
        # "Preprocess Latency": f"{avg_preprocess_ms:.2f} ms per image",
        # "Inference Latency": f"{avg_inference_ms:.2f} ms per image",
        # "Postprocess Latency": f"{avg_postprocess_ms:.2f} ms per image",
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

    print("\nAll output segmentation masks saved successfully.")


def main():
    setup_model()
    model = load_model(MODEL_PATH)
    preprocess(IMAGE_DIR)
    
    # 1. Execute Segmentation Inference
    results, perf_metrics = inference(model, IMAGE_DIR)
    
    # 2. Render pixel arrays and overlays to disk storage
    postprocess(results)   

    # 3. Parse and accumulate dense pixel maps out of the SemanticMask structure
    predictions_accumulator = []
    for result in results:
        filename = Path(result.path).name
        image_record = {
            "file_name": filename,
            "segmentations": []
        }
        
        # Verify the semantic mask array exists
        if not hasattr(result, "semantic_mask") or result.semantic_mask is None:
            predictions_accumulator.append(image_record)
            continue
            
        # Extract the dense integer class map [Shape: (H, W)]
        dense_class_map = result.semantic_mask.data.cpu().numpy()
        
        # Calculate overall pixel area for relative proportion metrics
        total_pixels = dense_class_map.size
        
        # Identify unique class IDs present in this specific frame and count their occurrences
        unique_classes, pixel_counts = np.unique(dense_class_map, return_counts=True)
        
        for cls_id, count in zip(unique_classes, pixel_counts):
            # Skip background mask references if designated as ignore boundary indexes (e.g., 255)
            if cls_id == 255:
                continue
                
            class_name = result.names[int(cls_id)]
            coverage_pct = (count / total_pixels) * 100
            
            image_record["segmentations"].append({
                "class_name": class_name,
                "pixel_area": int(count),
                "coverage": f"{coverage_pct:.2f}%"
            })
            
        predictions_accumulator.append(image_record)

    # 4. Gather System and Model Specifications
    sys_env = get_system_env_info()
    model_spec = get_model_details(model, MODEL_PATH)

    # 5. Build Unified Configuration Metadata Dictionary
    config_data = {

        "Model Name":str(MODEL_NAME),
        "Input Dimensions": f"{INPUT_SIZE[1]}x{INPUT_SIZE[0]} (WxH)",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **sys_env,
        **model_spec
    }


    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="pt",
        config_data=config_data,
        performance_data=perf_metrics,
        predictions=predictions_accumulator
    )

if __name__ == "__main__":
    main()