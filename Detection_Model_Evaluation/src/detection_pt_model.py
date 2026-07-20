import os
import time
import sys
import cv2
import torch
import platform
from pathlib import Path
import psutil
from configparser import ConfigParser
from ultralytics import YOLO
import ultralytics

# --- IMPORT REPORT GENERATOR ---
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
REPORT_PDF_PATH = REPORT_DIR / "detection_pt_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)



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


def get_model_details(model_path, yolo_model_obj):

    model_path = Path(model_path)
    
    try:
        params = f"{sum(p.numel() for p in yolo_model_obj.model.parameters()):,}"
    except Exception:
        params = "N/A"


    # 1. Model Architecture & Task
    task = getattr(yolo_model_obj, "task", "detect")
    architecture = f"YOLO ({task.capitalize()})"

    # 2. Total Parameters Calculation from PyTorch State Dict
    total_params = sum(p.numel() for p in yolo_model_obj.model.parameters())

    # 3. Precision Mode check based on model weights
    # Checks the first parameter's float layout structure
    first_param = next(yolo_model_obj.model.parameters(), None)
    if first_param is not None:
        dtype_str = str(first_param.dtype)
        if "float16" in dtype_str or "half" in dtype_str:
            precision = "FP16"
        elif "int8" in dtype_str:
            precision = "INT8"
        elif "float32" in dtype_str:
            precision = "FP32"
        else:
            precision = dtype_str.upper().replace("TORCH.", "")
    else:
        precision = "Unknown"

    # 4. File Size Calculation
    file_size_mb = f"{model_path.stat().st_size / (1024 * 1024):.2f} MB"

    # 5. Class Parsing out of Ultralytics Engine
    class_names_dict = getattr(yolo_model_obj, "names", {})
    class_names = list(class_names_dict.values()) if class_names_dict else []
    num_classes = len(class_names) if class_names else "Unknown"

    return {
        "Model Architecture": architecture,
        "Total Parameters": f"{int(total_params):,}" if total_params > 0 else "Unknown",
        "File Size": file_size_mb,
        "Number of Classes": num_classes,
        "Class Names": ", ".join(class_names) if class_names else "Unknown",
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
    print(f"Running on : {device}")
    model = YOLO(model_path)
    model.to(device)
    return model


def preprocess(folder_path):
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
    return image_paths



def inference(model, folder_path):
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

    # --- HIGH CONFIDENCE FILTER APPLIED HERE ---
    results = model.predict(
        source=str(folder_path),
        imgsz=INPUT_SIZE,
        device=device,
        verbose=False,
        conf=0.60,  # Filters out any detections below 60% confidence globally
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    total_time_ms = (end - start) * 1000
    batch_size = len(results)
    inference_time_from_results = sum(r.speed.get('inference', 0.0) for r in results)
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
        "Inference Time (time.perf_counter)": f"{total_time_ms:.2f} ms",
        "Inference Time (YOLO Results)": f"{inference_time_from_results:.2f} ms",
        # "Preprocess Latency": f"{avg_preprocess_ms:.2f} ms per image",
        # "Inference Latency": f"{avg_inference_ms:.2f} ms per image",
        # "Postprocess Latency": f"{avg_postprocess_ms:.2f} ms per image",
        "Throughput": f"{batch_size/(end-start):.2f} Images/sec",
        "Peak Memory Usage": memory_usage
    }

    print("\n" + "=" * 60)
    print("PYTORCH DETECTION PERFORMANCE")
    print("=" * 60)
    for k, v in perf_metrics.items():
        print(f"{k:<30}: {v}")

    return results, perf_metrics

def postprocess(results, image_paths):
    print("\n" + "=" * 60)
    print("POSTPROCESS")
    print("=" * 60)

    prediction_metadata = []

    for result, image_path in zip(results, image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Unable to read {image_path}")
            continue

        file_basename = os.path.basename(image_path)
        image_record = {
            "file_name": file_basename,
            "detections": []
        }

        if len(result.boxes) == 0:
            print(f"Image : {file_basename} -> No Detections")
            prediction_metadata.append(image_record)
            continue

        print(f"Image : {file_basename}")
        for box_tensor in result.boxes:
            class_id = int(box_tensor.cls.item())
            class_name = result.names[class_id]
            confidence = float(box_tensor.conf.item()) * 100
            x1, y1, x2, y2 = map(int, box_tensor.xyxy[0].tolist())
            
            image_record["detections"].append({
                "class_name": class_name,
                "confidence": confidence,
                "box": [x1, y1, x2, y2]
            })

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"{class_name} {confidence:.1f}%"
            cv2.putText(
                image, text, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        filename = Path(result.path).name
        save_path = OUTPUT_DIR / filename
        cv2.imwrite(str(save_path), image)
        prediction_metadata.append(image_record)

    return prediction_metadata


def main():
    setup_model()
    model = load_model(MODEL_PATH)
    image_paths = preprocess(IMAGE_DIR)

    # Inference & Postprocess pipelines
    results, perf_metrics = inference(model, IMAGE_DIR)
    prediction_metadata = postprocess(results, image_paths)


    sys_info = get_system_env_info()
    model_details = get_model_details(MODEL_PATH, model)

    # Merge into the final corporate reporting configuration structure
    config_dict = {
        "Model Name": str(MODEL_NAME),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **sys_info,
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