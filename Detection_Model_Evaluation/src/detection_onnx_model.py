import os
import time
import sys
import platform
import cv2
import torch
import numpy as np
import onnx
import ultralytics
import onnxruntime as ort
import pynvml
import threading
import ast
from pathlib import Path
from configparser import ConfigParser

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
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ONNX_OUTPUT_DIR"]
YAML_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]
LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]
REPORT_DIR = PROJECT_DIR / config['PATHS']["REPORT_OUTPUT_DIR"]

MODEL_NAME = config["PATHS"]["PT_MODEL_NAME"]
LABEL_NAME = config["PATHS"]["LABEL_NAME"]

MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = LABEL_DIR / LABEL_NAME

REPORT_PDF_PATH = REPORT_DIR / "detection_onnx_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
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


def get_model_details(onnx_model_path, label_path):
    onnx_model_path = Path(onnx_model_path)
    
    if not onnx_model_path.exists():
        return {
            "Model Architecture": "ONNX Runtime Engine",
            "Total Parameters": "Unknown",
            "File Size": "Unknown",
            "Number of Classes": "Unknown",
            "Class Names": "Unknown",
            "Precision Mode": "Unknown"
        }

    model_graph = onnx.load(str(onnx_model_path))
    session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
    
    metadata = session.get_modelmeta().custom_metadata_map

    task = metadata.get("task")
    architecture = f"YOLO ({task.capitalize()})" if task else (model_graph.graph.name or "ONNX Model")
    total_params = sum(np.prod(tensor.dims) for tensor in model_graph.graph.initializer)

    dtypes = {onnx.TensorProto.DataType.Name(t.data_type) for t in model_graph.graph.initializer}
    precision = "FP16" if "FLOAT16" in dtypes else "FP32" if "FLOAT" in dtypes else "INT8" if "INT8" in dtypes else "Unknown"
    file_size_mb = f"{onnx_model_path.stat().st_size / (1024 * 1024):.2f} MB"

    class_names = []
    num_classes = 'Unknown'

    if "names" in metadata:
        try:
            names = ast.literal_eval(metadata['names'])
            class_names = list(names.values()) if isinstance(names, dict) else names
            num_classes = len(class_names)
        except Exception:
            pass

    return {
        "Model Architecture": architecture,
        "Total Parameters": f"{int(total_params):,}" if total_params > 0 else "Unknown",
        "File Size": file_size_mb,
        "Number of Classes": num_classes,
        "Class Names": ", ".join(class_names) if class_names else "Unknown",
        "Precision Mode": precision
    }


def load_model(model_path):
    from ultralytics import YOLO
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Base YOLO initialization on: {device}")
    model = YOLO(model_path)
    return model


def export_onnx_model(model):   
    model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        batch=config.getint("MODEL", "ONNX_EXPORT_BATCH"),
        dynamic=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("\nONNX export completed successfully.")
    time.sleep(1)


def load_onnx_model(model_dir):
    pathlist = Path(model_dir).glob("**/*.onnx")
    filelist = sorted([str(file) for file in pathlist])

    if len(filelist) == 0:
        raise FileNotFoundError("No ONNX model found.")

    onnx_model = filelist[-1]
    print(f"ONNX Model Found: {onnx_model}")
    return onnx_model


def preprocess_onnx(folder_path):
    # Performance optimized NumPy & OpenCV pipeline 
    start_pre = time.perf_counter()

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
    
    end_pre = time.perf_counter()
    preprocess_time_ms = (end_pre - start_pre) * 1000
    return batch_numpy, image_files, preprocess_time_ms


def inference_onnx(onnx_model, batch_numpy):
    ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.required_cuda_compute_capability", "0") 
    
    session = ort.InferenceSession(
        onnx_model, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
        sess_options=opts
    )
    
    print("Execution Provider :", session.get_providers()[0])
    input_name = session.get_inputs()[0].name


    start = time.perf_counter()
    outputs = session.run(None, {input_name: batch_numpy})
    end = time.perf_counter()

    inference_time_ms = (end - start) * 1000
    return outputs[0], inference_time_ms


def postprocess_onnx(outputs, image_files, image_folder, label_path, confidence_threshold=0.6, input_size=640):
    start_post = time.perf_counter()

    print("=" * 60)
    print("POSTPROCESS")
    print("=" * 60)

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    input_h, input_w = input_size

    with open(label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    prediction_metadata = []

    for img_idx, image_name in enumerate(image_files):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        H, W = image.shape[:2]
        scale_x = W / input_w
        scale_y = H / input_h

        detections = outputs[img_idx]
        image_record = {"file_name": image_name, "detections": []}
        
        # Dictionary filtering logic to keep only the highest confidence match per class
        best_detections = {}

        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, score, cls = det[:6]
            else:
                continue

            if score < confidence_threshold:
                continue

            cls = int(cls)
            class_name = class_names[cls] if cls < len(class_names) else f"ID_{cls}"
            confidence_pct = float(score) * 100 if score <= 1.0 else float(score)

            if class_name not in best_detections or confidence_pct > best_detections[class_name]["confidence"]:
                best_detections[class_name] = {
                    "class_name": class_name,
                    "confidence": confidence_pct,
                    "raw_score": score,
                    "box": [x1, y1, x2, y2]
                }

        has_detections = len(best_detections) > 0

        for class_name, item in best_detections.items():
            x1, y1, x2, y2 = item["box"]
            
            x1_scaled = max(0, min(int(x1 * scale_x), W - 1))
            y1_scaled = max(0, min(int(y1 * scale_y), H - 1))
            x2_scaled = max(0, min(int(x2 * scale_x), W - 1))
            y2_scaled = max(0, min(int(y2 * scale_y), H - 1))

            image_record["detections"].append({
                "class_name": item["class_name"],
                "confidence": item["confidence"],
                "box": [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
            })

            cv2.rectangle(image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
            label_text = f"{class_name} {item['raw_score']:.2f}"
            cv2.putText(image, label_text, (x1_scaled, max(20, y1_scaled - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        save_path = OUTPUT_DIR / image_name
        cv2.imwrite(str(save_path), image)
        
        if has_detections:
            print(f"Processed & Saved Detections for: {save_path}")
        else:
            print(f"Processed: {image_name} (No Objects Found above threshold)")
            
        prediction_metadata.append(image_record)

    end_post = time.perf_counter()
    postprocess_time_ms = (end_post - start_post) * 1000
    return prediction_metadata, postprocess_time_ms


def main():
    try:
        onnx_model_path = load_onnx_model(MODEL_DIR)
    except FileNotFoundError:
        model = load_model(MODEL_PATH)
        export_onnx_model(model)
        onnx_model_path = load_onnx_model(MODEL_DIR)
    
    has_nvml = False
    peak_vram_bytes = 0
    stop_tracking = False
    my_pid = os.getpid()

    try:
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        has_nvml = True
    except Exception:
        print("Warning: NVML initialization failed. GPU memory tracking disabled.")

    # --- FIX 3: COOPERATIVE MULTI-THREADING TIMING STEP ---
    def track_peak_memory():
        nonlocal peak_vram_bytes
        while not stop_tracking:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(nvml_handle)
                for p in procs:
                    if p.pid == my_pid:
                        if p.usedGpuMemory > peak_vram_bytes:
                            peak_vram_bytes = p.usedGpuMemory
            except Exception:
                pass
            time.sleep(0.02)  # Relaxed to 20ms to protect CPU from GIL starvation

    if has_nvml:
        mem_thread = threading.Thread(target=track_peak_memory, daemon=True)
        mem_thread.start()

    batch_numpy, image_files, preprocess_ms = preprocess_onnx(folder_path=IMAGE_DIR)

    predictions, inference_ms = inference_onnx(
        onnx_model=onnx_model_path,
        batch_numpy=batch_numpy,
    )

    prediction_metadata, postprocess_ms = postprocess_onnx(
        outputs=predictions,
        image_files=image_files,
        image_folder=IMAGE_DIR,
        label_path=LABEL_PATH,
        confidence_threshold=0.6,
        input_size=INPUT_SIZE,
    )
    
    stop_tracking = True
    if has_nvml:
        mem_thread.join()

    peak_mem_mb = peak_vram_bytes / (1024 * 1024)
    if has_nvml:
        pynvml.nvmlShutdown()
        
    sys_env = get_system_env_info()
    model_spec = get_model_details(onnx_model_path, LABEL_PATH)

    batch_size = len(image_files)
    avg_preprocess = preprocess_ms / batch_size
    avg_inference = inference_ms / batch_size
    avg_postprocess = postprocess_ms / batch_size
    total_pipeline_ms = preprocess_ms + inference_ms + postprocess_ms

    print("=" * 60)
    print("ONNX RUNTIME PERFORMANCE")
    print("=" * 60)
    print(f"Total Images Processed : {batch_size}")
    print(f"Total Pipeline Time    : {total_pipeline_ms:.2f} ms")
    print(f"Preprocess Latency     : {avg_preprocess:.2f} ms per image")
    print(f"Inference Latency      : {avg_inference:.2f} ms per image")
    print(f"Postprocess Latency    : {avg_postprocess:.2f} ms per image")
    
    print(f"\nSummary -> preprocess: {avg_preprocess:.1f}ms, inference: {avg_inference:.1f}ms, postprocess: {avg_postprocess:.1f}ms per image\n")

    # --- FIX 4: ROBUST MODEL NAME EXTRACTION ---
    onnx_model_name = Path(onnx_model_path).name

    config_dict = {
        "Model Name": onnx_model_name,
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **sys_env,
        **model_spec
    }

    perf_metrics = {
        "Total Images Processed": batch_size,
        "Total Pipeline Time": f"{total_pipeline_ms:.2f} ms",
        "Total Inference Time in ms": f"{inference_ms:.2f}ms",
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Inference Latency": f"{avg_inference:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
        "Throughput": f"{batch_size / (inference_ms / 1000):.2f} Images/sec",
        "Peak Memory Usage": f"{peak_mem_mb:.2f} MB" if peak_mem_mb > 0 else "N/A"
    }

    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="onnx",
        config_data=config_dict,
        performance_data=perf_metrics,
        predictions=prediction_metadata
    )


if __name__ == "__main__":
    main()