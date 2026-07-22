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
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import torchvision.transforms as transforms
from configparser import ConfigParser
from src.report_generator import generate_pdf_report

PROJECT_DIR = Path.cwd()
config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")


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

INPUT_SIZE = (config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.to(device)
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
    filelist = sorted([str(f) for f in Path(model_dir).glob("**/*.onnx")])
    if not filelist:
        raise FileNotFoundError("No ONNX model found.")
    return filelist[-1]


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
    return {
        "OS": os_name,
        "Python Version": sys.version.split()[0],
        "PyTorch Version": torch.__version__,
        "Ultralytics Version": ultralytics.__version__,
        "Inference Device": device_model,
        "CUDA Version": cuda_version
    }


def get_model_details(onnx_model_path):
    onnx_model_path = Path(onnx_model_path)
    if not onnx_model_path.exists():
        return {"Total Parameters": "Unknown", "File Size": "Unknown", "Number of Classes": "Unknown", "Class Names": "Unknown", "Precision Mode": "Unknown"}

    model = onnx.load(str(onnx_model_path))
    session = ort.InferenceSession(str(onnx_model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map

    total_params = sum(np.prod(tensor.dims) for tensor in model.graph.initializer)
    dtypes = {onnx.TensorProto.DataType.Name(t.data_type) for t in model.graph.initializer}
    precision = "FP16" if "FLOAT16" in dtypes else "FP32" if "FLOAT" in dtypes else "INT8" if "INT8" in dtypes else "Unknown"

    class_names = []
    num_classes = "Unknown"
    if "names" in metadata:
        try:
            names = ast.literal_eval(metadata["names"])
            if isinstance(names, dict):
                class_names = list(names.values())
            num_classes = len(class_names)
        except Exception:
            pass

    return {
        "Total Parameters": f"{int(total_params):,}" if total_params > 0 else "Unknown",
        "File Size": f"{onnx_model_path.stat().st_size / (1024 * 1024):.2f} MB",
        "Number of Classes": num_classes,
        "Class Names": ", ".join(class_names) if class_names else "Unknown",
        "Precision Mode": precision
    }



def preprocess_onnx(image_files, image_folder, input_size):

    start_pre = time.perf_counter()
    transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])

    batch_images = []
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        batch_images.append(transform(image))

    batch_numpy = torch.stack(batch_images, dim=0).numpy().astype(np.float32)
    return batch_numpy, (time.perf_counter() - start_pre) * 1000


def inference_onnx(onnx_model, batch_numpy, warmup_runs=10):

    ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.required_cuda_compute_capability", "0") 
    
    session = ort.InferenceSession(
        str(onnx_model), 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
        sess_options=opts
    )

    input_name = session.get_inputs()[0].name
    input_feed = {input_name: batch_numpy}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_cold = time.perf_counter()
    outputs = session.run(None, input_feed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_cold_ms = (time.perf_counter() - start_cold) * 1000

    if warmup_runs > 0:
        for _ in range(warmup_runs):
            _ = session.run(None, input_feed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    start_warm = time.perf_counter()
    outputs = session.run(None, input_feed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_warm_ms = (time.perf_counter() - start_warm) * 1000
    
    return outputs[0], inference_cold_ms, inference_warm_ms



def postprocess_onnx(outputs, image_files, image_folder, label_path, confidence_threshold=0.6, input_size=640):
    start_post = time.perf_counter()

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
        onnx_model_path = Path(load_onnx_model(MODEL_DIR))
    except FileNotFoundError:
        model = load_model(MODEL_PATH)
        export_onnx_model(model)
        onnx_model_path = Path(load_onnx_model(MODEL_DIR))
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))
    ])

    if not image_files:
        raise ValueError(f"No valid source images found inside:{IMAGE_DIR}")
    
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

    batch_numpy, preprocess_ms = preprocess_onnx(image_files,IMAGE_DIR, INPUT_SIZE)

    predictions, inf_cold_ms, inf_warm_ms = inference_onnx(onnx_model=onnx_model_path,
                                                           batch_numpy=batch_numpy,
                                                           warmup_runs=1)

    prediction_metadata, postprocess_ms = postprocess_onnx(outputs=predictions,
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
    model_spec = get_model_details(onnx_model_path)

    batch_size = len(image_files)
    avg_preprocess = preprocess_ms / batch_size
    avg_postprocess = postprocess_ms / batch_size
    total_pipeline_cold = preprocess_ms + inf_cold_ms + postprocess_ms
    total_pipeline_warm = preprocess_ms + inf_warm_ms + postprocess_ms

    print("=" * 60)
    print("ONNX RUNTIME PROFILE RUN SUMMARY")
    print("=" * 60)
    print(f"Total Images Processed : {batch_size}")
    print(f"Preprocess Latency     : {avg_preprocess:.2f} ms per image")
    print(f"Inference (Cold Run)   : {inf_cold_ms / batch_size:.2f} ms per image")
    print(f"Inference (Warm Run)   : {inf_warm_ms / batch_size:.2f} ms per image")
    print(f"Postprocess Latency    : {avg_postprocess:.2f} ms per image")
    print(f"Peak VRAM Usage        : {peak_mem_mb:.2f} MB")
    

    config_dict = {
        "Model Name": str(onnx_model_path.name),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **sys_env,
        **model_spec
    }

    performance_metadata = {
        "Total Images Processed": str(batch_size),
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
        "Peak Memory Usage": f"{peak_mem_mb:.2f} MB",
        "Inference Latency (Without Warmup)": f"{inf_cold_ms / batch_size:.2f} ms per image",
        "Inference Latency (With Warmup)": f"{inf_warm_ms / batch_size:.2f} ms per image",
        "Throughput (Without Warmup)": f"{(batch_size / (total_pipeline_cold / 1000.0)):.2f} Images/sec",
        "Throughput (With Warmup)": f"{(batch_size / (total_pipeline_warm / 1000.0)):.2f} Images/sec",
        "Total Run Time (Without Warmup)": f"{total_pipeline_cold:.2f} ms",
        "Total Run Time (With Warmup)": f"{total_pipeline_warm:.2f} ms"
    }

    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="onnx",
        config_data=config_dict,
        performance_data=performance_metadata,
        predictions=prediction_metadata
    )


if __name__ == "__main__":
    main()