import os
import time
import sys
import platform
import cv2
import torch
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
from configparser import ConfigParser
import tensorrt as trt
import torchvision.transforms as transforms
from cuda.bindings import runtime as cudart
import json
import pynvml
import threading
import re
from src.report_generator import generate_pdf_report

PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")

# ==========================================================
# PATHS
# ==========================================================
MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = (PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]).resolve()
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]   
REPORT_DIR=PROJECT_DIR/config['PATHS']["REPORT_OUTPUT_DIR"]
LABEL_NAME = config["PATHS"]["LABEL_NAME"]

MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]

MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = LABEL_DIR / LABEL_NAME
REPORT_PDF_PATH = REPORT_DIR / "detection_engine_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True) 

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
print(f"Label Path : {LABEL_PATH}")
print(f"Output Dir : {OUTPUT_DIR}")
print(f"Input Size : {INPUT_SIZE}")
print("=" * 60)





def get_system_env_info():

    try:
        _, device = cudart.cudaGetDevice()
        _, prop = cudart.cudaGetDeviceProperties(device)
        device_model = prop.name.decode("utf-8")
        
        # Pull driver capabilities for versioning mapping
        err, cuda_version_int = cudart.cudaRuntimeGetVersion()
        cuda_version = f"{cuda_version_int // 1000}.{(cuda_version_int % 1000) // 10}" if err == cudart.cudaError_t.cudaSuccess else "Unknown"
    except Exception:
        device_model = "Unknown NVIDIA GPU"
        cuda_version = "N/A"
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
        "TensorRT Version": trt.__version__,
        "Inference Device": device_model,
        "CUDA Version": cuda_version
    }


def get_model_details(engine_model_path, label_path):

    engine_model_path = Path(engine_model_path)
    label_path = Path(label_path)
    
    if not engine_model_path.exists():
        return {
            "Total Parameters": "Unknown (Compiled)",
            "File Size": "Unknown",
            "Number of Classes": "Unknown",
            "Class Names": "Unknown",
            "Precision Mode": "Unknown"
        }

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_model_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context

    pattern = r'"Weights":\s*(\{[^}]+\})'

    weighted_sum = 0
    for idx in range(engine.num_layers):
        try:
            layer_info = inspector.get_layer_information(idx, trt.LayerInformationFormat.JSON)
            matches = re.findall(pattern, layer_info)

            for m in matches:
                dict_data = json.loads(m)
                weighted_sum += dict_data.get("Count",0)
            
        except Exception:
            continue        

    total_params = weighted_sum if weighted_sum > 0 else "N/A (Optimized Engine)"




    file_size_mb = f"{engine_model_path.stat().st_size / (1024 * 1024):.2f} MB"
    

    precision = "FP32"  
    if engine:
        try:
            tensor_name = engine.get_tensor_name(0)
            dtype = engine.get_tensor_dtype(tensor_name)
            if dtype == trt.DataType.HALF:
                precision = "FP16"
            elif dtype == trt.DataType.FLOAT:
                precision = "FP32"
            elif dtype == trt.DataType.INT8:
                precision = "INT8"
        except Exception:
            pass

    # Class Label Map Parsing
    class_names = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
            
    num_classes = len(class_names) if class_names else "Unknown"
    class_names_str = ", ".join(class_names) if class_names else "Unknown"

    return {
        "Total Parameters": total_params,
        "File Size": file_size_mb,
        "Number of Classes": num_classes,
        "Class Names": class_names_str,
        "Precision Mode": precision
    }



def load_onnx_model(model_dir):
    pathlist = Path(model_dir).glob("**/*.onnx")
    filelist = sorted([str(file) for file in pathlist])

    if len(filelist) == 0:
        raise FileNotFoundError("No ONNX model found.")
    return filelist[-1]


def export_engine_model(onnx_model_p):
    os.chdir(MODEL_DIR)
    subprocess.run([
        "trtexec",
        f"--onnx={onnx_model_p}",
        "--minShapes=images:1x3x640x640",
        "--optShapes=images:4x3x640x640",
        "--maxShapes=images:16x3x640x640",
        "--saveEngine=yolo26n.engine",
        "--profilingVerbosity=detailed"
    ], check=True)

    print("\nTensorRT engine export completed successfully.")
    time.sleep(1)


def load_engine_model(model_dir):
    pathlist = Path(model_dir).glob("**/*.engine")
    filelist = sorted([str(file) for file in pathlist])

    if len(filelist) == 0:
        raise FileNotFoundError("No engine model found.")
    return filelist[-1]


def check_cuda(err):
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error : {err}")


def preprocess_engine(image_folder, input_size):

    start_pre = time.perf_counter()

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    image_files = sorted([
        file for file in os.listdir(image_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in the specified folder: {image_folder}")

    batch_images = []

    print("="*60)
    print("PREPROCESS")
    print("="*60)

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        batch_images.append(tensor)

    batch_tensor = torch.stack(batch_images, dim=0)
    batch_numpy = batch_tensor.numpy().astype(np.float32)

    print(f"Number of Images : {len(image_files)}")
    print(f"Input Shape      : {batch_numpy.shape}")

    end_pre = time.perf_counter()
    preprocess_time_ms = (end_pre - start_pre) * 1000


    return batch_numpy, image_files, preprocess_time_ms


def inference_engine(batch_numpy, engine_model):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # --- TRUE PROCESS PEAK TRACKER INITIALIZATION ---
    has_nvml = False
    peak_vram_bytes = 0
    stop_tracking = False
    my_pid = os.getpid()

    try:
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        has_nvml = True
    except Exception:
        pass

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
            time.sleep(0.005)

    # Start tracking right before TensorRT deserialization pushes weights to VRAM
    if has_nvml:
        mem_thread = threading.Thread(target=track_peak_memory, daemon=True)
        mem_thread.start()

    with open(engine_model, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    context.set_input_shape(input_name, batch_numpy.shape)
    output_shape = tuple(context.get_tensor_shape(output_name))
    host_input = np.ascontiguousarray(batch_numpy)
    host_output = np.empty(output_shape, dtype=np.float32)

    err, d_input = cudart.cudaMalloc(host_input.nbytes)
    check_cuda(err)
    err, d_output = cudart.cudaMalloc(host_output.nbytes)
    check_cuda(err)

    err, stream = cudart.cudaStreamCreate()
    check_cuda(err)

    start = time.perf_counter()

    check_cuda(cudart.cudaMemcpyAsync(
        d_input, host_input.ctypes.data, host_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
    ))

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    success = context.execute_async_v3(stream)
    if not success:
        raise RuntimeError("TensorRT inference failed.")

    check_cuda(cudart.cudaMemcpyAsync(host_output.ctypes.data, d_output, host_output.nbytes,cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    check_cuda(cudart.cudaStreamSynchronize(stream))
    
    end = time.perf_counter()
    
    inference_time = (end - start) * 1000

   # Stop tracking right after synchronization completes
    stop_tracking = True
    if has_nvml:
        mem_thread.join()

    # Calculate final peak metric
    peak_gpu_usage_mb = peak_vram_bytes / (1024 * 1024)
    if has_nvml:
        pynvml.nvmlShutdown()

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    return host_output, inference_time, peak_gpu_usage_mb


def postprocess_engine(outputs, image_files, image_folder, label_path, confidence_threshold=0.6, input_size=640):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    input_h, input_w = input_size

    start_time = time.perf_counter()

    with open(label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    prediction_metadata = []

    for img_idx, image_name in enumerate(image_files):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Unable to read {image_path}")
            continue

        H, W = image.shape[:2]
        scale_x = W / input_w
        scale_y = H / input_h

        detections = outputs[img_idx]
        
        image_record = {
            "file_name": image_name,
            "detections": []
        }

        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, score, cls = det[:6]
            else:
                continue

            if score < confidence_threshold:
                continue

            cls = int(cls)
            class_label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            
            x1_scaled = max(0, min(int(x1 * scale_x), W - 1))
            y1_scaled = max(0, min(int(y1 * scale_y), H - 1))
            x2_scaled = max(0, min(int(x2 * scale_x), W - 1))
            y2_scaled = max(0, min(int(y2 * scale_y), H - 1))

            image_record["detections"].append({
                "class_name": class_label,
                "confidence": float(score) * 100,
                "box": [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
            })

            cv2.rectangle(image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
            label = f"{class_label} {score:.2f}"
            cv2.putText(image, label, (x1_scaled, max(20, y1_scaled - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        save_path = OUTPUT_DIR / image_name
        cv2.imwrite(str(save_path), image)
        prediction_metadata.append(image_record)

    postprocess_time = (time.perf_counter() - start_time) * 1000

    print("\nAll output images processed.")
    return prediction_metadata, postprocess_time


def main():
    try:
        engine_model_path = load_engine_model(MODEL_DIR)
        print(f"Using Existing Engine : {engine_model_path}")
    except FileNotFoundError:
        print("No TensorRT engine found. Exporting from ONNX...")
        onnx_model_path = load_onnx_model(MODEL_DIR)
        export_engine_model(onnx_model_path)
        engine_model_path = load_engine_model(MODEL_DIR)
        print(f"Using newly created engine: {engine_model_path}")

    batch_numpy, image_files, preprocess_ms = preprocess_engine(
        image_folder=IMAGE_DIR,
        input_size=INPUT_SIZE,
    )

    predictions, inference_ms, peak_gpu_mb = inference_engine(
        batch_numpy=batch_numpy,
        engine_model=engine_model_path,
    )

    prediction_metadata,postprocess_ms = postprocess_engine(
        outputs=predictions,
        image_files=image_files,
        image_folder=IMAGE_DIR,
        label_path=LABEL_PATH, 
        confidence_threshold=0.6,
        input_size=INPUT_SIZE,
    )


    sys_env = get_system_env_info()
    model_spec = get_model_details(engine_model_path, LABEL_PATH)
    model_details = get_model_details(engine_model_path, LABEL_PATH)
    engine_model_name =(engine_model_path.split('/')[-1:])
    config_dict = {
        "Model Path": str(engine_model_name[0]),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **sys_env,
        **model_spec
    }

    total_images = len(image_files)
    total_run_time_ms = preprocess_ms + inference_ms + postprocess_ms
    batch_size = len(image_files)
    avg_preprocess = preprocess_ms / batch_size
    avg_inference = inference_ms / batch_size
    avg_postprocess = postprocess_ms / batch_size


    performance_metadata = {
        "Total Images Processed": str(total_images),
        "Total Run Time": f"{total_run_time_ms:.2f} ms",
        "Total Inference Time in ms": f"{inference_ms:.2f}ms",
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Inference Latency": f"{avg_inference:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
        "Throughput": f"{(total_images / (inference_ms / 1000.0)):.2f} Images/sec",
        "Peak Memory Usage": f" {peak_gpu_mb:.2f} MB"
    }

 
    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="trt",
        config_data=config_dict,
        performance_data=performance_metadata,
        predictions=prediction_metadata
    )


if __name__ == "__main__":
    main()