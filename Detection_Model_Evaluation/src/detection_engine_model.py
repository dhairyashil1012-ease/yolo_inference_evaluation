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


MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]
REPORT_DIR = PROJECT_DIR / config['PATHS']["REPORT_OUTPUT_DIR"]
MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]
LABEL_NAME = config["PATHS"]["LABEL_NAME"]

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


def get_model_details(engine_model_path, label_path):
    engine_model_path = Path(engine_model_path)
    if not engine_model_path.exists():
        return {"Total Parameters": "Unknown", "File Size": "Unknown", "Number of Classes": "Unknown", "Class Names": "Unknown", "Precision Mode": "Unknown"}
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    class_names = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

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
                weighted_sum += json.loads(m).get("Count", 0)
        except Exception:
            continue        

    total_params = f"{weighted_sum:,}" if weighted_sum > 0 else "N/A (Optimized Engine)"
    try:
        input_tensor_name = engine.get_tensor_name(0)
        dtype_enum = engine.get_tensor_dtype(input_tensor_name)
        precision = "FP16" if dtype_enum == trt.DataType.HALF else "FP32" if dtype_enum == trt.DataType.FLOAT else "INT8" if dtype_enum == trt.DataType.INT8 else "Unknown"
    except Exception:
        precision = "Unknown"

    return {
        "Total Parameters": total_params,
        "File Size": f"{engine_model_path.stat().st_size / (1024 * 1024):.2f} MB",
        "Number of Classes": len(class_names) if class_names else "Unknown",
        "Class Names": ", ".join(class_names) if class_names else "Unknown",
        "Precision Mode": precision
    }



def load_onnx_model(model_dir):
    filelist = sorted([str(f) for f in Path(model_dir).glob("**/*.onnx")])
    if not filelist:
        raise FileNotFoundError("No ONNX model found to export engine from.")
    return filelist[-1]

def export_engine_model(onnx_model_p):
    current_dir = os.getcwd()
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
    os.chdir(current_dir)
    time.sleep(1)


def load_engine_model(model_dir):
    filelist = sorted([str(f) for f in Path(model_dir).glob("**/*.engine")])
    if not filelist:
        raise FileNotFoundError("No engine model found.")
    return filelist[-1]


def check_cuda(err):
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error : {err}")


def preprocess_engine(image_files, image_folder, input_size):
    print("=" * 60)
    print("PREPROCESS")
    print("=" * 60)

    start_pre = time.perf_counter()

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    batch_images = []

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        batch_images.append(tensor)

    batch_tensor = torch.stack(batch_images, dim=0)
    batch_numpy = batch_tensor.numpy().astype(np.float32)

    print(f"Number of Images : {len(image_files)}")
    print(f"Input Shape      : {batch_numpy.shape}")
    print()

    end_pre = time.perf_counter()
    preprocess_time_ms = (end_pre - start_pre) * 1000

    return batch_numpy, preprocess_time_ms


def inference_engine(batch_numpy, engine_model, warmup_iters=10):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    my_pid = os.getpid()

    with open(engine_model, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

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

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    has_nvml = False
    peak_vram_bytes = 0
    stop_tracking = False

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

    if has_nvml:
        mem_thread = threading.Thread(target=track_peak_memory, daemon=True)
        mem_thread.start()

    check_cuda(cudart.cudaMemcpyAsync(d_input, host_input.ctypes.data, host_input.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    
    start_cold = time.perf_counter()
    context.execute_async_v3(stream)
    check_cuda(cudart.cudaStreamSynchronize(stream))
    inference_cold_ms = (time.perf_counter() - start_cold) * 1000

    for _ in range(warmup_iters):
        context.execute_async_v3(stream)
    check_cuda(cudart.cudaStreamSynchronize(stream))


    check_cuda(cudart.cudaMemcpyAsync(d_input, host_input.ctypes.data, host_input.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    
    start_warm = time.perf_counter()
    context.execute_async_v3(stream)
    check_cuda(cudart.cudaMemcpyAsync(host_output.ctypes.data, d_output, host_output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    check_cuda(cudart.cudaStreamSynchronize(stream))
    inference_warm_ms = (time.perf_counter() - start_warm) * 1000

    print(f"Cold Inference (Without Warmup) : {inference_cold_ms:.2f} ms")
    print(f"Warm Inference (With Warmup)    : {inference_warm_ms:.2f} ms")

    stop_tracking = True
    if has_nvml:
        mem_thread.join()
        peak_gpu_usage_mb = peak_vram_bytes / (1024 * 1024)
        pynvml.nvmlShutdown()
    else:
        peak_gpu_usage_mb = 0.0

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    return host_output, inference_cold_ms, inference_warm_ms, peak_gpu_usage_mb



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
        engine_model_path = Path(load_engine_model(MODEL_DIR))

    except FileNotFoundError:
        onnx_model_path = load_onnx_model(MODEL_DIR)
        export_engine_model(onnx_model_path)
        engine_model_path = Path(load_engine_model(MODEL_DIR))

    image_files = sorted([
        file for file in os.listdir(IMAGE_DIR)
        if file.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))
    ])

    if not image_files:
        raise ValueError(f"No valid images found inside: {IMAGE_DIR}")
    
    batch_numpy, preprocess_ms = preprocess_engine(image_files, IMAGE_DIR, INPUT_SIZE)

    predictions,inf_cold_ms, inf_warm_ms, peak_gpu_mb = inference_engine(batch_numpy,engine_model_path,warmup_iters=1)

    prediction_metadata,postprocess_ms = postprocess_engine(
        outputs=predictions,
        image_files=image_files,
        image_folder=IMAGE_DIR,
        label_path=LABEL_PATH, 
        confidence_threshold=0.6,
        input_size=INPUT_SIZE,
    )


    sys_env = get_system_env_info()
    model_details = get_model_details(engine_model_path, LABEL_PATH)
    config_metadata = {
        "Model Name": engine_model_path.name,
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **sys_env,
        **model_details
    }


    batch_size = len(image_files)
    total_run_time_cold_ms = preprocess_ms + inf_cold_ms + postprocess_ms
    total_run_time_warm_ms = preprocess_ms + inf_warm_ms + postprocess_ms

    performance_metadata = {
        "Total Images Processed": str(batch_size),
        "Preprocess Latency": f"{preprocess_ms / batch_size:.2f} ms per image",
        "Postprocess Latency": f"{postprocess_ms / batch_size:.2f} ms per image",
        "Peak Memory Usage": f"{peak_gpu_mb:.2f} MB",
        "Inference Latency (Without Warmup)": f"{inf_cold_ms / batch_size:.2f} ms per image",
        "Inference Latency (With Warmup)": f"{inf_warm_ms / batch_size:.2f} ms per image",
        "Throughput (Without Warmup)": f"{(batch_size / (total_run_time_cold_ms / 1000.0)):.2f} Images/sec",
        "Throughput (With Warmup)": f"{(batch_size / (total_run_time_warm_ms / 1000.0)):.2f} Images/sec",
        "Total Run Time (Without Warmup)": f"{total_run_time_cold_ms:.2f} ms",
        "Total Run Time (With Warmup)": f"{total_run_time_warm_ms:.2f} ms"
    }

    print("=" * 60)
    print("PERFORMANCE SUMMARY FOR REPORT")
    print("=" * 60)
    for key, val in performance_metadata.items():
        print(f"{key:<35}: {val}")

    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="trt",
        config_data=config_metadata,
        performance_data=performance_metadata,
        predictions=prediction_metadata
    )


if __name__ == "__main__":
    main()