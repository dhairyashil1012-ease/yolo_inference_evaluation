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
import tensorrt as trt
from cuda.bindings import runtime as cudart
from src.report_generator import generate_pdf_report

PROJECT_DIR = Path.cwd()
config = ConfigParser()
config.read("config.txt")

MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]
REPORT_DIR = PROJECT_DIR / config['PATHS']["REPORT_OUTPUT_DIR"]
MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]
LABEL_NAME = config["PATHS"]["LABEL_NAME"]

MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = LABEL_DIR / LABEL_NAME
REPORT_PDF_PATH = REPORT_DIR / "classification_engine_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)


for d in [MODEL_DIR, OUTPUT_DIR, IMAGE_DIR, LABEL_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)



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
        "--minShapes=images:1x3x224x224",
        "--optShapes=images:4x3x224x224",
        "--maxShapes=images:16x3x224x224",
        "--saveEngine=yolo26n-cls.engine",
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
        raise RuntimeError(f"CUDA Error: {err}")



# def preprocess_engine(image_folder, input_size):
#     start_pre = time.perf_counter()
#     transform = transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.ToTensor(),
#     ])
#     image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
#     if not image_files:
#         raise ValueError("No images found in the specified folder.")

#     batch_images = []
#     for image_name in image_files:
#         image_path = os.path.join(image_folder, image_name)
#         image = Image.open(image_path).convert("RGB")
#         batch_images.append(transform(image))

#     batch_numpy = torch.stack(batch_images, dim=0).numpy().astype(np.float32)
#     return batch_numpy, image_files, (time.perf_counter() - start_pre) * 1000

def preprocess_engine(image_folder, input_size):
    start_pre = time.perf_counter()

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    
    # Read image filenames
    image_files = sorted([
        file for file in os.listdir(image_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    if len(image_files) == 0:
        raise ValueError("No images found in the specified folder.")

    batch_images = []

    print("=" * 60)
    print("PREPROCESS")
    print("=" * 60)

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        batch_images.append(tensor)

    # Stack into batch
    batch_tensor = torch.stack(batch_images, dim=0)

    # Convert to NumPy
    batch_numpy = batch_tensor.numpy().astype(np.float32)

    print(f"Number of Images : {len(image_files)}")
    print(f"Input Shape      : {batch_numpy.shape}")
    print()

    end_pre = time.perf_counter()
    preprocess_time_ms = (end_pre - start_pre) * 1000

    return batch_numpy, image_files, preprocess_time_ms



def inference_engine(batch_numpy, engine_model):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

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


    if has_nvml:
        mem_thread = threading.Thread(target=track_peak_memory, daemon=True)
        mem_thread.start()

  
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


    check_cuda(cudart.cudaMemcpyAsync(d_input, host_input.ctypes.data, host_input.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))

    start = time.perf_counter()
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    context.execute_async_v3(stream)

    check_cuda(cudart.cudaMemcpyAsync(host_output.ctypes.data, d_output, host_output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    check_cuda(cudart.cudaStreamSynchronize(stream))

    inference_time = (time.perf_counter() - start) * 1000

    stop_tracking = True
    if has_nvml:
        mem_thread.join()


    peak_gpu_usage_mb = peak_vram_bytes / (1024 * 1024)
    if has_nvml:
        pynvml.nvmlShutdown()


    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    return host_output, inference_time, peak_gpu_usage_mb




def postprocess_engine(predictions, image_files, folder_path, txt_label_path):
    start_time = time.perf_counter()
    with open(txt_label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    class_ids = np.argmax(predictions, axis=1)
    scores = np.max(predictions, axis=1)
    prediction_results = []

    for i, file_name in enumerate(image_files):
        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        class_id = int(class_ids[i])
        class_name = class_names[class_id] if class_id < len(class_names) else f"ID_{class_id}"
        confidence = float(scores[i] * 100)

        prediction_results.append({"file_name": file_name, "class_name": class_name, "confidence": confidence})
        text = f"{class_name}: {confidence:.1f}%"
        cv2.rectangle(image, (15, 15), (250, 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(OUTPUT_DIR / f"pred_{file_name}"), image)

    return prediction_results, (time.perf_counter() - start_time) * 1000



def main():

    try:

        engine_model_path = Path(load_engine_model(MODEL_DIR))

    except FileNotFoundError:    
        onnx_model_path = load_onnx_model(MODEL_DIR)
        export_engine_model(onnx_model_path)
        engine_model_path = Path(load_engine_model(MODEL_DIR))

    batch_numpy, image_files, preprocess_ms = preprocess_engine(IMAGE_DIR, INPUT_SIZE)
    predictions_raw, inference_ms,peak_gpu_mb = inference_engine(batch_numpy, engine_model_path)
    predictions_list, postprocess_ms = postprocess_engine(predictions_raw, image_files, IMAGE_DIR, LABEL_PATH)


    env_info = get_system_env_info()
    model_details = get_model_details(engine_model_path, LABEL_PATH)
    
    config_metadata = {
        "Model Name": engine_model_path.name,
        "Model Path": str(engine_model_path),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **env_info,
        **model_details
    }

    batch_size = len(image_files)
    total_run_time_ms = preprocess_ms + inference_ms + postprocess_ms

    performance_metadata = {
        "Total Images Processed": str(batch_size),
        "Total Run Time": f"{total_run_time_ms:.2f} ms",
        "Preprocess Latency": f"{preprocess_ms / batch_size:.2f} ms per image",
        "Inference Latency": f"{inference_ms / batch_size:.2f} ms per image",
        "Postprocess Latency": f"{postprocess_ms / batch_size:.2f} ms per image",
        "Throughput": f"{(batch_size / (total_run_time_ms / 1000.0)):.2f} Images/sec",
        "Peak Memory Usage": f"{peak_gpu_mb:.2f} MB"
    }

    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="trt",
        config_data=config_metadata,
        performance_data=performance_metadata,
        predictions=predictions_list
    )

if __name__ == "__main__":
    main()