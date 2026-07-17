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
import ast
import tensorrt as trt
from cuda.bindings import runtime as cudart
from src.report_generator import generate_pdf_report
PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read("config.txt")

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]
REPORT_DIR=PROJECT_DIR/config['PATHS']["REPORT_OUTPUT_DIR"]
MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]

LABEL_NAME = "label.txt" 
MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = LABEL_DIR / LABEL_NAME
REPORT_PDF_PATH = REPORT_DIR / "classification_engine_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Configuration")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
print(f"Label Path : {LABEL_PATH}")
print(f"Output     : {OUTPUT_DIR}")
print(f"Input Size : {INPUT_SIZE}")



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

    return {
        "OS": f"{platform.system()} {platform.release()}",
        "Python Version": sys.version.split()[0],
        "TensorRT Version": trt.__version__,
        "Inference Device": device_model,
        "CUDA Version": cuda_version
    }


def get_model_details(engine_model_path, label_path):

    engine_model_path = Path(engine_model_path)
    if not engine_model_path.exists():
        return {
            "Model Architecture": "TensorRT Engine",
            "Total Parameters": "Unknown",
            "File Size": "Unknown",
            "Number of Classes": "Unknown",
            "Class Names": "Unknown",
            "Precision Mode": "Unknown"
        }

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Read text class labels
    class_names = []
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    # Deserialize Engine to probe internal static profiles
    with open(engine_model_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context # OPTIONAL
    var1=inspector.get_layer_information(0, trt.LayerInformationFormat.JSON) # Print the information of the first layer in the engine.
    var2=inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    var3=var1+var2
    pattern = r'"Weights":\s*(\{[^}]+\})'
    matches = re.findall(pattern, var3)
    weighted_sum=0
    for i in matches:
        dict_data=json.loads(i)
        w=dict_data["Count"]
        weighted_sum+=w


    architecture = "YOLO (TensorRT Engine)"
    
 
    try:
        # Check primary input tensor profile data type
        input_tensor_name = engine.get_tensor_name(0)
        dtype_enum = engine.get_tensor_dtype(input_tensor_name)
        if dtype_enum == trt.DataType.HALF:
            precision = "FP16"
        elif dtype_enum == trt.DataType.FLOAT:
            precision = "FP32"
        elif dtype_enum == trt.DataType.INT8:
            precision = "INT8"
        else:
            precision = "Unknown"
    except Exception:
        precision = "Unknown"

    # 3. File Size Calculation
    file_size_mb = f"{engine_model_path.stat().st_size / (1024 * 1024):.2f} MB"

    return {
        "Model Architecture": architecture,
        "Total Parameters": weighted_sum,
        "File Size": file_size_mb,
        "Number of Classes": len(class_names) if class_names else "Unknown",
        "Class Names": ", ".join(class_names) if class_names else "Unknown",
        "Precision Mode": precision
    }


def load_onnx_model(model_dir):
    pathlist = Path(model_dir).glob("**/*.onnx")
    filelist = sorted([str(file) for file in pathlist])
    if len(filelist) == 0:
        raise FileNotFoundError("No ONNX model found to export engine from.")
    onnx_model = filelist[-1]
    return onnx_model


def export_engine_model(onnx_model_p):
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
    print("\nTensorRT engine export completed successfully.")
    time.sleep(1)


def load_engine_model(model_dir):
    pathlist = Path(model_dir).glob("**/*.engine")
    filelist = sorted([str(file) for file in pathlist])
    if len(filelist) == 0:
        raise FileNotFoundError("No engine model found.")
    engine_model = filelist[-1]
    print(f"engine Model : {engine_model}")
    return engine_model


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

  
    _, free_mem, total_mem = cudart.cudaMemGetInfo()
    peak_gpu_usage_mb = (total_mem - free_mem) / (1024 * 1024)

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

        prediction_results.append({
            "file_name": file_name,
            "class_name": class_name,
            "confidence": confidence
        })

        text = f"{class_name}: {confidence:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x, text_y = 20, 40
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        save_path = OUTPUT_DIR / f"pred_{file_name}"
        cv2.imwrite(str(save_path), image)

    postprocess_time = (time.perf_counter() - start_time) * 1000
    return prediction_results, postprocess_time



# -----------------------------
# Main Execution Entry Point
# -----------------------------
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

    # 1. Run Pipeline and Extract execution intervals
    batch_numpy, image_files, preprocess_ms = preprocess_engine(
        image_folder=IMAGE_DIR,
        input_size=INPUT_SIZE,
    )

    predictions_raw, inference_ms, peak_gpu_mb = inference_engine(
        batch_numpy=batch_numpy,
        engine_model=engine_model_path,
    )

    predictions_list, postprocess_ms = postprocess_engine(
        predictions=predictions_raw,
        image_files=image_files,
        folder_path=IMAGE_DIR,
        txt_label_path=LABEL_PATH,
    )


    env_info = get_system_env_info()
    model_details = get_model_details(engine_model_path, LABEL_PATH)

    config_metadata = {
        "Model Path": str(engine_model_path),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        "OS": env_info["OS"],
        "Python Version": env_info["Python Version"],
        "TensorRT Version": env_info["TensorRT Version"],
        "Inference Device": env_info["Inference Device"],
        "CUDA Version": env_info["CUDA Version"],
        "Model Architecture": model_details["Model Architecture"],
        "Total Parameters": model_details["Total Parameters"],
        "File Size": model_details["File Size"],
        "Number of Classes": model_details["Number of Classes"],
        "Precision Mode": model_details["Precision Mode"],
        "Class Names": model_details["Class Names"]
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
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Inference Latency": f"{avg_inference:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
        "Throughput": f"{(total_images / (inference_ms / 1000.0)):.2f} Images/sec",
        "Peak Memory Usage": f" {peak_gpu_mb:.2f} MB"
    }

  
    print("\nProcessing complete. Dispatching telemetry payloads to PDF compiler...")
    
    # Example Call Structure:
    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="trt",
        config_data=config_metadata,
        performance_data=performance_metadata,
        predictions=predictions_list
    )


if __name__ == "__main__":
    main()