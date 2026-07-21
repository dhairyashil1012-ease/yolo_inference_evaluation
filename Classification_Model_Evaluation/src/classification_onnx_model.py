import os
import time
import cv2
import torch
import sys
import ast
import onnx
import pynvml
import threading
import numpy as np
import ultralytics  
import platform
from pathlib import Path
from PIL import Image
from configparser import ConfigParser
from ultralytics import YOLO
import onnxruntime as ort
import torchvision.transforms as transforms
from src.report_generator import generate_pdf_report

PROJECT_DIR = Path.cwd()
config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")

MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ONNX_OUTPUT_DIR"]
YAML_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]
REPORT_DIR = PROJECT_DIR / config['PATHS']["REPORT_OUTPUT_DIR"]
MODEL_NAME = config["PATHS"]["PT_MODEL_NAME"]
LABEL_NAME = "label.txt" 

MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = YAML_DIR / LABEL_NAME
REPORT_PDF_PATH = REPORT_DIR / "classification_onnx_inference_report.pdf"

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

for d in [MODEL_DIR, IMAGE_DIR, OUTPUT_DIR, YAML_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_onnx_model(model_dir):
    filelist = sorted([str(f) for f in Path(model_dir).glob("**/*.onnx")])
    if not filelist:
        raise FileNotFoundError("No ONNX model found.")
    return filelist[-1]


def export_onnx_model(model):   
    model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        batch=config.getint("MODEL", "ONNX_EXPORT_BATCH"),
        dynamic=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    time.sleep(1)



def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.to(device)
    return model




def preprocess_onnx(image_folder, input_size):
    start_pre = time.perf_counter()
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
    if not image_files:
        raise ValueError("No images found in the specified folder.")

    batch_images = []
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        batch_images.append(transform(image))

    batch_numpy = torch.stack(batch_images, dim=0).numpy().astype(np.float32)
    return batch_numpy, image_files, (time.perf_counter() - start_pre) * 1000



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
    session = ort.InferenceSession(str(onnx_model_path), providers=["CPUExecutionProvider"])
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



# def inference_onnx(onnx_model, batch_numpy):
#     ort.preload_dlls()
#     opts = ort.SessionOptions()
#     opts.add_session_config_entry("session.required_cuda_compute_capability", "0") 
#     session = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], sess_options=opts)

#     input_name = session.get_inputs()[0].name

#     # Frame Execution Edge synchronization
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()

#     start_inf = time.perf_counter()
#     outputs = session.run(None, {input_name: batch_numpy})
    
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     end_inf = time.perf_counter()

#     return outputs[0], (end_inf - start_inf) * 1000
def inference_onnx(onnx_model, batch_numpy, warmup_runs=10):
    ort.preload_dlls()
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.required_cuda_compute_capability", "0") 
    session = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], sess_options=opts)

    input_name = session.get_inputs()[0].name

    # --- GPU Warm-up Phase ---
    if 'CUDAExecutionProvider' in session.get_providers():
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: batch_numpy})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    # -------------------------

    # Frame Execution Edge synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_inf = time.perf_counter()
    outputs = session.run(None, {input_name: batch_numpy})
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_inf = time.perf_counter()

    return outputs[0], (end_inf - start_inf) * 1000




def postprocess_onnx(predictions, image_files, folder_path, txt_label_path):
    start_post = time.perf_counter()
    with open(txt_label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    class_ids = np.argmax(predictions, axis=1)
    scores = np.max(predictions, axis=1)
    prediction_metadata = []

    for i, file_name in enumerate(image_files):
        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        class_id = int(class_ids[i])
        class_name = class_names[class_id] if class_id < len(class_names) else f"ID_{class_id}"
        confidence = scores[i] * 100

        cv2.rectangle(image, (15, 15), (250, 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, f"{class_name}: {confidence:.1f}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(OUTPUT_DIR / f"pred_{file_name}"), image)
        
        prediction_metadata.append({"file_name": file_name, "class_name": class_name, "confidence": confidence})

    return prediction_metadata, (time.perf_counter() - start_post) * 1000


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
            time.sleep(0.005)  # Sample every 5ms

    if has_nvml:
        mem_thread = threading.Thread(target=track_peak_memory, daemon=True)
        mem_thread.start()

    batch_numpy, image_files, preprocess_ms = preprocess_onnx(IMAGE_DIR, INPUT_SIZE)
    
    raw_predictions, inference_ms = inference_onnx(onnx_model=onnx_model_path, batch_numpy=batch_numpy)

    prediction_metadata, postprocess_ms = postprocess_onnx(
        predictions=raw_predictions,
        image_files=image_files,
        folder_path=IMAGE_DIR,
        txt_label_path=LABEL_PATH,
    )

    stop_tracking = True
    if has_nvml:
        mem_thread.join()

    peak_mem_mb = peak_vram_bytes / (1024 * 1024)
    if has_nvml:
        pynvml.nvmlShutdown()  # Safe shutdown after extraction

    batch_size = len(image_files)
    inference_time_in_second = inference_ms
    avg_preprocess = preprocess_ms / batch_size
    avg_inference = inference_ms / batch_size
    avg_postprocess = postprocess_ms / batch_size
    total_pipeline_ms = preprocess_ms + inference_ms + postprocess_ms
    avg_total_time_ms = avg_preprocess + avg_inference + avg_postprocess
    print("=" * 60)
    print("ONNX RUNTIME PERFORMANCE")
    print("=" * 60)
    print(f"Total Images Processed : {batch_size}")
    print(f"Total Pipeline Time    : {total_pipeline_ms:.2f} ms")
    print(f"Preprocess Latency     : {avg_preprocess:.2f} ms per image")
    print(f"Inference Latency      : {avg_inference:.2f} ms per image")
    print(f"Postprocess Latency    : {avg_postprocess:.2f} ms per image")
    print(f"Peak VRAM Usage        : {peak_mem_mb:.2f} MB")

    print(f"\nSummary -> preprocess: {avg_preprocess:.1f}ms, inference: {avg_inference:.1f}ms, postprocess: {avg_postprocess:.1f}ms per image\n")

    env_info = get_system_env_info()
    model_details = get_model_details(onnx_model_path)
    onnx_model_name =(onnx_model_path.split('/')[-1:]) #output: ./from_here/thefile.txt
    config_dict = {
        "Model Name": str(onnx_model_name[0]),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Label Map": str(LABEL_PATH),
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **env_info,
        **model_details
    }


    performance_metadata = {
        "Total Images Processed": str(batch_size),
        "Total Run Time": f"{total_pipeline_ms:.2f} ms",
        "Total Inference Time in ms": f"{inference_time_in_second:.2f}ms",
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Inference Latency": f"{avg_inference:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
        "Throughput": f"{(batch_size / (total_pipeline_ms / 1000.0)):.2f} Images/sec",
        "Peak Memory Usage": f" {peak_mem_mb:.2f} MB"
    }

        
    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="onnx",
        config_data=config_dict,
        performance_data=performance_metadata ,
        predictions=prediction_metadata
    )


if __name__ == "__main__":
    main()
    