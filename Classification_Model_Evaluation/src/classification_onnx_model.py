import os
import time
import cv2
import torch
import sys
import ast
import onnx
import pynvml
import torch
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
import contextlib



PROJECT_DIR = Path.cwd()


config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")


MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ONNX_OUTPUT_DIR"]
YAML_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]
REPORT_DIR=PROJECT_DIR/config['PATHS']["REPORT_OUTPUT_DIR"]
MODEL_NAME = config["PATHS"]["PT_MODEL_NAME"]


LABEL_NAME = "label.txt" 

MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = YAML_DIR / LABEL_NAME

REPORT_PDF_PATH = REPORT_DIR / "classification_onnx_inference_report.pdf"


INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
YAML_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
print(f"Label Path : {LABEL_PATH}")
print(f"Output Dir : {OUTPUT_DIR}")
print(f"Input Size : {INPUT_SIZE}")
print("=" * 60)



class Profile(contextlib.ContextDecorator):

    def __init__(self, t=0.0, sync_cuda=False):
        self.t = t
        self.sync_cuda = sync_cuda
        self.dt = 0.0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt

    def time(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()


# Load ONNX Model Path
def load_onnx_model(model_dir):
    pathlist = Path(model_dir).glob("**/*.onnx")
    filelist = sorted([str(file) for file in pathlist])

    if len(filelist) == 0:
        raise FileNotFoundError("No ONNX model found.")

    onnx_model = filelist[-1]
 
    return onnx_model



# Generate ONNX Model From Pt
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





# Load PyTorch/YOLO Model
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on : {device}")

    model = YOLO(model_path)
    model.to(device)

    
    return model


def preprocess_onnx(folder_path, sync_cuda=False):

    prof = Profile(sync_cuda=sync_cuda)
    
    with prof:
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
    
    preprocess_time_ms = prof.t * 1000
    return batch_numpy, image_files, preprocess_time_ms




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



# Get Model Details
def get_model_details(onnx_model_path):

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

    # Load ONNX structure and open metadata session
    model = onnx.load(str(onnx_model_path))
    session = ort.InferenceSession(
        str(onnx_model_path), 
        providers=["CPUExecutionProvider"]
    )
    metadata = session.get_modelmeta().custom_metadata_map



    # 2. Parameters
    total_params = sum(
        np.prod(tensor.dims)
        for tensor in model.graph.initializer
    )

    # 3. Precision Mode
    dtypes = {
        onnx.TensorProto.DataType.Name(t.data_type)
        for t in model.graph.initializer
    }
    if "FLOAT16" in dtypes:
        precision = "FP16"
    elif "FLOAT" in dtypes:
        precision = "FP32"
    elif "INT8" in dtypes:
        precision = "INT8"
    else:
        precision = "Unknown"

    # 4. File Size
    file_size_mb = f"{onnx_model_path.stat().st_size / (1024 * 1024):.2f} MB"

    # 5. Class Parsing
    class_names = []
    num_classes = "Unknown"
    if "names" in metadata:
        try:
            names = ast.literal_eval(metadata["names"])
            if isinstance(names, dict):
                class_names = list(names.values())
            elif isinstance(names, list):
                class_names = names
            num_classes = len(class_names)
        except Exception:
            pass

    return {
        "Total Parameters": f"{int(total_params):,}" if total_params > 0 else "Unknown",
        "File Size": file_size_mb,
        "Number of Classes": num_classes,
        "Class Names": ", ".join(class_names) if class_names else "Unknown",
        "Precision Mode": precision
    }



# Inference Code
def inference_onnx(onnx_model, batch_numpy,sync_cuda=False):
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


    # start_inf = time.perf_counter()
    prof = Profile(sync_cuda=sync_cuda)
    
    with prof:
        outputs = session.run(None, {input_name: batch_numpy})

    # end_inf = time.perf_counter()
    inference_time_ms = prof.t * 1000
    # inference_time_ms = (end_inf - start_inf) *1000

    return outputs[0], inference_time_ms



# Postproces

def postprocess_onnx(predictions, image_files, folder_path, txt_label_path, sync_cuda=False):
    prof = Profile(sync_cuda=sync_cuda)

    with prof:
        print("=" * 60)
        print("POSTPROCESS")
        print("=" * 60)

        with open(txt_label_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

        class_ids = np.argmax(predictions, axis=1)
        scores = np.max(predictions, axis=1)
        
        prediction_metadata = []

        for i, file_name in enumerate(image_files):
            img_path = os.path.join(folder_path, file_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Unable to read {img_path}")
                continue

            class_id = int(class_ids[i])
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f"ID_{class_id}"
                
            confidence = scores[i] * 100
            text = f"{class_name}: {confidence:.1f}%"
            print(f"Image: {file_name} -> {text}")

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            text_x, text_y = 20, 40
            cv2.rectangle(image, 
                          (text_x - 5, text_y - text_size[1] - 5), 
                          (text_x + text_size[0] + 5, text_y + baseline), 
                          (0, 0, 0), 
                          cv2.FILLED)

            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

            save_path = OUTPUT_DIR / f"pred_{file_name}"
            cv2.imwrite(str(save_path), image)
            
            prediction_metadata.append({
                "file_name": file_name,
                "class_name": class_name,
                "confidence": confidence
            })
            
    postprocess_time_ms = prof.t * 1000
    return prediction_metadata, postprocess_time_ms


# Main
def main():
    try:
        onnx_model_path = load_onnx_model(MODEL_DIR)
    except FileNotFoundError:
        model = load_model(MODEL_PATH)
        export_onnx_model(model)
        onnx_model_path = load_onnx_model(MODEL_DIR)
    sync_cuda = torch.cuda.is_available()
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
        print("Warning: NVML initialization failed. GPU memory tracking disabled.")

    # High-frequency sampling thread to catch micro-spikes
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


    batch_numpy, image_files, preprocess_ms = preprocess_onnx(folder_path=IMAGE_DIR, sync_cuda=sync_cuda)
    
    raw_predictions, inference_ms = inference_onnx(onnx_model=onnx_model_path, batch_numpy=batch_numpy,sync_cuda=sync_cuda)

    prediction_metadata, postprocess_ms = postprocess_onnx(
        predictions=raw_predictions,
        image_files=image_files,
        folder_path=IMAGE_DIR,
        txt_label_path=LABEL_PATH,
        sync_cuda=sync_cuda
    )


    stop_tracking = True
    if has_nvml:
        mem_thread.join()

    peak_mem_mb = peak_vram_bytes / (1024 * 1024)
    if has_nvml:
        pynvml.nvmlShutdown()  # Safe shutdown after extraction

    batch_size = len(image_files)
    avg_preprocess = preprocess_ms / batch_size
    avg_inference = inference_ms / batch_size
    avg_postprocess = postprocess_ms / batch_size
    total_pipeline_ms = preprocess_ms + inference_ms + postprocess_ms

    # Log metrics to console
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
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Inference Latency": f"{avg_inference:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
        "Throughput": f"{(batch_size / (inference_ms / 1000.0)):.2f} Images/sec",
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