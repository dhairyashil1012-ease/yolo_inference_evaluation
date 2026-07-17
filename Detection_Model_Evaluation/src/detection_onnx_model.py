import os
import time
import sys
import platform
import cv2
import torch
import numpy as np
import onnx
import onnxruntime as ort
import pynvml
import ast
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
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
    
    return {
        "OS": f"{platform.system()} {platform.release()}",
        "Python Version": sys.version.split()[0],
        "ONNX Runtime Version": ort.__version__,
        "Inference Device": device_model,
        "CUDA Version": cuda_version
    }


def get_model_details(onnx_model_path, label_path):

    onnx_model_path = Path(onnx_model_path)
    label_path = Path(label_path)
    
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
    session = ort.InferenceSession(str(onnx_model_path),
                                   providers=['CPUExecutionProvider'])
    
    metadata = session.get_modelmeta().custom_metadata_map

    task = metadata.get("task")
    if task:
        architecture = f"YOLO ({task.capitalize()})"
    else:
        architecture = model_graph.graph.name or "ONNX Model"

    total_params = sum(np.prod(tensor.dims) for tensor in model_graph.graph.initializer)

    dtypes = {onnx.TensorProto.DataType.Name(t.data_type) for t in model_graph.graph.initializer}
    if "FLOAT16" in dtypes:
        precision = "FP16"
    elif "FLOAT" in dtypes:
        precision = "FP32"
    elif "INT8" in dtypes:
        precision = "INT8"
    else:
        precision = "Unknown"


    file_size_mb = f"{onnx_model_path.stat().st_size / (1024 * 1024):.2f} MB"


    class_names = []
    num_classes = 'Unknown'

    if "names" in metadata:
        try:
            names = ast.literal_eval(metadata['names'])
            if isinstance(names,dict):
                class_names = list(names.values())
            elif isinstance(names,list):
                class_names = names
            num_classes =len(class_names)
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
    
    start_pre = time.perf_counter()

    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
    ])

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
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        image_list.append(tensor)

    batch_tensor = torch.stack(image_list)
    batch_numpy = batch_tensor.numpy().astype(np.float32)

    print("Number of Images :", len(image_files))
    print("Batch Shape      :", batch_numpy.shape)
    print("Input dtype      :", batch_numpy.dtype)
    print()
    end_pre = time.perf_counter()
    preprocess_time_ms = (end_pre - start_pre) * 1000
    return batch_numpy, image_files, preprocess_time_ms


def inference_onnx(onnx_model, batch_numpy):
    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.required_cuda_compute_capability", "0") 
    
    session = ort.InferenceSession(
        onnx_model, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], 
        sess_options=opts
    )
    
    active_provider = session.get_providers()[0]
    print("Execution Provider :", active_provider)

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    outputs = session.run(None, {input_name: batch_numpy})
    end = time.perf_counter()

    inference_time_ms = (end - start) * 1000

    return outputs[0], inference_time_ms



def postprocess_onnx(outputs, image_files, image_folder, label_path, confidence_threshold=0.6, input_size=640):
    start_post=time.perf_counter()

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    input_h, input_w = input_size

    # Load class names
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

       
        has_detections = False

        for det in detections:
            if len(det) >= 6:
                x1, y1, x2, y2, score, cls = det[:6]
            else:
                continue

            if score < confidence_threshold:
                continue

            has_detections = True
            cls = int(cls)
            class_name = class_names[cls] if cls < len(class_names) else f"ID_{cls}"

            # Rescale coordinate pairs back to source dimensions
            x1_scaled = max(0, min(int(x1 * scale_x), W - 1))
            y1_scaled = max(0, min(int(y1 * scale_y), H - 1))
            x2_scaled = max(0, min(int(x2 * scale_x), W - 1))
            y2_scaled = max(0, min(int(y2 * scale_y), H - 1))

            # Store detection dictionaries
            image_record["detections"].append({
                "class_name": class_name,
                "confidence": float(score) * 100 if score <= 1.0 else float(score),
                "box": [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
            })

            
            cv2.rectangle(image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), (0, 255, 0), 2)
            label_text = f"{class_name} {score:.2f}"
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

    print("\nAll output images saved successfully.")
    return prediction_metadata,postprocess_time_ms


def main():
    try:
        onnx_model_path = load_onnx_model(MODEL_DIR)
    except FileNotFoundError:
        model = load_model(MODEL_PATH)
        export_onnx_model(model)
        onnx_model_path = load_onnx_model(MODEL_DIR)
    
    has_nvml = False
    try:
        pynvml.nvmlInit()
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        has_nvml = True
    except Exception:
        print("Warning: NVML initialization failed. GPU memory tracking disabled.")

    baseline_mem = 0
    if has_nvml:
        baseline_mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used


    batch_numpy, image_files,preprocess_ms = preprocess_onnx(folder_path=IMAGE_DIR)

    predictions, inference_ms = inference_onnx(
        onnx_model=onnx_model_path,
        batch_numpy=batch_numpy,
    )

    peak_mem_mb = 0
    if has_nvml:
        current_mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle).used
        peak_mem_mb = max(0, (current_mem - baseline_mem) / (1024 * 1024))
        pynvml.nvmlShutdown() 

    prediction_metadata,postprocess_ms = postprocess_onnx(
        outputs=predictions,
        image_files=image_files,
        image_folder=IMAGE_DIR,
        label_path=LABEL_PATH,
        confidence_threshold=0.6,
        input_size=INPUT_SIZE,
    )


    sys_env = get_system_env_info()
    model_spec = get_model_details(onnx_model_path, LABEL_PATH)

    batch_size = len(image_files)
    avg_preprocess = preprocess_ms / batch_size
    avg_inference = inference_ms / batch_size
    avg_postprocess = postprocess_ms / batch_size
    total_pipeline_ms = preprocess_ms + inference_ms + postprocess_ms

    # Log metrics to console matching required format
    print("=" * 60)
    print("ONNX RUNTIME PERFORMANCE")
    print("=" * 60)
    print(f"Total Images Processed : {batch_size}")
    print(f"Total Pipeline Time    : {total_pipeline_ms:.2f} ms")
    print(f"Preprocess Latency     : {avg_preprocess:.2f} ms per image")
    print(f"Inference Latency      : {avg_inference:.2f} ms per image")
    print(f"Postprocess Latency    : {avg_postprocess:.2f} ms per image")
    
    print(f"\nSummary -> preprocess: {avg_preprocess:.1f}ms, inference: {avg_inference:.1f}ms, postprocess: {avg_postprocess:.1f}ms per image\n")


    # Construct Configuration Dictionary matching Phase 1 Table requirements
    config_dict = {
        "Model Path": str(onnx_model_path),
        "Input Dimensions": f"{INPUT_SIZE[0]}x{INPUT_SIZE[1]}",
        "Source Images Directory": str(IMAGE_DIR),
        "Processed Images Output": str(OUTPUT_DIR),
        **sys_env,
        **model_spec
    }

    perf_metrics = {
        "Total Images Processed": batch_size,
        "Total Pipeline Time": f"{total_pipeline_ms:.2f} ms",
        "Preprocess Latency": f"{avg_preprocess:.2f} ms per image",
        "Inference Latency": f"{avg_inference:.2f} ms per image",
        "Postprocess Latency": f"{avg_postprocess:.2f} ms per image",
        "Throughput": f"{batch_size / (total_pipeline_ms / 1000):.2f} Images/sec",
        "Peak Memory Usage": f"{peak_mem_mb:.2f} MB" if peak_mem_mb > 0 else "N/A"
    }

    # Generate Structured Report PDF
    # pdf_report_path = OUTPUT_DIR / "onnx_inference_report.pdf"
    generate_pdf_report(
        output_pdf_path=REPORT_PDF_PATH,
        backend="onnx",
        config_data=config_dict,
        performance_data=perf_metrics,
        predictions=prediction_metadata
    )


if __name__ == "__main__":
    main()