import os
import subprocess
import time
import cv2
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from configparser import ConfigParser

import tensorrt as trt
from cuda.bindings import runtime as cudart

import torchvision.transforms as transforms

PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read("config.txt")

# -----------------------------
# Paths
# -----------------------------
# MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
# IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
# OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]

# MODEL_NAME = config["PATHS"]["ENGINE_MODEL_NAME"]
MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]
# CHANGED: Read from the generated label.txt inside the target folder instead of the yaml file
LABEL_NAME = "label.txt" 
# MODEL_ONNX_PATH = MODEL_DIR / MODEL_NAME
MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = LABEL_DIR / LABEL_NAME

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Engine model not found at: {MODEL_PATH}")

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"Label file not found at: {LABEL_PATH}")

print("=" * 60)
print("Configuration")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
print(f"Label Path : {LABEL_PATH}")
print(f"Output     : {OUTPUT_DIR}")
print(f"Input Size : {INPUT_SIZE}")
# print(MODEL_DIR)



# Load ONNX Model
def load_onnx_model(model_dir):
    pathlist = Path(model_dir).glob("**/*.onnx")
    filelist = sorted([str(file) for file in pathlist])

    if len(filelist) == 0:
        raise FileNotFoundError("No ONNX model found.")

    onnx_model = filelist[-1]
    # print(f"ONNX Model : {onnx_model}")
    return onnx_model





def export_engine_model(onnx_model_p):
    os.chdir(MODEL_DIR)
    subprocess.run([
        "trtexec",
        f"--onnx={onnx_model_p}",
        "--minShapes=images:1x3x224x224",
        "--optShapes=images:4x3x224x224",
        "--maxShapes=images:16x3x224x224",
        "--saveEngine=yolo26n-cls.engine"
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





# Check Function for Checking error logs
def check_cuda(err):
    if isinstance(err, tuple):
        err = err[0]

    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error : {err}")


# Preprocess
def preprocess_engine(image_folder, input_size):
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

    return batch_numpy, image_files


# Inference
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

    # Allocate GPU memory
    err, d_input = cudart.cudaMalloc(host_input.nbytes)
    check_cuda(err)

    err, d_output = cudart.cudaMalloc(host_output.nbytes)
    check_cuda(err)

    # Create CUDA stream
    err, stream = cudart.cudaStreamCreate()
    check_cuda(err)

    start = time.perf_counter()

    # Host -> Device
    check_cuda(
        cudart.cudaMemcpyAsync(
            d_input,
            host_input.ctypes.data,
            host_input.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
    )

    # Bind tensors
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # TensorRT inference
    context.execute_async_v3(stream)

    # Device -> Host
    check_cuda(
        cudart.cudaMemcpyAsync(
            host_output.ctypes.data,
            d_output,
            host_output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
    )

    # Wait for GPU
    check_cuda(cudart.cudaStreamSynchronize(stream))

    end = time.perf_counter()
    inference_time = (end - start)
    batch_size = batch_numpy.shape[0]

    print("\n" + "=" * 60)
    print("TENSORRT PERFORMANCE")
    print("=" * 60)
    print(f"Batch Size           : {batch_size}")
    print(f"Batch Inference Time : {inference_time:.3f} ")
    print(f"Per Image Latency    : {inference_time/batch_size:.3f} ")

    # Cleanup
    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    return host_output


# PostProcess
# CHANGED: Replaced imagenet_yaml with txt_label_path parameter
def postprocess_engine(predictions, image_files, folder_path, txt_label_path):
    print("=" * 60)
    print("POSTPROCESS")
    print("=" * 60)

    # CHANGED: Load class names directly from line-separated label text file instead of YAML parsing
    with open(txt_label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    class_ids = np.argmax(predictions, axis=1)
    scores = np.max(predictions, axis=1)

    for i, file_name in enumerate(image_files):
        # 1. Load image using OpenCV for manipulation and drawing
        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Unable to read {img_path}")
            continue

        # 2. Match prediction to index in label list
        class_id = int(class_ids[i])
        
        # CHANGED: Map index position cleanly to text element list boundaries
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"ID_{class_id}"
            
        confidence = scores[i] * 100  # Convert to standard percentage format

        # 3. Create annotation string (e.g., "Cat: 94.5%")
        text = f"{class_name}: {confidence:.1f}%"
        print(f"Image: {file_name} -> {text}")

        # 4. Draw background bounding box for text readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        text_x, text_y = 20, 40
        # Draw a small filled dark rectangle behind text
        cv2.rectangle(image, 
                      (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + baseline), 
                      (0, 0, 0), 
                      cv2.FILLED)

        # 5. Burn text onto the image (Bright Neon Green font color)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        # 6. Save target image directly into your configured OUTPUT_DIR
        save_path = OUTPUT_DIR / f"pred_{file_name}"
        cv2.imwrite(str(save_path), image)
        print(f"Saved: {save_path}")


# Main
def main():

    try :
        engine_model_path=load_engine_model(MODEL_DIR)
        print(f"Using Existing Engine :{engine_model_path}")
    
    except FileNotFoundError:
        print("No TensorRT engine found. Exporting from ONNX...")

        # Load ONNX model
        onnx_model_path = load_onnx_model(MODEL_DIR)

        # Export TensorRT engine
        export_engine_model(onnx_model_path)

        # Load newly created engine
        engine_model_path = load_engine_model(MODEL_DIR)
        print(f"Using newly created engine: {engine_model_path}")




    batch_numpy, image_files = preprocess_engine(
        image_folder=IMAGE_DIR,
        input_size=INPUT_SIZE,
    )

    predictions = inference_engine(
        batch_numpy=batch_numpy,
        engine_model=engine_model_path,
    )

    # CHANGED: Passing the target label text path setup
    postprocess_engine(
        predictions=predictions,
        image_files=image_files,
        folder_path=IMAGE_DIR,
        txt_label_path=LABEL_PATH,
    )


if __name__ == "__main__":
    main()