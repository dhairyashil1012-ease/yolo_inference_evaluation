import time
import sys
import cv2
import os
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tensorrt as trt
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms as transforms
from cuda.bindings import runtime as cudart
import shutil
from zipfile import ZipFile
import ultralytics
import onnx
import onnxruntime as ort
import time
from pathlib import Path




MODEL_NAME = "yolo26n-cls.engine"
INPUT_SIZE = (224, 224)
YAML_NAME  = "ImageNet.yaml"

PROJECT_DIR = Path.cwd()


MODEL_DIR = PROJECT_DIR / "Classification _Models"
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "ENGINE_Output"
YAML_DIR = PROJECT_DIR / "cls-yaml"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / MODEL_NAME
YAML_PATH = YAML_DIR / YAML_NAME
MODEL_PATH = MODEL_DIR / MODEL_NAME
print(MODEL_PATH)





# Check Function for Checking error logs
def check_cuda(err):

    if isinstance(err,tuple):
        err=err[0]

    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error : {err}")
    



# Preprocess

def preprocess_engine(image_folder,input_size):


    # Convert integer to (H, W)

    input_size = input_size

    transform = transforms.Compose([transforms.Resize(size=input_size),transforms.ToTensor()])

    # Read image filenames
    image_files = sorted([
        file for file in os.listdir(image_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])

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
def inference_engine(batch_numpy,engine_model):


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

    # -----------------------------
    # End Timer
    # -----------------------------
    end = time.perf_counter()

    # inference_time = (end - start) * 1000
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


def postprocess_engine(predictions,image_files,folder_path,imagenet_yaml):


    # Load YAML file containing class indices and names
    with open(imagenet_yaml, "r") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]


    class_ids = np.argmax(predictions, axis=1)
    scores = np.max(predictions, axis=1)




    for i, file_name in enumerate(image_files):
        
        # 1. Load image using OpenCV for manipulation and drawing
        img_path = os.path.join(folder_path, file_name)
        image = cv2.imread(img_path)
 
        # 2. Match prediction to YAML class mapping
        class_id = int(class_ids[i])
        class_name = class_names.get(class_id, class_names.get(str(class_id), f"ID_{class_id}"))
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

    batch_numpy, image_files = preprocess_engine(IMAGE_DIR,INPUT_SIZE)

    # Inference
    predictions= inference_engine(batch_numpy,MODEL_PATH)

    # Postprocess
    postprocess_engine(predictions,image_files,IMAGE_DIR,YAML_PATH)


if __name__ == "__main__":
    main()

