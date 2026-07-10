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




MODEL_NAME = "yolo26n.engine"
YAML_NAME  = "coco.yaml"
INPUT_SIZE = (640, 640)

PROJECT_DIR = Path.cwd()

MODEL_DIR = PROJECT_DIR / "Detection_Models"
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "ENGINE_Output"
YAML_DIR = PROJECT_DIR / "detection-yaml"

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





def postprocess_engine(outputs,image_files,image_folder,yaml_path,confidence_threshold=0.6,input_size=640):

    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    input_h, input_w = input_size

    # Load class names
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    class_names = data["names"]

    if isinstance(class_names, dict):
    
        names = []
    
        for key in sorted(class_names.keys()):
            names.append(class_names[key])
    
        class_names = names

 
    for img_idx, image_name in enumerate(image_files):

        image_path = os.path.join(image_folder, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        scale_x = W / input_w
        scale_y = H / input_h

        detections = outputs[img_idx]

        for det in detections:

            x1, y1, x2, y2, score, cls = det

            if score < confidence_threshold:
                continue

            cls = int(cls)

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W - 1))
            y2 = max(0, min(y2, H - 1))

            label = f"{class_names[cls]} {score:.2f}"

            cv2.rectangle(image,(x1, y1),(x2, y2),(0, 255, 0),2)

            cv2.putText(image,label,(x1, max(20, y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255, 0, 0),2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            filename=image_name
            
            save_path = OUTPUT_DIR / filename
            cv2.imwrite(str(save_path), image)
            
            print(f"Saved : {save_path}")

        print("\nAll output images saved successfully.")

        # plt.figure(figsize=(8, 8))
        # plt.imshow(image)
        # plt.title(image_name)
        # plt.axis("off")
        # plt.show()        
def main():


    batch_numpy, image_files = preprocess_engine(IMAGE_DIR,INPUT_SIZE)

    # Inference
    predictions= inference_engine(batch_numpy,MODEL_PATH)

    # Postprocess
    postprocess_engine(predictions,image_files,IMAGE_DIR,YAML_PATH,confidence_threshold=0.6,input_size=INPUT_SIZE)

if __name__ == "__main__":
    main()

