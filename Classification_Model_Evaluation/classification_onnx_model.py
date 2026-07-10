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




MODEL_NAME = "yolo26n-cls.pt"
INPUT_SIZE = (224, 224)
YAML_NAME  = "ImageNet.yaml"
PROJECT_DIR = Path.cwd()

MODEL_DIR = PROJECT_DIR / "Classification _Models"
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "ONNX_Output"
YAML_DIR = PROJECT_DIR / "cls-yaml"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / MODEL_NAME
YAML_PATH = YAML_DIR / YAML_NAME



# Load .pt Model
def load_model(model_path):

    device = torch.device("cuda")


    model = YOLO(model_path)

    model.to(device)

    return model


def export_onnx_model(model):   

    model.export(format="onnx",imgsz=(224,224),batch=16,dynamic=True,device="cuda")
    print("\nONNX export completed successfully.")
    time.sleep(1)



# Load ONNX Model
def load_onnx_model(MODEL_DIR):

    pathlist = Path(MODEL_DIR).glob('**/*.onnx')

    filelist = sorted( [str(file) for file in pathlist] )

    for file in filelist:
        print( file )
    
    return file





# Preprocess Phase 
def preprocess_onnx(folder_path):

    transform = transforms.Compose([transforms.Resize(INPUT_SIZE),transforms.ToTensor()])

    image_files = sorted([file
                          for file in os.listdir(folder_path)
                          if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])

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


    return batch_numpy, image_files






# Inference Phase
def inference_onnx(onnx_model, batch_numpy):
    opts = ort.SessionOptions()

    # Force ONNX Runtime to raise an exception if an op cannot run on the requested ExecutionProvider
    opts.add_session_config_entry("session.required_cuda_compute_capability", "0") 

    # This configuration forces strict provider matching
    session = ort.InferenceSession(onnx_model, providers=['CUDAExecutionProvider','CPUExecutionProvider'], sess_options=opts)

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()

    outputs = session.run(None,{input_name: batch_numpy})

    end = time.perf_counter()

    # inference_time = (end - start) * 1000
    inference_time = (end - start)

    batch_size = batch_numpy.shape[0]

    print("=" * 60)
    print("ONNX RUNTIME PERFORMANCE")
    print("=" * 60)
    print(f"Batch Size           : {batch_size}")
    print(f"Batch Inference Time : {inference_time:.3f}")
    print(f"Per Image Latency    : {inference_time / batch_size:.3f}")
    print()

    return outputs[0], inference_time






# Postprocess Phase

def postprocess_onnx(predictions, image_files, folder_path, imagenet_yaml):
    print("=" * 60)
    print("POSTPROCESS")
    print("=" * 60)

    # Load YAML file containing class indices and names
    with open(imagenet_yaml, "r") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]



    # Extract class IDs and actual probability scores
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

        # 6. Save target image directly into your configured OUTPUT_DIR
        save_path = OUTPUT_DIR / f"pred_{file_name}"
        cv2.imwrite(str(save_path), image)
        print(f"Saved: {save_path}")


        

# Main
def main():

    model=load_model(MODEL_PATH)

    export_onnx_model(model)

    onnx_model_path=load_onnx_model(MODEL_DIR)

    batch_numpy, image_files = preprocess_onnx(IMAGE_DIR)

    predictions,inference_time = inference_onnx(onnx_model_path,batch_numpy)

    postprocess_onnx(predictions,image_files,IMAGE_DIR,YAML_PATH)


if __name__ == "__main__":
    main()

