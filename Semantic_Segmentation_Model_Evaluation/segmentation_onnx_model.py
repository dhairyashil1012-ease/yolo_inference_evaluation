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




MODEL_NAME = "yolo26n-sem.pt"
INPUT_SIZE = (1024, 2048)

PROJECT_DIR = Path.cwd()

MODEL_DIR = PROJECT_DIR / "Sem_Models"
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "ONNX_Output"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / MODEL_NAME


# Load .pt Model
def load_model(model_path):

    device = torch.device("cuda")


    model = YOLO(model_path)

    model.to(device)

    return model


# Export .pt Model To ONNX Model
def export_onnx_model(model):   

    model.export(format="onnx",imgsz=(1024, 2048),batch=16,dynamic=True,device="cuda",simplify=False,opset=18)
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




# PostProcess

# PostProcess

def postprocess_onnx(outputs,image_files,image_folder):

    
    for img_idx, image_name in enumerate(image_files):

        image_path = os.path.join(image_folder, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]

        detections = outputs[img_idx]

        
        # # Convert scores to class IDs
        mask = np.argmax(detections, axis=0).astype(np.uint8)


        # # Resize mask if model output size differs from original image
        if mask.shape != (H, W):
            mask = cv2.resize(mask,(W, H),interpolation=cv2.INTER_NEAREST)

    
        # # Create colored mask
        
        color_mask = cv2.applyColorMap(mask * 15, cv2.COLORMAP_JET)

        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image,0.7,color_mask,0.3,0)
        
        # filename = Path(image_path).
        filename=image_name

        save_path = OUTPUT_DIR / filename

        cv2.imwrite(str(save_path), blended)

        print(f"Saved : {save_path}")

        print("\nAll output images saved successfully.")
    
        # # Display
        # plt.figure(figsize=(14,5))
        # plt.imshow(blended)
        # plt.title(image_name)
        # plt.axis("off")
        # plt.show()
        



def main():

    model=load_model(MODEL_PATH)

    export_onnx_model(model)

    onnx_model_path=load_onnx_model(MODEL_DIR)

    batch_numpy, image_files = preprocess_onnx(IMAGE_DIR)

    predictions,inference_time = inference_onnx(onnx_model_path,batch_numpy)

    postprocess_onnx(predictions,image_files,IMAGE_DIR )

if __name__ == "__main__":
    main()

