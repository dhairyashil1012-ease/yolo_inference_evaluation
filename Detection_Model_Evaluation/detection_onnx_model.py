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




MODEL_NAME = "yolo26n.pt"
INPUT_SIZE = (640, 640)
YAML_NAME  = "coco.yaml"
PROJECT_DIR = Path.cwd()

MODEL_DIR = PROJECT_DIR / "Detection_Models"
IMAGE_DIR = PROJECT_DIR / "Images"
OUTPUT_DIR = PROJECT_DIR / "ONNX_Output"
YAML_DIR = PROJECT_DIR / "detection-yaml"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / MODEL_NAME
YAML_PATH = YAML_DIR / YAML_NAME

print(YAML_PATH)
# Load .pt Model
def load_model(model_path):

    device = torch.device("cuda")


    model = YOLO(model_path)

    model.to(device)

    return model


# Export .pt Model To ONNX Model
def export_onnx_model(model):   

    model.export(format="onnx",imgsz=640,batch=16,dynamic=True,device="cuda")
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

def postprocess_onnx(outputs,image_files,image_folder,yaml_path,confidence_threshold=0.6,input_size=640):

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

    postprocess_onnx(predictions,image_files,IMAGE_DIR,YAML_PATH, confidence_threshold=0.6,input_size=INPUT_SIZE)

    # # Inference
    # predictions,inference_time = inference_onnx(session,input_name,batch_numpy)

    # # Postprocess
    # postprocess_onnx(predictions,image_files,image_folder,coco_yaml,confidence_threshold=0.6,input_size=img_size)

if __name__ == "__main__":
    main()

