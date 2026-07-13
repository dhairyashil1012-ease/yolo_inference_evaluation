import os
import time
import cv2
import subprocess
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from configparser import ConfigParser
import tensorrt as trt
import torchvision.transforms as transforms
from cuda.bindings import runtime as cudart




PROJECT_DIR = Path.cwd()

config = ConfigParser()
config.read(PROJECT_DIR / "config.txt")



# ==========================================================
# PATHS
# ==========================================================
MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
IMAGE_DIR = (PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]).resolve() # Resolve relative path safely
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]   
LABEL_NAME = config["PATHS"]["LABEL_NAME"]


# MODEL_NAME = config["PATHS"]["ENGINE_MODEL_NAME"]
MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]

MODEL_PATH = MODEL_DIR / MODEL_NAME
LABEL_PATH = LABEL_DIR / LABEL_NAME

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

# Directories setup (Fixed undefined YAML_DIR bug)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True) 

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"Label file not found at: {LABEL_PATH}")

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
print(f"Label Path : {LABEL_PATH}")
print(f"Output Dir : {OUTPUT_DIR}")
print(f"Input Size : {INPUT_SIZE}")
print("=" * 60)





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
        "--minShapes=images:1x3x640x640",
        "--optShapes=images:4x3x640x640",
        "--maxShapes=images:16x3x640x640",
        "--saveEngine=yolo26n.engine"
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
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])

    image_files = sorted([
        file for file in os.listdir(image_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in the specified folder: {image_folder}")

    batch_images = []
    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image)
        batch_images.append(tensor)

    batch_tensor = torch.stack(batch_images, dim=0)
    batch_numpy = batch_tensor.numpy().astype(np.float32)

    print(f"Number of Images : {len(image_files)}")
    print(f"Input Shape      : {batch_numpy.shape}")

    return batch_numpy, image_files





def inference_engine(batch_numpy, engine_model):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(engine_model, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

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
    check_cuda(cudart.cudaMemcpyAsync(
        d_input, host_input.ctypes.data, host_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream
    ))

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    success = context.execute_async_v3(stream)
    if not success:
        raise RuntimeError("TensorRT inference failed.")

    # Device -> Host
    check_cuda(cudart.cudaMemcpyAsync(
        host_output.ctypes.data, d_output, host_output.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream
    ))

    check_cuda(cudart.cudaStreamSynchronize(stream))
    end = time.perf_counter()
    inference_time = end - start

    batch_size = batch_numpy.shape[0]
    print("\n" + "=" * 60)
    print("TENSORRT PERFORMANCE")
    print("=" * 60)
    print(f"Batch Size           : {batch_size}")
    print(f"Batch Inference Time : {inference_time:.3f} seconds")
    print(f"Per Image Latency    : {inference_time/batch_size:.3f} seconds")

    cudart.cudaFree(d_input)
    cudart.cudaFree(d_output)
    cudart.cudaStreamDestroy(stream)

    return host_output

def postprocess_engine(outputs, image_files, image_folder, label_path, confidence_threshold=0.6, input_size=640):
    if isinstance(input_size, int):
        input_size = (input_size, input_size)

    input_h, input_w = input_size

    with open(label_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

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

        for det in detections:
            x1, y1, x2, y2, score, cls = det

            if score < confidence_threshold:
                continue

            cls = int(cls)
            x1 = max(0, min(int(x1 * scale_x), W - 1))
            y1 = max(0, min(int(y1 * scale_y), H - 1))
            x2 = max(0, min(int(x2 * scale_x), W - 1))
            y2 = max(0, min(int(y2 * scale_y), H - 1))

            # Defensive bound checking for class index strings
            class_label = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            label = f"{class_label} {score:.2f}"

            # Keep operations uniformly in BGR since cv2.imread loads it as BGR
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        # Save image ONCE per file, outside the bounding box loop
        save_path = OUTPUT_DIR / image_name
        cv2.imwrite(str(save_path), image)
        print(f"Saved : {save_path}")

    print("\nAll output images processed.")

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

    # Fixed: Passing variable names that actually exist
    postprocess_engine(
        outputs=predictions,
        image_files=image_files,
        image_folder=IMAGE_DIR,
        label_path=LABEL_PATH, 
        confidence_threshold=0.6,
        input_size=INPUT_SIZE,
    )

if __name__ == "__main__":
    main()