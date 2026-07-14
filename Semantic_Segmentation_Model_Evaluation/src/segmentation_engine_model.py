import os
import time
import cv2
import torch
import subprocess
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
IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]

# MODEL_NAME = config["PATHS"]["ENGINE_MODEL_NAME"]
MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]
MODEL_PATH = MODEL_DIR / MODEL_NAME

INPUT_SIZE = (
    config.getint("MODEL", "INPUT_HEIGHT"),
    config.getint("MODEL", "INPUT_WIDTH"),
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)


print("=" * 60)
print("CONFIGURATION")
print("=" * 60)
print(f"Model Path : {MODEL_PATH}")
print(f"Image Path : {IMAGE_DIR}")
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
        "--minShapes=images:1x3x1024x2048",
        "--optShapes=images:4x3x1024x2048",
        "--maxShapes=images:16x3x1024x2048",
        "--saveEngine=yolo26n-sem.engine"
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

    if isinstance(err,tuple):
        err=err[0]

    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Error : {err}")
    


def preprocess_engine(image_folder,input_size):


    # Convert integer to (H, W)

    input_size = input_size

    transform = transforms.Compose([transforms.Resize(input_size),transforms.ToTensor(),])

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
    success = context.execute_async_v3(stream)

    if not success:
        raise RuntimeError("TensorRT inference failed.")

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





def postprocess_engine(outputs,image_files,image_folder):

 
    for img_idx, image_name in enumerate(image_files):

        image_path = os.path.join(image_folder, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Unable to read {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W = image.shape[:2]


        detections = outputs[img_idx]
        
        # Convert scores to class IDs
        mask = np.argmax(detections, axis=0).astype(np.uint8)

        # Resize mask if model output size differs from original image
        if mask.shape != (H, W):
            mask = cv2.resize(
                mask,
                (W, H),
                interpolation=cv2.INTER_NEAREST)
    
        # Create colored mask
        color_mask = cv2.applyColorMap(mask * 15, cv2.COLORMAP_JET)
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

    
        # Overlay
        blended = cv2.addWeighted(image,0.7,color_mask,0.3,0)
        
        filename=image_name

        save_path = OUTPUT_DIR / filename

        cv2.imwrite(str(save_path), blended)

        print(f"Saved : {save_path}")

    print("\nAll output images saved successfully.")
    
    




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

    postprocess_engine(
        outputs=predictions,
        image_files=image_files,
        image_folder=IMAGE_DIR,
    )


if __name__ == "__main__":
    main()


