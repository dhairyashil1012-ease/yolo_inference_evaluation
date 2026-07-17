# import os
# import subprocess
# import time
# import cv2
# import torch
# import sys
# import numpy as np
# import platform

# from pathlib import Path
# from PIL import Image
# from configparser import ConfigParser

# import tensorrt as trt
# from cuda.bindings import runtime as cudart

# import torchvision.transforms as transforms
# from ultralytics import YOLO
# PROJECT_DIR = Path.cwd()

# config = ConfigParser()
# config.read("config.txt")

# # -----------------------------
# # Paths
# # -----------------------------
# MODEL_DIR = PROJECT_DIR / config["PATHS"]["MODEL_DIR"]
# IMAGE_DIR = PROJECT_DIR / config["PATHS"]["IMAGE_DIR"]
# OUTPUT_DIR = PROJECT_DIR / config["PATHS"]["ENGINE_OUTPUT_DIR"]
# LABEL_DIR = PROJECT_DIR / config["PATHS"]["YAML_DIR"]

# MODEL_NAME = config["PATHS"]["ONNX_MODEL_NAME"]
# MODEL_NAME_PT = config["PATHS"]["PT_MODEL_NAME"]
# LABEL_NAME = "label.txt" 
# MODEL_PATH = MODEL_DIR / MODEL_NAME
# LABEL_PATH = LABEL_DIR / LABEL_NAME

# INPUT_SIZE = (
#     config.getint("MODEL", "INPUT_HEIGHT"),
#     config.getint("MODEL", "INPUT_WIDTH"),
# )

# MODEL_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# IMAGE_DIR.mkdir(parents=True, exist_ok=True)
# LABEL_DIR.mkdir(parents=True, exist_ok=True)

# from pathlib import Path

# def load_model(model_dir):
#     pathlist = Path(model_dir).glob("**/*.pt")
#     filelist = sorted([str(file) for file in pathlist])

#     if len(filelist) == 0:
#         raise FileNotFoundError("No ONNX model found.")

#     pt_model = filelist[-1]
#     return pt_model


# # def load_model(model_path):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Running on : {device}")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = YOLO("/home/easemyai/Documents/yolo_inference_evaluation/Classification_Model_Evaluation/Classification_Models/yolo26n-cls.pt")
# model.to(device)
# g=f"{model.info()[3]:.2f}"
# print(g)

# # model = load_model(MODEL_DIR)
# # m=load_model(model)
# # print(model)
# # print(m.info())


import onnxruntime as ort

# 1. Load the model and create an inference session
model_path = "/home/easemyai/Documents/yolo_inference_evaluation/Classification_Model_Evaluation/Classification_Models/yolo26n-cls.onnx"
# session = ort.InferenceSession(model_path)

# 2. Retrieve model metadata
# model_meta = session.get_modelmeta()

# Assuming the queried attribute is a custom map (like version or specific metrics)
# or a specific metadata array, e.g., model_meta.custom_metadata_map['some_key']
# or model_meta.metadata_props[index]

# 3. Access your target and format it
# import onnx
# import ast

# # 1. Load the ONNX model
# model = onnx.load(model_path)

# # 2. Extract metadata properties
# props = { p.key : p.value for p in model.metadata_props }

# for p in model.metadata_props:
#     print(p.value)


# # # 3. Print the number of labels and the actual labels
# if 'names' in props:
#     labels = ast.literal_eval(props['names'])
#     print(f"Total Labels: {len(labels)}")
#     print(f"Labels: {labels}")
# else:
#     print("No labels found in model metadata.")
# for value in props.values():
#     print(value)
# for v in props.values():
#     print(v)

# import onnx
# from onnx import numpy_helper

# # model_path = "your_model.onnx"
# model = onnx.load(model_path)

# # Sum the size of all initializers (weights/biases) in the graph
# total_params = sum(numpy_helper.to_array(initializer).size for initializer in model.graph.initializer)

# print(f"{total_params:,}")


import onnx_tool

def get_onnx_gflops(model_path):
    try:
        # Generate model profile and get total MACs 
        profile_results = onnx_tool.model_profile(model_path)
        macs = profile_results[0]
        gflops = (macs * 2) / 1e9  # Convert MACs to GFLOPs
        return f"{gflops:.2f}"
    except Exception:
        return "Unknown"

# Usage
model_gflops = get_onnx_gflops("/home/easemyai/Documents/yolo_inference_evaluation/Classification_Model_Evaluation/Classification_Models/yolo26n-cls.onnx")
print(model_gflops)
