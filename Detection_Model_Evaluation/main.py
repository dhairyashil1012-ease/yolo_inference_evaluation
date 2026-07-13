import argparse
import subprocess
import sys
# python3 main.py -m pt
SCRIPT_MAP = {
    "pt": "src/detection_pt_model.py",
    "onnx": "src/detection_onnx_model.py",
    "engine": "src/detection_engine_model.py",
}

parser = argparse.ArgumentParser(
    description="YOLO Classification Inference"
)

parser.add_argument(
    "-m",
    choices=SCRIPT_MAP.keys(),
    required=True,
    help="Inference backend"
)

args = parser.parse_args()

script = SCRIPT_MAP[args.m]

subprocess.run([sys.executable, script], check=True)