import os
import sys
from src import (classification_pt_model,classification_engine_model,classification_onnx_model)
# Wrap imports inside functional execution blocks

def run_pt():
    classification_pt_model.main()

def run_onnx():
    classification_onnx_model.main()

def run_engine():
    classification_engine_model.main()



# Map the strings to the execution functions
MODEL_MAP = {
    "pt": run_pt,
    "onnx": run_onnx,
    "engine": run_engine,
}



def main():
    backend = os.getenv("MODEL_TYPE", "pt").lower()


    if backend not in MODEL_MAP:
        print(f"Invalid MODEL_TYPE: {backend}")
        print(f"Supported values: {', '.join(MODEL_MAP.keys())}")
        sys.exit(1)

    print("=" * 50)
    print(f"Selected Backend : {backend}")
    print("=" * 50)
    #
    b=MODEL_MAP[backend]()

if __name__ == "__main__":
    main()
