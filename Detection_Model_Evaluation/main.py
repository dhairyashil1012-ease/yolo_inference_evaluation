import os
import sys

# Wrap imports inside functional execution blocks
def run_pt():
    from src import detection_pt_model
    detection_pt_model.main()

def run_onnx():
    from src import detection_onnx_model
    detection_onnx_model.main()

def run_engine():
    from src import detection_engine_model
    detection_engine_model.main()



# Map the strings to the execution functions
MODEL_MAP = {
    "pt": run_pt,
    "onnx": run_onnx,
    "engine": run_engine,
}



def main():
    backend = os.getenv("MODEL_TYPE", "pt").lower()


    # if backend not in MODEL_MAP:
    #     print(f"Invalid MODEL_TYPE: {backend}")
    #     print(f"Supported values: {', '.join(MODEL_MAP.keys())}")
    #     sys.exit(1)

    print("=" * 50)
    print(f"Selected Backend : {backend}")
    print("=" * 50)


    b=MODEL_MAP[backend]()
    print(f"value of B:{b}")
if __name__ == "__main__":
    main()
