import yaml
from pathlib import Path

def extract_labels_to_txt(yaml_path, txt_output_path):
    # 1. Load and parse the YAML file content
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    # 2. Extract the names block from the dataset dict
    class_names_dict = data.get("names", {})
    
    if not class_names_dict:
        raise ValueError("Could not find a 'names' key inside the provided YAML file.")
        
    # 3. Sort by the numeric class key ID to guarantee correct ordering
    sorted_keys = sorted(class_names_dict.keys())
    ordered_labels = [class_names_dict[key] for key in sorted_keys]
    
    # 4. Write out the plain text labels row by row
    with open(txt_output_path, "w", encoding="utf-8") as f_out:
        for label in ordered_labels:
            f_out.write(f"{label}\n")
            
    print(f"Successfully extracted {len(ordered_labels)} labels into: {txt_output_path}")

# ==========================================
# Run Execution Setup
# ==========================================
# Point this to where your source yaml file lives
SOURCE_YAML = Path("/home/easemyai/Downloads/yolo_inference_evaluation/Detection_Model_Evaluation/detection-yaml/coco.yaml") 
OUTPUT_TXT = Path("/home/easemyai/Downloads/yolo_inference_evaluation/Detection_Model_Evaluation/detection-yaml/label.txt")

# Ensure the parent folders exist
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)

# Run the extraction function
if __name__ == "__main__":
    # For testing, creating a mock file if it doesn't exist yet
    if not SOURCE_YAML.exists():
        print(f"Please put your YAML contents inside a file at: {SOURCE_YAML}")
    else:
        extract_labels_to_txt(SOURCE_YAML, OUTPUT_TXT)