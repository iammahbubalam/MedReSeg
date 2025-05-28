def load_reasoning_prompts(json_path='/kaggle/input/res-seg-dataset-448448/processed_data/glioma_reasoning_prompts.json'):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            prompts = [item["prompt"] for item in data["prompts"]]
            return prompts
    except Exception as e:
        print(f"Error loading reasoning prompts: {e}")
        # Fallback prompts if file not found
        return ["Segment the regions of abnormal signal intensity that may represent tumor tissue.",
                "Identify regions that might show abnormalities in this brain MRI."]
