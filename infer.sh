#!/bin/bash

# Script to run the MedCLIPUNet inference

# --- Configuration ---
# TODO: Replace these placeholder paths with your actual paths
MODEL_PATH="model_best.pth.tar"
CSV_PATH="dataset_subset\subset_image_to_mask_mapping_with_metadata.csv"
DATA_ROOT_DIR="dataset_subset"

# Optional: Adjust these parameters if needed
NUM_CLASSES=1
IMG_SIZE=256
BATCH_SIZE=1
NUM_WORKERS=0 # Use 0 for main process, or >0 for multiprocessing
DEVICE="cuda" # "cuda" or "cpu"
PLOT_SAVE_DIR="inference_plots"
NUM_PLOTS_TO_SAVE=100

# --- Activate your Python environment if you have one ---
# e.g., source /path/to/your/venv/bin/activate
# e.g., conda activate your_env_name

# --- Run the inference script ---
echo "Starting inference..."
python infer.py \
    --model_path "${MODEL_PATH}" \
    --csv_path "${CSV_PATH}" \
    --data_root_dir "${DATA_ROOT_DIR}" \
    --num_classes ${NUM_CLASSES} \
    --img_size ${IMG_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --device "${DEVICE}" \
    --plot_save_dir "${PLOT_SAVE_DIR}" \
    --num_plots_to_save ${NUM_PLOTS_TO_SAVE}

echo "Inference script finished."

# --- Deactivate environment if you activated one ---
# e.g., deactivate
# e.g., conda deactivate
