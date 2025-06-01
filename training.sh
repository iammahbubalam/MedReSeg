#!/bin/bash

# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate 
# Example: source venv/bin/activate (if venv is in the project root)
# If you are using Conda:
# conda activate your_env_name

# --- Training Configuration ---
SEED=42
# Adjust these paths if your dataset is located elsewhere
CSV_PATH="datasets/image_to_mask_mapping_with_metadata.csv"
DATA_ROOT_DIR="datasets/"
VAL_SIZE=0.2
NUM_WORKERS=0 # Set to 0 for Windows, or a positive integer for Linux/macOS if beneficial

NUM_CLASSES=1
IMG_SIZE=256

BATCH_SIZE=4
NUM_EPOCHS=20 # Increased epochs for a more substantial run
LEARNING_RATE=0.0001
GRAD_ACC_STEPS=2

EPOCH_CKPT_DIR="checkpoints/epochs"
BEST_CKPT_DIR="checkpoints/best"
MODEL_SAVE_PATH="trained_models/medclip_unet_run1.pt"


LOG_FILE="logs/training_run1.log" # Example: specific log file for this run
LOG_LEVEL="INFO"                  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Execute the Python Training Script ---
echo "Starting training with the following configuration:"
echo "SEED: $SEED"
echo "CSV_PATH: $CSV_PATH"
echo "DATA_ROOT_DIR: $DATA_ROOT_DIR"
echo "VAL_SIZE: $VAL_SIZE"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "NUM_CLASSES: $NUM_CLASSES"
echo "IMG_SIZE: $IMG_SIZE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "GRAD_ACC_STEPS: $GRAD_ACC_STEPS"
echo "EPOCH_CKPT_DIR: $EPOCH_CKPT_DIR"
echo "BEST_CKPT_DIR: $BEST_CKPT_DIR"
echo "MODEL_SAVE_PATH: $MODEL_SAVE_PATH"
echo "----------------------------------------------------"

python main.py \
    --seed $SEED \
    --csv_path "$CSV_PATH" \
    --data_root_dir "$DATA_ROOT_DIR" \
    --val_size $VAL_SIZE \
    --num_workers $NUM_WORKERS \
    --num_classes $NUM_CLASSES \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --grad_accumulation_steps $GRAD_ACC_STEPS \
    --epoch_checkpoint_dir "$EPOCH_CKPT_DIR" \
    --best_checkpoint_dir "$BEST_CKPT_DIR" \
    --model_save_path "$MODEL_SAVE_PATH"\
    --log_file "$LOG_FILE" \
    --log_level "$LOG_LEVEL"

echo "----------------------------------------------------"
echo "Training script finished."