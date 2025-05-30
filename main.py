

from data.dataloader import create_dataloader
from components.medclip_unet import MedCLIPUNet 
from traintest.train import train_model
import torch
import os
import random
import numpy as np



if __name__ == "__main__":
    # Set paths to processed data
    IMAGE_DIR = "/kaggle/input/res-seg-dataset-448448/processed_data/enhanced_images"
    MASK_DIR = "/kaggle/input/res-seg-dataset-448448/processed_data/masks"
    
    print("Starting medical image segmentation model training with Transformer Decoder Fusion")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Mask directory: {MASK_DIR}")
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.abspath(os.path.join(current_script_dir, '..')) 
    
    csv_path = os.path.join(project_base_dir, 'dataset', 'SAMed2Dv1', 'SAMed2D_image_metadata_per_mask_with_questions.csv')
    data_root_dir = os.path.join(project_base_dir, 'dataset', 'SAMed2Dv1')

    print(f"Attempting to load data from: {csv_path} with base directory: {data_root_dir}")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at: {csv_path}")
    elif not os.path.isdir(data_root_dir):
        print(f"Error: Data root directory not found at: {data_root_dir}")
    else:
        train_dataloader, train_dataset = create_dataloader(
            csv_file_path=csv_path,
            data_base_dir=data_root_dir,
            batch_size=4,
            shuffle=True,
            transform=None, 
            num_workers=0
        )
    
    # Initialize model with img_size=448 to match input images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedCLIPUNet(num_classes=1, img_size=448).to(device)
    print(f"Model initialized on device: {device}")
    
    # Train model with multi-task loss
    trained_model = train_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_loader,
        device=device,
        num_epochs=1,
        learning_rate=1e-4,
        batch_size=4
    )
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "med_clip_unet_transformer_reasoning.pt")
    print("Model training complete and saved to med_clip_unet_transformer_reasoning.pt")