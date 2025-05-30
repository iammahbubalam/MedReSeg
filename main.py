

from data.dataloader import create_dataloaders
from model.med_clip_unet import MedCLIPUNet 
from train import train_model
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
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        batch_size=4
    )
    
    # Initialize model with img_size=448 to match input images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedCLIPUNet(num_classes=1, img_size=448).to(device)
    print(f"Model initialized on device: {device}")
    
    # Train model with multi-task loss
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=1,
        learning_rate=1e-4,
        batch_size=4
    )
    
    # Save the trained model
    torch.save(trained_model.state_dict(), "med_clip_unet_transformer_reasoning.pt")
    print("Model training complete and saved to med_clip_unet_transformer_reasoning.pt")