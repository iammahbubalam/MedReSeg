from torch.utils.data import DataLoader
import torch
import os
import random
from PIL import Image
from dataset import MedicalSegmentationDataset
from util.util import load_reasoning_prompts

# Create data loaders with evenly distributed reasoning prompts
def create_dataloaders(image_dir, mask_dir, batch_size=8, val_split=0.2, transform=None):
    # Load all reasoning prompts
    reasoning_prompts = load_reasoning_prompts()
    
    # Create dataset with evenly distributed prompts
    dataset = MedicalSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform,
        reasoning_prompts=reasoning_prompts
    )
    
    # Split into train and validation while maintaining prompt distribution
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    val_size = int(val_split * dataset_size)
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Created dataloaders with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print(f"Using {len(reasoning_prompts)} unique reasoning prompts distributed evenly across the dataset")
    
    return train_loader, val_loader

