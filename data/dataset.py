
from torch.utils.data import Dataset
import os
import random
import numpy as np
from PIL import Image
import torch
from util.util import load_reasoning_prompts


class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, reasoning_prompts=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Collect image filenames
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        
        # Load reasoning prompts
        if reasoning_prompts is None:
            self.reasoning_prompts = load_reasoning_prompts()
        else:
            self.reasoning_prompts = reasoning_prompts
        
        # Evenly distribute prompts across all images
        self.image_to_prompt = {}
        num_images = len(self.image_files)
        num_prompts = len(self.reasoning_prompts)
        
        # Create a list with prompts repeated as needed to cover all images
        prompt_assignments = []
        while len(prompt_assignments) < num_images:
            prompt_assignments.extend(self.reasoning_prompts)
        prompt_assignments = prompt_assignments[:num_images]
        
        # Shuffle the assignments to avoid having sequential images with the same prompt patterns
        random.shuffle(prompt_assignments)
        
        # Assign prompts to images
        for i, img_file in enumerate(self.image_files):
            self.image_to_prompt[img_file] = prompt_assignments[i]
        
        # Statistics about prompt distribution
        prompt_counts = {}
        for prompt in self.reasoning_prompts:
            prompt_counts[prompt] = 0
        for prompt in prompt_assignments:
            prompt_counts[prompt] += 1
        
        # Log statistics
        min_count = min(prompt_counts.values())
        max_count = max(prompt_counts.values())
        print(f"Prompt distribution - Min: {min_count}, Max: {max_count}, Avg: {num_images/num_prompts:.2f}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # Same filename for mask
        
        # Open as RGB
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale for mask
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            # Default conversion to tensors
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        
        # Get the pre-assigned prompt for this image
        prompt = self.image_to_prompt[img_name]
        
        return {
            "image": image, 
            "mask": mask, 
            "prompt": prompt,
            "filename": img_name
        }
