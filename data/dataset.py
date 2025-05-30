from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from skimage.exposure import equalize_adapthist
from PIL import UnidentifiedImageError


# Helper collate function to skip None samples
def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # Or however your training loop handles an empty batch
    return torch.utils.data.dataloader.default_collate(batch)


class MedicalSegmentationDataset(Dataset):
    def __init__(self, csv_file_path, data_base_dir, transform=None):
        """
        Args:
            csv_file_path (string): Path to the csv file with annotations.
            data_base_dir (string): Base directory for image and mask paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file_path)
        self.data_base_dir = data_base_dir
        self.transform = transform

        # Ensure 'question' column exists and fill NaN with a default or raise error
        if 'question' not in self.data_frame.columns:
            raise ValueError("CSV file must contain a 'question' column.")
        self.data_frame['question'] = self.data_frame['question'].fillna("No question provided.")
        
        # Ensure other required columns exist
        required_columns = ['image_path', 'mask_path', 'image_type', 'ID_details']
        for col in required_columns:
            if col not in self.data_frame.columns:
                raise ValueError(f"CSV file must contain a '{col}' column.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_rel_path = self.data_frame.loc[idx, 'image_path']
        mask_rel_path = self.data_frame.loc[idx, 'mask_path']
        image_type = self.data_frame.loc[idx, 'image_type']
        id_details = str(self.data_frame.loc[idx, 'ID_details']).replace('_'," ")
        question = self.data_frame.loc[idx, 'question']

        img_full_path = os.path.join(self.data_base_dir, img_rel_path)
        mask_full_path = os.path.join(self.data_base_dir, mask_rel_path)

        # Check image file validity (exists, not 0KB, and loadable)
        try:
            if not os.path.exists(img_full_path):
                print(f"WARNING: Image file does not exist: {img_full_path} for index {idx}. Skipping sample.")
                return None
            if os.path.getsize(img_full_path) == 0:
                print(f"WARNING: Image file is 0KB: {img_full_path} for index {idx}. Skipping sample.")
                return None
            image = Image.open(img_full_path).convert('RGB')
        except FileNotFoundError: # Defensive, os.path.exists should catch this
            print(f"ERROR: Image file not found (during open): {img_full_path} for index {idx}. Skipping sample.")
            return None
        except UnidentifiedImageError:
            print(f"ERROR: Cannot identify image file (corrupted/invalid format): {img_full_path} for index {idx}. Skipping sample.")
            return None
        except Exception as e:
            print(f"ERROR: Failed to load image {img_full_path} for index {idx} due to: {e}. Skipping sample.")
            return None

        # Check mask file validity (exists, not 0KB, and loadable)
        try:
            if not os.path.exists(mask_full_path):
                print(f"WARNING: Mask file does not exist: {mask_full_path} for index {idx}. Skipping sample.")
                return None
            if os.path.getsize(mask_full_path) == 0:
                print(f"WARNING: Mask file is 0KB: {mask_full_path} for index {idx}. Skipping sample.")
                return None
            mask = Image.open(mask_full_path).convert('L')
        except FileNotFoundError: # Defensive
            print(f"ERROR: Mask file not found (during open): {mask_full_path} for index {idx}. Skipping sample.")
            return None
        except UnidentifiedImageError:
            print(f"ERROR: Cannot identify mask file (corrupted/invalid format): {mask_full_path} for index {idx}. Skipping sample.")
            return None
        except Exception as e:
            print(f"ERROR: Failed to load mask {mask_full_path} for index {idx} due to: {e}. Skipping sample.")
            return None
        
        # Apply transformations if any
        # Store original image size if needed by transforms or later use
        # original_size = image.size 

        if self.transform:
            # User-provided transform is responsible for all processing,
            # including resizing if needed.
            # It's common for transforms to handle image and mask differently
            # or be applied sequentially. For simplicity here, we assume
            # self.transform can correctly process both.
            # A more robust way might be to have self.image_transform and self.mask_transform.
            transformed_image = self.transform(image)
            # If the same transform is used for mask, ensure it's appropriate.
            # Often, masks need simpler transforms (e.g., ToTensor, Resize with NEAREST).
            transformed_mask = self.transform(mask) 
        else:
            # Default preprocessing: Resize image, apply CLAHE, then convert to tensor.
            
            # Preprocess image
            # 1. Resize image
            image_resized_pil = image.resize((256, 256), Image.Resampling.BILINEAR)
            
            # 2. Convert to NumPy array (uint8, H, W, C)
            image_resized_np = np.array(image_resized_pil)
            
            # 3. Apply CLAHE. Input is uint8 [0,255]. Output is float64 [0,1].
            # equalize_adapthist handles RGB images by applying to each channel.
            image_clahe_np = equalize_adapthist(image_resized_np, clip_limit=0.03) # Result is float64 in [0,1]
            
            # 4. Convert to PyTorch tensor (float32, C, H, W)
            # Permute H, W, C to C, H, W. .float() converts float64 to float32.
            transformed_image = torch.from_numpy(image_clahe_np).permute(2, 0, 1).float()

            # Preprocess mask
            # Resize mask to match image dimensions using NEAREST interpolation
            mask_resized = mask.resize((256, 256), Image.Resampling.NEAREST)
            mask_np = np.array(mask_resized) # H, W
            # Add channel dimension and normalize
            transformed_mask = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0
            # For binary masks (0 or 255), this normalization results in 0.0 or 1.0.
            # If mask values are class labels (0, 1, 2, ...), normalization might not be desired
            # or should be handled differently (e.g., just convert to LongTensor without division).
            # Assuming binary mask [0, 255] scaled to [0, 1] for now.
        
        sample = {
            "image": transformed_image,
            "mask": transformed_mask,
            "image_type": image_type,
            "id_details": id_details,
            "prompt": question,
            "image_path": img_full_path, # Storing full path for reference
            "mask_path": mask_full_path   # Storing full path for reference
            # "original_size": original_size 
        }
        
        return sample

# Example usage (optional, for testing)
if __name__ == '__main__':
    # Define base directory assuming this script is in MedReSeg/data
    # and the dataset CSV and images/masks are in MedReSeg/dataset/SAMed2Dv1/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_base_dir = os.path.abspath(os.path.join(current_script_dir, '..')) # MedReSeg directory
    
    csv_path = os.path.join(project_base_dir, 'dataset', 'SAMed2Dv1', 'SAMed2D_image_metadata_per_mask_with_questions.csv')
    # The data_base_dir should point to the directory where 'images/' and 'masks/' folders from the CSV are located
    # In this case, it's 'MedReSeg/dataset/SAMed2Dv1/'
    data_root_dir = os.path.join(project_base_dir, 'dataset', 'SAMed2Dv1')

    if not os.path.exists(csv_path):
        print(f"CSV file not found at: {csv_path}")
    elif not os.path.isdir(data_root_dir):
         print(f"Data root directory not found at: {data_root_dir}")
    else:
        print(f"Loading dataset from CSV: {csv_path}")
        print(f"Using data base directory: {data_root_dir}")
        
        from torchvision import transforms
        # Example transform (e.g., from torchvision)
        # transform = transforms.Compose([
        #     transforms.Resize((256, 256)), # Ensure resize is appropriate for your model
        #     transforms.ToTensor() # Converts PIL image to tensor and scales to [0,1]
        # ])
        # dataset = MedicalSegmentationDataset(csv_file_path=csv_path, data_base_dir=data_root_dir, transform=transform)
        
        # Without external transforms, using default tensor conversion and CLAHE
        dataset = MedicalSegmentationDataset(csv_file_path=csv_path, data_base_dir=data_root_dir)

        if len(dataset) > 0:
            print(f"Dataset loaded successfully with {len(dataset)} samples (potential skips not accounted for in this count).")
            
            # Count 0KB image files
            zero_kb_image_count = 0
            print("\nChecking for 0KB image files...")
            for i in range(len(dataset.data_frame)):
                img_rel_path = dataset.data_frame.loc[i, 'image_path']
                img_full_path = os.path.join(dataset.data_base_dir, img_rel_path)
                if os.path.exists(img_full_path):
                    if os.path.getsize(img_full_path) == 0:
                        zero_kb_image_count += 1
                        # print(f"Found 0KB image: {img_full_path} at index {i}") # Optional: print each 0KB file
                else:
                    # This case should ideally not happen if CSV is accurate and files are present
                    # print(f"Image file listed in CSV not found: {img_full_path} at index {i}")
                    pass # Or handle as an error/warning
            
            print(f"Total 0KB image files found: {zero_kb_image_count}")

            # Test getting a single sample
            print("\nTesting single sample retrieval (index 500):")
            sample_idx = 500
            # Create a dummy 0KB file for testing the skip logic if needed
            # For example, find dataset.data_frame.loc[sample_idx, 'image_path'] and make it 0KB
            # Ensure you have a backup or do this on a test copy.
            # e.g., open(os.path.join(data_root_dir, dataset.data_frame.loc[sample_idx, 'image_path']), 'w').close()

            sample = dataset[sample_idx] 
            
            if sample:
                print(f"\nSample {sample_idx} details:")
                print(f"  Image shape: {sample['image'].shape}")
                print(f"  Mask shape: {sample['mask'].shape}")
                print(f"  Image type: {sample['image_type']}")
                print(f"  ID details: {sample['id_details']}")
                print(f"  Question: {sample['question']}")
                print(f"  Image path: {sample['image_path']}")
                print(f"  Mask path: {sample['mask_path']}")
            else:
                print(f"\nSample {sample_idx} was skipped (returned None).")

            # Example of using DataLoader with the custom collate_fn
            # from torch.utils.data import DataLoader
            # print("\nTesting with DataLoader and collate_fn_skip_none:")
            # dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn_skip_none, num_workers=0)
            # for i, batch_data in enumerate(dataloader):
            #     if batch_data is None:
            #         print(f"Batch {i} was empty after skipping None samples.")
            #         continue
            #     print(f"Batch {i} loaded. Image batch shape: {batch_data['image'].shape}")
            #     if i >= 2: # Print first few batches
            #         break 
        else:
            print("Dataset is empty or failed to load.")
