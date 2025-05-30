from torch.utils.data import DataLoader
import os
import random
import torch # Add torch import for torch.utils.data.Subset

if __name__ == '__main__' and (__package__ is None or __package__ == ''):
    # If run as a script (e.g., python data/dataloader.py)
    # and not as a module (e.g., python -m data.dataloader)
    import sys
    # Get the absolute path of the project's root directory (MedReSeg)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Add the project root to the Python path if it's not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Now, import using the absolute path from the project root
    from data.dataset import MedicalSegmentationDataset, collate_fn_skip_none
else:
    # If imported as a module within a package, use the original relative import
    from .dataset import MedicalSegmentationDataset, collate_fn_skip_none



def create_dataloader(csv_file_path, 
                      data_base_dir, 
                      batch_size, 
                      shuffle=True, 
                      transform=None, 
                      num_workers=0,
                      pin_memory=True,
                      num_samples=None): # Added num_samples parameter
    """
    Creates a DataLoader for the MedicalSegmentationDataset.

    Args:
        csv_file_path (string): Path to the csv file with annotations.
        data_base_dir (string): Base directory for image and mask paths.
        batch_size (int): How many samples per batch to load.
        shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to True.
        transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        num_workers (int, optional): How many subprocesses to use for data loading. 
                                     0 means that the data will be loaded in the main process. Defaults to 0.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory
                                     before returning them. Defaults to True.
        num_samples (int or str, optional): Number of samples to load. 
                                            If an int, loads that many samples. 
                                            If "full" or None, loads all samples. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: The configured DataLoader.
        MedicalSegmentationDataset or torch.utils.data.Subset: The dataset instance (or subset).
    """
    # Instantiate the dataset
    full_dataset = MedicalSegmentationDataset(
        csv_file_path=csv_file_path,
        data_base_dir=data_base_dir,
        transform=transform
    )

    if len(full_dataset) == 0:
        print(f"Warning: Dataset created from {csv_file_path} is empty.")
        return None, full_dataset
    
    dataset_to_load = full_dataset
    if num_samples is not None and isinstance(num_samples, int) and num_samples > 0:
        if num_samples < len(full_dataset):
            print(f"Loading a subset of {num_samples} samples.")
            # Create a subset of the dataset
            # Ensure indices are within the range of the dataset
            # If shuffle is True for the dataloader, the subset itself doesn't strictly need to be shuffled here,
            # but if you want a specific random subset each time (even with shuffle=False in DataLoader),
            # you might want to shuffle indices before selecting.
            # For simplicity, taking the first num_samples.
            # If specific random N samples are needed, one could do:
            # indices = random.sample(range(len(full_dataset)), num_samples)
            # dataset_to_load = torch.utils.data.Subset(full_dataset, indices)
            dataset_to_load = torch.utils.data.Subset(full_dataset, list(range(num_samples)))
        else:
            print(f"num_samples ({num_samples}) is >= dataset size ({len(full_dataset)}). Loading full dataset.")
    elif isinstance(num_samples, str) and num_samples.lower() == "full":
        print("Loading full dataset as per 'num_samples' parameter.")
    elif num_samples is not None:
        print(f"Warning: Invalid value for num_samples: {num_samples}. Loading full dataset.")


    # Create the DataLoader
    dataloader = DataLoader(
        dataset_to_load, # Use the potentially subsetted dataset
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_skip_none, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader, dataset_to_load

if __name__ == '__main__':
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

        if train_dataloader and train_dataset:
            print(f"DataLoader and Dataset created successfully.")
            print(f"Dataset contains {len(train_dataset)} samples (pre-skip count).")
            print(f"DataLoader will serve batches of size 4.")

            print("\nTesting DataLoader iteration (first few batches):")
            for i, batch_data in enumerate(train_dataloader):
                if batch_data is None: 
                    print(f"Batch {i} was skipped (all samples in batch were None).")
                    continue
                
                print(f"Batch {i} loaded.")
                if 'image' in batch_data and hasattr(batch_data['image'], 'shape'):
                    print(f"  Image batch shape: {batch_data['image'].shape}")
                if 'mask' in batch_data and hasattr(batch_data['mask'], 'shape'):
                    print(f"  Mask batch shape: {batch_data['mask'].shape}")
                
                if i >= 2: 
                    break
            print("\nDataLoader test finished.")
        else:
            print("Failed to create DataLoader or Dataset.")
        
        # Example 1: Load a small subset (e.g., 10 samples)
        print("\n--- Testing with a subset of 10 samples ---")
        subset_dataloader, subset_dataset = create_dataloader(
            csv_file_path=csv_path,
            data_base_dir=data_root_dir,
            batch_size=2, # Smaller batch size for small dataset
            shuffle=False, # Usually False for test/debug subsets
            transform=None, 
            num_workers=0,
            num_samples=10 # Requesting 10 samples
        )

        if subset_dataloader and subset_dataset:
            print(f"Subset DataLoader and Dataset created successfully.")
            print(f"Subset Dataset contains {len(subset_dataset)} samples.")
            print(f"Subset DataLoader will serve batches of size {subset_dataloader.batch_size}.")
            for i, batch_data in enumerate(subset_dataloader):
                if batch_data is None: 
                    print(f"Batch {i} was skipped.")
                    continue
                print(f"Batch {i} loaded. Image batch shape: {batch_data['image'].shape}")
                if i >= 4: # Print a few batches
                    break
            print("\n--- Subset test finished ---\n")

        # Example 2: Load the full dataset (explicitly)
        print("\n--- Testing with the full dataset ---")
        full_train_dataloader, full_train_dataset = create_dataloader(
            csv_file_path=csv_path,
            data_base_dir=data_root_dir,
            batch_size=4,
            shuffle=True,
            transform=None, 
            num_workers=0,
            num_samples="full" # Requesting full dataset
        )

        if full_train_dataloader and full_train_dataset:
            print(f"Full DataLoader and Dataset created successfully.")
            print(f"Full Dataset contains {len(full_train_dataset)} samples (pre-skip count).")
            print(f"Full DataLoader will serve batches of size {full_train_dataloader.batch_size}.")

            print("\nTesting Full DataLoader iteration (first few batches):")
            for i, batch_data in enumerate(full_train_dataloader):
                if batch_data is None: 
                    print(f"Batch {i} was skipped (all samples in batch were None).")
                    continue
                
                print(f"Batch {i} loaded.")
                if 'image' in batch_data and hasattr(batch_data['image'], 'shape'):
                    print(f"  Image batch shape: {batch_data['image'].shape}")
                if 'mask' in batch_data and hasattr(batch_data['mask'], 'shape'):
                    print(f"  Mask batch shape: {batch_data['mask'].shape}")
                
                if i >= 2: 
                    break
            print("\nFull DataLoader test finished.")
        else:
            print("Failed to create Full DataLoader or Dataset.")

