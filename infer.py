import sys
import os
import shutil

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import numpy as np
import pandas as pd # Required by MedicalSegmentationDataset
from PIL import Image # Required by MedicalSegmentationDataset for default transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback

from components.medclip_unet import MedCLIPUNet
from components.loss import MultiTaskLoss
from data.dataset import MedicalSegmentationDataset, collate_fn_skip_none
from torch.utils.data import DataLoader
from monai.metrics.meandice import DiceMetric # Corrected import
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric # Corrected import
# from skimage.exposure import equalize_adapthist # Uncomment if MedicalSegmentationDataset uses it and it's not vendored

def denormalize_image(tensor_image):
    """
    Basic denormalization assuming image tensor is in [0, 1] range and (C, H, W).
    Adjust if more complex normalization (e.g., mean/std) was used.
    """
    img = tensor_image.cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1) # Ensure it's in [0,1]
    return img

def plot_results(image_tensor, true_mask_tensor, pred_mask_tensor, plot_idx, save_dir="inference_plots"):
    """
    Plots the input image, true mask, and predicted mask, then saves the figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    img_to_plot = denormalize_image(image_tensor)
    true_mask_to_plot = true_mask_tensor.cpu().squeeze().numpy()
    # Apply sigmoid to logits and threshold for binary mask
    pred_mask_to_plot = torch.sigmoid(pred_mask_tensor).cpu().squeeze().detach().numpy() > 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_to_plot)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(true_mask_to_plot, cmap='gray')
    axes[1].set_title("True Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_mask_to_plot, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    plt.suptitle(f"Sample {plot_idx}")
    plot_filename = os.path.join(save_dir, f"inference_sample_{plot_idx}.png")
    try:
        plt.savefig(plot_filename)
        print(f"Saved plot to {plot_filename}")
    except Exception as e:
        print(f"Failed to save plot {plot_filename}: {e}")
    plt.close(fig)


def run_inference(args):
    """
    Main function to run the inference process.
    """
    device_str = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # 1. Load Model
    print(f"Loading model from user-provided path: {args.model_path}")
    model = MedCLIPUNet(num_classes=args.num_classes, img_size=args.img_size)
    
    model_file_to_load = args.model_path  # Initialize with the user-provided path

    # Check if the provided path is a directory or a file
    if os.path.isdir(args.model_path):
        print(f"Warning: The provided model_path ('{args.model_path}') is a directory.")
        print(f"torch.load() typically expects a file path to a checkpoint.")
        try:
            files_in_dir = os.listdir(args.model_path)
            print(f"Contents of this directory: {files_in_dir}")

            # Look for .pt files specifically
            pt_files = [f for f in files_in_dir if f.endswith('.pt')]
            pth_files = [f for f in files_in_dir if f.endswith(('.pth', '.pth.tar'))]
            all_model_files = pt_files + pth_files

            if pt_files:
                if 'medclip_unet_run1.pt' in pt_files:
                    model_file_to_load = os.path.join(args.model_path, 'medclip_unet_run1.pt')
                    print(f"Found target model file: '{model_file_to_load}'")
                else:
                    model_file_to_load = os.path.join(args.model_path, pt_files[0])
                    print(f"Using first .pt file found: '{model_file_to_load}'")
            elif pth_files:
                # Fallback to .pth files
                best_checkpoints = [f for f in pth_files if 'model_best' in f]
                if best_checkpoints:
                    model_file_to_load = os.path.join(args.model_path, best_checkpoints[0])
                    print(f"Found best model checkpoint: '{model_file_to_load}'")
                else:
                    model_file_to_load = os.path.join(args.model_path, pth_files[0])
                    print(f"Using first .pth file found: '{model_file_to_load}'")
            else:
                print(f"Error: No model files (.pt, .pth, .pth.tar) found in directory '{args.model_path}'.")
                return
        except OSError as e:
            print(f"Error listing contents of directory '{args.model_path}': {e}")
            return
    else:
        # Direct file path provided
        if not os.path.isfile(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            return

    try:
        print(f"Attempting to load model state from: {model_file_to_load}")
        
        # Check if file exists and get its size for diagnostics
        if not os.path.isfile(model_file_to_load):
            print(f"Error: Model file not found at {model_file_to_load}")
            return
            
        file_size = os.path.getsize(model_file_to_load)
        print(f"Model file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
        
        if file_size < 1000:  # Very small file, likely corrupted
            print(f"Warning: Model file is very small ({file_size} bytes). It may be corrupted.")
        
        # Load the model - try different strategies for .pt files
        print(f"=> Loading model '{model_file_to_load}'")
        
        try:
            # First try: Load as direct state_dict (common for .pt files)
            print("=> Attempting to load as direct state_dict...")
            state_dict = torch.load(model_file_to_load, map_location=device, weights_only=False)
            
            if isinstance(state_dict, dict):
                # Check if it's a nested checkpoint or direct state_dict
                if 'state_dict' in state_dict:
                    print("=> Found nested 'state_dict' key in loaded data")
                    actual_state_dict = state_dict['state_dict']
                    
                    # Print additional info if available
                    if 'epoch' in state_dict:
                        print(f"=> Model was saved after epoch {state_dict['epoch']}")
                    if 'best_metric' in state_dict:
                        print(f"=> Best metric: {state_dict['best_metric']:.4f}")
                else:
                    print("=> Loading as direct state_dict")
                    actual_state_dict = state_dict
                
                # Verify state_dict is valid
                if not isinstance(actual_state_dict, dict) or len(actual_state_dict) == 0:
                    print("=> Error: state_dict is empty or invalid")
                    return
                
                print(f"=> state_dict contains {len(actual_state_dict)} parameter tensors")
                
                # Load state_dict into model
                model.load_state_dict(actual_state_dict)
                print("=> Successfully loaded model state_dict")
                
            else:
                print(f"=> Error: Expected dict, got {type(state_dict)}")
                return
                
        except Exception as load_error:
            print(f"=> Error loading model: {load_error}")
            
            # Provide specific guidance for common issues
            error_str = str(load_error).lower()
            if "size mismatch" in error_str:
                print("=> This appears to be a model architecture mismatch.")
                print("=> Verify that --num_classes and --img_size match your trained model.")
                print(f"=> Current settings: num_classes={args.num_classes}, img_size={args.img_size}")
            elif "key" in error_str and "missing" in error_str:
                print("=> Some model parameters are missing. This might be due to:")
                print("   1. Architecture changes between training and inference")
                print("   2. Partial model save during training")
            elif "unexpected key" in error_str:
                print("=> Extra parameters found. This might be due to:")
                print("   1. Different model architecture")
                print("   2. Including optimizer/scheduler states")
            
            return
            
    except Exception as e:
        print(f"=> Critical error loading model from {model_file_to_load}: {e}")
        print("=> Full traceback:")
        traceback.print_exc()
        return

    model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    # 2. Load Data
    print(f"Loading inference data from CSV: {args.csv_path}")
    print(f"Data root directory: {args.data_root_dir}")
    
    # Validate that the CSV file exists
    if not os.path.isfile(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        print("Please ensure the CSV file exists and contains columns: image_path, mask_path, question")
        return
    
    # Validate that the data root directory exists
    if not os.path.isdir(args.data_root_dir):
        print(f"Error: Data root directory not found at {args.data_root_dir}")
        print("Please ensure the dataset_subset directory exists and contains the image/mask data")
        return
    
    try:
        # Create the dataset
        inference_dataset = MedicalSegmentationDataset(
            csv_file_path=args.csv_path,
            data_base_dir=args.data_root_dir,
            transform=None  # Use default transforms from dataset
        )
        
        if len(inference_dataset) == 0:
            print("Error: Inference dataset is empty.")
            print("Please check:")
            print(f"  1. CSV file format and content: {args.csv_path}")
            print(f"  2. Data directory structure: {args.data_root_dir}")
            print("  3. File paths in CSV are correct relative to data_root_dir")
            return

        inference_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            collate_fn=collate_fn_skip_none
        )
        print(f"Inference DataLoader created successfully:")
        print(f"  - Dataset size: {len(inference_dataset)} samples")
        print(f"  - Number of batches: {len(inference_loader)}")
        print(f"  - Batch size: {args.batch_size}")
        
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        print("This might be due to:")
        print("  1. Incorrect CSV format")
        print("  2. Missing image/mask files")
        print("  3. Incorrect file paths in CSV")
        print("Full traceback:")
        traceback.print_exc()
        return
    
    if len(inference_loader) == 0:
        print("Inference loader is empty after creation. No data to process.")
        return

    # 3. Initialize Loss and Metrics
    criterion = MultiTaskLoss( # Using default weights from loss.py, override with args if needed
        dice_weight=args.dice_weight, ce_weight=args.ce_weight, boundary_weight=args.boundary_weight,
        l2_weight=args.l2_weight, contrastive_weight=args.contrastive_weight,
        focal_weight=args.focal_weight, tversky_weight=args.tversky_weight
    )
    dice_metric = DiceMetric(include_background=args.include_background_dice, reduction="mean", get_not_nans=False)
    hausdorff_metric = HausdorffDistanceMetric(include_background=args.include_background_hd, reduction="mean", get_not_nans=False, percentile=args.hd_percentile)

    total_loss_sum = 0.0
    all_loss_components = {}
    num_valid_batches = 0
    plotted_samples_count = 0

    # 4. Inference Loop
    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(inference_loader, desc="Inference Progress")):
            if batch_data is None:
                print(f"Skipping empty batch {batch_idx+1}/{len(inference_loader)} due to collate_fn.")
                continue

            inputs = batch_data["image"].to(device)
            masks = batch_data["mask"].to(device) # Ground truth masks for loss/metrics
            prompts = batch_data["prompt"]       # List of text prompts

            if inputs.nelement() == 0:
                print(f"Skipping batch {batch_idx+1} due to zero elements after collation.")
                continue
            
            try:
                outputs = model(inputs, prompts) # Model output are logits
            except Exception as e:
                print(f"Error during model forward pass on batch {batch_idx+1}: {e}")
                print(f"Input shapes: image={inputs.shape if hasattr(inputs, 'shape') else 'N/A'}")
                print(f"Prompts: {prompts}")
                continue 

            # Calculate loss
            # MultiTaskLoss.forward returns (total_loss_tensor, loss_dict_batch_scalars)
            loss_tensor, loss_dict_batch = criterion(outputs, masks, None, None, None) # Pass None for optional features
            
            if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
                print(f"Warning: NaN or Inf loss encountered in batch {batch_idx+1}. Skipping loss accumulation for this batch.")
            else:
                total_loss_sum += loss_tensor.item()
                for key, value in loss_dict_batch.items(): # value is already .item() from MultiTaskLoss
                    all_loss_components[key] = all_loss_components.get(key, 0.0) + value
            
            # Calculate metrics (MONAI metrics expect channel-first, sigmoid/softmax applied predictions)
            pred_probs = torch.sigmoid(outputs) # Apply sigmoid to logits for binary segmentation
            dice_metric(y_pred=pred_probs, y=masks)
            hausdorff_metric(y_pred=pred_probs, y=masks)

            num_valid_batches += 1

            # Plot results for a few samples
            if plotted_samples_count < args.num_plots_to_save:
                for sample_in_batch_idx in range(inputs.size(0)):
                    if plotted_samples_count < args.num_plots_to_save:
                        current_plot_idx = batch_idx * args.batch_size + sample_in_batch_idx
                        print(f"Plotting sample {current_plot_idx} (Overall sample {plotted_samples_count + 1})")
                        plot_results(inputs[sample_in_batch_idx], 
                                     masks[sample_in_batch_idx], 
                                     outputs[sample_in_batch_idx], # Pass logits to plot_results
                                     current_plot_idx, 
                                     args.plot_save_dir)
                        plotted_samples_count += 1
                    else:
                        break 
    
    # 5. Print Results
    if num_valid_batches > 0:
        avg_total_loss = total_loss_sum / num_valid_batches
        print(f"\\n--- Inference Results ---")
        print(f"Processed {num_valid_batches} valid batches.")
        print(f"Average Total Loss: {avg_total_loss:.4f}")

        print("Average Loss Components:")
        for key, value in all_loss_components.items():
            print(f"  {key}: {value / num_valid_batches:.4f}")

        try:
            # When reduction="mean" and get_not_nans=False, aggregate() returns a single tensor value.
            aggregated_dice = dice_metric.aggregate()
            avg_dice = aggregated_dice.item() if torch.is_tensor(aggregated_dice) else aggregated_dice
            print(f"Average Dice Score: {avg_dice:.4f}")
        except Exception as e:
            print(f"Could not compute Dice score: {e}")
        
        try:
            # When reduction="mean" and get_not_nans=False, aggregate() returns a single tensor value.
            aggregated_hd = hausdorff_metric.aggregate()
            avg_hausdorff = aggregated_hd.item() if torch.is_tensor(aggregated_hd) else aggregated_hd
            print(f"Average Hausdorff Distance ({args.hd_percentile}th percentile): {avg_hausdorff:.4f}")
        except Exception as e:
            print(f"Could not compute Hausdorff score: {e}")

        # Reset metrics
        dice_metric.reset()
        hausdorff_metric.reset()
    else:
        print("No valid batches were processed. Cannot compute average loss or metrics.")

    print("Inference finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedCLIPUNet Inference Script")

    # Paths
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model state_dict (.pt) or checkpoint (.pth.tar)')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file for inference data (must include image_path, mask_path, question)')
    parser.add_argument('--data_root_dir', type=str, required=True, help='Root directory of the inference dataset')
    parser.add_argument('--plot_save_dir', type=str, default="inference_plots", help='Directory to save plotted images')

    # Model parameters (must match the trained model)
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes for the model')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (height and width) for the model')

    # DataLoader parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference (adjust based on GPU memory)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (0 for main process, >0 for multiprocessing)')

    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for inference (cuda or cpu)')

    # Plotting
    parser.add_argument('--num_plots_to_save', type=int, default=5, help='Total number of sample plots to save from the beginning of inference.')

    # Loss weights (defaulted from MultiTaskLoss, adjust if needed for analysis)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--boundary_weight', type=float, default=0.5)
    parser.add_argument('--l2_weight', type=float, default=0.3)
    parser.add_argument('--contrastive_weight', type=float, default=0.2)
    parser.add_argument('--focal_weight', type=float, default=0.3)
    parser.add_argument('--tversky_weight', type=float, default=0.2)

    # Metric parameters
    parser.add_argument('--include_background_dice', action='store_true', help="Include background for Dice (default is False).")
    parser.add_argument('--include_background_hd', action='store_true', help="Include background for Hausdorff (default is False).")
    parser.set_defaults(include_background_dice=False, include_background_hd=False)
    parser.add_argument('--hd_percentile', type=float, default=95.0, help="Hausdorff distance percentile (default: 95.0).")

    args = parser.parse_args()
    run_inference(args)
