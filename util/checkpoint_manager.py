import torch
import os
import shutil
import glob
import re

def save_checkpoint(state, is_best, epoch_checkpoint_dir='checkpoints/epochs', best_checkpoint_dir='checkpoints/best'):
    """
    Saves the current model checkpoint.
    - Saves epoch-specific checkpoint to `epoch_checkpoint_dir`, removing the previous one.
    - If `is_best`, copies the epoch checkpoint to `best_checkpoint_dir`.

    Args:
        state (dict): Contains model's state_dict, optimizer's state_dict, epoch, etc.
                      Requires 'epoch' key (current completed epoch number).
        is_best (bool): True if this is the best model seen so far.
        epoch_checkpoint_dir (str): Directory to save epoch-specific checkpoints.
        best_checkpoint_dir (str): Directory to save the best model checkpoint.
    """
    if 'epoch' not in state:
        raise ValueError("State dictionary must contain an 'epoch' key (current completed epoch).")
    
    current_epoch = state['epoch'] # This is the epoch that just finished
    
    os.makedirs(epoch_checkpoint_dir, exist_ok=True)
    os.makedirs(best_checkpoint_dir, exist_ok=True)

    base_name = "checkpoint"
    extension = ".pth.tar"
    
    # 1. Save current epoch-specific checkpoint
    current_epoch_filename = f"{base_name}_epoch_{current_epoch}{extension}"
    current_filepath = os.path.join(epoch_checkpoint_dir, current_epoch_filename)
    torch.save(state, current_filepath)
    print(f"Epoch {current_epoch} checkpoint saved to {current_filepath}")

    # 2. Remove previous epoch-specific checkpoint (if it exists)
    if current_epoch > 0: # No previous epoch if current_epoch is 0 (or 1 if 0-indexed start)
        # If state['epoch'] is the epoch that just finished (e.g. 50),
        # then the next epoch to run is state['epoch'] + 1.
        # The checkpoint being saved is for 'current_epoch'.
        # We want to remove 'current_epoch - 1'.
        previous_epoch_to_remove = current_epoch - 1
        if previous_epoch_to_remove >= 0: # Ensure we don't try to remove epoch -1
            previous_epoch_filename = f"{base_name}_epoch_{previous_epoch_to_remove}{extension}"
            previous_filepath = os.path.join(epoch_checkpoint_dir, previous_epoch_filename)
            if os.path.exists(previous_filepath):
                try:
                    os.remove(previous_filepath)
                    print(f"Removed previous epoch checkpoint: {previous_filepath}")
                except OSError as e:
                    print(f"Error removing previous checkpoint {previous_filepath}: {e}")

    # 3. Handle is_best: copy current epoch-specific checkpoint to best_filename
    if is_best:
        best_filename = f"model_best{extension}"
        best_filepath = os.path.join(best_checkpoint_dir, best_filename)
        shutil.copyfile(current_filepath, best_filepath)
        print(f"Best model (from epoch {current_epoch}) saved to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.
        device (str): Device to load the checkpoint onto ('cpu' or 'cuda').

    Returns:
        tuple: (start_epoch, best_metric)
               start_epoch (int): The epoch to resume training from (epoch after the one saved).
               best_metric (float): The best validation metric from the checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        # Return next epoch as 0 (start from scratch), and default best_metric
        return 0, float('inf') 

    print(f"=> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    
    # The 'epoch' in the checkpoint is the epoch that was just completed.
    # So, training should start from epoch + 1.
    completed_epoch = checkpoint.get('epoch', -1) # if -1, then start_epoch will be 0
    start_epoch = completed_epoch + 1
    
    best_metric = checkpoint.get('best_metric', float('inf')) # Default for loss-based metric

    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Loaded optimizer state.")
        except Exception as e:
            print(f"Could not load optimizer state: {e}. Optimizer will be re-initialized.")
    elif optimizer:
        print("=> Optimizer state not found in checkpoint. Optimizer will be re-initialized.")

    print(f"=> Loaded checkpoint '{checkpoint_path}' (completed epoch {completed_epoch}, best_metric: {best_metric:.4f}). Resuming from epoch {start_epoch}.")
    return start_epoch, best_metric

def get_latest_checkpoint(epoch_checkpoint_dir='checkpoints/epochs'):
    """
    Gets the path to the latest epoch-specific checkpoint file based on epoch number.

    Args:
        epoch_checkpoint_dir (str): Directory where epoch-specific checkpoints are saved.

    Returns:
        str or None: Path to the latest checkpoint, or None if not found.
    """
    os.makedirs(epoch_checkpoint_dir, exist_ok=True) # Ensure dir exists
    
    base_name = "checkpoint"
    extension = ".pth.tar"
    glob_pattern = os.path.join(epoch_checkpoint_dir, f"{base_name}_epoch_*{extension}")
    
    checkpoints = glob.glob(glob_pattern)
    if not checkpoints:
        return None

    latest_epoch = -1
    latest_checkpoint_path = None
    
    # Regex to extract epoch number from filenames like 'checkpoint_epoch_10.pth.tar'
    epoch_capture_regex = re.compile(f"^{re.escape(base_name)}_epoch_(\\d+){re.escape(extension)}$")

    for cp_path in checkpoints:
        filename = os.path.basename(cp_path)
        match = epoch_capture_regex.match(filename)
        if match:
            try:
                epoch_num = int(match.group(1))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint_path = cp_path
            except ValueError:
                print(f"Warning: Could not parse epoch number from filename {filename}")
                continue
                
    if latest_checkpoint_path:
        print(f"Found latest checkpoint: {latest_checkpoint_path} for epoch {latest_epoch}")
    else:
        print(f"No valid epoch-numbered checkpoints found in {epoch_checkpoint_dir} matching pattern.")
        
    return latest_checkpoint_path

