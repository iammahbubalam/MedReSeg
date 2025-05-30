import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from tqdm import tqdm
from monai.metrics.meandice import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from components.loss import MultiTaskLoss
from util.training_visualizer import TrainingVisualizer
from util.checkpoint_manager import save_checkpoint, load_checkpoint, get_latest_checkpoint

import logging
import io

# Helper class to redirect tqdm output to logger
class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to a logger.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self, logger_instance, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger_instance
        self.level = level or logging.INFO
    
    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')
    
    def flush(self):
        if self.buf: # Avoid logging empty lines from tqdm
            self.logger.log(self.level, self.buf)
        self.buf = '' # Clear buffer after logging

def setup_logger(logger_name, log_file, level=logging.INFO, append=True):
    """Sets up a logger instance."""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False # Prevent log duplication if root logger is also configured

    if not logger.handlers: # Add handlers only if they haven't been added before
        # File Handler
        file_mode = 'a' if append else 'w'
        fh = logging.FileHandler(log_file, mode=file_mode)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                      datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def train_model(model, train_loader, val_loader, device, 
                num_epochs=10, learning_rate=1e-4, batch_size=8, 
                grad_accumulation_steps=2, 
                epoch_checkpoint_dir='checkpoints/epochs', 
                best_checkpoint_dir='checkpoints/best',   
                best_val_metric=float('inf'), metric_mode='min',
                log_file="logs/training.log", log_level_str="INFO"):
    
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger = setup_logger(f"train_model.{model.__class__.__name__}", log_file, level=log_level)
    
    tqdm_logger_stream = TqdmToLogger(logger, level=log_level)

    logger.info("Starting training with configuration:")
    logger.info(f"- Target epochs: {num_epochs}")
    logger.info(f"- Batch size (for dataloader, not effective batch): {batch_size}") # Clarified batch_size meaning
    logger.info(f"- Learning rate: {learning_rate}")
    logger.info(f"- Gradient accumulation steps: {grad_accumulation_steps}")
    logger.info(f"- Device: {device}")
    logger.info(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"- GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"- Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"- Epoch checkpoint directory: {epoch_checkpoint_dir}")
    logger.info(f"- Best checkpoint directory: {best_checkpoint_dir}")
    logger.info(f"- Initial best_val_metric: {best_val_metric}")
    logger.info(f"- Metric mode: {metric_mode}")
    
    criterion = MultiTaskLoss(
        dice_weight=1.0, ce_weight=1.0, boundary_weight=0.5,
        l2_weight=0.3, contrastive_weight=0.2, focal_weight=0.3, tversky_weight=0.2
    )
    logger.info("MultiTaskLoss initialized.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False # verbose=True would print to console
    ) # verbose for scheduler can be logged manually if needed after scheduler.step()
    logger.info("Optimizer (Adam) and LR Scheduler (ReduceLROnPlateau) initialized.")

    scaler = torch.GradScaler(enabled=(device.type == 'cuda'))
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
    # visualizer = TrainingVisualizer() # Visualizer might print or save images, handle its output separately if needed

    start_epoch = 0
    # Note: get_latest_checkpoint and load_checkpoint might have their own print statements.
    # To fully redirect, those would also need modification or to accept a logger.
    latest_checkpoint_path = get_latest_checkpoint(epoch_checkpoint_dir=epoch_checkpoint_dir)
    if latest_checkpoint_path:
        logger.info(f"Attempting to resume from checkpoint: {latest_checkpoint_path}")
        # Assuming load_checkpoint prints its own status, or modify it to use logger
        start_epoch, best_val_metric_loaded = load_checkpoint(latest_checkpoint_path, model, optimizer, device)
        if best_val_metric_loaded is not None:
            if metric_mode == 'min':
                best_val_metric = min(best_val_metric, best_val_metric_loaded)
            else: # metric_mode == 'max'
                best_val_metric = max(best_val_metric, best_val_metric_loaded) 
        logger.info(f"Resumed from epoch {start_epoch}. Best validation metric from checkpoint: {best_val_metric_loaded if best_val_metric_loaded is not None else 'N/A'}. Current best: {best_val_metric:.4f}")
    else:
        logger.info("No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"=== Epoch {epoch + 1}/{num_epochs} (Index {epoch}) ===") 
        model.train()
        epoch_loss = 0
        epoch_loss_components = {} # For accumulating individual loss components
        
        optimizer.zero_grad(set_to_none=True) 
        
        # tqdm will write to the TqdmToLogger stream, which then logs
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False, 
            disable=False, # Set to True if tqdm bar in logs is too noisy
            file=tqdm_logger_stream,
            mininterval=1.0 # Update progress bar in log less frequently
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            inputs, masks, prompts = batch_data["image"].to(device), batch_data["mask"].to(device), batch_data["prompt"]
            
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs, prompts)
                features = model.get_last_features() # Assuming this is needed for loss
                attention_maps = features.get('attention_maps') # Use .get for safety
                img_features = features.get('image_features')
                text_features = features.get('text_features')
                
                loss, loss_dict = criterion(
                    outputs, masks, attention_maps, img_features, text_features
                )
                loss = loss / grad_accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            current_batch_loss = loss.item() * grad_accumulation_steps 
            epoch_loss += current_batch_loss
            for loss_name, loss_val in loss_dict.items(): # Changed loss_val_tensor to loss_val
                # If loss_val is a tensor, call .item(), otherwise use it directly
                current_component_loss = loss_val.item() if torch.is_tensor(loss_val) else loss_val
                epoch_loss_components[loss_name] = epoch_loss_components.get(loss_name, 0) + current_component_loss

            # Postfix for tqdm is handled by tqdm writing to the logger stream
            progress_bar.set_postfix({
                'loss': f"{current_batch_loss:.4f}",
                'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}"
            })
            
            del inputs, masks, prompts, outputs, features, attention_maps, img_features, text_features, loss, loss_dict

        avg_epoch_loss = epoch_loss / len(train_loader) # Average per batch
        logger.info(f"End of Epoch {epoch + 1}/{num_epochs} Training:")
        logger.info(f"  Average Training Loss: {avg_epoch_loss:.4f}")
        for loss_name, total_loss_val in epoch_loss_components.items():
            logger.info(f"    - Avg {loss_name}: {total_loss_val/len(train_loader):.4f}")
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        logger.info("--- Starting validation ---")
        model.eval()
        val_loss_total = 0.0 # Renamed to avoid conflict
        val_loss_components_total = {}
        
        val_progress = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Valid]",
            leave=False, 
            disable=False, # Set to True if tqdm bar in logs is too noisy
            file=tqdm_logger_stream,
            mininterval=1.0
        )
        
        with torch.no_grad():
            for val_data in val_progress:
                val_inputs, val_masks, val_prompts = val_data["image"].to(device), val_data["mask"].to(device), val_data["prompt"]
                
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    val_outputs = model(val_inputs, val_prompts)
                    val_features = model.get_last_features()
                    val_attention_maps = val_features.get('attention_maps')
                    val_img_features = val_features.get('image_features')
                    val_text_features = val_features.get('text_features')
                    
                    val_loss_batch, val_loss_dict_batch = criterion(
                        val_outputs, val_masks, val_attention_maps, val_img_features, val_text_features
                    )
                
                current_val_batch_loss = val_loss_batch.item()
                val_loss_total += current_val_batch_loss
                for loss_name, loss_val in val_loss_dict_batch.items(): # Changed loss_val_tensor to loss_val
                    # If loss_val is a tensor, call .item(), otherwise use it directly
                    current_component_loss = loss_val.item() if torch.is_tensor(loss_val) else loss_val
                    val_loss_components_total[loss_name] = val_loss_components_total.get(loss_name, 0) + current_component_loss

                val_progress.set_postfix({'val_loss': f"{current_val_batch_loss:.4f}"})
                
                val_outputs_metric = (val_outputs > 0.5).float()
                dice_metric(y_pred=val_outputs_metric, y=val_masks)
                hausdorff_metric(y_pred=val_outputs_metric, y=val_masks)

                del val_inputs, val_masks, val_prompts, val_outputs, val_features
                del val_attention_maps, val_img_features, val_text_features, val_loss_batch, val_loss_dict_batch, val_outputs_metric
                
            avg_val_loss = val_loss_total / len(val_loader)
            logger.info(f"End of Epoch {epoch + 1}/{num_epochs} Validation:")
            logger.info(f"  Average Validation Loss: {avg_val_loss:.4f}")
            for loss_name, total_loss_val in val_loss_components_total.items():
                logger.info(f"    - Avg {loss_name}: {total_loss_val/len(val_loader):.4f}")

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            aggregated_dice = dice_metric.aggregate() # MONAI 1.0+ returns a tensor
            val_dice_score = aggregated_dice.item() if torch.is_tensor(aggregated_dice) else aggregated_dice[0].item() # Handle older MONAI too
            
            aggregated_hausdorff = hausdorff_metric.aggregate()
            val_hausdorff_score = aggregated_hausdorff.item() if torch.is_tensor(aggregated_hausdorff) else aggregated_hausdorff[0].item()
            
            dice_metric.reset()
            hausdorff_metric.reset()

            logger.info(f"  Validation Dice Score: {val_dice_score:.4f}")
            logger.info(f"  Validation Hausdorff Distance: {val_hausdorff_score:.4f}")
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                logger.info(f"Learning rate reduced from {old_lr} to {new_lr} by scheduler.")


            current_val_metric_to_monitor = avg_val_loss # Using average validation loss to monitor
            is_best = False
            if metric_mode == 'min':
                if current_val_metric_to_monitor < best_val_metric:
                    best_val_metric = current_val_metric_to_monitor
                    is_best = True
                    logger.info(f"New best model found with validation metric ({metric_mode}): {best_val_metric:.4f} at epoch {epoch + 1}")
            else: # metric_mode == 'max'
                if current_val_metric_to_monitor > best_val_metric:
                    best_val_metric = current_val_metric_to_monitor
                    is_best = True
                    logger.info(f"New best model found with validation metric ({metric_mode}): {best_val_metric:.4f} at epoch {epoch + 1}")
            
            checkpoint_state = {
                'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), # Save scheduler state
                'best_metric': best_val_metric,
                'metric_mode': metric_mode
            }
            
            # save_checkpoint might also print, consider passing logger or modifying it
            save_checkpoint(checkpoint_state, is_best, 
                            epoch_checkpoint_dir=epoch_checkpoint_dir, 
                            best_checkpoint_dir=best_checkpoint_dir)
            
    logger.info("=== Training Loop Finished ===")
    return model
# Note: The original "Training Complete" print after return model was unreachable and removed.
# The calling script (main.py) should handle the final completion message.