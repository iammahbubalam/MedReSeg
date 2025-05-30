import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from tqdm import tqdm
from monai.metrics.meandice import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from components.loss import MultiTaskLoss # Assuming MultiTaskLoss is in components/loss.py
from util.training_visualizer import TrainingVisualizer # Assuming TrainingVisualizer is in util/training_visualizer.py
from util.checkpoint_manager import save_checkpoint, load_checkpoint, get_latest_checkpoint # Correct if util/checkpoint_manager.py exists


def train_model(model, train_loader, val_loader, device, 
                num_epochs=10, learning_rate=1e-4, batch_size=8, 
                grad_accumulation_steps=2, 
                epoch_checkpoint_dir='checkpoints/epochs', 
                best_checkpoint_dir='checkpoints/best',   
                best_val_metric=float('inf'), metric_mode='min'):
    # VRAM Optimization Note:
    # - Gradient Accumulation (grad_accumulation_steps > 1) is used to simulate a larger batch size
    #   while using less VRAM.
    # - Automatic Mixed Precision (AMP) with torch.GradScaler and torch.autocast is used,
    #   which can reduce VRAM and speed up training on compatible GPUs.
    # - Consider reducing 'batch_size' if VRAM issues persist.
    # - Ensure DataLoader uses pin_memory=True and an appropriate num_workers for efficient data transfer,
    #   though these are configured outside this function.
    print(f"Starting training with:")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"- Device: {device}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        print(f"- Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    criterion = MultiTaskLoss(
        dice_weight=1.0,
        ce_weight=1.0,
        boundary_weight=0.5,
        l2_weight=0.3,
        contrastive_weight=0.2,
        focal_weight=0.3,
        tversky_weight=0.2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler = torch.GradScaler(enabled=(device.type == 'cuda'))
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
    visualizer = TrainingVisualizer()
    
    start_epoch = 0
    latest_checkpoint_path = get_latest_checkpoint(epoch_checkpoint_dir=epoch_checkpoint_dir)
    if latest_checkpoint_path:
        print(f"Resuming from checkpoint: {latest_checkpoint_path}")
        start_epoch, best_val_metric_loaded = load_checkpoint(latest_checkpoint_path, model, optimizer, device)
        if best_val_metric_loaded is not None:
            if metric_mode == 'min':
                best_val_metric = min(best_val_metric, best_val_metric_loaded)
            else:
                best_val_metric = max(best_val_metric, best_val_metric_loaded) 
        print(f"Resumed from epoch {start_epoch}. Best validation metric from checkpoint: {best_val_metric_loaded if best_val_metric_loaded is not None else 'N/A'}. Current best: {best_val_metric:.4f}")
    else:
        print("No checkpoint found, starting from scratch.")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} (Index {epoch}) ===") 
        model.train()
        epoch_loss = 0
        epoch_loss_components = {}
        step = 0
        optimizer.zero_grad(set_to_none=True) # Modified for potential efficiency
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False, 
            disable=False  
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            step += 1
            inputs, masks, prompts = batch_data["image"].to(device), batch_data["mask"].to(device), batch_data["prompt"]
            
            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs, prompts)
                # Consider removing duplicate call: outputs = model(inputs, prompts)
                features = model.get_last_features()
                attention_maps = features['attention_maps']
                img_features = features['image_features']
                text_features = features['text_features']
                loss, loss_dict = criterion(
                    outputs, 
                    masks, 
                    attention_maps,
                    img_features, 
                    text_features
                )
                loss = loss / grad_accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True) # Modified for potential efficiency
            
            curr_loss = loss.item() * grad_accumulation_steps # Make sure loss.item() is called before loss is deleted
            epoch_loss += curr_loss
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_components:
                    epoch_loss_components[loss_name] = 0
                epoch_loss_components[loss_name] += loss_value # Ensure loss_value is used before loss_dict is deleted
            
            progress_bar.set_postfix({
                'loss': f"{curr_loss:.4f}",
                'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}"
            })

            # Explicitly delete tensors to free VRAM
            del inputs, masks, prompts, outputs, features, attention_maps, img_features, text_features, loss, loss_dict
            # Removed torch.cuda.empty_cache() from here to avoid frequent calls
        
        avg_epoch_loss = epoch_loss / step
        avg_loss_components = {k: v / step for k, v in epoch_loss_components.items()}
        print(f"\n★ Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
        for loss_name, loss_avg in avg_loss_components.items():
            print(f"  - {loss_name}: {loss_avg:.4f}")
        
        # Call empty_cache less frequently, e.g., end of training epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print("\n--- Starting validation ---")
        model.eval()
        val_loss = 0.0
        val_loss_components = {}
        
        val_progress = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Valid]",
            leave=False, 
            disable=False
        )
        
        with torch.no_grad():
            for val_idx, val_data in enumerate(val_progress):
                val_inputs, val_masks, val_prompts = val_data["image"].to(device), val_data["mask"].to(device), val_data["prompt"]
                # For validation, autocast can also be used if model benefits from it,
                # but ensure it's consistent with training if using AMP for validation loss comparison.
                # Since we are only doing inference and then calculating loss on full precision masks,
                # autocast might not be strictly necessary here unless model inference itself is slow.
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    val_outputs = model(val_inputs, val_prompts)
                    val_features = model.get_last_features()
                    val_attention_maps = val_features['attention_maps']
                    val_img_features = val_features['image_features']
                    val_text_features = val_features['text_features']
                    # Loss calculation should ideally be in the same precision as training if comparing directly
                    # or be aware of potential differences.
                    val_loss_batch, val_loss_dict = criterion(
                        val_outputs, 
                        val_masks,
                        val_attention_maps,
                        val_img_features,
                        val_text_features
                    )
                
                val_loss += val_loss_batch.item() # Make sure .item() is called before deletion
                for loss_name, loss_value in val_loss_dict.items():
                    if loss_name not in val_loss_components:
                        val_loss_components[loss_name] = 0
                    val_loss_components[loss_name] += loss_value # Ensure used before deletion

                val_progress.set_postfix({
                    'val_loss': f"{val_loss_batch.item():.4f}"
                })
                # Thresholding for metrics should be done on CPU or after bringing tensor to CPU
                # if val_outputs is on GPU to avoid GPU sync issues in loop if not careful.
                # However, MONAI metrics handle GPU tensors.
                val_outputs_metric = (val_outputs > 0.5).float()
                dice_metric(y_pred=val_outputs_metric, y=val_masks)
                hausdorff_metric(y_pred=val_outputs_metric, y=val_masks)

                # Explicitly delete tensors to free VRAM
                del val_inputs, val_masks, val_prompts, val_outputs, val_features
                del val_attention_maps, val_img_features, val_text_features, val_loss_batch, val_loss_dict, val_outputs_metric
                # Removed torch.cuda.empty_cache() from here to avoid frequent calls
                
            avg_val_loss = val_loss / len(val_loader)
            avg_val_loss_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
            print(f"\n★ Validation Loss: {avg_val_loss:.4f}")
            for loss_name, loss_avg in avg_val_loss_components.items():
                print(f"  - {loss_name}: {loss_avg:.4f}")

            # Call empty_cache less frequently, e.g., end of validation loop
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Assuming aggregate() returns a tuple (metric_tensor, ...) and metric_tensor is the first element
            aggregated_dice = dice_metric.aggregate()
            val_dice_score = aggregated_dice[0].item() if isinstance(aggregated_dice, tuple) else aggregated_dice.item()
            
            aggregated_hausdorff = hausdorff_metric.aggregate()
            val_hausdorff_score = aggregated_hausdorff[0].item() if isinstance(aggregated_hausdorff, tuple) else aggregated_hausdorff.item()
            
            dice_metric.reset()
            hausdorff_metric.reset()

            print(f"  - Dice Score: {val_dice_score:.4f}")
            print(f"  - Hausdorff Distance: {val_hausdorff_score:.4f}")
            scheduler.step(avg_val_loss)

            current_val_metric = avg_val_loss 
            is_best = False
            if metric_mode == 'min':
                if current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric
                    is_best = True
                    print(f"\n✨ New best model found with validation metric: {best_val_metric:.4f} at epoch {epoch + 1}")
            else: 
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    is_best = True
                    print(f"\n✨ New best model found with validation metric: {best_val_metric:.4f} at epoch {epoch + 1}")
            
            checkpoint_state = {
                'epoch': epoch, 
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_val_metric,
            }
            save_checkpoint(checkpoint_state, is_best, 
                            epoch_checkpoint_dir=epoch_checkpoint_dir, 
                            best_checkpoint_dir=best_checkpoint_dir)

    print("\n=== Training Complete ===")
