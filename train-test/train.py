
def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-4, batch_size=8, grad_accumulation_steps=2):
    print(f"Starting training with:")
    print(f"- Batch size: {batch_size}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"- Device: {device}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name(0)}")
        print(f"- Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Define multi-task loss function and optimizer
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
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler()
    
    # Define metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean")
    
    # Setup visualizer
    visualizer = TrainingVisualizer()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        model.train()
        epoch_loss = 0
        epoch_loss_components = {}
        step = 0
        optimizer.zero_grad()  # Zero gradients once at the beginning of accumulation
        
        # Create progress bar for training
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False, 
            disable=False  
        )
        
        for batch_idx, batch_data in enumerate(progress_bar):
            step += 1
            inputs, masks, prompts = batch_data["image"].to(device), batch_data["mask"].to(device), batch_data["prompt"]
            
            # Use mixed precision
            with torch.amp.autocast(device_type='cuda'):
                # Forward pass
                outputs = model(inputs, prompts)
                
                # Get features for loss calculation
                features = model.get_last_features()
                attention_maps = features['attention_maps']
                img_features = features['image_features']
                text_features = features['text_features']
                
                # Calculate multi-task loss
                loss, loss_dict = criterion(
                    outputs, 
                    masks, 
                    attention_maps,
                    img_features, 
                    text_features
                )
                
                # Scale loss by accumulation steps
                loss = loss / grad_accumulation_steps
                
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            
            # Only step and zero grad when we've accumulated enough gradients
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Update metrics
            curr_loss = loss.item() * grad_accumulation_steps
            epoch_loss += curr_loss
            
            # Accumulate loss components for epoch average
            for loss_name, loss_value in loss_dict.items():
                if loss_name not in epoch_loss_components:
                    epoch_loss_components[loss_name] = 0
                epoch_loss_components[loss_name] += loss_value
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{curr_loss:.4f}",
                'avg_loss': f"{epoch_loss/(batch_idx+1):.4f}"
            })
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / step
        
        # Calculate average for each loss component
        avg_loss_components = {k: v / step for k, v in epoch_loss_components.items()}
        
        # Print loss components
        print(f"\n★ Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
        for loss_name, loss_avg in avg_loss_components.items():
            print(f"  - {loss_name}: {loss_avg:.4f}")
        
        # Validation
        print("\n--- Starting validation ---")
        model.eval()
        val_loss = 0.0
        val_loss_components = {}
        
        # Create progress bar for validation
        val_progress = tqdm(
            val_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs} [Valid]",
            leave=False, 
            disable=False  # Disable progress bar on Kaggle
        )
        
        with torch.no_grad():
            for val_idx, val_data in enumerate(val_progress):
                val_inputs, val_masks, val_prompts = val_data["image"].to(device), val_data["mask"].to(device), val_data["prompt"]
                
                # Forward pass in validation
                val_outputs = model(val_inputs, val_prompts)
                
                # Get features for loss calculation
                val_features = model.get_last_features()
                val_attention_maps = val_features['attention_maps']
                val_img_features = val_features['image_features']
                val_text_features = val_features['text_features']
                
                # Calculate validation loss
                val_loss_batch, val_loss_dict = criterion(
                    val_outputs, 
                    val_masks,
                    val_attention_maps,
                    val_img_features,
                    val_text_features
                )
                
                val_loss += val_loss_batch.item()
                
                # Accumulate loss components
                for loss_name, loss_value in val_loss_dict.items():
                    if loss_name not in val_loss_components:
                        val_loss_components[loss_name] = 0
                    val_loss_components[loss_name] += loss_value
                
                # Update validation progress bar
                val_progress.set_postfix({
                    'val_loss': f"{val_loss_batch.item():.4f}"
                })
                
                # Convert output to binary mask for metrics
                val_outputs = (val_outputs > 0.5).float()
                
                # Calculate metrics
                dice_metric(y_pred=val_outputs, y=val_masks)
                hausdorff_metric(y_pred=val_outputs, y=val_masks)
                
            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate average for each validation loss component
            avg_val_loss_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
            
            # Print validation loss components
            print(f"\n★ Validation Loss: {avg_val_loss:.4f}")
            for loss_name, loss_avg in avg_val_loss_components.items():
                print(f"  - {loss_name}: {loss_avg:.4f}")
            
            # Aggregate the metrics
            val_dice_score = dice_metric.aggregate().item()
            val_hausdorff_score = hausdorff_metric.aggregate().item()
            dice_metric.reset()
            hausdorff_metric.reset()
            
            print(f"★ Dice Score: {val_dice_score:.4f}")
            print(f"★ Hausdorff Distance: {val_hausdorff_score:.4f}")
            
            # Update visualizer
            visualizer.update_metrics(
                epoch=epoch+1, 
                train_loss=avg_epoch_loss, 
                val_dice=val_dice_score, 
                val_hausdorff=val_hausdorff_score
            )
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            # Plot and save metrics every epoch
            metrics_path = visualizer.plot_metrics()
            print(f"Metrics plot saved to {metrics_path}")
    
    print("\nTraining completed!")
    
    # Final plot of all metrics
    final_metrics_path = visualizer.plot_metrics()
    print(f"Final metrics plot saved to {final_metrics_path}")
    
    return model