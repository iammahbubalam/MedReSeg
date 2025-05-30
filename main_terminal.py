from data.dataloader import  create_train_val_dataloaders
from components.medclip_unet import MedCLIPUNet 
from traintest.train import train_model
import torch
import os
import random
import numpy as np

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    print("Starting medical image segmentation model training with MedCLIPUNet")
    print("=" * 60)
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)
    
    # Set up data paths - modify these according to your dataset location
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Option 1: Use SAMed2D dataset structure
    csv_path = os.path.join(current_script_dir, 'dataset_subset', 'SAMed2Dv1', 'subset_SAMed2D_image_metadata_per_mask_with_questions.csv')
    data_root_dir = os.path.join(current_script_dir, 'dataset_subset', 'SAMed2Dv1')
    
    print(f"Looking for data at:")
    print(f"CSV file: {csv_path}")
    print(f"Data root: {data_root_dir}")
    
    # Check if data paths exist
    if not os.path.exists(csv_path):
        print(f"\n❌ Error: CSV file not found at: {csv_path}")
        print("Please ensure your dataset is properly set up or modify the paths in main.py")
        exit(1)
    elif not os.path.isdir(data_root_dir):
        print(f"\n❌ Error: Data root directory not found at: {data_root_dir}")
        print("Please ensure your dataset is properly set up or modify the paths in main.py")
        exit(1)
    else:
        print("✅ Data paths found!")
        
        # Create train/validation dataloaders
        print("\n📊 Creating train/validation dataloaders...")
        try:
            train_dataloader, val_dataloader, train_dataset, val_dataset = create_train_val_dataloaders(
                csv_file_path=csv_path,
                data_base_dir=data_root_dir,
                batch_size=4,  # Adjust based on your GPU memory
                val_size=0.2,  # 20% for validation
                random_state=42,
                shuffle=True,
                transform=None, 
                num_workers=0  # Set to 0 for Windows compatibility
            )
            
            if train_dataloader is None or val_dataloader is None:
                print("❌ Error: Failed to create dataloaders. Check your data paths and CSV file.")
                exit(1)
            
            print(f"✅ Successfully created dataloaders:")
            print(f"   - Training batches: {len(train_dataloader)}")
            print(f"   - Validation batches: {len(val_dataloader)}")
            print(f"   - Training samples: {len(train_dataset)}")
            print(f"   - Validation samples: {len(val_dataset)}")
            
        except Exception as e:
            print(f"❌ Error creating dataloaders: {str(e)}")
            exit(1)
    
    # Initialize model
    print("\n🔧 Initializing MedCLIPUNet model...")
    try:
        model = MedCLIPUNet(num_classes=1, img_size=256).to(device)
        print(f"✅ Model initialized successfully on {device}")
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error initializing model: {str(e)}")
        exit(1)
    
    # Training configuration
    training_config = {
        'num_epochs': 10,  # Adjust as needed
        'learning_rate': 1e-4,
        'batch_size': 4,
        'grad_accumulation_steps': 2,  # For simulating larger batch sizes
    }
    
    print("\n🚀 Starting training...")
    print(f"Training Configuration:")
    for key, value in training_config.items():
        print(f"   - {key}: {value}")
    print("=" * 60)
    
    # Train model
    try:
        trained_model = train_model(
            model=model,
            train_loader=train_dataloader,
            val_loader=val_dataloader,
            device=device,
            **training_config
        )
        
        # Save the trained model
        model_save_path = "med_clip_unet_transformer_256x256.pt"
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"\n🎉 Training completed successfully!")
        print(f"Model saved to: {model_save_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
