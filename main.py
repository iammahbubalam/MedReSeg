from data.dataloader import create_train_val_dataloaders
from components.medclip_unet import MedCLIPUNet
from traintest.train import train_model
import torch
import os
import random
import numpy as np
import argparse # Import argparse

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Keep deterministic for reproducibility, benchmark can be False if speed is critical and input sizes don't vary
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)

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

    # Use paths from arguments
    csv_path = args.csv_path
    data_root_dir = args.data_root_dir

    print(f"Looking for data at:")
    print(f"CSV file: {csv_path}")
    print(f"Data root: {data_root_dir}")

    # Check if data paths exist
    if not os.path.exists(csv_path):
        print(f"\n Error: CSV file not found at: {csv_path}")
        print("Please ensure your dataset is properly set up or modify the paths.")
        exit(1)
    elif not os.path.isdir(data_root_dir):
        print(f"\n Error: Data root directory not found at: {data_root_dir}")
        print("Please ensure your dataset is properly set up or modify the paths.")
        exit(1)
    else:
        print(" Data paths found!")

        # Create train/validation dataloaders
        print("\n Creating train/validation dataloaders...")
        try:
            train_dataloader, val_dataloader, train_dataset, val_dataset = create_train_val_dataloaders(
                csv_file_path=csv_path,
                data_base_dir=data_root_dir,
                batch_size=args.batch_size,
                val_size=args.val_size,
                random_state=args.seed, # Use the same seed for data splitting
                shuffle=True,
                transform=None,
                num_workers=args.num_workers
            )

            if train_dataloader is None or val_dataloader is None:
                print("❌ Error: Failed to create dataloaders. Check your data paths and CSV file.")
                exit(1)

            print(f"    Successfully created dataloaders:")
            print(f"   - Training batches: {len(train_dataloader)}")
            print(f"   - Validation batches: {len(val_dataloader)}")
            print(f"   - Training samples: {len(train_dataset)}")
            print(f"   - Validation samples: {len(val_dataset)}")

        except Exception as e:
            print(f" Error creating dataloaders: {str(e)}")
            import traceback
            traceback.print_exc()
            exit(1)

    # Initialize model
    print("\n Initializing MedCLIPUNet model...")
    try:
        model = MedCLIPUNet(num_classes=args.num_classes, img_size=args.img_size).to(device)
        print(f" Model initialized successfully on {device}")

        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f" Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Training configuration from arguments
    training_config = {
    'num_epochs': args.num_epochs,
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'grad_accumulation_steps': args.grad_accumulation_steps,
    'epoch_checkpoint_dir': args.epoch_checkpoint_dir,
    'best_checkpoint_dir': args.best_checkpoint_dir,
    'log_file': args.log_file, # Add this
    'log_level_str': args.log_level # Add this
}

    print("\n Starting training...")
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

        if trained_model is None:
            print(f"\n Training failed: train_model returned None.")
            exit(1)

        # Save the trained model
        model_save_path = args.model_save_path
        # Ensure directory for model_save_path exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"\n Training completed successfully!")
        print(f"Model saved to: {model_save_path}")

    except Exception as e:
        print(f"\n Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedCLIPUNet Training Script")

    # General arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Data arguments
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_path = os.path.join(current_script_dir, 'dataset_subset', 'SAMed2Dv1', 'subset_SAMed2D_image_metadata_per_mask_with_questions.csv')
    default_data_root_dir = os.path.join(current_script_dir, 'dataset_subset', 'SAMed2Dv1')

    parser.add_argument('--csv_path', type=str, default=default_csv_path, help='Path to the training CSV file')
    parser.add_argument('--data_root_dir', type=str, default=default_data_root_dir, help='Root directory of the dataset')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of dataset to use for validation')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader (0 for Windows compatibility)')

    # Model arguments
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes for the model')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (height and width) for the model')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--grad_accumulation_steps', type=int, default=2, help='Number of gradient accumulation steps')
    
    # Checkpoint arguments (passed to train_model)
    parser.add_argument('--epoch_checkpoint_dir', type=str, default='checkpoints/epochs', help='Directory to save epoch checkpoints')
    parser.add_argument('--best_checkpoint_dir', type=str, default='checkpoints/best', help='Directory to save best model checkpoints')


    # Output arguments
    parser.add_argument('--model_save_path', type=str, default="med_clip_unet_transformer_256x256.pt", help='Path to save the final trained model')
    parser.add_argument('--log_file', type=str, default='logs/training.log', help='Path to the log file')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')
    args = parser.parse_args()
    main(args)