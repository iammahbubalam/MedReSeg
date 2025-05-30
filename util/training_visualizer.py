import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np





class TrainingVisualizer:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrics to track
        self.train_losses = []
        self.val_losses = []
        self.dice_scores = []
        self.hausdorff_distances = []
        self.epochs = []
        
    def update_metrics(self, epoch, train_loss, val_dice=None, val_hausdorff=None):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        
        if val_dice is not None:
            self.dice_scores.append(val_dice)
        
        if val_hausdorff is not None:
            self.hausdorff_distances.append(val_hausdorff)
    
    def plot_metrics(self):
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot training loss
        axes[0].plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0].set_title('Training Loss per Epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot Dice score and Hausdorff distance if available
        if self.dice_scores:
            ax2 = axes[1]
            ax2.plot(self.epochs, self.dice_scores, 'g-', label='Dice Score')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Dice Score', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.grid(True)
            
            # Create second y-axis for Hausdorff Distance
            if self.hausdorff_distances:
                ax3 = ax2.twinx()
                ax3.plot(self.epochs, self.hausdorff_distances, 'r-', label='Hausdorff Distance')
                ax3.set_ylabel('Hausdorff Distance', color='r')
                ax3.tick_params(axis='y', labelcolor='r')
                
                # Add legend for both metrics
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax3.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
            else:
                ax2.legend()
                
        plt.tight_layout()
        
        # Save the figure
        metrics_path = os.path.join(self.log_dir, f"metrics_{self.timestamp}.png")
        plt.savefig(metrics_path)
        plt.close()
        
        return metrics_path