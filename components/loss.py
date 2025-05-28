import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from monai.losses import DiceLoss, DiceCELoss, TverskyLoss, FocalLoss, HausdorffDTLoss
from torch.nn.functional import mse_loss, cosine_embedding_loss



class MultiTaskLoss(nn.Module):
    def __init__(self, 
                 dice_weight=1.0, 
                 ce_weight=1.0, 
                 boundary_weight=0.5, 
                 l2_weight=0.3, 
                 contrastive_weight=0.2,
                 focal_weight=0.3,
                 tversky_weight=0.2):
        super().__init__()
        
        # Segmentation losses
        self.dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
        self.cross_entropy = DiceCELoss(to_onehot_y=False, sigmoid=True)
        self.tversky_loss = TverskyLoss(to_onehot_y=False, sigmoid=True, alpha=0.7, beta=0.3)
        self.focal_loss = FocalLoss(to_onehot_y=False, gamma=2.0)
        
        # Boundary/Edge losses
        self.boundary_loss = HausdorffDTLoss(include_background=False)
        
        # Assign loss weights
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.l2_weight = l2_weight
        self.contrastive_weight = contrastive_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        
    def forward(self, 
                prediction, 
                target, 
                attention_maps=None, 
                img_features=None, 
                text_features=None):

        loss_dict = {}
        
        # 1. Segmentation losses
        dice = self.dice_loss(prediction, target)
        loss_dict['dice_loss'] = dice.item()
        
        ce = self.cross_entropy(prediction, target)
        loss_dict['ce_loss'] = ce.item()
        
        tversky = self.tversky_loss(prediction, target)
        loss_dict['tversky_loss'] = tversky.item()
        
        focal = self.focal_loss(prediction, target)
        loss_dict['focal_loss'] = focal.item()
        
        # 2. Boundary/Edge losses
        # Convert predictions to binary for boundary loss
        binary_pred = (prediction > 0.5).float()
        boundary = self.boundary_loss(binary_pred, target)
        loss_dict['boundary_loss'] = boundary.item()
        
        # 3. Initialize total loss with weighted components
        total_loss = (
            self.dice_weight * dice + 
            self.ce_weight * ce + 
            self.boundary_weight * boundary +
            self.focal_weight * focal +
            self.tversky_weight * tversky
        )
        
        # 4. Add anomaly localization loss (L2 loss) if attention maps are provided
        if attention_maps is not None and target is not None:
            # Use target mask as proxy for anomaly region
            # Resize target to match attention map size if needed
            if attention_maps.shape[-2:] != target.shape[-2:]:
                target_resized = F.interpolate(target, size=attention_maps.shape[-2:], 
                                               mode='nearest')
            else:
                target_resized = target
                
            # Calculate L2 loss between attention maps and target mask
            l2_loss_val = mse_loss(attention_maps, target_resized)
            total_loss += self.l2_weight * l2_loss_val
            loss_dict['l2_loss'] = l2_loss_val.item()
        
        # 5. Add multimodal alignment loss if image and text features are provided
        if img_features is not None and text_features is not None:
            # Ensure features have the same dimensions for cosine similarity
            if img_features.size(1) != text_features.size(1):
                # Project to common space if dimensions don't match
                projection_layer = nn.Linear(img_features.size(1), text_features.size(1)).to(img_features.device)
                img_features_proj = projection_layer(img_features)
                
                # Calculate cosine similarity loss (expect high similarity for matched pairs)
                target_sim = torch.ones(img_features.size(0)).to(img_features.device)
                contrastive_loss_val = cosine_embedding_loss(
                    img_features_proj, text_features, target_sim
                )
            else:
                # Calculate cosine similarity loss directly
                target_sim = torch.ones(img_features.size(0)).to(img_features.device)
                contrastive_loss_val = cosine_embedding_loss(
                    img_features, text_features, target_sim
                )
                
            total_loss += self.contrastive_weight * contrastive_loss_val
            loss_dict['contrastive_loss'] = contrastive_loss_val.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
