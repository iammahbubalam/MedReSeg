
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from transformers import BertTokenizer, BertModel
from components import TransformerDecoderFusion
from components.feat_guided_unet import FeatureGuidedUNet



class MedCLIPUNet(nn.Module):
    def __init__(self, num_classes=1, img_size=448):
        super().__init__()
        
        # Load pre-trained MedCLIP model
        self.clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
        
        # Freeze the CLIP model to preserve the pre-trained weights
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # BERT text encoder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze the text encoder to preserve the pre-trained weights
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Dimensions
        self.image_embed_dim = self.clip_model.visual.output_dim
        self.text_embed_dim = 768  # BERT base hidden size
        
        # Transformer Decoder Fusion module
        self.fusion = TransformerDecoderFusion(
            text_dim=self.text_embed_dim, 
            image_dim=self.image_embed_dim,
            hidden_dim=768,
            nhead=8,
            num_layers=3
        )
        
        # Create a feature converter to prepare for UNet
        self.feature_converter = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        
        # UNet for segmentation
        self.unet = FeatureGuidedUNet(
            in_channels=3,
            out_channels=num_classes,
            feature_channels=256
        )
        
        # Feature reshape parameters
        self.img_size = img_size
        self.patch_size = 28  # Adjusted to 28 for compatibility with 448x448 input
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # Store intermediate features for loss calculation
        self.last_image_features = None
        self.last_text_features = None
        self.last_attention_maps = None
        
    def encode_text(self, text):
        # Tokenize text
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=77, 
                              return_tensors="pt").to(next(self.parameters()).device)
        
        # Get BERT embeddings for all tokens
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            
        # Use all token embeddings (excluding padding tokens)
        attention_mask = inputs['attention_mask']
        text_features = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        
        return text_features, attention_mask
    
    def encode_image(self, x):
        # Extract image features from CLIP's vision encoder
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
        return image_features
        
    def forward(self, images, text_prompts):
        batch_size = images.shape[0]
        
        # Encode images 
        image_features = self.encode_image(images)  # [batch_size, embed_dim]
        self.last_image_features = image_features  # Store for loss calculation
        
        # Encode text with all token embeddings
        text_features, attention_mask = self.encode_text(text_prompts)  # [batch_size, seq_length, text_embed_dim]
        self.last_text_features = text_features.mean(dim=1)  # Average tokens for simple feature vector
        
        # Add sequence dimension to image features
        image_features = image_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Apply transformer decoder fusion with token-level text features
        fused_features = self.fusion(image_features, text_features, attention_mask)  # [batch_size, 1, hidden_dim]
        
        # Convert fused features to UNet-compatible format
        features = self.feature_converter(fused_features.squeeze(1))  # [batch_size, 256]
        
        # Reshape features to spatial grid for UNet
        h = w = int(self.img_size // self.patch_size)  # For 448/28 = 16
        features = features.view(batch_size, 256, 1, 1).expand(batch_size, 256, h, w)
        
        # Store attention maps for loss calculation
        # We use feature maps as a proxy for attention, since they highlight regions of interest
        self.last_attention_maps = features.mean(dim=1, keepdim=True)
        
        # Resize original images to match the UNet input size
        resized_images = F.interpolate(images, size=(h, w), 
                                      mode='bilinear', align_corners=False)
        
        # Pass through feature-guided UNet
        segmentation = self.unet(resized_images, features)
        
        # Upsample the segmentation back to original image size
        segmentation = F.interpolate(segmentation, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)
        
        return segmentation
    
    def get_last_features(self):
        """Return the last computed features for loss calculation"""
        return {
            'image_features': self.last_image_features,
            'text_features': self.last_text_features,
            'attention_maps': self.last_attention_maps
        }
