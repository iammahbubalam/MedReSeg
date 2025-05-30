import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
# from transformers import BertModel, BertTokenizer
from .trans_de_fusion import TransformerDecoderFusion
from .feat_guided_unet import FeatureGuidedUNet

from transformers import AutoTokenizer, AutoModel


class MedCLIPUNet(nn.Module):
    def __init__(self, num_classes=1, img_size=256):  # Changed default to 256
        super().__init__()
        
        # Load pre-trained MedCLIP model
        self.clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:luhuitong/CLIP-ViT-L-14-448px-MedICaT-ROCO')
        
        # Freeze the CLIP model to preserve the pre-trained weights
        for param in self.clip_model.parameters(): 
            param.requires_grad = False
            
        # BERT text encoder
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
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
            nhead=12,
            num_layers=6,
            dim_feedforward=3072,
            dropout=0.1
    
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
        
        # Feature reshape parameters - Fixed for proper sizing
        self.img_size = img_size
        # Use a smaller patch size to get reasonable feature map size
        if img_size <= 256:
            self.patch_size = 16  # 256/16 = 16x16 feature map
        elif img_size <= 448:
            self.patch_size = 28  # 448/28 = 16x16 feature map
        else:
            self.patch_size = img_size // 16  # Always aim for ~16x16 grid
            
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
        
        if x.shape[-1] != 448:
            x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=False)
        # Extract image features from CLIP's vision encoder
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)
        return image_features
        
    def forward(self, images, text_prompts):
        batch_size = images.shape[0]
        
        # Encode images (resized to 448x448 for CLIP)
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
        
        # Reshape features to spatial grid for UNet - FIXED calculation
        h = w = int(self.img_size // self.patch_size)  # For 256/16 = 16x16
        features = features.view(batch_size, 256, 1, 1).expand(batch_size, 256, h, w)
        
        # Store attention maps for loss calculation
        self.last_attention_maps = features.mean(dim=1, keepdim=True)
        
        # Resize original images to match the feature map size
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

if __name__ == '__main__':
    # Configuration for the test
    num_classes_test = 1
    img_size_test = 256 # Should match the default or be passed to MedCLIPUNet
    batch_size_test = 2

    print(f"--- Testing MedCLIPUNet with dummy data ---")
    print(f"Model parameters: num_classes={num_classes_test}, img_size={img_size_test}")
    print(f"Test data: batch_size={batch_size_test}")

    # Instantiate the model
    # Ensure the device is consistent, defaulting to CPU for this test if CUDA not explicitly handled
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = MedCLIPUNet(num_classes=num_classes_test, img_size=img_size_test).to(device)
        model.eval() # Set to evaluation mode for testing (disables dropout, etc.)
        print("MedCLIPUNet model instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating MedCLIPUNet model: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # Create dummy input data
    # Dummy images: (batch_size, 3, img_size, img_size)
    # Assuming input images are 3-channel RGB
    dummy_images = torch.randn(batch_size_test, 3, img_size_test, img_size_test).to(device)
    
    # Dummy text prompts
    dummy_text_prompts = [
        "Segment the liver in this CT scan.",
        "Highlight the tumor region in the MRI image."
    ]
    if batch_size_test == 1:
        dummy_text_prompts = [dummy_text_prompts[0]]
    elif batch_size_test > 2:
        dummy_text_prompts = dummy_text_prompts * (batch_size_test // 2) + dummy_text_prompts[:batch_size_test % 2]


    print(f"\\nPerforming forward pass...")
    print(f"  Input images shape: {dummy_images.shape}")
    print(f"  Input text prompts (example): '{dummy_text_prompts[0]}'")

    try:
        # Forward pass
        with torch.no_grad(): # No need to track gradients during testing
            segmentation_output = model(dummy_images, dummy_text_prompts)
        
        print(f"\\nForward pass successful.")
        print(f"  Output segmentation mask shape: {segmentation_output.shape}")

        # Verify output shape
        expected_output_shape = (batch_size_test, num_classes_test, img_size_test, img_size_test)
        if segmentation_output.shape == expected_output_shape:
            print(f"  Output shape is as expected.")
        else:
            print(f"  WARNING: Output shape {segmentation_output.shape} does not match expected {expected_output_shape}.")

        # Test get_last_features
        print(f"\\nTesting get_last_features()...")
        last_features = model.get_last_features()
        if last_features['image_features'] is not None:
            print(f"  Shape of last_image_features: {last_features['image_features'].shape}")
        else:
            print(f"  last_image_features is None")
        if last_features['text_features'] is not None:
            print(f"  Shape of last_text_features: {last_features['text_features'].shape}")
        else:
            print(f"  last_text_features is None")
        if last_features['attention_maps'] is not None:
            print(f"  Shape of last_attention_maps: {last_features['attention_maps'].shape}")
        else:
            print(f"  last_attention_maps is None")

    except Exception as e:
        print(f"\\nError during forward pass or feature retrieval: {e}")
        import traceback
        traceback.print_exc()

    print("\\n--- MedCLIPUNet test finished ---")
