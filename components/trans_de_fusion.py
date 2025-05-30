import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderFusion(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim=768, nhead=8, dim_feedforward=2048, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Project text and image features to the same dimension
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        
        # Position encodings for the transformer
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Create transformer decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Final projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, image_features, text_features, text_mask=None):
        # Project both to the same dimension
        image_features = self.image_projection(image_features)  # [batch_size, 1, hidden_dim]
        text_features = self.text_projection(text_features)     # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding to image features
        image_features = image_features + self.pos_encoder
        
        # Create attention mask from text mask
        if text_mask is not None:
            # Convert boolean mask to attention mask (1.0 for tokens to attend to, 0.0 for padding)
            # Transform from [batch_size, seq_len] -> [batch_size, 1, seq_len]
            attn_mask = text_mask.unsqueeze(1).float()
            # Invert the mask and set to -inf (0 -> -inf, 1 -> 0)
            attn_mask = (1.0 - attn_mask) * -10000.0
        else:
            attn_mask = None
        
        # In transformer decoder:
        # - tgt = image features (what we're generating/modifying)
        # - memory = text features (what we're attending to)
        # Here image features are considered the target that attends to text features as memory
        fused_features = self.transformer_decoder(
            tgt=image_features,
            memory=text_features,
            tgt_key_padding_mask=None,  # No padding in image features
            memory_key_padding_mask=None if text_mask is None else ~text_mask.bool()  # Convert mask from [1=token, 0=pad] to [False=token, True=pad]
        )
        
        # Final projection
        output = self.output_projection(fused_features)
        
        return output