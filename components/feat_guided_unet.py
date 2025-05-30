import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureGuidedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=256):
        super().__init__()
        
        # Encoder path
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Feature attention modules
        self.feat_attn1 = self._feature_attention(64, feature_channels)
        self.feat_attn2 = self._feature_attention(128, feature_channels)
        self.feat_attn3 = self._feature_attention(256, feature_channels)
        self.feat_attn4 = self._feature_attention(512, feature_channels)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder path with skip connections
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)  # 512 + 512 from skip
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)  # 256 + 256 from skip
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)  # 128 + 128 from skip
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)  # 64 + 64 from skip
        
        # Final output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def _feature_attention(self, channels, feature_channels):
        return nn.Sequential(
            nn.Conv2d(feature_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, features):
        # Encoder
        enc1 = self.enc1(x)
        # Level 1 attention - original size (16x16)
        features_level1 = features
        enc1_attn = enc1 * self.feat_attn1(features_level1)
        
        x = self.pool(enc1_attn)
        enc2 = self.enc2(x)
        # Level 2 attention - downsample features by 2 (8x8)
        features_level2 = F.interpolate(features, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        enc2_attn = enc2 * self.feat_attn2(features_level2)
        
        x = self.pool(enc2_attn)
        enc3 = self.enc3(x)
        # Level 3 attention - downsample features by 4 (4x4)
        features_level3 = F.interpolate(features, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        enc3_attn = enc3 * self.feat_attn3(features_level3)
        
        x = self.pool(enc3_attn)
        enc4 = self.enc4(x)
        # Level 4 attention - downsample features by 8 (2x2)
        features_level4 = F.interpolate(features, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        enc4_attn = enc4 * self.feat_attn4(features_level4)
        
        # Bottleneck
        x = self.pool(enc4_attn)
        bottleneck = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(bottleneck)
        x = torch.cat((x, enc4_attn), dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat((x, enc3_attn), dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat((x, enc2_attn), dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat((x, enc1_attn), dim=1)
        x = self.dec1(x)
        
        # Final output
        return self.final_conv(x)
