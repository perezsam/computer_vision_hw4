import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample by a factor of 2
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Convolutions after concatenating with the skip connection
        self.conv = nn.Sequential(
            nn.Conv2d((in_channels // 2) + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Automatically handle any 1-pixel rounding differences during upsampling
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PretrainedResNet34UNet(nn.Module):
    def __init__(self, out_classes=6):
        super().__init__()
        # Load the pre-trained ImageNet encoder
        encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Extract specific layers to use as the U-Net contracting path (Encoder)
        self.enc0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu) 
        self.maxpool = encoder.maxpool 
        self.enc1 = encoder.layer1     
        self.enc2 = encoder.layer2     
        self.enc3 = encoder.layer3     
        self.enc4 = encoder.layer4     # Bottleneck layer

        # Build the expanding path (Decoder)
        self.dec4 = DecoderBlock(512, 256, 256) 
        self.dec3 = DecoderBlock(256, 128, 128) 
        self.dec2 = DecoderBlock(128, 64, 64)   
        self.dec1 = DecoderBlock(64, 64, 64)    

        # Final upsample to return to the original image resolution
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, out_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        s0 = self.enc0(x)
        x = self.maxpool(s0)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.enc4(s3) # Bottleneck

        # --- Decoder Path ---
        d4 = self.dec4(b, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        # --- Output ---
        out = self.final_up(d1)
        return self.final_conv(out)