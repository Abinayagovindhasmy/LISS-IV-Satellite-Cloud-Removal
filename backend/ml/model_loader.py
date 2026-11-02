import os
import sys
import json
import logging
import torch
import torch.nn as nn

# Ensure project root is in sys.path for importing tools
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.config import settings

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LISS4CloudRemovalUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=4):
        super(LISS4CloudRemovalUNet, self).__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)

from tools.train_pix2pix_gan import Pix2PixGenerator
from tools.train_diffusion import ConditionalDiffusionUNet, DDPMScheduler

class ModelRegistry:
    def __init__(self):
        self.models_dir = settings.MODEL_DIR

    def get_available_models(self):
        return [
            {
                "model_id": "synthetic_liss4_unet",
                "name": "Generative U-Net (LISS-IV Baseline)",
                "version": "2.0",
                "architecture": "Deep Residual U-Net",
                "band_order": ["Green", "Red", "NIR", "SWIR"],
                "input_channels": ["Green", "Red", "NIR", "SWIR", "cloud_mask"],
                "patch_size": 512,
                "overlap": 64
            },
            {
                "model_id": "pix2pix_gan",
                "name": "Pix2Pix Conditional GAN (cGAN)",
                "version": "1.0",
                "architecture": "cGAN with 70x70 PatchGAN Discriminator",
                "band_order": ["Green", "Red", "NIR", "SWIR"],
                "input_channels": ["Green", "Red", "NIR", "SWIR", "cloud_mask"],
                "patch_size": 512,
                "overlap": 64
            },
            {
                "model_id": "diffusion_model",
                "name": "Conditional Diffusion Model (DDPM)",
                "version": "1.0",
                "architecture": "Time-Embedded Denoising Diffusion Model",
                "band_order": ["Green", "Red", "NIR", "SWIR"],
                "input_channels": ["Green", "Red", "NIR", "SWIR", "cloud_mask"],
                "patch_size": 512,
                "overlap": 64
            }
        ]

    def load_model(self, model_id: str = "synthetic_liss4_unet", in_channels: int = 5, out_channels: int = 4):
        if model_id == "pix2pix_gan":
            model = Pix2PixGenerator(in_channels=in_channels, out_channels=out_channels)
            weights_name = "pix2pix_gan.pt"
        elif model_id == "diffusion_model":
            model = ConditionalDiffusionUNet(in_channels=4, cond_channels=5, out_channels=4)
            weights_name = "diffusion_model.pt"
        else:
            model = LISS4CloudRemovalUNet(in_channels=in_channels, out_channels=out_channels)
            weights_name = "synthetic_liss4_unet.pt"

        weights_path = os.path.join(self.models_dir, weights_name)
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights for '{model_id}' were not found: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load compatible weights for model '{model_id}': {exc}") from exc

        logger.info(f"Loaded weights for model '{model_id}' from {weights_path}")
            
        model.eval()
        return model

model_registry = ModelRegistry()
