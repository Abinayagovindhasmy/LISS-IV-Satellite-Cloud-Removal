import os
import sys
import json
import time
# pyrefly: ignore[missing-import]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.visualizer import generate_liss4_comparison_figure, calculate_metrics
from tools.train_model import generate_paired_dataset, ConvBlock

class Pix2PixGenerator(nn.Module):
    """
    Generator Network for Pix2Pix cGAN (U-Net Architecture with Skip Connections).
    Input: 5 channels (Green, Red, NIR, SWIR, Cloud Mask)
    Output: 4 channels (Clear Green, Red, NIR, SWIR)
    """
    def __init__(self, in_channels=5, out_channels=4):
        super(Pix2PixGenerator, self).__init__()
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


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator for Pix2Pix cGAN.
    Inputs: Concat of Condition (5 ch) + Target/Generated Image (4 ch) = 9 channels.
    """
    def __init__(self, in_channels=9):
        super(PatchGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, cond, img):
        x = torch.cat([cond, img], dim=1)
        return self.net(x)


def train_pix2pix_gan(epochs=20, batch_size=4, lr=2e-4, lambda_l1=100.0):
    print("==========================================================")
    print("Starting Pix2Pix Conditional GAN Training Pipeline...")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Generate paired dataset
    clear_imgs, cloudy_imgs, masks = generate_paired_dataset(num_samples=32, img_size=256)
    inputs_5ch = np.concatenate([cloudy_imgs, masks[:, None, :, :]], axis=1) # (N, 5, H, W)
    targets_4ch = clear_imgs # (N, 4, H, W)

    train_x = torch.tensor(inputs_5ch, dtype=torch.float32)
    train_y = torch.tensor(targets_4ch, dtype=torch.float32)

    netG = Pix2PixGenerator(in_channels=5, out_channels=4).to(device)
    netD = PatchGANDiscriminator(in_channels=9).to(device)

    optG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    num_samples = len(train_x)
    print(f"Dataset size: {num_samples} paired samples. Training Pix2Pix cGAN for {epochs} epochs...")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        loss_G_sum = 0.0
        loss_D_sum = 0.0

        for i in range(0, num_samples, batch_size):
            indices = perm[i:i + batch_size]
            cond = train_x[indices].to(device)
            real_img = train_y[indices].to(device)
            bs = cond.size(0)

            # ---------------------
            #  1. Train Discriminator
            # ---------------------
            optD.zero_grad()
            
            fake_img = netG(cond)

            # Real patch evaluation
            pred_real = netD(cond, real_img)
            label_real = torch.ones_like(pred_real).to(device)
            loss_D_real = criterion_bce(pred_real, label_real)

            # Fake patch evaluation
            pred_fake = netD(cond, fake_img.detach())
            label_fake = torch.zeros_like(pred_fake).to(device)
            loss_D_fake = criterion_bce(pred_fake, label_fake)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optD.step()

            # -----------------
            #  2. Train Generator
            # -----------------
            optG.zero_grad()
            
            pred_fake_for_G = netD(cond, fake_img)
            loss_GAN = criterion_bce(pred_fake_for_G, label_real)
            loss_L1 = criterion_l1(fake_img, real_img)

            loss_G = loss_GAN + lambda_l1 * loss_L1
            loss_G.backward()
            optG.step()

            loss_G_sum += loss_G.item() * bs
            loss_D_sum += loss_D.item() * bs

        loss_G_avg = loss_G_sum / num_samples
        loss_D_avg = loss_D_sum / num_samples

        if epoch == 1 or epoch % 4 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss G: {loss_G_avg:.4f} | Loss D: {loss_D_avg:.4f}")

    training_time = time.time() - start_time
    print(f"Pix2Pix cGAN Training completed in {training_time:.2f} seconds!")

    def to_numpy(tensor):
        try:
            return tensor.cpu().detach().numpy()
        except Exception:
            return np.array(tensor.cpu().detach().tolist(), dtype=np.float32)

    # Evaluate model
    netG.eval()
    with torch.no_grad():
        test_in = train_x[0:1].to(device)
        test_out_t = netG(test_in).squeeze(0)
        test_out = to_numpy(test_out_t)
        test_cloudy = to_numpy(train_x[0, :4])
        test_ref = to_numpy(train_y[0])

    psnr_val, ssim_val = calculate_metrics(test_out, test_ref)
    print(f"Pix2Pix cGAN Evaluation -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    # Save generator weights
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/ml/models"))
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "pix2pix_gan.pt")
    torch.save(netG.state_dict(), weights_path)
    print(f"Saved Pix2Pix Generator weights to: {weights_path}")

    # Export comparison figure
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/outputs"))
    fig_path = os.path.join(outputs_dir, "pix2pix_gan_comparison.png")
    generate_liss4_comparison_figure(
        cloudy_img=test_cloudy,
        reconstructed_img=test_out,
        reference_img=test_ref,
        output_png_path=fig_path,
        psnr_val=psnr_val,
        ssim_val=ssim_val
    )
    print(f"Generated Pix2Pix cGAN 3-panel comparison output at: {fig_path}")
    return psnr_val, ssim_val, fig_path
if __name__ == "__main__":
    train_pix2pix_gan(epochs=20)
