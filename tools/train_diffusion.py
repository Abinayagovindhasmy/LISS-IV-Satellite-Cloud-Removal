import os
import sys
import math
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.visualizer import generate_liss4_comparison_figure, calculate_metrics
from tools.train_model import generate_paired_dataset

def get_timestep_embedding(timesteps, embedding_dim=64):
    """
    Sinusoidal positional embedding for diffusion timesteps.
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x, time_emb):
        h = self.relu(self.bn1(self.conv1(x)))
        # Add time embedding spatially
        t_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + t_emb
        h = self.relu(self.bn2(self.conv2(h)))
        return h


class ConditionalDiffusionUNet(nn.Module):
    """
    Conditional Denoising Diffusion UNet (DDPM) for Satellite Cloud Removal.
    Inputs:
      - x_t: Noisy ground truth target (4 channels)
      - t: Timesteps (1D Tensor)
      - cond: Cloudy condition image + mask (5 channels)
    Output:
      - Predicted noise epsilon (4 channels)
    """
    def __init__(self, in_channels=4, cond_channels=5, out_channels=4, time_dim=64):
        super(ConditionalDiffusionUNet, self).__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        input_dim = in_channels + cond_channels # 4 + 5 = 9 channels
        
        self.res1 = ResidualBlock(input_dim, 32, time_dim)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.res2 = ResidualBlock(32, 64, time_dim)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = ResidualBlock(64, 128, time_dim)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(128, 64, time_dim)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(64, 32, time_dim)
        
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x_t, t, cond):
        t_emb = get_timestep_embedding(t, embedding_dim=64)
        t_emb = self.time_mlp(t_emb)
        
        x = torch.cat([x_t, cond], dim=1) # (B, 9, H, W)
        
        e1 = self.res1(x, t_emb)
        p1 = self.pool1(e1)
        
        e2 = self.res2(p1, t_emb)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2, t_emb)
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb)
        
        return self.out_conv(d1)


class DDPMScheduler:
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model, cond, shape):
        model.eval()
        bs = cond.size(0)
        x = torch.randn(shape, device=self.device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((bs,), i, device=self.device, dtype=torch.long)
            pred_noise = model(x, t, cond)
            
            beta_t = self.betas[i]
            sqrt_alpha_t = torch.sqrt(self.alphas[i])
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[i]
            
            # Predict x_0 and step backward
            mean = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
                
        return torch.clamp(x, 0.0, 1.0)


def train_diffusion_model(epochs=15, batch_size=4, lr=1e-3, num_timesteps=100):
    print("==========================================================")
    print("Starting Conditional Diffusion Model (DDPM) Training...")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Generate paired dataset
    clear_imgs, cloudy_imgs, masks = generate_paired_dataset(num_samples=32, img_size=256)
    inputs_5ch = np.concatenate([cloudy_imgs, masks[:, None, :, :]], axis=1) # (N, 5, H, W)
    targets_4ch = clear_imgs # (N, 4, H, W)

    train_x = torch.tensor(inputs_5ch, dtype=torch.float32)
    train_y = torch.tensor(targets_4ch, dtype=torch.float32)

    model = ConditionalDiffusionUNet(in_channels=4, cond_channels=5, out_channels=4).to(device)
    scheduler = DDPMScheduler(timesteps=num_timesteps, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    num_samples = len(train_x)
    print(f"Dataset size: {num_samples} paired samples. Training DDPM for {epochs} epochs...")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0

        model.train()
        for i in range(0, num_samples, batch_size):
            indices = perm[i:i + batch_size]
            cond = train_x[indices].to(device)
            real_img = train_y[indices].to(device)
            bs = cond.size(0)

            # Random timesteps for batch
            t = torch.randint(0, scheduler.timesteps, (bs,), device=device).long()
            
            # Add noise according to scheduler
            noisy_img, noise = scheduler.add_noise(real_img, t)

            optimizer.zero_grad()
            pred_noise = model(noisy_img, t, cond)

            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * bs

        epoch_loss /= num_samples

        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Noise Prediction MSE Loss: {epoch_loss:.6f}")

    training_time = time.time() - start_time
    print(f"Conditional Diffusion Model Training completed in {training_time:.2f} seconds!")

    # Evaluate model via reverse sampling
    print("Running reverse diffusion sampling for evaluation...")
    test_in = train_x[0:1].to(device)
    sampled_out = scheduler.sample(model, test_in, shape=(1, 4, 256, 256))
    
    test_out = sampled_out.cpu().squeeze(0).numpy()
    test_cloudy = train_x[0, :4].numpy()
    test_ref = train_y[0].numpy()

    psnr_val, ssim_val = calculate_metrics(test_out, test_ref)
    print(f"Conditional Diffusion Model Evaluation -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    # Save model weights
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/ml/models"))
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "diffusion_model.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved Conditional Diffusion Model weights to: {weights_path}")

    # Export comparison figure
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/outputs"))
    fig_path = os.path.join(outputs_dir, "diffusion_model_comparison.png")
    generate_liss4_comparison_figure(
        cloudy_img=test_cloudy,
        reconstructed_img=test_out,
        reference_img=test_ref,
        output_png_path=fig_path,
        psnr_val=psnr_val,
        ssim_val=ssim_val
    )
    print(f"Generated Conditional Diffusion 3-panel comparison output at: {fig_path}")
    return psnr_val, ssim_val, fig_path


if __name__ == "__main__":
    train_diffusion_model(epochs=15)
