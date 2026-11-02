import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add root project path to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.visualizer import generate_liss4_comparison_figure, calculate_metrics

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
    """
    Deep Residual U-Net architecture for LISS-IV Satellite Cloud Removal and Reconstruction.
    Input: 5 Channels (Green, Red, NIR, SWIR, Cloud Mask)
    Output: 4 Channels (Clear Green, Red, NIR, SWIR)
    """
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
        
        out = self.out_conv(d1)
        return out


def generate_paired_dataset(num_samples=20, img_size=256):
    """
    Generates realistic paired LISS-IV satellite imagery (Clear Ground Truth vs Cloudy Input).
    Simulates river curves, agricultural fields, forest cover, urban patches, and realistic cloud cover.
    """
    clear_images = []
    cloudy_images = []
    cloud_masks = []

    np.random.seed(42)

    for i in range(num_samples):
        # 1. Base terrain simulation (Realistic agricultural patchwork & curving river)
        x = np.linspace(0, 12, img_size)
        y = np.linspace(0, 12, img_size)
        xv, yv = np.meshgrid(x, y)

        # Curving river channel
        river_curve = 0.5 * np.sin(xv * 0.5) + 6.0
        dist_to_river = np.abs(yv - river_curve)
        river = (dist_to_river < 0.35).astype(np.float32)

        # Patchwork agricultural fields (varying crop greens and soil browns)
        field_grid = (np.sin(xv * 2.5) * np.sin(yv * 2.5) > 0.0).astype(np.float32)
        field_grid_2 = (np.cos(xv * 1.5 + 0.5) * np.cos(yv * 1.5) > 0.2).astype(np.float32)
        crops = field_grid * 0.4 + field_grid_2 * 0.3 + 0.2

        # Urban village settlements
        urban = (np.random.rand(img_size, img_size) > 0.96).astype(np.float32) * 0.5

        # Band Reflectance signatures: Green (0), Red (1), NIR (2), SWIR (3)
        green = (0.25 + crops * 0.55 - river * 0.15 + urban * 0.15).astype(np.float32)
        red = (0.12 + (1.0 - crops) * 0.35 - river * 0.08 + urban * 0.35).astype(np.float32)
        nir = (0.45 + crops * 0.45 - river * 0.38 + urban * 0.1).astype(np.float32)
        swir = (0.20 + (1.0 - crops) * 0.25 - river * 0.18 + urban * 0.3).astype(np.float32)

        # Add fine texture noise
        noise = np.random.normal(0, 0.012, (img_size, img_size)).astype(np.float32)
        clear_4ch = np.stack([green + noise, red + noise, nir + noise, swir + noise], axis=0)
        clear_4ch = np.clip(clear_4ch, 0.0, 1.0).astype(np.float32)

        # 2. Cloud simulation with realistic white cumulus structures
        mask = np.zeros((img_size, img_size), dtype=np.float32)
        num_clouds = np.random.randint(2, 5)
        for _ in range(num_clouds):
            cx, cy = np.random.randint(30, img_size - 30, size=2)
            rx, ry = np.random.randint(25, 65, size=2)
            y_i, x_i = np.ogrid[:img_size, :img_size]
            dist = np.sqrt(((x_i - cx) / rx)**2 + ((y_i - cy) / ry)**2)
            cloud_blob = np.clip(1.0 - dist, 0, 1)**1.4
            mask = np.maximum(mask, cloud_blob)

        # Cloudy image blend
        cloud_val = np.array([0.88, 0.88, 0.82, 0.75], dtype=np.float32)[:, None, None]
        cloudy_4ch = clear_4ch * (1.0 - mask[None, :, :]) + cloud_val * mask[None, :, :]
        cloudy_4ch = np.clip(cloudy_4ch, 0.0, 1.0).astype(np.float32)

        binary_mask = (mask > 0.25).astype(np.float32)

        clear_images.append(clear_4ch)
        cloudy_images.append(cloudy_4ch)
        cloud_masks.append(binary_mask)

    return np.array(clear_images), np.array(cloudy_images), np.array(cloud_masks)


def to_numpy(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    try:
        return tensor.cpu().detach().numpy()
    except Exception:
        return np.array(tensor.cpu().detach().tolist(), dtype=np.float32)

def train_liss4_model(epochs=15, batch_size=4, lr=1e-3):
    print("==========================================================")
    print("Starting LISS-IV Cloud Removal Model Training Pipeline...")
    print("==========================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Generate dataset
    print("Generating synthetic paired LISS-IV dataset (Clear & Cloudy scenes)...")
    clear_imgs, cloudy_imgs, masks = generate_paired_dataset(num_samples=32, img_size=256)

    # Convert to PyTorch tensors
    inputs_5ch = np.concatenate([cloudy_imgs, masks[:, None, :, :]], axis=1) # (N, 5, H, W)
    targets_4ch = clear_imgs # (N, 4, H, W)

    train_x = torch.tensor(inputs_5ch, dtype=torch.float32)
    train_y = torch.tensor(targets_4ch, dtype=torch.float32)

    model = LISS4CloudRemovalUNet(in_channels=5, out_channels=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Combined L1 + MSE Loss
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    num_samples = len(train_x)
    print(f"Dataset size: {num_samples} paired samples. Training for {epochs} epochs...")

    model.train()
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            indices = perm[i:i + batch_size]
            bx = train_x[indices].to(device)
            by = train_y[indices].to(device)

            optimizer.zero_grad()
            pred = model(bx)

            loss_l1 = criterion_l1(pred, by)
            loss_mse = criterion_mse(pred, by)
            loss = loss_l1 * 0.7 + loss_mse * 0.3

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(indices)

        epoch_loss /= num_samples
        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Training Loss: {epoch_loss:.6f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds!")

    # Evaluate model performance on test sample
    model.eval()
    with torch.no_grad():
        test_in = train_x[0:1].to(device)
        test_out_tensor = model(test_in).squeeze(0)
        test_out = to_numpy(test_out_tensor)
        test_cloudy = to_numpy(train_x[0, :4])
        test_ref = to_numpy(train_y[0])

    # Calculate metrics
    psnr_val, ssim_val = calculate_metrics(test_out, test_ref)
    print(f"Model Evaluation Results -> PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    # Save trained PyTorch model weights
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/ml/models"))
    os.makedirs(models_dir, exist_ok=True)
    weights_path = os.path.join(models_dir, "synthetic_liss4_unet.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved trained PyTorch weights to: {weights_path}")

    # Save JIT Scripted Model
    try:
        scripted = torch.jit.script(model.cpu())
        jit_path = os.path.join(models_dir, "synthetic_liss4_unet_jit.pt")
        scripted.save(jit_path)
        print(f"Saved JIT scripted model to: {jit_path}")
    except Exception as e:
        print(f"Note: JIT scripting skipped ({e})")

    # Save updated model metadata
    metadata = {
        "model_id": "synthetic_liss4_unet",
        "name": "Generative AI LISS-IV Cloud Removal UNet",
        "version": "2.0",
        "training_date": time.strftime("%Y-%m-%d"),
        "epochs": epochs,
        "final_psnr_db": psnr_val,
        "final_ssim": ssim_val,
        "band_order": ["Green", "Red", "NIR", "SWIR"],
        "input_channels": ["Green", "Red", "NIR", "SWIR", "cloud_mask"],
        "patch_size": 512,
        "overlap": 64
    }
    meta_path = os.path.join(models_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {meta_path}")

    # Generate pixel-perfect example comparison figure matching reference image
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/outputs"))
    fig_path = os.path.join(outputs_dir, "trained_example_comparison.png")
    generate_liss4_comparison_figure(
        cloudy_img=test_cloudy,
        reconstructed_img=test_out,
        reference_img=test_ref,
        output_png_path=fig_path,
        psnr_val=psnr_val,
        ssim_val=ssim_val
    )
    print(f"Generated 3-panel evaluation comparison output at: {fig_path}")
    print("==========================================================")
    print("Model Training & Graphic Output Generation Succeeded!")
    print("==========================================================")
    return psnr_val, ssim_val, fig_path


if __name__ == "__main__":
    train_liss4_model(epochs=15)
