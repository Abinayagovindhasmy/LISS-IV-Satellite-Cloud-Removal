import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFilter

sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.ml.model_loader import LISS4CloudRemovalUNet

try:
    import rasterio
except ImportError:
    rasterio = None


def read_image(path):
    if rasterio is not None and path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            raw = source.read()
        data = raw.astype(np.float32)
        if raw.dtype == np.uint8:
            data /= 255.0
        elif raw.dtype == np.uint16:
            data /= 10000.0 if float(raw.max()) <= 10000.0 else 65535.0
        else:
            maximum = float(np.nanmax(data))
            if maximum > 1.0:
                data /= 10000.0 if maximum <= 10000.0 else maximum
    else:
        data = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        data = np.transpose(data, (2, 0, 1))
        data /= 255.0
        # Match inference.py's RGB-to-four-channel convention: simulate SWIR
        # from the first two displayed channels, rather than copying blue.
        swir = (data[0:1] * 0.5) + (data[1:2] * 0.5)
        data = np.concatenate([data, swir], axis=0)

    # Match inference.py's GeoTIFF handling for rasters with fewer than four bands.
    if data.shape[0] < 4:
        data = np.concatenate([data, np.tile(data[0:1], (4 - data.shape[0], 1, 1))], axis=0)
    return np.clip(data[:4], 0.0, 1.0)


def read_mask(path, shape):
    if path is None:
        return None
    mask = read_image(path)[0]
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {shape}")
    return (mask > 0.5).astype(np.float32)


def detect_cloud_mask(image):
    """Keep the training mask convention aligned with inference.py."""
    visible = np.clip(image[:3], 0.0, 1.0)
    high = np.max(visible, axis=0)
    low = np.min(visible, axis=0)
    brightness = np.mean(visible, axis=0)
    valid = high > 0.02
    valid_brightness = brightness[valid]
    if valid_brightness.size == 0:
        return np.zeros(brightness.shape, dtype=np.float32)

    threshold = float(np.clip(np.percentile(valid_brightness, 78), 0.28, 0.68))
    candidates = valid & (brightness >= threshold) & (low / np.maximum(high, 1e-6) >= 0.62)
    mask = Image.fromarray(candidates.astype(np.uint8) * 255)
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = mask.filter(ImageFilter.MaxFilter(3))
    return (np.asarray(mask) > 0).astype(np.float32)


def crop_pair(cloudy, clear, mask, patch_size):
    height, width = cloudy.shape[1:]
    if clear.shape[1:] != (height, width):
        raise ValueError("Cloudy and clear images must have identical dimensions")
    if height < patch_size or width < patch_size:
        raise ValueError(f"Images must be at least {patch_size}x{patch_size}")
    y = np.random.randint(0, height - patch_size + 1)
    x = np.random.randint(0, width - patch_size + 1)
    cloudy_crop = cloudy[:, y:y + patch_size, x:x + patch_size]
    clear_crop = clear[:, y:y + patch_size, x:x + patch_size]
    if mask is None:
        mask_crop = detect_cloud_mask(cloudy)[y:y + patch_size, x:x + patch_size]
    else:
        mask_crop = mask[y:y + patch_size, x:x + patch_size]
    return np.concatenate([cloudy_crop, mask_crop[None]], axis=0), clear_crop


def train(cloudy_dir, clear_dir, mask_dir, epochs, steps_per_epoch, patch_size, learning_rate):
    clear_files = {path.stem: path for path in Path(clear_dir).iterdir() if path.is_file()}
    pairs = []
    for cloudy_path in Path(cloudy_dir).iterdir():
        if cloudy_path.is_file() and cloudy_path.stem in clear_files:
            mask_path = None if mask_dir is None else Path(mask_dir) / cloudy_path.name
            pairs.append((cloudy_path, clear_files[cloudy_path.stem], mask_path if mask_path and mask_path.exists() else None))
    if not pairs:
        raise ValueError("No matching cloudy/clear files found by filename stem")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LISS4CloudRemovalUNet().to(device)
    weights_path = Path(__file__).parent.parent / "backend" / "ml" / "models" / "synthetic_liss4_unet.pt"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    l1_loss = nn.L1Loss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            cloudy_path, clear_path, mask_path = pairs[np.random.randint(len(pairs))]
            cloudy = read_image(cloudy_path)
            clear = read_image(clear_path)
            mask = read_mask(mask_path, cloudy.shape[1:])
            input_patch, target_patch = crop_pair(cloudy, clear, mask, patch_size)
            inputs = torch.from_numpy(input_patch[None]).to(device)
            targets = torch.from_numpy(target_patch[None]).to(device)
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = l1_loss(prediction, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} loss={running_loss / steps_per_epoch:.6f}")

    model.eval()
    torch.save(model.cpu().state_dict(), weights_path)
    print(f"Saved fine-tuned weights to {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LISS-IV cloud removal on aligned image pairs")
    parser.add_argument("--cloudy-dir", required=True)
    parser.add_argument("--clear-dir", required=True)
    parser.add_argument("--mask-dir")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    train(**vars(args))
