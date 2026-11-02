import os
import json
import torch
import torch.nn as nn

class TinyUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=4):
        super(TinyUNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        out = self.conv3(h)
        return out

def main():
    model_dir = "backend/ml/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Instantiate and save model weights
    model = TinyUNet(in_channels=5, out_channels=4)
    model_path = os.path.join(model_dir, "synthetic_liss4_unet.pt")
    
    # Save the state dict
    torch.save(model.state_dict(), model_path)
    print(f"Saved synthetic model to {model_path}")
    
    # Also save as a scripted module for easy trace/inference loading if needed
    try:
        scripted = torch.jit.script(model)
        jit_path = os.path.join(model_dir, "synthetic_liss4_unet_jit.pt")
        scripted.save(jit_path)
        print(f"Saved JIT-scripted model to {jit_path}")
    except Exception as e:
        print(f"Failed to script model: {e}")
        
    # Write metadata
    metadata = {
        "model_id": "synthetic_liss4_unet",
        "name": "Synthetic LISS-IV Cloud Removal UNet",
        "version": "1.0",
        "training_date": "2026-08-06",
        "band_order": ["Green", "Red", "NIR", "SWIR"],
        "input_channels": [
            "Green",
            "Red",
            "NIR",
            "SWIR",
            "cloud_mask"
        ],
        "normalization": {
            "mean": [0.15, 0.18, 0.35, 0.22, 0.0],
            "std": [0.08, 0.10, 0.15, 0.12, 1.0]
        },
        "patch_size": 512,
        "overlap": 64
    }
    
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {meta_path}")

if __name__ == "__main__":
    main()
