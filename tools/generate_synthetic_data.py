import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def generate_synthetic_liss4():
    data_dir = "backend/uploads"
    os.makedirs(data_dir, exist_ok=True)
    
    # Image specifications
    width, height = 1024, 1024
    crs = "EPSG:4326"
    
    # Define affine transform (somewhere around Dehradun, India)
    west, north = 78.0, 30.3
    pixel_size = 0.00005 # ~5m resolution
    transform = from_origin(west, north, pixel_size, pixel_size)
    
    # 1. Generate background bands (reflectance between 0.0 and 1.0)
    # Band 1: Green, Band 2: Red, Band 3: NIR, Band 4: SWIR
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    xv, yv = np.meshgrid(x, y)
    
    # Base terrain (river, forest, urban)
    river = (np.sin(xv + yv * 0.5) > 0.8).astype(np.float32) * 0.05
    forest = (np.sin(xv * 0.3) * np.cos(yv * 0.3) > 0.0).astype(np.float32) * 0.2 + 0.15
    urban = (np.random.rand(height, width) > 0.95).astype(np.float32) * 0.4
    
    green = 0.1 + forest * 0.8 + river * 0.2 + urban * 0.3
    red = 0.08 + forest * 0.2 + river * 0.1 + urban * 0.5
    nir = 0.4 + forest * 0.9 + river * 0.02 + urban * 0.2
    swir = 0.2 + forest * 0.3 + river * 0.01 + urban * 0.4
    
    # Add noise
    noise = np.random.normal(0, 0.01, (height, width)).astype(np.float32)
    green = np.clip(green + noise, 0, 1)
    red = np.clip(red + noise, 0, 1)
    nir = np.clip(nir + noise, 0, 1)
    swir = np.clip(swir + noise, 0, 1)
    
    # 2. Simulate clouds
    # Center cloud at (400, 400) and (700, 300)
    cloud_mask = np.zeros((height, width), dtype=np.float32)
    for cx, cy, r in [(400, 400, 180), (700, 300, 120)]:
        y_indices, x_indices = np.ogrid[:height, :width]
        dist = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        # Soft-edged cloud
        mask_part = np.clip((r - dist) / 40.0, 0, 1)
        cloud_mask = np.maximum(cloud_mask, mask_part)
        
    # Cloud values are bright in all bands (reflectance near 0.8 - 0.9)
    green = green * (1 - cloud_mask) + 0.85 * cloud_mask
    red = red * (1 - cloud_mask) + 0.85 * cloud_mask
    nir = nir * (1 - cloud_mask) + 0.80 * cloud_mask
    swir = swir * (1 - cloud_mask) + 0.75 * cloud_mask
    
    # Write cloudy GeoTIFF (4 bands)
    cloudy_path = os.path.join(data_dir, "synthetic_liss4_cloudy.tif")
    with rasterio.open(
        cloudy_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=4,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=0.0
    ) as dst:
        dst.write(green, 1)
        dst.write(red, 2)
        dst.write(nir, 3)
        dst.write(swir, 4)
        dst.set_band_description(1, "Green")
        dst.set_band_description(2, "Red")
        dst.set_band_description(3, "NIR")
        dst.set_band_description(4, "SWIR")
        
    print(f"Saved synthetic cloudy GeoTIFF to {cloudy_path}")
    
    # Write binary cloud mask GeoTIFF (1 band)
    mask_path = os.path.join(data_dir, "synthetic_liss4_mask.tif")
    binary_mask = (cloud_mask > 0.3).astype(np.uint8)
    with rasterio.open(
        mask_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='uint8',
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(binary_mask, 1)
        dst.set_band_description(1, "Cloud Mask")
        
    print(f"Saved synthetic cloud mask GeoTIFF to {mask_path}")

if __name__ == "__main__":
    generate_synthetic_liss4()
