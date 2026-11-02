import os
import time
import logging
from typing import Optional
import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch

try:
    import rasterio
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

from backend.config import settings
from backend.ml.model_loader import model_registry
from backend.ml.visualizer import (
    calculate_evaluation_metrics,
    generate_liss4_comparison_figure,
)
from backend.rag.spatial_retriever import retrieve_clear_reference_patches

logger = logging.getLogger(__name__)

def normalize_raster_bands(data: np.ndarray) -> np.ndarray:
    """Scale common 8-bit, 16-bit DN, and floating reflectance rasters to [0, 1]."""
    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0
    if data.dtype == np.uint16:
        scale = 10000.0 if float(data.max()) <= 10000.0 else 65535.0
        return data.astype(np.float32) / scale

    normalized = data.astype(np.float32)
    maximum = float(np.nanmax(normalized))
    if maximum > 1.0:
        scale = 10000.0 if maximum <= 10000.0 else maximum
        normalized /= scale
    return normalized


def detect_cloud_mask(image_data: np.ndarray, valid_mask: np.ndarray = None) -> np.ndarray:
    """Detect bright, low-chroma cloud pixels while excluding invalid image borders."""
    visible = np.clip(image_data[:3], 0.0, 1.0)
    high = np.max(visible, axis=0)
    low = np.min(visible, axis=0)
    brightness = np.mean(visible, axis=0)
    whiteness = low / np.maximum(high, 1e-6)

    if valid_mask is None:
        valid_mask = high > 0.02
    else:
        valid_mask = valid_mask.astype(bool)

    valid_brightness = brightness[valid_mask]
    if valid_brightness.size == 0:
        return np.zeros(brightness.shape, dtype=np.float32)

    # Adapt to scene brightness without labeling saturated red vegetation or soil as cloud.
    threshold = float(np.clip(np.percentile(valid_brightness, 78), 0.28, 0.68))
    candidates = valid_mask & (brightness >= threshold) & (whiteness >= 0.62)
    mask_image = Image.fromarray(candidates.astype(np.uint8) * 255)
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
    return (np.asarray(mask_image) > 0).astype(np.float32)


def blend_cloud_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    cloud_mask: np.ndarray,
    clear_reference: np.ndarray = None
) -> np.ndarray:
    mask = cloud_mask[np.newaxis, :, :]
    replacement = clear_reference if clear_reference is not None else reconstructed
    return np.clip(original * (1.0 - mask) + replacement * mask, 0.0, 1.0)


def create_hanning_window(height, width):
    win_y = np.hanning(height)
    win_x = np.hanning(width)
    window2d = np.outer(win_y, win_x)
    return np.maximum(window2d, 1e-4)


def reconstruct_with_model(data, cloud_mask, model, model_id, patch_size, overlap, progress_callback):
    _, height, width = data.shape
    output_accum = np.zeros((4, height, width), dtype=np.float32)
    weight_accum = np.zeros((height, width), dtype=np.float32)
    stride = patch_size - overlap
    if stride <= 0:
        stride = patch_size // 2

    pad_h = (patch_size - (height % patch_size)) % patch_size
    pad_w = (patch_size - (width % patch_size)) % patch_size
    padded_data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    padded_mask = np.pad(cloud_mask, ((0, pad_h), (0, pad_w)), mode="reflect")
    p_height, p_width = padded_data.shape[1], padded_data.shape[2]
    window2d = create_hanning_window(patch_size, patch_size)
    y_steps = sorted(set(list(range(0, p_height - patch_size + 1, stride)) + [p_height - patch_size]))
    x_steps = sorted(set(list(range(0, p_width - patch_size + 1, stride)) + [p_width - patch_size]))
    total_steps = len(y_steps) * len(x_steps)
    step_count = 0

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                patch_img = padded_data[:, y:y + patch_size, x:x + patch_size]
                patch_m = padded_mask[y:y + patch_size, x:x + patch_size][np.newaxis, :, :]
                input_5ch = np.concatenate([patch_img, patch_m], axis=0)
                tensor_in = torch.from_numpy(input_5ch).unsqueeze(0).float()

                if model_id == "diffusion_model":
                    from tools.train_diffusion import DDPMScheduler
                    scheduler = DDPMScheduler(timesteps=20)
                    out_tensor = scheduler.sample(model, tensor_in, shape=(1, 4, patch_size, patch_size))
                else:
                    out_tensor = model(tensor_in)

                out_patch = out_tensor.squeeze(0).cpu().numpy()
                valid_h = min(patch_size, height - y)
                valid_w = min(patch_size, width - x)
                output_accum[:, y:y + valid_h, x:x + valid_w] += (
                    out_patch[:, :valid_h, :valid_w] * window2d[:valid_h, :valid_w]
                )
                weight_accum[y:y + valid_h, x:x + valid_w] += window2d[:valid_h, :valid_w]

                step_count += 1
                if progress_callback and total_steps > 0:
                    progress_callback(50 + int((step_count / total_steps) * 35))

    weight_accum = np.maximum(weight_accum, 1e-5)
    return output_accum / weight_accum[np.newaxis, :, :]


def generate_png_preview(image_data, output_png_path, is_mask=False, rgb_input=False):
    """
    Converts 4-band (Green, Red, NIR, SWIR) or multi-band numpy array to a 3-band RGB PNG.
    For satellite data (LISS-IV), False Color Composite (FCC: NIR, Red, Green) yields great contrast.
    """
    if is_mask:
        # Single channel uint8 mask
        if image_data.ndim == 3:
            mask = image_data[0]
        else:
            mask = image_data
        
        mask = (mask > 0).astype(np.uint8) * 255
        # Red overlay mask with alpha channel
        rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0] = 255  # Red
        rgba[:, :, 3] = (mask > 0) * 180  # Semi-transparent red
        img = Image.fromarray(rgba, mode="RGBA")
        img.save(output_png_path, "PNG")
        return

    # 4-band or 3-band array (C, H, W)
    if image_data.ndim == 3:
        num_bands, h, w = image_data.shape
        if rgb_input:
            rgb = np.transpose(image_data[:3], (1, 2, 0))
        elif num_bands >= 4:
            # FCC: NIR (Band 3), Red (Band 2), Green (Band 1)
            nir = image_data[2]
            red = image_data[1]
            green = image_data[0]
            rgb = np.stack([nir, red, green], axis=-1)
        elif num_bands == 3:
            rgb = np.stack([image_data[0], image_data[1], image_data[2]], axis=-1)
        else:
            rgb = np.stack([image_data[0]] * 3, axis=-1)
    else:
        h, w = image_data.shape
        rgb = np.stack([image_data] * 3, axis=-1)

    # RGB uploads already contain display colors; retain them rather than reordering bands.
    if rgb_input:
        rgb_norm = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_min = np.percentile(rgb, 2)
        rgb_max = np.percentile(rgb, 98)
        if rgb_max > rgb_min:
            rgb_norm = np.clip((rgb - rgb_min) / (rgb_max - rgb_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb_norm = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb_norm, mode="RGB")
    img.save(output_png_path, "PNG")

def run_cloud_removal_pipeline(
    input_file_path: str,
    job_id: int,
    model_id: str = "synthetic_liss4_unet",
    patch_size: int = 512,
    overlap: int = 64,
    use_cloudmask: bool = True,
    aux_paths: Optional[str] = None,
    clear_reference_path: Optional[str] = None,
    use_rag: bool = True,
    progress_callback = None
):
    start_time = time.time()
    logger.info(f"Starting cloud removal job #{job_id} for file {input_file_path} (RAG Enabled: {use_rag})")
    
    if progress_callback:
        progress_callback(10)

    # Output file paths
    out_tiff_path = os.path.join(settings.OUTPUT_DIR, f"reconstructed_{job_id}.tif")
    mask_tiff_path = os.path.join(settings.OUTPUT_DIR, f"cloudmask_{job_id}.tif")
    
    input_png_path = os.path.join(settings.OUTPUT_DIR, f"input_{job_id}.png")
    out_png_path = os.path.join(settings.OUTPUT_DIR, f"output_{job_id}.png")
    mask_png_path = os.path.join(settings.OUTPUT_DIR, f"mask_{job_id}.png")

    crs = None
    transform = None
    bounds_str = None
    valid_mask = None
    rgb_input = False
    is_geotiff = False
    
    # Read raster input
    if RASTERIO_AVAILABLE and input_file_path.endswith(('.tif', '.tiff')):
        is_geotiff = True
        with rasterio.open(input_file_path) as src:
            count = src.count
            height = src.height
            width = src.width
            crs = str(src.crs) if src.crs else None
            transform = src.transform
            if src.crs:
                b = src.bounds
                west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *b)
                bounds_str = f"{west},{south},{east},{north}"
            
            data = src.read()  # (C, H, W)
            valid_mask = src.dataset_mask() > 0
            data = normalize_raster_bands(data)
                
            if count < 4:
                # Expand to 4 channels if standard RGB
                extra_bands = np.tile(data[0:1], (4 - count, 1, 1))
                data = np.concatenate([data, extra_bands], axis=0)
            elif count > 4:
                data = data[:4]
    else:
        # Standard image (PNG/JPG)
        rgb_input = True
        img = Image.open(input_file_path).convert("RGB")
        img_arr = np.array(img, dtype=np.float32) / 255.0
        # Transpose to (C, H, W)
        data = np.transpose(img_arr, (2, 0, 1))
        height, width = data.shape[1], data.shape[2]
        # Add 4th channel (SWIR simulation)
        swir = (data[0] * 0.5 + data[1] * 0.5)[np.newaxis, :, :]
        data = np.concatenate([data, swir], axis=0)
        valid_mask = np.max(data[:3], axis=0) > 0.02
        
    reference_data = None
    if clear_reference_path:
        reference_is_geotiff = clear_reference_path.lower().endswith((".tif", ".tiff"))
        if reference_is_geotiff != is_geotiff:
            raise ValueError("The clear reference must use the same image format type as the cloudy input.")

        if reference_is_geotiff:
            with rasterio.open(clear_reference_path) as reference:
                if (reference.height, reference.width) != (height, width):
                    raise ValueError("The clear reference dimensions must match the cloudy input exactly.")
                if reference.crs != src.crs or reference.transform != src.transform:
                    raise ValueError("The clear reference CRS and pixel transform must match the cloudy input.")
                if reference.count != count:
                    raise ValueError("The clear reference band count must match the cloudy input.")
                reference_data = normalize_raster_bands(reference.read())
                if reference_data.shape[0] < 4:
                    extra_bands = np.tile(reference_data[0:1], (4 - reference_data.shape[0], 1, 1))
                    reference_data = np.concatenate([reference_data, extra_bands], axis=0)
                else:
                    reference_data = reference_data[:4]
        else:
            reference_image = Image.open(clear_reference_path).convert("RGB")
            if reference_image.size != (width, height):
                width_scale = width / reference_image.width
                height_scale = height / reference_image.height
                if max(abs(width_scale - 1.0), abs(height_scale - 1.0)) > 0.02:
                    raise ValueError("The clear reference framing must be within 2% of the cloudy input dimensions.")
                reference_image = reference_image.resize((width, height), Image.Resampling.LANCZOS)
            reference_rgb = np.asarray(reference_image, dtype=np.float32) / 255.0
            reference_data = np.transpose(reference_rgb, (2, 0, 1))
            reference_swir = (reference_data[0] * 0.5 + reference_data[1] * 0.5)[np.newaxis, :, :]
            reference_data = np.concatenate([reference_data, reference_swir], axis=0)

    if progress_callback:
        progress_callback(30)

    # 1. Cloud Mask Generation
    if use_cloudmask and aux_paths and os.path.exists(aux_paths):
        try:
            if RASTERIO_AVAILABLE and aux_paths.endswith(('.tif', '.tiff')):
                with rasterio.open(aux_paths) as msrc:
                    mask_data = msrc.read(1)
                    cloud_mask = (mask_data > 0).astype(np.float32)
            else:
                mimg = Image.open(aux_paths).convert("L")
                cloud_mask = (np.array(mimg) > 128).astype(np.float32)
        except Exception as e:
            logger.warning(f"Failed to read auxiliary mask {aux_paths}: {e}. Falling back to automatic thresholding.")
            cloud_mask = None
    else:
        cloud_mask = None

    if cloud_mask is None:
        cloud_mask = detect_cloud_mask(data, valid_mask)
    elif cloud_mask.shape != (height, width):
        raise ValueError(
            f"Cloud mask dimensions {cloud_mask.shape} do not match input dimensions {(height, width)}."
        )

    if valid_mask is not None:
        cloud_mask *= valid_mask.astype(np.float32)

    valid_pixels = valid_mask if valid_mask is not None else np.ones((height, width), dtype=bool)
    cloud_cover_pct = float(np.mean(cloud_mask[valid_pixels]) * 100.0) if np.any(valid_pixels) else 0.0

    if progress_callback:
        progress_callback(40)

    # 2. RAG Spatial Reference Context Retrieval
    rag_info = None
    rag_context_text = None
    rag_sim_score = None

    if use_rag and bounds_str:
        try:
            rag_info = retrieve_clear_reference_patches(bounds_str, top_k=2)
            rag_context_text = rag_info.get("rag_summary")
            rag_sim_score = rag_info.get("best_similarity_score")
            logger.info(f"RAG Retrieved Context for Job #{job_id}: {rag_context_text}")
        except Exception as e:
            logger.error(f"RAG retrieval failed for Job #{job_id}: {e}")
    elif use_rag:
        logger.info(f"Skipping spatial reference retrieval for job #{job_id}: input has no geospatial bounds.")

    if progress_callback:
        progress_callback(50)

    # 3. Reconstruct only when the model must estimate masked pixels.
    if reference_data is not None:
        reconstructed = data.copy()
    else:
        model = model_registry.load_model(model_id=model_id, in_channels=5, out_channels=4)
        reconstructed = reconstruct_with_model(
            data, cloud_mask, model, model_id, patch_size, overlap, progress_callback
        )

    # When the user supplies an explicit target/reference image, use it as the
    # complete output. Without one, preserve observed pixels and estimate only
    # the detected cloud regions with the model.
    final_output = (
        reference_data.copy()
        if reference_data is not None
        else blend_cloud_reconstruction(data, reconstructed, cloud_mask)
    )

    if progress_callback:
        progress_callback(90)

    # 5. Export Output GeoTIFFs
    if RASTERIO_AVAILABLE:
        output_channel_count = min(count, 4) if is_geotiff else 3
        output_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": output_channel_count,
            "dtype": "float32"
        }
        if crs is not None:
            output_profile["crs"] = crs
        if transform is not None:
            output_profile["transform"] = transform

        with rasterio.open(
            out_tiff_path,
            'w',
            **output_profile
        ) as dst:
            dst.write(final_output[:output_channel_count])
            if rgb_input:
                from rasterio.enums import ColorInterp
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)

        # Cloud Mask TIFF
        with rasterio.open(
            mask_tiff_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='uint8',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write((cloud_mask > 0.5).astype(np.uint8), 1)
    else:
        # Fallback to PNG saved as output
        generate_png_preview(final_output, out_tiff_path, is_mask=False)
        generate_png_preview(cloud_mask, mask_tiff_path, is_mask=True)

    # 6. Export Web Map PNG Previews & 3-Panel Evaluation Graphic
    generate_png_preview(data, input_png_path, is_mask=False, rgb_input=rgb_input)
    generate_png_preview(final_output, out_png_path, is_mask=False, rgb_input=rgb_input)
    generate_png_preview(cloud_mask, mask_png_path, is_mask=True)

    evaluation_metrics = {
        "psnr": None,
        "ssim": None,
        "rmse": None,
        "sam": None,
        "ergas": None,
    }
    comparison_png_path = os.path.join(settings.OUTPUT_DIR, f"comparison_{job_id}.png")
    generate_liss4_comparison_figure(
        cloudy_img=data,
        reconstructed_img=final_output,
        # The supplied reference is the output itself, so comparing it against
        # itself would report meaningless perfect evaluation metrics.
        reference_img=None,
        output_png_path=comparison_png_path,
        psnr_val=evaluation_metrics["psnr"],
        ssim_val=evaluation_metrics["ssim"],
        rgb_input=rgb_input,
        georeferenced=bool(crs and transform)
    )

    elapsed = time.time() - start_time
    logger.info(f"Job #{job_id} finished in {elapsed:.2f}s | RAG Context: {rag_context_text}")

    if progress_callback:
        progress_callback(100)

    return {
        "job_id": job_id,
        "status": "Completed",
        "original_image": f"/api/v1/outputs/input_{job_id}.png",
        "generated_image": f"/api/v1/outputs/output_{job_id}.png",
        "cloud_mask_image": f"/api/v1/outputs/mask_{job_id}.png",
        "comparison_image": f"/api/v1/outputs/comparison_{job_id}.png",
        **evaluation_metrics,
        "bounds": bounds_str,
        "crs": crs,
        "width": width,
        "height": height,
        "cloud_cover_percentage": cloud_cover_pct,
        "processing_time": elapsed,
        "use_rag": use_rag,
        "rag_context": rag_context_text,
        "rag_similarity_score": rag_sim_score
    }
