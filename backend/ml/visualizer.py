import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
from PIL import Image

def calculate_metrics(img1: np.ndarray, img2: np.ndarray):
    """
    Calculates PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)
    between output reconstructed image and ground truth clear image.
    Both images should be float32 in range [0, 1] or uint8 in range [0, 255].
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.max() > 1.0 or img2.max() > 1.0:
        img1 = img1 / 255.0
        img2 = img2 / 255.0

    # MSE and PSNR calculation
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 10 * np.log10(1.0 / mse)

    # Simplified channel-wise SSIM calculation
    try:
        from skimage.metrics import structural_similarity as ssim_func
        # Determine channel axis if 3D
        if img1.ndim == 3:
            if img1.shape[0] in [3, 4]:
                img1_c = np.transpose(img1, (1, 2, 0))
                img2_c = np.transpose(img2, (1, 2, 0))
            else:
                img1_c = img1
                img2_c = img2
            
            img1_c = img1_c[:, :, :3]
            img2_c = img2_c[:, :, :3]
            ssim_val = ssim_func(img1_c, img2_c, data_range=1.0, channel_axis=2)
        else:
            ssim_val = ssim_func(img1, img2, data_range=1.0)
    except Exception:
        # Fallback SSIM approximation
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))

    return round(float(psnr), 2), round(float(ssim_val), 4)


def calculate_evaluation_metrics(
    reconstructed: np.ndarray,
    reference: np.ndarray
) -> dict[str, float]:
    """Calculate full-reference quality metrics for aligned channel-first images."""
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reconstructed.shape != reference.shape:
        raise ValueError("Reconstructed and reference images must have identical shapes.")

    psnr, ssim = calculate_metrics(reconstructed, reference)
    difference = reconstructed - reference
    rmse = float(np.sqrt(np.mean(np.square(difference))))

    reconstructed_spectra = np.moveaxis(reconstructed, 0, -1).reshape(-1, reconstructed.shape[0])
    reference_spectra = np.moveaxis(reference, 0, -1).reshape(-1, reference.shape[0])
    dot_products = np.sum(reconstructed_spectra * reference_spectra, axis=1)
    magnitudes = (
        np.linalg.norm(reconstructed_spectra, axis=1)
        * np.linalg.norm(reference_spectra, axis=1)
    )
    valid_spectra = magnitudes > 1e-12
    if np.any(valid_spectra):
        cosines = np.clip(dot_products[valid_spectra] / magnitudes[valid_spectra], -1.0, 1.0)
        sam = float(np.degrees(np.mean(np.arccos(cosines))))
    else:
        sam = 0.0

    band_rmse = np.sqrt(np.mean(np.square(difference), axis=(1, 2)))
    band_means = np.mean(reference, axis=(1, 2))
    valid_bands = np.abs(band_means) > 1e-12
    ergas = (
        float(100.0 * np.sqrt(np.mean(np.square(band_rmse[valid_bands] / band_means[valid_bands]))))
        if np.any(valid_bands)
        else 0.0
    )

    return {
        "psnr": psnr,
        "ssim": ssim,
        "rmse": round(rmse, 6),
        "sam": round(sam, 4),
        "ergas": round(ergas, 4),
    }


def add_north_arrow(ax):
    """Adds a standard GIS North Arrow to top right of plot."""
    x, y = 0.92, 0.90
    ax.text(x, y + 0.04, 'N', transform=ax.transAxes, color='white',
            fontsize=12, fontweight='bold', ha='center', va='bottom',
            path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
    
    # Draw arrow body
    ax.annotate('', xy=(x, y + 0.04), xytext=(x, y - 0.03),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(facecolor='white', edgecolor='black', width=1.5, headwidth=8, headlength=8))


def add_scale_bar(ax):
    """Adds GIS Scale Bar (0 2 4 8 km) to bottom left of plot."""
    x0, y0 = 0.05, 0.06
    w = 0.30
    
    # Scale line
    ax.plot([x0, x0 + w], [y0, y0], color='white', lw=2.5, transform=ax.transAxes,
            path_effects=[patheffects.withStroke(linewidth=4, foreground='black')])
    
    # Scale ticks
    for frac in [0.0, 0.25, 0.5, 1.0]:
        tx = x0 + w * frac
        ax.plot([tx, tx], [y0, y0 + 0.02], color='white', lw=1.8, transform=ax.transAxes,
                path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])
    
    # Scale labels
    labels = [('0', x0), ('2', x0 + w * 0.25), ('4', x0 + w * 0.5), ('8 km', x0 + w * 1.0)]
    for text, tx in labels:
        ax.text(tx, y0 + 0.03, text, transform=ax.transAxes, color='white',
                fontsize=9, fontweight='bold', ha='center', va='bottom',
                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])


def generate_liss4_comparison_figure(
    cloudy_img: np.ndarray,
    reconstructed_img: np.ndarray,
    output_png_path: str,
    reference_img: np.ndarray = None,
    psnr_val: float = None,
    ssim_val: float = None,
    rgb_input: bool = False,
    georeferenced: bool = True
):
    """
    Renders the exact 3-panel side-by-side visual graphic requested by the user:
    [Input: Cloudy LISS-IV Image] | [Output: Reconstructed Cloud-Free Image] | [Reference: Actual Clear LISS-IV Image]
    Along with Evaluation Metrics (PSNR, SSIM) banner and disclaimer.
    """
    def prep_rgb(arr):
        if arr.ndim == 3 and arr.shape[0] in [3, 4]:
            arr = np.transpose(arr, (1, 2, 0))
        if rgb_input:
            arr = np.clip(arr[:, :, :3], 0.0, 1.0)
            return (arr * 255.0).astype(np.uint8)
        if arr.shape[-1] >= 4:
            # Band 0: Green, Band 1: Red, Band 2: NIR, Band 3: SWIR
            # Create natural vegetation/earth RGB mapping:
            green_b = arr[:, :, 0]
            red_b = arr[:, :, 1]
            nir_b = arr[:, :, 2]
            
            # Natural Color Synthesis: R=Red, G=(Green*0.6 + NIR*0.4), B=Green*0.8
            r = red_b
            g = green_b * 0.5 + nir_b * 0.5
            b = green_b * 0.7 + red_b * 0.3
            arr = np.stack([r, g, b], axis=-1)
        elif arr.shape[-1] == 3:
            arr = arr[:, :, :3]

        arr = np.nan_to_num(arr)
        if arr.dtype != np.uint8:
            arr_min, arr_max = np.percentile(arr, (1, 99))
            if arr_max > arr_min:
                arr = np.clip((arr - arr_min) / (arr_max - arr_min) * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr

    cloudy_rgb = prep_rgb(cloudy_img)
    reconstructed_rgb = prep_rgb(reconstructed_img)
    reference_rgb = prep_rgb(reference_img) if reference_img is not None else None

    if reference_img is not None and (psnr_val is None or ssim_val is None):
        calc_psnr, calc_ssim = calculate_metrics(reconstructed_rgb, reference_rgb)
        psnr_val = psnr_val or calc_psnr
        ssim_val = ssim_val or calc_ssim

    # Setup figure canvas (16:7 aspect ratio)
    fig = plt.figure(figsize=(15, 7.2), dpi=200, facecolor='white')
    
    # Overall Title
    fig.suptitle("Generative AI-Based Cloud Removal and Reconstruction Using LISS-IV Satellite Images",
                 fontsize=15, fontweight='bold', fontfamily='serif', y=0.97, color='#111111')

    # Grid layout: 3 columns for panels, space for header boxes and metrics box
    panel_configs = [
        ("Input: Cloudy LISS-IV Image", cloudy_rgb, "#f5c6cb", "#d9534f", 0.03),
        ("Output: Reconstructed Cloud-Free Image", reconstructed_rgb, "#c3e6cb", "#28a745", 0.36)
    ]
    if reference_rgb is not None:
        panel_configs.append(("Reference: Actual Clear LISS-IV Image", reference_rgb, "#b8daff", "#007bff", 0.69))

    for title, img_data, header_bg, border_color, left_pos in panel_configs:
        # Create panel axes [left, bottom, width, height]
        ax_box = fig.add_axes([left_pos, 0.22, 0.28, 0.68])
        ax_box.axis('off')
        
        # Draw outer colored border around panel
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax_box.transAxes,
                                 facecolor='none', edgecolor=border_color, linewidth=2.5, zorder=5)
        ax_box.add_patch(rect)

        # Draw header bar at top of panel
        header_rect = patches.Rectangle((0, 0.90), 1, 0.10, transform=ax_box.transAxes,
                                        facecolor=header_bg, edgecolor=border_color, linewidth=1.5, zorder=4)
        ax_box.add_patch(header_rect)

        # Header Title Text
        ax_box.text(0.5, 0.95, title, transform=ax_box.transAxes,
                    fontsize=12, fontweight='bold', fontfamily='serif',
                    ha='center', va='center', color='#111111', zorder=6)

        # Inner Image Axes
        ax_img = fig.add_axes([left_pos + 0.008, 0.22 + 0.008, 0.28 - 0.016, 0.68 * 0.90 - 0.010])
        ax_img.imshow(img_data)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)

        if georeferenced:
            add_north_arrow(ax_img)
            add_scale_bar(ax_img)

    # Metrics Box at Bottom Center
    ax_metrics = fig.add_axes([0.28, 0.04, 0.44, 0.13])
    ax_metrics.axis('off')
    
    # Gray background box for metrics
    m_box = patches.FancyBboxPatch((0, 0), 1, 1, transform=ax_metrics.transAxes,
                                  boxstyle="square,pad=0",
                                  facecolor='#ebebeb', edgecolor='#d0d0d0', linewidth=1.2)
    ax_metrics.add_patch(m_box)

    ax_metrics.text(0.5, 0.70, "Evaluation Metrics", transform=ax_metrics.transAxes,
                    fontsize=12, fontweight='bold', fontfamily='serif', ha='center', va='center', color='#222222')

    metrics_str = (
        f"PSNR: {psnr_val:.2f} dB    |    SSIM: {ssim_val:.4f}"
        if psnr_val is not None and ssim_val is not None
        else "Metrics require an independent clear reference image"
    )
    ax_metrics.text(0.5, 0.30, metrics_str, transform=ax_metrics.transAxes,
                    fontsize=15, fontweight='bold', fontfamily='serif', ha='center', va='center', color='#111111')

    # Disclaimer Note at Bottom Right
    fig.text(0.97, 0.02, "Note: The supplied reference is used as the full output; otherwise cloud-covered details are model estimates.",
             fontsize=8, fontstyle='italic', fontfamily='serif', ha='right', va='bottom', color='#555555')

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_png_path
