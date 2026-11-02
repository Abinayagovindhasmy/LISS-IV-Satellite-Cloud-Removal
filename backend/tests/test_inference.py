import unittest
import tempfile
import os

import numpy as np
from PIL import Image

from backend.ml.inference import (
    blend_cloud_reconstruction,
    detect_cloud_mask,
    generate_png_preview,
    normalize_raster_bands
)
from backend.ml.visualizer import calculate_evaluation_metrics


class InferenceHelpersTests(unittest.TestCase):
    def test_cloud_mask_selects_bright_neutral_clouds_not_red_land_or_nodata(self):
        image = np.zeros((4, 40, 40), dtype=np.float32)
        image[0] = 0.7
        image[1] = 0.12
        image[2] = 0.08
        image[:, 12:28, 12:28] = 0.85
        valid = np.ones((40, 40), dtype=bool)
        valid[:4, :] = False

        mask = detect_cloud_mask(image, valid)

        self.assertEqual(mask[20, 20], 1.0)
        self.assertEqual(mask[8, 8], 0.0)
        self.assertEqual(mask[0, 20], 0.0)

    def test_clear_reference_fills_masked_pixels_and_preserves_clear_input(self):
        original = np.full((4, 3, 3), 0.2, dtype=np.float32)
        reconstructed = np.full_like(original, 0.4)
        reference = np.full_like(original, 0.8)
        mask = np.zeros((3, 3), dtype=np.float32)
        mask[1, 1] = 1.0

        result = blend_cloud_reconstruction(original, reconstructed, mask, reference)

        np.testing.assert_array_equal(result[:, 1, 1], reference[:, 1, 1])
        np.testing.assert_array_equal(result[:, 0, 0], original[:, 0, 0])

    def test_uint16_liss_reflectance_uses_10000_dn_scale(self):
        values = np.array([[[0, 10000]]], dtype=np.uint16)

        np.testing.assert_allclose(normalize_raster_bands(values), [[[0.0, 1.0]]])

    def test_rgb_preview_preserves_uploaded_channel_order(self):
        image = np.zeros((3, 2, 2), dtype=np.float32)
        image[:, 0, 0] = [0.8, 0.2, 0.1]
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, "preview.png")

            generate_png_preview(image, output_path, rgb_input=True)

            pixel = np.asarray(Image.open(output_path))[0, 0]
            np.testing.assert_array_equal(pixel, [204, 51, 25])

    def test_evaluation_metrics_are_zero_for_identical_images(self):
        image = np.full((4, 8, 8), 0.5, dtype=np.float32)

        metrics = calculate_evaluation_metrics(image, image.copy())

        self.assertEqual(metrics["psnr"], 100.0)
        self.assertEqual(metrics["ssim"], 1.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["sam"], 0.0)
        self.assertEqual(metrics["ergas"], 0.0)

    def test_evaluation_metrics_reject_misaligned_reference(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            calculate_evaluation_metrics(np.zeros((3, 8, 8)), np.zeros((3, 7, 8)))


if __name__ == "__main__":
    unittest.main()
