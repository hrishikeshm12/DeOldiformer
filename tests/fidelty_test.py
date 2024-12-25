import os
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from glob import glob
import numpy as np
import cv2
import pyiqa
from PIL import Image

# Set up logging
logging.basicConfig(filename='model_execution.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create SSIM metric
ssim_metric = pyiqa.create_metric('ssim')

def calculate_psnr(original_image_path, restored_image_path):
    original_image = cv2.imread(original_image_path)
    restored_image = cv2.imread(restored_image_path)

    # Check if images are loaded successfully
    if original_image is None or restored_image is None:
        raise ValueError(f"Error: Could not read images {original_image_path} or {restored_image_path}")

    # Ensure images have the same dimensions
    if original_image.shape != restored_image.shape:
        raise ValueError(f"Error: Images {original_image_path} and {restored_image_path} have different dimensions")

    # Calculate PSNR
    psnr_value = psnr(original_image, restored_image, data_range=255)
    return psnr_value

def calculate_ssim(test_image_path, transformed_image_path):
    # Open images using PIL
    test_image_pil = Image.open(test_image_path).convert("RGB")
    transformed_image_pil = Image.open(transformed_image_path).convert("RGB")

    # Compute SSIM score for the image pair
    score_ssim = ssim_metric(test_image_pil, transformed_image_pil).item()
    return score_ssim

def main(reference_images_directory, fidelity_weights):
    try:
        # Gather all reference image paths
        reference_image_paths = sorted(glob(os.path.join(reference_images_directory, "*.png")))
        if not reference_image_paths:
            logging.error("No reference images found in the directory.")
            return

        # Prepare a dictionary to store the results
        results = {w: {"psnr": [], "ssim": []} for w in fidelity_weights}

        for w in fidelity_weights:
            restored_images_folder = f"./resized_codeformer_results_w_{w}/final_results"
            restored_image_paths = sorted(glob(os.path.join(restored_images_folder, "*.png")))

            if len(reference_image_paths) != len(restored_image_paths):
                logging.error(f"Number of images in {reference_images_directory} and {restored_images_folder} do not match.")
                continue

            # Iterate over each pair of images
            for ref_img_path, res_img_path in zip(reference_image_paths, restored_image_paths):
                try:
                    # Calculate PSNR
                    psnr_value = calculate_psnr(ref_img_path, res_img_path)
                    results[w]["psnr"].append(psnr_value)

                    # Calculate SSIM
                    ssim_value = calculate_ssim(ref_img_path, res_img_path)
                    results[w]["ssim"].append(ssim_value)

                    logging.info(f"Image: {ref_img_path}, Fidelity Weight: {w}, PSNR: {psnr_value}, SSIM: {ssim_value}")
                except Exception as e:
                    logging.error(f"Error processing images {ref_img_path} and {res_img_path}: {str(e)}")

        # Calculate average metrics for each fidelity weight
        avg_results = {w: {"psnr": np.mean(metrics["psnr"]) if metrics["psnr"] else 0,
                           "ssim": np.mean(metrics["ssim"]) if metrics["ssim"] else 0}
                       for w, metrics in results.items()}

        for w, metrics in avg_results.items():
            logging.info(f"Fidelity Weight: {w}, Average PSNR: {metrics['psnr']}, Average SSIM: {metrics['ssim']}")
            print(f"Fidelity Weight: {w}, Average PSNR: {metrics['psnr']}, Average SSIM: {metrics['ssim']}")

    except Exception as e:
        logging.error(f"Error occurred during the evaluation: {str(e)}")

if __name__ == "__main__":
    reference_images_directory = "resized_div2k_images"  # Path to your reference images directory
    fidelity_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Example range of fidelity weights
    main(reference_images_directory, fidelity_weights)
