import pyiqa
import os
from PIL import Image

# Create SSIM metric
ssim_metric = pyiqa.create_metric('ssim')

def compute_average_ssim_score(test_images_folder, transformed_images_folder):
    # Initialize a list to store SSIM scores for all image pairs
    ssim_scores = []
    # Iterate over each image in the test images folder
    for test_image_file in os.listdir(test_images_folder):
        # Construct the paths for the test image and transformed image
        test_image_path = os.path.join(test_images_folder, test_image_file)
        transformed_image_path = os.path.join(transformed_images_folder, test_image_file.replace('.png', '_deoldify.png'))
        print(transformed_image_path)
        # Open images using PIL
        test_image_pil = Image.open(test_image_path)
        transformed_image_pil = Image.open(transformed_image_path)

        # Compute SSIM score for the image pair
        score_ssim = ssim_metric(test_image_pil, transformed_image_pil)
       
        ssim_scores.append(score_ssim)

    # Compute average SSIM score
    avg_ssim_score = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0
    return avg_ssim_score

if __name__ == "__main__":
    # Path to the folder containing test images (JPEG format)
    test_images_folder = "./resized_div2k_images/"

    # Path to the folder containing transformed images (PNG format)
    transformed_images_folder = "./resized_codeformer_results_w_0.9/"

    # Compute average SSIM score
    avg_ssim_score = compute_average_ssim_score(test_images_folder, transformed_images_folder)
    print("Average SSIM score:", avg_ssim_score)
