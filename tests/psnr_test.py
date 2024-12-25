import os
import cv2
import numpy as np

def calculate_psnr(folder1, folder2):
    psnr_values = []
    
    # Get the list of image filenames in each folder and sort them
    images1 = sorted(os.listdir(folder1))
    images2 = sorted(os.listdir(folder2))

    # Iterate over each pair of images
    for img1, img2 in zip(images1, images2):
        # Construct the file paths
        img1_path = os.path.join(folder1, img1)
        img2_path = os.path.join(folder2, img2)
        
        # Read the images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Check if images are loaded successfully
        if img1 is None or img2 is None:
            print(f"Error: Could not read images {img1_path} or {img2_path}")
            continue

        # Ensure images have the same dimensions
        if img1.shape != img2.shape:
            print(f"Error: Images {img1_path} and {img2_path} have different dimensions")
            continue

        # Calculate PSNR
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        psnr_values.append(psnr)

    # Calculate average PSNR
    avg_psnr = np.mean(psnr_values)
    return avg_psnr

if __name__ == "__main__":
    # Path to the first folder containing images
    folder1 = "./resized_div2k_images/"

    # Path to the second folder containing images
    folder2 = "./resized_codeformer_results_w_0.9/"

    # Calculate PSNR between the two sets of images
    avg_psnr = calculate_psnr(folder1, folder2)
    print("Average PSNR:", avg_psnr)
