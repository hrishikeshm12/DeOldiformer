
import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(512, 512)):
    """
    Resize images in the input folder to the target size and save them in the output folder.

    Parameters:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder where resized images will be saved.
    - target_size (tuple): Target size of the images in the format (width, height).
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Construct the input and output file paths
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the image
        with Image.open(input_path) as img:
            # Resize the image to the target size
            resized_img = img.resize(target_size, Image.BICUBIC)

            # Convert the image to RGB mode (if not already)
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
                
            # Save the resized image as JPEG
            resized_img.save(output_path, format='JPEG')

# Example usage:
input_folder = "./codeformer_results_rf/codeformer_results_rf_10/final_results"
output_folder = "./resized_codeformer_results_rf_10/"
target_size = (512, 512)

# Resize images in the input folder to 512x512 pixels and save them in the output folder
resize_images(input_folder, output_folder, target_size)
