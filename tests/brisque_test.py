from brisque import BRISQUE
import numpy as np
from PIL import Image
import os

def calculate_average_brisque_score(folder_path):
    # Initialize BRISQUE object
    brisque_obj = BRISQUE(url=False)
    
    # List all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # Check if there are images in the folder
    if not image_files:
        print(f"No images found in the folder: {folder_path}")
        return None
    
    scores = []

    # Loop through all image files and calculate BRISQUE score
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path)
        ndarray = np.asarray(img)
        score = brisque_obj.score(img=ndarray)
        scores.append(score)
        print(f"BRISQUE score for {image_file}: {score}")
    
    # Calculate average BRISQUE score
    average_score = np.mean(scores)
    return average_score

# Example usage
folder_path = "./resized_GFPGAN_synthetic"
average_brisque_score = calculate_average_brisque_score(folder_path)
if average_brisque_score is not None:
    print(f"Average BRISQUE score for the folder: {average_brisque_score}")
