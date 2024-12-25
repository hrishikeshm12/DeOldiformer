import os
import pyiqa

# Path to the folders containing images
distorted_images_folder = './resized_Real-ESRGAN_result'
reference_images_folder = './resized_div2k_images'

# Create the MUSIQ metric
iqa_metric = pyiqa.create_metric('musiq')

# Initialize variables to store the sum of MUSIQ scores and the count of images
total_score = 0.0
num_images = 0

# Iterate over all files in the distorted images folder
for i in range(801,900):  # Assuming there are 10 images, adjust the range accordingly
    # Construct the path to the distorted image file
    distorted_image_path = os.path.join(distorted_images_folder, f'0{i}.png')
    
    # Construct the corresponding path to the reference image file
    reference_image_path = os.path.join(reference_images_folder, f'0{i}.png')
    
    # Check if both the distorted and reference image files exist
    if os.path.exists(distorted_image_path) and os.path.exists(reference_image_path):
        # Calculate the MUSIQ score for the pair of images
        score = iqa_metric(distorted_image_path, reference_image_path)
        
        # Accumulate the MUSIQ score
        total_score += score
        num_images += 1

# Calculate the average MUSIQ score
avg_score = total_score / num_images if num_images > 0 else 0.0

# Display the average MUSIQ score
print("Average MUSIQ score:", avg_score)
