import os
from PIL import Image
import torch
import lpips
import torchvision.transforms as transforms

def compute_lpips_score(im0, im1):
    """
    Compute the LPIPS score between two images.

    Parameters:
    - im0 (torch.Tensor): Tensor representing the first image.
    - im1 (torch.Tensor): Tensor representing the second image.

    Returns:
    - float: LPIPS score between the two images.
    """
    # Initialize the LPIPS model
    loss_fn = lpips.LPIPS(net='alex')

    # Resize images to have the same size
    resize = transforms.Resize((min(im0.size[0], im1.size[0]), min(im0.size[1], im1.size[1])))
    im0 = resize(im0)
    im1 = resize(im1)

    # Convert images to torch tensors
    transform = transforms.ToTensor()
    im0 = transform(im0).unsqueeze(0)
    im1 = transform(im1).unsqueeze(0)

    # Compute the LPIPS score between im0 and im1
    lpips_score = loss_fn.forward(im0, im1)

    return lpips_score.item()

def compute_average_lpips_score(test_images_folder, transformed_images_folder):
    # Initialize a list to store LPIPS scores for all image pairs
    lpips_scores = []

    # Iterate over each image in the test images folder
    for test_image_file in os.listdir(test_images_folder):
        # Construct the paths for the test image and transformed image
        test_image_path = os.path.join(test_images_folder, test_image_file)
        transformed_image_path = os.path.join(transformed_images_folder, test_image_file.replace('.png', '.png'))

        # Open images using PIL
        test_image_pil = Image.open(test_image_path).convert('RGB')
        transformed_image_pil = Image.open(transformed_image_path).convert('RGB')

        # Compute LPIPS score for the image pair
        lpips_score = compute_lpips_score(test_image_pil, transformed_image_pil)
        lpips_scores.append(lpips_score)

    # Compute average LPIPS score
    avg_lpips_score = sum(lpips_scores) / len(lpips_scores)
    return avg_lpips_score

if __name__ == "__main__":
    # Path to the folder containing test images (JPEG format)
    test_images_folder = "./resized_div2k_images/"

    # Path to the folder containing transformed images (PNG format)
    transformed_images_folder = "./resized_instcolorization_results_div2k"

    # Compute average LPIPS score
    avg_lpips_score = compute_average_lpips_score(test_images_folder, transformed_images_folder)
    print("Average LPIPS score:", avg_lpips_score)
