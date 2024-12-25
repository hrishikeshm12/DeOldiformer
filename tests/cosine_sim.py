import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to load the image and transform it to the required format
def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Function to extract features using VGG16
def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model(image_tensor).flatten().numpy()
    return features

# Main function to compute cosine similarity
def compute_cosine_similarity(ground_truth_folder, restored_images_folder):
    # Initialize the pre-trained VGG16 model
    vgg = vgg16(pretrained=True)
    # Remove the classification layers to use it as a feature extractor
    model = nn.Sequential(*list(vgg.children())[:-1])
    model.eval()

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Lists to store feature vectors
    ground_truth_features = []
    restored_images_features = []

    # Iterate over each image in the ground truth folder
    for image_file in os.listdir(ground_truth_folder):
        ground_truth_path = os.path.join(ground_truth_folder, image_file)
        restored_image_path = os.path.join(restored_images_folder, image_file.replace('.png', '_deoldify.png'))

        if not os.path.exists(restored_image_path):
            print(f"Restored image for {image_file} not found.")
            continue

        # Load and transform images
        ground_truth_image = load_image(ground_truth_path, transform)
        restored_image = load_image(restored_image_path, transform)

        # Extract features
        ground_truth_feature = extract_features(model, ground_truth_image)
        restored_image_feature = extract_features(model, restored_image)

        ground_truth_features.append(ground_truth_feature)
        restored_images_features.append(restored_image_feature)

    # Compute cosine similarity for each pair of feature vectors
    cosine_similarities = []
    for gt_feature, res_feature in zip(ground_truth_features, restored_images_features):
        cos_sim = cosine_similarity(gt_feature.reshape(1, -1), res_feature.reshape(1, -1))
        cosine_similarities.append(cos_sim[0][0])

    # Compute average cosine similarity
    avg_cosine_similarity = np.mean(cosine_similarities)
    return avg_cosine_similarity

if __name__ == "__main__":
    # Define paths to the folders
    ground_truth_folder = "./resized_div2k_images/"
    restored_images_base_folder = "./resized_codeformer_results_w/resized_codeformer_results_w_"

    # Render factors to iterate over
    render_factors = np.arange(0.1, 1.0, 0.1)

    # Initialize an empty list to store cosine similarities
    cosine_similarities = []

    # Loop through each render factor
    for rf in render_factors:
        restored_images_folder = f"{restored_images_base_folder}{rf}/"
        avg_cosine_similarity = compute_cosine_similarity(ground_truth_folder, restored_images_folder)
        print(f"Average Cosine Similarity for render factor {rf}: {avg_cosine_similarity}")
        cosine_similarities.append((rf, avg_cosine_similarity))

    # Display the results
    print("\nCosine Similarity for each render factor:")
    for rf, cos_sim in cosine_similarities:
        print(f"Render Factor {rf}: {cos_sim}")
