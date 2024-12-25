from arcface import ArcFace
import os

# Initialize ArcFace
face_rec = ArcFace.ArcFace()

# Path to the folders containing images
folder1 = "./resized_test_images/"
folder2 = "./resized_result_images/"

# Get the list of files in both folders
files1 = os.listdir(folder1)
files2 = os.listdir(folder2)

# Make sure both folders have the same number of files
if len(files1) != len(files2):
    print("Number of files in both folders must be the same.")
    exit()

# Calculate embeddings and distances for each pair of images
total_distance = 0.0
num_pairs = 0

for filename1, filename2 in zip(files1, files2):
    image_path1 = os.path.join(folder1, filename1)
    image_path2 = os.path.join(folder2, filename2)

    # Calculate embeddings for both images
    embedding1 = face_rec.calc_emb(image_path1)
    embedding2 = face_rec.calc_emb(image_path2)

    # Compute distance between embeddings
    distance = face_rec.get_distance_embeddings(embedding1, embedding2)

    total_distance += distance
    num_pairs += 1

# Calculate average distance
if num_pairs > 0:
    average_distance = total_distance / num_pairs
else:
    average_distance = 0.0

print("Average distance between embeddings:", average_distance)


from arcface import ArcFace

# Initialize ArcFace
face_rec = ArcFace.ArcFace()

# Path to the two images
image_path1 = "./resized_result_images/test_image0_deoldify.png"
image_path2 = "./resized_test_images/test_image0.jpg"

# Calculcleate embeddings for both images
embedding1 = face_rec.calc_emb(image_path1)
embedding2 = face_rec.calc_emb(image_path2)

# Compute distance between embeddings
distance = face_rec.get_distance_embeddings(embedding1, embedding2)

print("Distance between embeddings:", distance)
