import os
import torch
from cleanfid import fid
import numpy as np

print(torch.cuda.is_available())

# Define the path to the original images and the base path for the transformed images
fdir1 = './resized_div2k_images/'
base_fdir2 = 'resized_codeformer_results_w/resized_codeformer_results_w_'

# Render factors to iterate over
render_factors = np.arange(0.1, 1.1, 0.1)

# Initialize an empty list to store FID scores
fid_scores = []

# Loop through each render factor
for rf in render_factors:
    fdir2 = f"{base_fdir2}{rf}/"
    score = fid.compute_fid(fdir1, fdir2, mode="clean", num_workers=0)
    print(f"FID for w {rf}: {score}")
    fid_scores.append((rf, score))

# Display the results
print("\nFID Scores for each w:")
for rf, score in fid_scores:
    print(f"w {rf}: {score}")
