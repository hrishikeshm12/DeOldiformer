import requests
import os

# Create a directory to save the DIV2K data
os.makedirs("DIV2K/validation", exist_ok=True)

# URL of the DIV2K validation data
url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/validation.tar"

# Download the data
response = requests.get(url, stream=True)
file_path = os.path.join("DIV2K", "validation.tar")
with open(file_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

# Extract the downloaded tar file
import tarfile

with tarfile.open(file_path, "r") as tar:
    tar.extractall("DIV2K/validation")

# Remove the downloaded tar file
os.remove(file_path)

print("DIV2K validation data downloaded and extracted successfully.")
