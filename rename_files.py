import os

def rename_files(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    # Initialize a counter
    counter = 0
    # Iterate over each file
    for file in files:
        # Construct the new filename
        new_name = f"test_image{counter}.jpg"  # Change the extension if your files have a different format
        # Construct the full paths for the old and new filenames
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        # Increment the counter
        counter += 1

# Specify the directory containing the files
directory = './test_images'

# Call the function to rename the files
rename_files(directory)
