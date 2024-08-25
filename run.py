import subprocess
import logging
import os
import sys
import gc
import time
import psutil
import shutil
import time
import torch

from DeOldify.deoldify.device_id import DeviceId
from DeOldify.deoldify import device

# Configure logging
logging.basicConfig(filename='model_execution.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_first_model(image_path):
    """
    Run the first model.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - str: Path to the output folder of the first model.
    """

    try:
        render_factor = 35  # Setting render factor to 35
        
        # Activate the virtual environment
        venv_activate_script = os.path.join("venv", "Scripts", "activate")
        activate_command = f'"{venv_activate_script}" &&'

        # Construct the command to run the first model
        command = [
            activate_command,
            f'"{sys.executable}"', "-m", "DeOldify.deoldify_execute",
            f'--source_url "{image_path}"',
            f'--render_factor {render_factor}'
        ]
        
        # Join the command parts into a single string
        command_string = " ".join(command)

        # Execute the command and capture the output
        subprocess.run(command_string, shell=True, check=True, capture_output=True, text=True)
        logging.info("First model execution successful")

        # Extract the output folder path from the command output
        

    except Exception as e:
        logging.error(f"Error occurred while running the first model: {str(e)}")
        raise




  
def run_second_model(input_folder):
    """
    Run the second model.

    Parameters:
    - input_folder (str): Path to the input folder of the second model.

    Returns:
    - None
    """
    try:
        # Activate the virtual environment
        venv_activate_script = os.path.join("venv", "Scripts", "activate")
        activate_command = f'"{venv_activate_script}" &&'

        # Construct the command to run the second model
        command = [
            activate_command,
            f'"{sys.executable}"', "-m", "CodeFormer.inference_codeformer",
            f'-w 0.7', "--bg_upsampler realesrgan --face_upsample", 
            f'--input_path "{input_folder}"'
        ]
        
        # Join the command parts into a single string
        command_string = " ".join(command)

        # Execute the command
        subprocess.run(command_string, shell=True, check=True)
        logging.info("Second model execution successful")
    except Exception as e:
        logging.error(f"Error occurred while running the second model: {str(e)}")
        raise

def free_up_resources():
    """
    Free up system resources after running the first model and before running the second model.
    """
    try:
        logging.info("Freeing up resources...")
        
        # Release GPU memory
        torch.cuda.empty_cache()

        # Manually trigger garbage collection
        gc.collect()

        # Check system memory usage and clear caches if necessary
        memory_usage = psutil.virtual_memory().used
        logging.info(f"System memory usage before cleanup: {memory_usage} bytes")

        # Clear DNS cache
        subprocess.run(['ipconfig', '/flushdns'], capture_output=True, text=True)

        # Clear Windows system file cache
        subprocess.run(['sfc', '/purgecache'], capture_output=True, text=True)

        # Wait for a few seconds to allow resources to be released
        #logging.info("Waiting for 2 seconds after freeing up resources...")
        #time.sleep(2)

        logging.info("Resource cleanup complete.")
    except Exception as e:
        logging.error(f"Error occurred while freeing up resources: {str(e)}")



def clear_directory():

    directory = "./deoldify_results"
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate over the contents of the directory
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            # Check if it's a file
            if os.path.isfile(item_path):
                try:
                    # Remove the file
                    os.remove(item_path)
                    print(f"Removed file: {item_path}")
                except Exception as e:
                    print(f"Error removing file {item_path}: {e}")
            # Check if it's a directory
            elif os.path.isdir(item_path):
                try:
                    # Remove the directory and its contents recursively
                    shutil.rmtree(item_path)
                    print(f"Removed directory and its contents: {item_path}")
                except Exception as e:
                    print(f"Error removing directory {item_path}: {e}")
        print(f"Cleared contents of directory '{directory}'")
    else:
        print(f"Directory '{directory}' does not exist.")


def main(image_path):
    try:
        start_time = time.time()  # Record the start time of the main function

        # Free up resources
        free_up_resources()

        # Run first model
        run_first_model(image_path)

        input_folder = "./deoldify_results"

        # Free up resources
        free_up_resources()

        # Run second model
        run_second_model(input_folder)

        # Clear directory
        # clear_directory()

        end_time = time.time()  # Record the end time of the main function
        execution_time = end_time - start_time  # Calculate the execution time of the main function

        print(f"Execution time for main function: {execution_time} seconds")

    except Exception as e:
        logging.error(f"Error occurred for image {image_path}: {str(e)}")

if __name__ == "__main__":
    image_path = "degraded_images_new/0801.png"  # Adjust the path to your image
    main(image_path)
