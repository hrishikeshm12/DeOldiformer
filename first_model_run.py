import subprocess
import logging
import os
import sys
import gc
import time
import psutil
import shutil
import torch
from glob import glob

from DeOldify.deoldify.device_id import DeviceId
from DeOldify.deoldify import device

# Configure logging
logging.basicConfig(filename='model_execution.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_first_model(image_path, render_factor, output_folder):
    """
    Run the first model.

    Parameters:
    - image_path (str): Path to the input image.
    - render_factor (int): Render factor to use for the model.
    - output_folder (str): Path to the output folder.
    """
    try:
        # Activate the virtual environment
        venv_activate_script = os.path.join("venv", "Scripts", "activate")
        activate_command = f'"{venv_activate_script}" &&'

        # Construct the command to run the first model
        command = [
            activate_command,
            f'"{sys.executable}"', "-m", "DeOldify.deoldify_execute",
            f'--source_url "{image_path}"',
            f'--render_factor {render_factor}',
            f'--results_directory "{output_folder}"'
        ]
        
        # Join the command parts into a single string
        command_string = " ".join(command)

        # Execute the command and capture the output
        subprocess.run(command_string, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"First model execution successful for render factor {render_factor}")
    except Exception as e:
        logging.error(f"Error occurred while running the first model for render factor {render_factor}: {str(e)}")
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

        logging.info("Resource cleanup complete.")
    except Exception as e:
        logging.error(f"Error occurred while freeing up resources: {str(e)}")

def main(image_folder):
    try:
    
        # Free up resources
        free_up_resources()

        render_factors = range(10, 45, 5)
        images = glob(os.path.join(image_folder, "*.png"))

        for render_factor in render_factors:
            output_folder = f"./deoldify_results_rf_{render_factor}"
            os.makedirs(output_folder, exist_ok=True)

            for image_path in images:
                run_first_model(image_path, render_factor, output_folder)

            # Free up resources after processing each render factor
            free_up_resources()


    except Exception as e:
        logging.error(f"Error occurred during processing: {str(e)}")

if __name__ == "__main__":
    image_folder = "resized_degraded_images_new"  # Adjust the path to your folder of images
    main(image_folder)
