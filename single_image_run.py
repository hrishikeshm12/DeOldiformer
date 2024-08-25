import subprocess
import logging
import os
import gc
import psutil
import shutil
import torch

from DeOldify.deoldify.device_id import DeviceId
from DeOldify.deoldify import device

# Configure logging
logging.basicConfig(filename='model_execution.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_model(command):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True)
        logging.info(f"Successfully executed command: {command}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running command: {e}")
        raise

def run_first_model(image_path):
    """
    Run the first model.

    Parameters:
    - image_path (str): Path to the input image.
    """
    render_factor = 35
    venv_python = os.path.join("venv", "bin", "python")  # Adjust for Windows if needed
    command = f'{venv_python} -m DeOldify.deoldify_execute --source_url "{image_path}" --render_factor {render_factor}'
    run_model(command)

def run_second_model(input_folder):
    """
    Run the second model.

    Parameters:
    - input_folder (str): Path to the input folder of the second model.
    """
    venv_python = os.path.join("venv", "bin", "python")  # Adjust for Windows if needed
    command = f'{venv_python} -m CodeFormer.inference_codeformer -w 0.7 --bg_upsampler realesrgan --face_upsample --input_path "{input_folder}"'
    run_model(command)

def free_up_resources():
    """Free up system resources."""
    try:
        logging.info("Freeing up resources...")
        torch.cuda.empty_cache()
        gc.collect()
        memory_usage = psutil.virtual_memory().used
        logging.info(f"System memory usage before cleanup: {memory_usage} bytes")
        subprocess.run(['ipconfig', '/flushdns'], capture_output=True, text=True)
        subprocess.run(['sfc', '/purgecache'], capture_output=True, text=True)
        logging.info("Resource cleanup complete.")
    except Exception as e:
        logging.error(f"Error occurred while freeing up resources: {str(e)}")

def clear_directory(directory="./deoldify_results"):
    """Clear the contents of a directory."""
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    logging.info(f"Removed file: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logging.info(f"Removed directory and its contents: {item_path}")
            except Exception as e:
                logging.error(f"Error removing {item_path}: {e}")
        logging.info(f"Cleared contents of directory '{directory}'")
    else:
        logging.info(f"Directory '{directory}' does not exist.")

def process_single_image(image_path):
    """Process a single image through the models."""
    try:
        free_up_resources()
        run_first_model(image_path)
        free_up_resources()
        run_second_model("./deoldify_results")
        clear_directory()
    except Exception as e:
        logging.error(f"Error occurred for image {image_path}: {str(e)}")

if __name__ == "__main__":
    image_path = "degraded_images_new/0801.png"  # Change this to your image path
    process_single_image(image_path)
