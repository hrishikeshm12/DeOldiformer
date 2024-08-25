import io
import os
from  urllib.parse import urlparse
import shutil
import urllib.request
import subprocess
import logging
import sys
import gc
import time
import requests
from PIL import Image
import psutil
import torch
import gradio as gr


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
            f'-w 0.7', "--face_upsample", 
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

        # Check system memory usage
        memory_usage = psutil.virtual_memory().used
        logging.info(f"System memory usage before cleanup: {memory_usage} bytes")

        # Clear DNS cache
        subprocess.run(['ipconfig', '/flushdns'], capture_output=True, text=True)

        # Clear Windows system file cache
        subprocess.run(['sfc', '/purgecache'], capture_output=True, text=True)

        logging.info("Resource cleanup complete.")
    except Exception as e:
        logging.error(f"Error occurred while freeing up resources: {str(e)}")


def fetch_uploaded_image(input_image):
    try:
        print(input_image)
        pil_image = Image.open(input_image)
        save_directory = "./inputs"
        file_path = os.path.join(save_directory, "uploaded_image.jpg")
        pil_image.save(file_path)
        print(file_path)
        return file_path
    except Exception as e:
        print(f"Error: {e}")
        return None

def clear_and_delete_directory():
    # Define the directories to delete and clear
    directory_to_delete = "./codeformer_results"
    directory_to_clear = "./deoldify_results"

    # Delete directory
    if os.path.exists(directory_to_delete):
        try:
            shutil.rmtree(directory_to_delete)
            print(f"Deleted directory '{directory_to_delete}'")
        except Exception as e:
            print(f"Error deleting directory '{directory_to_delete}': {e}")

    # Clear directory
    if os.path.exists(directory_to_clear):
        try:
            shutil.rmtree(directory_to_clear)
            print(f"Cleared contents of directory '{directory_to_clear}'")
        except Exception as e:
            print(f"Error clearing directory '{directory_to_clear}': {e}")



def image_restoration(local_image):
        try:

            initial_image_path=""
            clear_and_delete_directory()
            if local_image is not None:
                # Local image was uploaded
                initial_image_path=fetch_uploaded_image(local_image)
                print(f"Initial image path for local is {initial_image_path}")

            else:
                print("No input provided.")


            # Free up resources
            free_up_resources()

            # Run first model
            run_first_model(initial_image_path)
            
            input_folder = "./deoldify_results"

            # Free up resources
            free_up_resources()

            # Run second model
            run_second_model(str(input_folder))

            # Open and return the final restored image
            folder_path = "./codeformer_results/final_results"
            image_files = (file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg')))
            try:
                image_file = next(image_files)
                image_path = os.path.join(folder_path, image_file)
                # Open the image using PIL
                restored_image = Image.open(image_path)
                return restored_image
            except StopIteration:
                print("Error: No image file found in the directory.")
                return None
            except Exception as e:
                print(f"Error: {e}")
                return None

            
        except Exception as e:
            logging.error(f"Error occurred during image restoration: {str(e)}")
            return None



def main():
   

    # Create Gradio interface
    inputs = [
    gr.Image(label="Input Image (Local)", type="filepath")  # Option 1: Upload from local device
    
    ]
    
    outputs = gr.Image(label="Restored Image")

    title = "Perceptual NoGAN-Enhanced CodeFormer for Image Reconstruction"
    description = "Upload an image to restore and enhance its quality."
    examples = [["./examples/307127084_426710346225153_4314088907171612186_n.jpg"], ["./examples/307127084_426710346225153_4314088907171612186_n_deoldify.png"],
                ["./examples/62aab33f1fa53-62a854c1100ed_101639832_3245698148813561_2469869223292174336_n__700.jpg"],["examples/62aab33f1fa53-62a854c1100ed_101639832_3245698148813561_2469869223292174336_n__700_deoldify.png"]
                ,["examples/test1 _car.jpg"],["examples/test1 _car_final.png"]
                ,["examples/test2_man.jpg"],["examples/test2_man_final.png"]
                ,["examples/test3_man.jpg"],["examples/test3_man_final.png"]]

    gr.Interface(image_restoration, inputs, outputs, title=title, description=description, examples=examples).launch()

if __name__ == "__main__":
    main()
