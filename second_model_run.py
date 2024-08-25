import subprocess
import logging
import os
import gc
import psutil
import torch
import sys

# Configure logging
logging.basicConfig(filename='model_execution.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_second_model(input_folder, output_folder):
    """
    Run the second model.

    Parameters:
    - input_folder (str): Path to the input folder of the second model.
    - output_folder (str): Path to the output folder of the second model.
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
            f'--input_path "{input_folder}"',
            f'--output_path "{output_folder}"'
        ]
        
        # Join the command parts into a single string
        command_string = " ".join(command)

        # Execute the command
        subprocess.run(command_string, shell=True, check=True)
        logging.info(f"Second model execution successful for folder {input_folder}")
    except Exception as e:
        logging.error(f"Error occurred while running the second model for folder {input_folder}: {str(e)}")
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

def main():
    try:
        render_factors = range(10, 45, 5)

        for render_factor in render_factors:
            input_folder = f"./deoldify_results_rf_{render_factor}"
            output_folder = f"./codeformer_results_rf_{render_factor}"
            os.makedirs(output_folder, exist_ok=True)

            run_second_model(input_folder, output_folder)

            # Free up resources after processing each render factor
            free_up_resources()

    except Exception as e:
        logging.error(f"Error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
