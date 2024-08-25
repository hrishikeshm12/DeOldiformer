from PIL import Image
from pathlib import Path
import os 
class ImageProcessor:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def _save_result_image(self, source_path: Path, image: Image, results_dir: Path = None) -> Path:
        if results_dir is None:
            results_dir = Path(self.results_dir)
        suffix = 'deoldify'
        print(results_dir)
        print(source_path.stem)
        print(suffix)
        print(source_path.suffix)
        result_path = os.path.join(results_dir, f"{source_path.stem}_{suffix}{source_path.suffix}")
        image.save(result_path)
        return result_path

# Create an instance of the class
image_processor = ImageProcessor(results_dir="deoldify_results_rf_10")

# Assuming colorizer is an instance of the relevant class and has the _save_result_image method
source_image_path = Path("resized_degraded_images_new/0801.png")
results_directory = Path("deoldify_results_rf_10")

# Open an image (this would normally be your transformed image)
image = Image.open(source_image_path)

# Save the image using the method
saved_image_path = image_processor._save_result_image(source_path=source_image_path, image=image, results_dir=results_directory)

print(f"Saved image at: {saved_image_path}")
