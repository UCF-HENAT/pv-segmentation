import os
from PIL import Image
from pathlib import Path
import time


def convert_bmp_to_png_and_separate(source_folder, target_base_folder):
    # Walk through the source directory
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".bmp"):
                # Construct the full file path
                file_path = Path(root) / file
                # Extract the dataset name (e.g., 'PV01') from the path
                dataset_name = Path(root).parts[-2]  # Adjust based on your directory structure

                # Determine the specific target folder ('img' or 'mask') within the dataset folder
                if "_label" in file:
                    target_folder = Path(target_base_folder) / dataset_name / "mask"
                else:
                    target_folder = Path(target_base_folder) / dataset_name / "img"
                
                # Ensure the target folder exists
                target_folder.mkdir(parents=True, exist_ok=True)
                
                # Define the target file path
                target_file_path = target_folder / file.replace("_label.bmp", ".png").replace(".bmp", ".png")
                
                # Load the BMP image
                image = Image.open(file_path)

                # Convert and save the image as PNG
                image.save(target_file_path)

    print(f"Conversion and separation complete. Data organized under '{target_base_folder}'.")

# Example usage
source_folder = "/pv-segmentation/datasets/PV"
target_base_folder = "/pv-segmentation/datasets/png_PV"  # Replace PV folder with these images (manually)

start_time = time.time()

convert_bmp_to_png_and_separate(source_folder, target_base_folder)

end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")