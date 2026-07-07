import os
import shutil

def flatten_dataset(base_dir):
    """
    Flatten the dataset structure by moving all images from subfolders
    into the parent 'fake' and 'real' folders. Handles duplicate filenames.
    """
    for category in ['fake', 'real']:
        category_path = os.path.join(base_dir, category)
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file)
                    if os.path.isfile(file_path):
                        # Check if the file already exists in the destination
                        destination_path = os.path.join(category_path, file)
                        if os.path.exists(destination_path):
                            # Rename the file to avoid overwriting
                            base, ext = os.path.splitext(file)
                            new_file = f"{base}_{os.urandom(4).hex()}{ext}"
                            destination_path = os.path.join(category_path, new_file)
                        shutil.move(file_path, destination_path)
                # Remove the now-empty subfolder
                os.rmdir(subfolder_path)

if __name__ == "__main__":
    # Replace with the path to your test directory
    base_dir = r"C:\Users\BMS\program files\Desktop\programming languages\princess\lisive_model.tflite\data\test"
    flatten_dataset(base_dir)