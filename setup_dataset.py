# setup_dataset.py
import os
import shutil
import random
from tqdm import tqdm

def organize_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Organize raw data into train/validation/test splits
    Expected structure:
    source_dir/
        real/  (folder with real images)
        fake/  (folder with fake images)
    """
    
    # Create target directories
    for split in ['train', 'validation', 'test']:
        for cls in ['real', 'fake']:
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)
    
    # Process each class
    for class_name in ['real', 'fake']:
        source_path = os.path.join(source_dir, class_name)
        images = os.listdir(source_path)
        random.shuffle(images)
        
        # Split indices
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy files
        for img in tqdm(train_images, desc=f"Copying {class_name} train"):
            shutil.copy(
                os.path.join(source_path, img),
                os.path.join(target_dir, 'train', class_name, img)
            )
        
        for img in tqdm(val_images, desc=f"Copying {class_name} validation"):
            shutil.copy(
                os.path.join(source_path, img),
                os.path.join(target_dir, 'validation', class_name, img)
            )
        
        for img in tqdm(test_images, desc=f"Copying {class_name} test"):
            shutil.copy(
                os.path.join(source_path, img),
                os.path.join(target_dir, 'test', class_name, img)
            )
    
    print(f"Dataset organized at: {target_dir}")
    print(f"Train: {len(train_images)} per class")
    print(f"Validation: {len(val_images)} per class")
    print(f"Test: {len(test_images)} per class")

if __name__ == "__main__":
    organize_dataset(
        source_dir="raw_data",  # Your raw data folder
        target_dir="data",       # Output folder
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )