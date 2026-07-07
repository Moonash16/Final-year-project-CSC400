"""
Sample dataset to reduce number of images for faster training
Takes 1000 images from each class in train, validation, test
"""

import os
import random
import shutil

# Configuration
SOURCE_ROOT = "dataset"          # Original dataset folder
TARGET_ROOT = "sampled_dataset"  # New folder with limited samples
SAMPLES_PER_CLASS = 1000          # Number of images per class per split

def sample_images():
    """Copy 1000 random images from each class to new folder"""
    
    splits = ['train', 'validation', 'test']
    classes = ['fake', 'real']
    
    # Create target folder structure
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(TARGET_ROOT, split, cls), exist_ok=True)
    
    total_copied = 0
    
    for split in splits:
        print(f"\n📁 Processing {split}...")
        
        for cls in classes:
            source_dir = os.path.join(SOURCE_ROOT, split, cls)
            target_dir = os.path.join(TARGET_ROOT, split, cls)
            
            # List all images in source
            images = [f for f in os.listdir(source_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"   {split}/{cls}: {len(images)} images found")
            
            # Randomly select up to SAMPLES_PER_CLASS
            selected = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
            
            # Copy selected images to target
            for img in selected:
                shutil.copy2(os.path.join(source_dir, img), 
                             os.path.join(target_dir, img))
            
            print(f"   → Copied {len(selected)} images to {target_dir}")
            total_copied += len(selected)
    
    print(f"\n✅ Done! Copied {total_copied} images to '{TARGET_ROOT}'")

if __name__ == "__main__":
    print("=" * 50)
    print("Dataset Sampler - Reducing to 1000 images per class")
    print("=" * 50)
    sample_images()
    print(f"\nNow run: python retrain_model.py (it will use sampled_dataset)")