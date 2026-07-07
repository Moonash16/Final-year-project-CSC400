from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image

def load_and_preprocess_data(raw_data_dir, target_size=(224, 224)):
    """
    Load and preprocess images from the raw data directory.
    
    Parameters:
    - raw_data_dir: Directory containing raw images.
    - target_size: Desired size for the images (default is 224x224).
    
    Returns:
    - images: Numpy array of preprocessed images.
    - labels: List of labels corresponding to the images.
    """
    images = []
    labels = []
    
    for class_name in ['real', 'fake']:
        class_dir = os.path.join(raw_data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(class_name)
    
    return np.array(images), np.array(labels)

def augment_data(images, labels):
    """
    Augment the dataset using ImageDataGenerator.
    
    Parameters:
    - images: Numpy array of images.
    - labels: Numpy array of labels.
    
    Returns:
    - generator: Data generator for augmented images.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    generator = datagen.flow(images, labels, batch_size=32)
    return generator