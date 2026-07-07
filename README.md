# deepfake-detection/deepfake-detection/README.md

# Deepfake Detection Project

This project aims to develop a deepfake detection model using TensorFlow/Keras, optimized for mobile deployment. The model is trained on a dataset of real and fake images, and it is exported as a TensorFlow Lite model for use in mobile applications.

## Project Structure

- **data/**: Contains raw and processed images for training.
  - **raw/**: Directory for raw images.
  - **processed/**: Directory for processed images ready for training.
  - **README.md**: Documentation about the data structure and usage.

- **models/**: Stores the trained models.
  - **model.h5**: The trained Keras model in HDF5 format.
  - **model.tflite**: The exported TensorFlow Lite model for mobile deployment.

- **notebooks/**: Contains Jupyter notebooks for data exploration.
  - **exploration.ipynb**: Notebook for exploratory data analysis and visualization.

- **src/**: Source code for data processing, model training, and exporting.
  - **data_preprocessing.py**: Functions for loading and preprocessing the dataset.
  - **model_training.py**: Defines the model architecture and training loop.
  - **model_export.py**: Handles conversion of the Keras model to TensorFlow Lite format.
  - **utils.py**: Utility functions for logging and evaluation metrics.

- **requirements.txt**: Lists dependencies required for the project, including TensorFlow and Keras.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd deepfake-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset by placing raw images in the `data/raw` directory.

4. Run the data preprocessing script to process the images:
   ```
   python src/data_preprocessing.py
   ```

5. Train the model using:
   ```
   python src/model_training.py
   ```

6. Export the trained model to TensorFlow Lite format:
   ```
   python src/model_export.py
   ```

## Usage

The trained model can be used for detecting deepfakes in images. Load the TensorFlow Lite model in your mobile application and use it to classify images as real or fake.

## Acknowledgments

This project is inspired by the growing need for reliable deepfake detection methods in the digital age.