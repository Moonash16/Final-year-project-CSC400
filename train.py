"""
LiSive Deepfake Detection Model Training Script
University of Eswatini - Department of Computer Science
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
class Config:
    # Paths (using your exact folder structure)
    TRAIN_DIR = "data/train"
    VAL_DIR = "data/validation"
    TEST_DIR = "data/test"
    MODEL_SAVE_PATH = "models/lisive_deepfake_model.h5"
    TFLITE_SAVE_PATH = "models/lisive_deepfake_model.tflite"
    
    # Training parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 15  # Reduced since you have good data size
    LEARNING_RATE = 0.0001
    
    # Model architecture
    DROPOUT_RATE = 0.3
    NUM_CLASSES = 2  # real, fake
    
config = Config()

print("=" * 60)
print("LiSive Deepfake Detection Model Training")
print("University of Eswatini - Department of Computer Science")
print("=" * 60)
print(f"Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Epochs: {config.EPOCHS}")
print(f"Learning Rate: {config.LEARNING_RATE}")
print("=" * 60)

# Step 1: Load and Augment Data
print("\n[1/5] Loading and augmenting data...")

# Data augmentation for training (helps prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for validation and test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    config.VAL_DIR,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Loading test data...")
test_generator = test_datagen.flow_from_directory(
    config.TEST_DIR,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\n✅ Dataset loaded!")
print(f"Classes: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Batches per epoch: {train_generator.samples // config.BATCH_SIZE}")

# Step 2: Build the Model
print("\n[2/5] Building model architecture...")

# Load pre-trained MobileNetV2 (optimized for mobile)
base_model = MobileNetV2(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers initially
base_model.trainable = False

# Build the complete model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(config.DROPOUT_RATE),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(config.NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

print("✅ Model built successfully!")
print(f"Total parameters: {model.count_params():,}")

# Step 3: Set up Callbacks
print("\n[3/5] Setting up training callbacks...")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

callbacks_list = [
    # Save the best model
    callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    # Early stopping if no improvement
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateauing
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("✅ Callbacks configured")

# Step 4: Train the Model
print("\n[4/5] Training model...")
print("=" * 60)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
    epochs=config.EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // config.BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

print("\n✅ Initial training complete!")

# Step 5: Fine-tune the model
print("\n[4.5/5] Fine-tuning model (unfreezing base model)...")

# Unfreeze the last 50 layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

print("Fine-tuning with learning rate:", config.LEARNING_RATE / 10)

# Continue training
fine_tune_epochs = 10
total_epochs = config.EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_generator,
    validation_steps=val_generator.samples // config.BATCH_SIZE,
    callbacks=callbacks_list,
    verbose=1
)

# Combine histories
for key in history.history.keys():
    history.history[key].extend(history_fine.history[key])

print("\n✅ Fine-tuning complete!")

# Step 6: Evaluate the Model
print("\n[5/5] Evaluating model...")
print("=" * 60)

# Evaluate on test set
test_results = model.evaluate(test_generator, verbose=1)

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print(f"Test Precision: {test_results[2]:.4f}")
print(f"Test Recall: {test_results[3]:.4f}")
print(f"Test AUC: {test_results[4]:.4f}")
print("=" * 60)

# Save the final model
model.save(config.MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {config.MODEL_SAVE_PATH}")

# Step 7: Plot training history
print("\n[6/6] Generating training plots...")

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy plot
    axes[0, 0].plot(history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision plot
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], 'b-', label='Train Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], 'r-', label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall plot
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], 'b-', label='Train Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], 'r-', label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("✅ Training plot saved as 'training_history.png'")
    plt.show()

plot_training_history(history)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Model saved at: {config.MODEL_SAVE_PATH}")
print("Next step: Run convert_to_tflite.py to create Android-ready model")