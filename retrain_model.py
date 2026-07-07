"""
LiSive Deepfake Detection Model - Training with Sampled Dataset
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
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Paths - USE SAMPLED DATASET (1000 images per class)
    DATASET_ROOT = "sampled_dataset"   # <-- CHANGE THIS
    TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
    VAL_DIR = os.path.join(DATASET_ROOT, "validation")
    TEST_DIR = os.path.join(DATASET_ROOT, "test")
    
    # Output paths
    MODEL_SAVE_PATH = "models/lisive_deepfake_model.h5"
    TFLITE_SAVE_PATH = "models/lisive_deepfake_model_quantized.tflite"
    TFLITE_FLOAT_PATH = "models/lisive_deepfake_model_float.tflite"
    
    # Training parameters - REDUCED EPOCHS
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 10               # <-- REDUCED from 30 to 10
    FINE_TUNE_EPOCHS = 5      # <-- REDUCED from 10 to 5
    LEARNING_RATE = 0.0001
    
    # Model architecture
    BASE_MODEL = "mobilenetv2"
    DROPOUT_RATE = 0.3
    NUM_CLASSES = 2
    
    # Data augmentation
    USE_AUGMENTATION = True
    
config = Config()

# ============================================
# VERIFY DATASET STRUCTURE
# ============================================

def verify_dataset():
    """Check if dataset folders exist and have images"""
    print("\n" + "=" * 60)
    print("VERIFYING DATASET STRUCTURE")
    print("=" * 60)
    
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(config.DATASET_ROOT, split)
        if not os.path.exists(split_path):
            print(f"❌ Missing folder: {split_path}")
            return False
        
        for cls in ['fake', 'real']:
            cls_path = os.path.join(split_path, cls)
            if not os.path.exists(cls_path):
                print(f"❌ Missing folder: {cls_path}")
                return False
            
            num_images = len([f for f in os.listdir(cls_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   {split}/{cls}: {num_images} images")
    
    return True

# ============================================
# DATA LOADING WITH AUGMENTATION
# ============================================

def load_data():
    """Load and augment datasets"""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    if config.USE_AUGMENTATION:
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
        print("✅ Data augmentation ENABLED")
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
        print("⚠️ Data augmentation DISABLED")
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Classes: {train_generator.class_indices}")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Test samples: {test_generator.samples}")
    print(f"   Batches per epoch: {train_generator.samples // config.BATCH_SIZE}")
    
    return train_generator, val_generator, test_generator

# ============================================
# BUILD MODEL
# ============================================

def build_model():
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    
    base_model = MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    print(f"✅ Using MobileNetV2 base model")
    
    base_model.trainable = False
    
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
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model, base_model

# ============================================
# TRAINING PHASE 1
# ============================================

def train_model(model, train_generator, val_generator):
    print("\n" + "=" * 60)
    print(f"TRAINING PHASE 1 - Frozen Base Model ({config.EPOCHS} epochs)")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    earlystop_cb = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )
    reducelr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
    callbacks_list = [checkpoint_cb, earlystop_cb, reducelr_cb]
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // config.BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )
    return history

# ============================================
# TRAINING PHASE 2 (FINE-TUNING)
# ============================================

def fine_tune_model(model, base_model, train_generator, val_generator, history):
    print("\n" + "=" * 60)
    print(f"TRAINING PHASE 2 - Fine-Tuning ({config.FINE_TUNE_EPOCHS} epochs)")
    print("=" * 60)
    
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    print(f"✅ Fine-tuning with learning rate: {config.LEARNING_RATE / 10}")
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    earlystop_cb = callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )
    callbacks_list = [checkpoint_cb, earlystop_cb]
    
    total_epochs = config.EPOCHS + config.FINE_TUNE_EPOCHS
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
    
    for key in history.history.keys():
        history.history[key].extend(history_fine.history[key])
    
    return history

# ============================================
# EVALUATION
# ============================================

def evaluate_model(model, test_generator):
    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    results = model.evaluate(test_generator, verbose=1)
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']
    print("\n📊 Test Results:")
    for name, value in zip(metrics, results):
        print(f"   {name}: {value:.4f}")
    
    test_generator.reset()
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    cm = confusion_matrix(true_classes, predicted_classes)
    print("\n📊 Confusion Matrix:")
    print(f"   True Fake: {cm[0,0]} correct, {cm[0,1]} misclassified as Real")
    print(f"   True Real: {cm[1,1]} correct, {cm[1,0]} misclassified as Fake")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE', 'REAL'],
                yticklabels=['FAKE', 'REAL'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")
    plt.close()
    
    return results

# ============================================
# CONVERT TO TFLITE
# ============================================

def convert_to_tflite(model):
    print("\n" + "=" * 60)
    print("CONVERTING TO TENSORFLOW LITE")
    print("=" * 60)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    print("📦 Applying INT8 quantization...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, config.IMG_SIZE, config.IMG_SIZE, 3).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(config.TFLITE_SAVE_PATH), exist_ok=True)
    with open(config.TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Also save float version
    converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter_float.convert()
    with open(config.TFLITE_FLOAT_PATH, 'wb') as f:
        f.write(tflite_float)
    
    keras_size = os.path.getsize(config.MODEL_SAVE_PATH) / (1024 * 1024)
    quant_size = os.path.getsize(config.TFLITE_SAVE_PATH) / (1024 * 1024)
    float_size = os.path.getsize(config.TFLITE_FLOAT_PATH) / (1024 * 1024)
    
    print("\n📊 Model Size Comparison:")
    print(f"   Original Keras model: {keras_size:.2f} MB")
    print(f"   Quantized TFLite: {quant_size:.2f} MB")
    print(f"   Float32 TFLite: {float_size:.2f} MB")
    print(f"   Reduction: {(1 - quant_size/keras_size) * 100:.1f}%")
    print(f"\n✅ Quantized model saved to: {config.TFLITE_SAVE_PATH}")

# ============================================
# PLOT TRAINING HISTORY
# ============================================

def plot_history(history):
    print("\n" + "=" * 60)
    print("GENERATING TRAINING PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history.history['accuracy'], 'b-', label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], 'r-', label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history.history['loss'], 'b-', label='Train')
    axes[0, 1].plot(history.history['val_loss'], 'r-', label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], 'b-', label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], 'r-', label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], 'b-', label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], 'r-', label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("✅ Training history saved as 'training_history.png'")
    plt.close()

# ============================================
# MAIN PIPELINE
# ============================================

def main():
    print("=" * 60)
    print("LISIVE DEEPFAKE DETECTION - SAMPLED DATASET (1000 per class)")
    print("University of Eswatini - Department of Computer Science")
    print("=" * 60)
    
    if not verify_dataset():
        print("\n❌ Dataset verification failed. Run 'python sample_dataset.py' first.")
        return
    
    train_gen, val_gen, test_gen = load_data()
    model, base_model = build_model()
    history = train_model(model, train_gen, val_gen)
    history = fine_tune_model(model, base_model, train_gen, val_gen, history)
    evaluate_model(model, test_gen)
    plot_history(history)
    convert_to_tflite(model)
    
    print("\n✅ TRAINING COMPLETE! Use the .tflite file in your Android project.")

if __name__ == "__main__":
    main()