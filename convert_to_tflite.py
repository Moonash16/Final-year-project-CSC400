"""
Convert trained deepfake detection model to TensorFlow Lite for Android deployment
University of Eswatini - Department of Computer Science
"""

import tensorflow as tf
import numpy as np
import os

def convert_to_tflite(keras_model_path, output_path, quantize=True):
    """
    Convert Keras model to TFLite format
    
    Args:
        keras_model_path: Path to the .h5 Keras model
        output_path: Path to save the .tflite file
        quantize: Whether to apply quantization (reduces file size)
    """
    
    print("=" * 60)
    print("TensorFlow Lite Model Converter")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(keras_model_path):
        print(f"❌ Model not found at: {keras_model_path}")
        print("Make sure you have run train.py first!")
        return None
    
    print(f"📁 Loading model from: {keras_model_path}")
    
    # Load the trained model
    model = tf.keras.models.load_model(keras_model_path)
    print(f"✅ Model loaded successfully!")
    
    # Print model summary
    model.summary()
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("\n📦 Applying INT8 quantization for mobile optimization...")
        
        # Quantization for smaller file size and faster inference
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization calibration
        # This helps maintain accuracy after quantization
        def representative_dataset():
            # Generate random samples in the range [0, 255] (uint8 range)
            for _ in range(100):
                # Random 224x224 RGB image normalized to [0,1]
                data = np.random.rand(1, 224, 224, 3).astype(np.float32)
                yield [data]
        
        converter.representative_dataset = representative_dataset
        
        # Set target specification for INT8 quantization
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        print("   - Quantization type: INT8")
        print("   - Input type: uint8 (0-255)")
        print("   - Output type: uint8")
    else:
        print("\n📦 Converting without quantization (FP32)...")
    
    print("\n🔄 Converting to TFLite...")
    
    # Convert the model
    try:
        tflite_model = converter.convert()
        print("✅ Conversion successful!")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    original_size_mb = os.path.getsize(keras_model_path) / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("CONVERSION RESULTS")
    print("=" * 60)
    print(f"📁 Original Keras model: {original_size_mb:.2f} MB")
    print(f"📱 TFLite model: {size_mb:.2f} MB")
    print(f"💾 Size reduction: {(1 - size_mb/original_size_mb) * 100:.1f}%")
    print(f"✅ TFLite model saved to: {output_path}")
    print("=" * 60)
    
    # Verify the TFLite model
    print("\n🔍 Verifying TFLite model...")
    verify_tflite_model(output_path)
    
    return tflite_model

def verify_tflite_model(tflite_path):
    """
    Verify the TFLite model loads correctly and print details
    """
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\n📊 Model Details:")
        print(f"   Input name: {input_details[0]['name']}")
        print(f"   Input shape: {input_details[0]['shape']}")
        print(f"   Input type: {input_details[0]['dtype']}")
        print(f"   Output name: {output_details[0]['name']}")
        print(f"   Output shape: {output_details[0]['shape']}")
        print(f"   Output type: {output_details[0]['dtype']}")
        
        # Test inference with random input
        input_shape = input_details[0]['shape']
        random_input = np.random.rand(*input_shape).astype(input_details[0]['dtype'])
        
        # For uint8 input, scale to [0, 255]
        if input_details[0]['dtype'] == np.uint8:
            random_input = (random_input * 255).astype(np.uint8)
        
        interpreter.set_tensor(input_details[0]['index'], random_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n✅ TFLite model verified! Test inference successful.")
        print(f"   Test output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TFLite verification failed: {e}")
        return False

def convert_float_model(keras_model_path, output_path):
    """
    Convert without quantization (float32) - better accuracy, larger file size
    """
    print("=" * 60)
    print("Converting Float32 Model (No Quantization)")
    print("=" * 60)
    
    if not os.path.exists(keras_model_path):
        print(f"❌ Model not found at: {keras_model_path}")
        return None
    
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # No quantization
    converter.optimizations = []
    
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Float32 TFLite model saved to: {output_path}")
    print(f"📦 Model size: {size_mb:.2f} MB")
    
    return tflite_model

def compare_models(keras_path, quantized_path, float_path=None):
    """
    Compare original vs quantized model sizes
    """
    print("\n" + "=" * 60)
    print("MODEL SIZE COMPARISON")
    print("=" * 60)
    
    keras_size = os.path.getsize(keras_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    
    print(f"Original Keras model:    {keras_size:.2f} MB")
    print(f"Quantized TFLite model:  {quantized_size:.2f} MB")
    print(f"Reduction:               {(1 - quantized_size/keras_size) * 100:.1f}%")
    
    if float_path and os.path.exists(float_path):
        float_size = os.path.getsize(float_path) / (1024 * 1024)
        print(f"Float32 TFLite model:    {float_size:.2f} MB")

if __name__ == "__main__":
    # Paths to your models
    KERAS_MODEL_PATH = "models/lisive_deepfake_model.h5"
    QUANTIZED_OUTPUT_PATH = "models/lisive_deepfake_model_quantized.tflite"
    FLOAT_OUTPUT_PATH = "models/lisive_deepfake_model_float.tflite"
    
    print("\n🚀 Starting TFLite Conversion Process")
    print("=" * 60)
    
    # Option 1: Convert with quantization (RECOMMENDED for mobile)
    print("\n📱 Option 1: Converting with INT8 Quantization (Recommended for Android)")
    convert_to_tflite(
        keras_model_path=KERAS_MODEL_PATH,
        output_path=QUANTIZED_OUTPUT_PATH,
        quantize=True
    )
    
    # Option 2: Convert without quantization (better accuracy, larger file)
    print("\n\n💻 Option 2: Converting without quantization (Float32)")
    convert_float_model(
        keras_model_path=KERAS_MODEL_PATH,
        output_path=FLOAT_OUTPUT_PATH
    )
    
    # Compare models
    compare_models(KERAS_MODEL_PATH, QUANTIZED_OUTPUT_PATH, FLOAT_OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 60)
    print("\n📱 For Android deployment, use:")
    print(f"   {QUANTIZED_OUTPUT_PATH}")
    print("\n📋 Next steps:")
    print("   1. Copy the .tflite file to your Android project")
    print("   2. Place it in: app/src/main/assets/")
    print("   3. Update MODEL_PATH in deepfakedetector.java to match the filename")
    print("=" * 60)