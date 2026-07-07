import tensorflow as tf
import numpy as np
import os

# Paths
KERAS_MODEL_PATH = "models/lisive_deepfake_model.h5"
QUANTIZED_OUTPUT = "models/lisive_deepfake_model_quantized.tflite"
FLOAT_OUTPUT = "models/lisive_deepfake_model_float.tflite"

print("Loading model...")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# 1. Quantized model (small, fast, use this for Android)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_quant = converter.convert()
with open(QUANTIZED_OUTPUT, "wb") as f:
    f.write(tflite_quant)

# 2. Float model (backup, larger)
converter_float = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float = converter_float.convert()
with open(FLOAT_OUTPUT, "wb") as f:
    f.write(tflite_float)

# Show sizes
keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
quant_size = os.path.getsize(QUANTIZED_OUTPUT) / (1024 * 1024)
float_size = os.path.getsize(FLOAT_OUTPUT) / (1024 * 1024)

print("\n✅ Conversion complete")
print(f"Keras model: {keras_size:.2f} MB")
print(f"Quantized TFLite (for Android): {quant_size:.2f} MB")
print(f"Float TFLite (backup): {float_size:.2f} MB")
print(f"\nCopy '{QUANTIZED_OUTPUT}' to your Android project's 'app/src/main/assets/'")