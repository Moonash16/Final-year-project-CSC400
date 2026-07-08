package com.lisive.detector;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class deepfakedetector {
    
    private static final String TAG = "deepfakedetector";
    private static final String MODEL_PATH = "premune_deepfake_model_quantized.tflite";
    private static final int INPUT_SIZE = 224;
    private static final int NUM_CLASSES = 2; // [Fake, Real] output from Softmax
    
    private Interpreter tflite;
    private GpuDelegate gpuDelegate;
    private Context context;
    private forensicanalyzer forensicAnalyzer;
    private boolean isInitialized = false;
    private boolean useGpu = false;
    
    // Detection thresholds - Recalibrated for specialized sensitivity
    private static final float DEEPFAKE_THRESHOLD = 0.45f;
    private static final float FORENSIC_WEIGHT = 0.70f;
    private static final float ML_WEIGHT = 0.30f;
    
    public deepfakedetector(Context context) {
        this.context = context;
        this.forensicAnalyzer = new forensicanalyzer();
    }
    
    public interface InitializationCallback {
        void onSuccess();
        void onError(String error);
    }
    
    public interface DetectionCallback {
        void onResult(boolean isFake, float confidence, forensicanalyzer.ForensicResult forensicResult);
        void onProgress(String message);
        void onError(String error);
    }
    
    public void initialize(InitializationCallback callback) {
        try {
            // Load the TensorFlow Lite model
            MappedByteBuffer modelBuffer = loadModelFile();
            
            // Configure interpreter options
            Interpreter.Options options = new Interpreter.Options();
            
            // Try to enable GPU delegation for faster inference
            try {
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
                useGpu = true;
                Log.d(TAG, "GPU delegate enabled for faster detection");
            } catch (Exception e) {
                Log.w(TAG, "GPU delegate not available, using CPU: " + e.getMessage());
                useGpu = false;
            }
            
            // Set number of threads for CPU (optimize for your device)
            options.setNumThreads(4);
            
            tflite = new Interpreter(modelBuffer, options);
            
            // Log model details
            logModelDetails();
            
            isInitialized = true;
            callback.onSuccess();
            
        } catch (IOException e) {
            Log.e(TAG, "Error loading model file", e);
            callback.onError("Model file not found: " + e.getMessage());
        } catch (Exception e) {
            Log.e(TAG, "Unexpected error during initialization", e);
            callback.onError("Initialization failed: " + e.getMessage());
        }
    }
    
    private void logModelDetails() {
        try {
            // Log input tensor details
            org.tensorflow.lite.Tensor inputTensor = tflite.getInputTensor(0);
            Log.d(TAG, "Input tensor shape: " + java.util.Arrays.toString(inputTensor.shape()));
            Log.d(TAG, "Input tensor data type: " + inputTensor.dataType());
            
            // Log output tensor details
            org.tensorflow.lite.Tensor outputTensor = tflite.getOutputTensor(0);
            Log.d(TAG, "Output tensor shape: " + java.util.Arrays.toString(outputTensor.shape()));
            Log.d(TAG, "Output tensor data type: " + outputTensor.dataType());
            
        } catch (Exception e) {
            Log.w(TAG, "Could not log model details: " + e.getMessage());
        }
    }
    
    private MappedByteBuffer loadModelFile() throws IOException {
        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_PATH);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            fileDescriptor.close();
            inputStream.close();
            Log.d(TAG, "Model loaded successfully. Size: " + declaredLength + " bytes");
            return buffer;
        } catch (IOException e) {
            Log.e(TAG, "Failed to load model from assets. Make sure mobilenet_model.tflite exists in assets folder");
            throw e;
        }
    }
    
    public void detectdeepfake(Uri imageUri, DetectionCallback callback) {
        detectdeepfake(imageUri, null, callback);
    }

    public void detectdeepfake(Uri imageUri, Bitmap providedBitmap, DetectionCallback callback) {
        if (!isInitialized) {
            callback.onError("Detector not initialized. Please wait for model to load.");
            return;
        }
        
        new Thread(() -> {
            try {
                callback.onProgress("Loading image...");
                
                // Load and preprocess image
                Bitmap originalBitmap;
                if (providedBitmap != null) {
                    originalBitmap = providedBitmap.copy(providedBitmap.getConfig(), true);
                } else if (imageUri != null) {
                    originalBitmap = loadBitmapFromUri(imageUri);
                } else {
                    callback.onError("No image provided");
                    return;
                }

                if (originalBitmap == null) {
                    callback.onError("Failed to load image");
                    return;
                }
                
                callback.onProgress("Running forensic analysis...");
                
                // Run forensic analysis (manipulation artifacts)
                forensicanalyzer.ForensicResult forensicResult = 
                    forensicAnalyzer.analyzeImage(originalBitmap);
                
                callback.onProgress("Analyzing with AI model...");
                
                // Run TensorFlow Lite inference
                float mlScore = runModelInference(originalBitmap);
                
                callback.onProgress("Combining results...");
                
                // Combine forensic and ML scores
                float finalConfidence = calculateFinalConfidence(forensicResult, mlScore);
                boolean isFake = finalConfidence > DEEPFAKE_THRESHOLD;
                
                // Clean up
                originalBitmap.recycle();
                
                // Return result
                callback.onResult(isFake, finalConfidence, forensicResult);
                
            } catch (Exception e) {
                Log.e(TAG, "Detection error", e);
                callback.onError("Detection failed: " + e.getMessage());
            }
        }).start();
    }
    
    private float runModelInference(Bitmap bitmap) {
        try {
            // Preprocess bitmap into a ByteBuffer for quantized uint8 input
            java.nio.ByteBuffer inputBuffer = preprocessBitmapToBuffer(bitmap);
            
            // Prepare output array (1 x NUM_CLASSES) as bytes for uint8
            byte[][] output = new byte[1][NUM_CLASSES];
            
            // Run inference
            long startTime = System.currentTimeMillis();
            tflite.run(inputBuffer, output);
            long inferenceTime = System.currentTimeMillis() - startTime;
            
            Log.d(TAG, "Quantized Inference completed in " + inferenceTime + "ms");
            
            // Convert byte output (0-255) to float probability (0-1)
            float deepfakeScore = extractQuantizedProbability(output[0]);
            
            return deepfakeScore;
            
        } catch (Exception e) {
            Log.e(TAG, "Model inference failed", e);
            return 0.5f;
        }
    }

    private java.nio.ByteBuffer preprocessBitmapToBuffer(Bitmap bitmap) {
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        
        // 1 * 224 * 224 * 3 bytes (1 byte per channel for uint8)
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * 3);
        buffer.order(java.nio.ByteOrder.nativeOrder());
        
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        
        for (int pixel : pixels) {
            buffer.put((byte) ((pixel >> 16) & 0xFF)); // R
            buffer.put((byte) ((pixel >> 8) & 0xFF));  // G
            buffer.put((byte) (pixel & 0xFF));         // B
        }
        
        resizedBitmap.recycle();
        return buffer;
    }

    private float extractQuantizedProbability(byte[] output) {
        // Convert unsigned byte (0-255) to float (0-1)
        // In Java, bytes are signed, so we use & 0xFF to get the unsigned value
        if (output.length >= 2) {
            // Index 0 is 'Fake' (alphabetical order in Keras flow_from_directory)
            return (output[0] & 0xFF) / 255.0f;
        } else if (output.length == 1) {
            return (output[0] & 0xFF) / 255.0f;
        }
        return 0.0f;
    }
    
    private Integer[] getTopKIndices(float[] array, int k) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }
        
        // Sort indices by probability (descending)
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(array[b], array[a]));
        
        // Return top k
        Integer[] result = new Integer[Math.min(k, indices.length)];
        System.arraycopy(indices, 0, result, 0, result.length);
        return result;
    }
    
    private float calculateCategoryConsistency(Integer[] topCategories) {
        // Check if top categories are semantically related
        // For now, use a simplified approach
        if (topCategories.length < 2) return 1.0f;
        
        // In a real implementation, you would check category relationships
        // For quantized model, assume categories 0-500 are "person/face" related
        int faceRelatedCount = 0;
        for (int category : topCategories) {
            if (category < 500) { // Approximate: person/face categories in ImageNet
                faceRelatedCount++;
            }
        }
        
        return faceRelatedCount / (float) topCategories.length;
    }
    
    private float calculateFinalConfidence(forensicanalyzer.ForensicResult forensicResult, float mlScore) {
        double forensicBase = forensicResult.overallScore;

        // Apply a mild power curve to spread scores away from the center
        // Scores < 0.45 (threshold) get lower, scores > 0.45 get higher
        float spreadForensic;
        if (forensicBase > DEEPFAKE_THRESHOLD) {
            spreadForensic = (float) Math.pow(forensicBase, 1.1);
        } else {
            spreadForensic = (float) Math.pow(forensicBase, 1.4);
        }
        
        float weightedScore = (spreadForensic * FORENSIC_WEIGHT) + (mlScore * ML_WEIGHT);
        
        // Agreement Boost/Penalty
        if (forensicBase > 0.55 && mlScore > 0.3) {
            weightedScore = Math.min(0.98f, weightedScore + 0.12f);
        } else if (forensicBase < 0.4 && mlScore < 0.15) {
            weightedScore = Math.max(0.02f, weightedScore - 0.12f);
        }
        
        // AI Artifacts are a very strong indicator.
        if (forensicResult.aiArtifactScore > 0.85) {
            weightedScore = Math.max(weightedScore, 0.82f);
        }
        
        return Math.max(0.01f, Math.min(0.99f, weightedScore));
    }
    
    private Bitmap loadBitmapFromUri(Uri uri) throws IOException {
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.RGB_565;
            options.inSampleSize = 1;
            
            Bitmap bitmap;
            if ("content".equals(uri.getScheme())) {
                // Handle content URI (from Gallery)
                bitmap = BitmapFactory.decodeStream(context.getContentResolver().openInputStream(uri), null, options);
            } else {
                // Handle file URI
                bitmap = BitmapFactory.decodeFile(uri.getPath(), options);
            }
            
            if (bitmap == null) {
                throw new IOException("Could not decode image from URI: " + uri);
            }
            
            // Limit bitmap size to prevent OOM
            if (bitmap.getWidth() > 1024 || bitmap.getHeight() > 1024) {
                float scale = Math.min(1024f / bitmap.getWidth(), 1024f / bitmap.getHeight());
                int newWidth = Math.round(bitmap.getWidth() * scale);
                int newHeight = Math.round(bitmap.getHeight() * scale);
                Bitmap scaled = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
                bitmap.recycle();
                bitmap = scaled;
            }
            
            return bitmap;
            
        } catch (Exception e) {
            Log.e(TAG, "Error loading bitmap", e);
            throw new IOException("Failed to load image: " + e.getMessage());
        }
    }
    
    public boolean isInitialized() {
        return isInitialized;
    }
    
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
            gpuDelegate = null;
        }
        isInitialized = false;
    }
}