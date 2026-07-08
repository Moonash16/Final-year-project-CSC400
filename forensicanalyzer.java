package com.lisive.detector;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Rect;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;
import com.google.android.gms.tasks.Tasks;
import java.util.List;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;

public class forensicanalyzer {
    
    public static class ForensicResult {
        public double elaScore;
        public double noiseScore;
        public double compressionScore;
        public double edgeScore;
        public double aiArtifactScore;
        public double blurInconsistencyScore;
        public double colorInconsistencyScore;
        public double noiseInconsistencyScore;
        public double portraitInconsistencyScore;
        public double overallScore;
        public boolean isManipulated;
        
        public ForensicResult(double ela, double noise, double compression, double edge, double ai, double blur, double color, double noiseInc, double portrait) {
            this.elaScore = ela;
            this.noiseScore = noise;
            this.compressionScore = compression;
            this.edgeScore = edge;
            this.aiArtifactScore = ai;
            this.blurInconsistencyScore = blur;
            this.colorInconsistencyScore = color;
            this.noiseInconsistencyScore = noiseInc;
            this.portraitInconsistencyScore = portrait;
            
            // Optimized exclusively for Deepfakes (Face Swaps & AI Generations)
            this.overallScore = (ai * 0.50 + portrait * 0.30 + ela * 0.10 + noiseInc * 0.10);
            this.isManipulated = this.overallScore > 0.45; // Sensitive threshold for specialized detection
        }
    }
    
    public static ForensicResult analyzeImage(Bitmap bitmap) {
        // Run face detection first to guide forensic checks
        List<Face> faces = detectFaces(bitmap);
        
        // Use higher resolution for artifact and ELA detection to preserve micro-patterns
        double aiArtifactScore = detectAIArtifacts(bitmap, faces);
        double elaScore = computeELA(bitmap);
        
        // Resize for macro-analysis metrics (performance)
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, 512, 512, true);
        
        double noiseScore = analyzeNoise(scaledBitmap);
        double compressionScore = detectCompression(scaledBitmap);
        double edgeScore = analyzeEdges(scaledBitmap);
        double blurScore = detectBlurInconsistency(scaledBitmap);
        double colorScore = detectColorInconsistency(scaledBitmap);
        double noiseIncScore = analyzeNoiseInconsistency(scaledBitmap);
        
        // Enhanced portrait check using detected face locations
        double portraitScore = detectPortraitInconsistency(bitmap, faces);
        
        scaledBitmap.recycle();
        
        return new ForensicResult(elaScore, noiseScore, compressionScore, edgeScore, aiArtifactScore, blurScore, colorScore, noiseIncScore, portraitScore);
    }

    private static List<Face> detectFaces(Bitmap bitmap) {
        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .build();
        FaceDetector detector = FaceDetection.getClient(options);
        InputImage image = InputImage.fromBitmap(bitmap, 0);
        try {
            return Tasks.await(detector.process(image));
        } catch (Exception e) {
            return new ArrayList<>();
        }
    }
    
    private static double computeELA(Bitmap bitmap) {
        try {
            // ELA: Error Level Analysis
            // Increased resolution to 512 for better local detail detection
            Bitmap scaled = Bitmap.createScaledBitmap(bitmap, 512, 512, true);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            scaled.compress(Bitmap.CompressFormat.JPEG, 70, out);
            byte[] data = out.toByteArray();
            Bitmap resaved = BitmapFactory.decodeByteArray(data, 0, data.length);

            if (resaved == null) return 0.2;

            double diffSum = 0;
            double maxDiff = 0;
            
            for (int y = 0; y < 128; y += 2) {
                for (int x = 0; x < 128; x += 2) {
                    int p1 = scaled.getPixel(x, y);
                    int p2 = resaved.getPixel(x, y);
                    
                    int rDiff = Math.abs(Color.red(p1) - Color.red(p2));
                    int gDiff = Math.abs(Color.green(p1) - Color.green(p2));
                    int bDiff = Math.abs(Color.blue(p1) - Color.blue(p2));
                    
                    double diff = (rDiff + gDiff + bDiff) / 3.0;
                    diffSum += diff;
                    if (diff > maxDiff) maxDiff = diff;
                }
            }
            
            double avgDiff = diffSum / (256 * 256);
            scaled.recycle();
            resaved.recycle();
            
            // If max difference is much higher than average, it indicates local manipulation
            double variance = maxDiff / (avgDiff + 1.0);
            return Math.min(1.0, variance / 15.0); // Higher denominator to lower baseline for clean images
            
        } catch (Exception e) {
            return 0.1;
        }
    }
    
    private static double analyzeNoise(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        double laplacianSum = 0;
        int count = 0;
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int p = (Color.red(bitmap.getPixel(x,y)) + Color.green(bitmap.getPixel(x,y)) + Color.blue(bitmap.getPixel(x,y))) / 3;
                int p1 = (Color.red(bitmap.getPixel(x-1,y)) + Color.green(bitmap.getPixel(x-1,y)) + Color.blue(bitmap.getPixel(x-1,y))) / 3;
                int p2 = (Color.red(bitmap.getPixel(x+1,y)) + Color.green(bitmap.getPixel(x+1,y)) + Color.blue(bitmap.getPixel(x+1,y))) / 3;
                int p3 = (Color.red(bitmap.getPixel(x,y-1)) + Color.green(bitmap.getPixel(x,y-1)) + Color.blue(bitmap.getPixel(x,y-1))) / 3;
                int p4 = (Color.red(bitmap.getPixel(x,y+1)) + Color.green(bitmap.getPixel(x,y+1)) + Color.blue(bitmap.getPixel(x,y+1))) / 3;
                
                // Laplacian filter (edge/noise detector)
                int laplacian = Math.abs(4 * p - p1 - p2 - p3 - p4);
                laplacianSum += laplacian;
                count++;
            }
        }
        
        double avgLaplacian = count > 0 ? laplacianSum / count : 0;
        // Adjusted denominator: iPhone photos often have high clarity (high Laplacian) 
        // which was being mistaken for artifacts. 180.0 is safer for modern phone sensors.
        return Math.min(1.0, avgLaplacian / 180.0);
    }
    
    private static double detectCompression(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        double artifactScore = 0;
        int blockCount = 0;
        
        int blockSize = 8;
        for (int y = 0; y < height - blockSize; y += blockSize) {
            for (int x = 0; x < width - blockSize; x += blockSize) {
                double variance = 0;
                for (int dy = 0; dy < blockSize; dy++) {
                    for (int dx = 0; dx < blockSize; dx++) {
                        int pixel = bitmap.getPixel(x + dx, y + dy);
                        int brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3;
                        variance += Math.pow(brightness - 128, 2);
                    }
                }
                variance = variance / (blockSize * blockSize);
                if (variance < 100) { 
                    artifactScore += 0.1;
                }
                blockCount++;
            }
        }
        
        double result = artifactScore / blockCount;
        return Math.min(1.0, result);
    }
    
    private static double analyzeEdges(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        double edgeInconsistency = 0;
        int edgeCount = 0;
        
        for (int y = 1; y < height - 1; y += 5) {
            for (int x = 1; x < width - 1; x += 5) {
                int gx = 0, gy = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int pixel = bitmap.getPixel(x + kx, y + ky);
                        int brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3;
                        gx += kx * brightness;
                        gy += ky * brightness;
                    }
                }
                
                int gradient = (int) Math.sqrt(gx * gx + gy * gy);
                if (gradient > 100) { 
                    edgeCount++;
                    if (gradient > 200) {
                        edgeInconsistency += 0.5;
                    }
                }
            }
        }
        
        double result = edgeCount > 0 ? edgeInconsistency / edgeCount : 0;
        return Math.min(1.0, result);
    }

    private static double detectAIArtifacts(Bitmap bitmap, List<Face> faces) {
        // AI models (GANs/Diffusion) often leave structured, repeating patterns (checkerboards or grids).
        // Scaling to small sizes (like 256x256) destroys these patterns, leading to false negatives.
        // We use a cluster-based detection approach on a high-res buffer to distinguish from random noise.
        Bitmap workBitmap = bitmap;
        boolean recycled = false;
        if (bitmap.getWidth() > 1024 || bitmap.getHeight() > 1024) {
            float scale = 1024f / Math.max(bitmap.getWidth(), bitmap.getHeight());
            workBitmap = Bitmap.createScaledBitmap(bitmap, (int)(bitmap.getWidth() * scale), (int)(bitmap.getHeight() * scale), false);
            recycled = true;
        }

        int w = workBitmap.getWidth();
        int h = workBitmap.getHeight();
        int artifactClusters = 0;
        int totalWindows = 0;
        
        // Scan in 16x16 windows to find clusters of periodic patterns
        int winSize = 16;
        for (int y = 0; y < h - winSize; y += winSize) {
            for (int x = 0; x < w - winSize; x += winSize) {
                int localMatches = 0;
                for (int dy = 0; dy < winSize - 2; dy += 2) {
                    for (int dx = 0; dx < winSize - 2; dx += 2) {
                        int p1 = Color.red(workBitmap.getPixel(x + dx, y + dy));
                        int p2 = Color.red(workBitmap.getPixel(x + dx + 1, y + dy));
                        int p3 = Color.red(workBitmap.getPixel(x + dx, y + dy + 1));
                        int p4 = Color.red(workBitmap.getPixel(x + dx + 1, y + dy + 1));
                        
                        // Look for structured oscillation [A B; B A]
                        // Natural noise is random; AI artifacts are repeating and structured.
                        if (Math.abs(p1 - p4) < 3 && Math.abs(p2 - p3) < 3 && Math.abs(p1 - p2) > 6) {
                            localMatches++;
                        }
                    }
                }
                
                // If a window has more than 3 matches (out of 64 checked), it's a structured cluster.
                // Random sensor noise rarely forms dense clusters of checkerboard patterns.
                if (localMatches >= 3) {
                    artifactClusters++;
                }
                totalWindows++;
            }
        }
        
        if (recycled) workBitmap.recycle();
        
        if (totalWindows == 0) return 0;
        double density = (double) artifactClusters / totalWindows;
        
        // AI-generated images typically have artifact clusters across > 5-15% of the image.
        // Natural photos may have sparse noise (< 1%) but rarely structured clusters.
        // Adjust weight to be more sensitive to density.
        return Math.min(1.0, density * 18.0);
    }

    private static double detectBlurInconsistency(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        
        // Finer 8x8 grid to catch local blur/sharpening differences
        int grid = 8;
        int cellW = width / grid;
        int cellH = height / grid;
        double[] variances = new double[grid * grid];
        
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                double sum = 0;
                double sumSq = 0;
                int count = 0;
                
                for (int y = gy * cellH + 1; y < (gy + 1) * cellH - 1; y += 2) {
                    for (int x = gx * cellW + 1; x < (gx + 1) * cellW - 1; x += 2) {
                        int p = Color.red(bitmap.getPixel(x, y));
                        int px = Color.red(bitmap.getPixel(x + 1, y));
                        int py = Color.red(bitmap.getPixel(x, y + 1));
                        
                        double grad = Math.sqrt(Math.pow(p - px, 2) + Math.pow(p - py, 2));
                        sum += grad;
                        sumSq += grad * grad;
                        count++;
                    }
                }
                
                if (count > 0) {
                    double mean = sum / count;
                    variances[gy * grid + gx] = (sumSq / count) - (mean * mean);
                }
            }
        }
        
        double maxVar = 0;
        double minVar = Double.MAX_VALUE;
        for (double v : variances) {
            if (v > maxVar) maxVar = v;
            if (v < minVar && v > 1.0) minVar = v;
        }
        
        if (maxVar == 0) return 0;
        
        // High ratio means some parts are much sharper than others (common in face swaps)
        double ratio = maxVar / (minVar + 0.05);
        return Math.min(1.0, ratio / 18.0); // Increased denominator for better variance
    }

    private static double detectColorInconsistency(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        
        int grid = 8; // Finer grid
        int cellW = width / grid;
        int cellH = height / grid;
        double[] cbMeans = new double[grid * grid];
        double[] crMeans = new double[grid * grid];
        
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                double cbSum = 0;
                double crSum = 0;
                int count = 0;
                
                for (int y = gy * cellH; y < (gy + 1) * cellH; y += 2) {
                    for (int x = gx * cellW; x < (gx + 1) * cellW; x += 2) {
                        int p = bitmap.getPixel(x, y);
                        int r = Color.red(p);
                        int g = Color.green(p);
                        int b = Color.blue(p);
                        
                        double cb = 128 - 0.168 * r - 0.331 * g + 0.5 * b;
                        double cr = 128 + 0.5 * r - 0.418 * g - 0.081 * b;
                        
                        cbSum += cb;
                        crSum += cr;
                        count++;
                    }
                }
                
                if (count > 0) {
                    cbMeans[gy * grid + gx] = cbSum / count;
                    crMeans[gy * grid + gx] = crSum / count;
                }
            }
        }
        
        double cbVar = calculateVariance(cbMeans);
        double crVar = calculateVariance(crMeans);
        
        // Very high denominator to only catch extreme "pasted face" color mismatches
        return Math.min(1.0, (cbVar + crVar) / 3000.0);
    }

    private static double analyzeNoiseInconsistency(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        
        int grid = 8;
        int cellW = width / grid;
        int cellH = height / grid;
        double[] noiseLevels = new double[grid * grid];
        
        for (int gy = 0; gy < grid; gy++) {
            for (int gx = 0; gx < grid; gx++) {
                double sumLap = 0;
                int count = 0;
                for (int y = gy * cellH + 1; y < (gy + 1) * cellH - 1; y += 2) {
                    for (int x = gx * cellW + 1; x < (gx + 1) * cellW - 1; x += 2) {
                        int p = Color.red(bitmap.getPixel(x, y));
                        int p1 = Color.red(bitmap.getPixel(x-1, y));
                        int p2 = Color.red(bitmap.getPixel(x+1, y));
                        int p3 = Color.red(bitmap.getPixel(x, y-1));
                        int p4 = Color.red(bitmap.getPixel(x, y+1));
                        sumLap += Math.abs(4 * p - p1 - p2 - p3 - p4);
                        count++;
                    }
                }
                if (count > 0) noiseLevels[gy * grid + gx] = sumLap / count;
            }
        }
        
        double variance = calculateVariance(noiseLevels);
        // Low-light indoor shots can have extreme noise variance naturally
        return Math.min(1.0, variance / 4000.0);
    }

    private static double detectPortraitInconsistency(Bitmap bitmap, List<Face> faces) {
        if (faces == null || faces.isEmpty()) {
            // Fallback to center-based check if no faces detected
            return detectPortraitInconsistencyLegacy(bitmap);
        }

        double maxInconsistency = 0;
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        for (Face face : faces) {
            Rect bounds = face.getBoundingBox();
            
            // Define inner (face) and outer (halo/boundary) regions
            int centerX = bounds.centerX();
            int centerY = bounds.centerY();
            int innerRadius = bounds.width() / 3;
            int outerRadiusStart = bounds.width() / 2;
            int outerRadiusEnd = (int)(bounds.width() * 0.8);

            double innerNoise = 0;
            int innerCount = 0;
            double outerNoise = 0;
            int outerCount = 0;

            // Sample pixels around the detected face
            for (int y = Math.max(0, centerY - outerRadiusEnd); y < Math.min(height, centerY + outerRadiusEnd); y += 4) {
                for (int x = Math.max(0, centerX - outerRadiusEnd); x < Math.min(width, centerX + outerRadiusEnd); x += 4) {
                    int p = Color.red(bitmap.getPixel(x, y));
                    int p1 = Color.red(bitmap.getPixel(Math.max(0, x - 1), y));
                    int p2 = Color.red(bitmap.getPixel(Math.min(width - 1, x + 1), y));
                    int noise = Math.abs(2 * p - p1 - p2);

                    double dist = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
                    if (dist < innerRadius) {
                        innerNoise += noise;
                        innerCount++;
                    } else if (dist > outerRadiusStart && dist < outerRadiusEnd) {
                        outerNoise += noise;
                        outerCount++;
                    }
                }
            }

            if (innerCount > 0 && outerCount > 0) {
                double avgInner = innerNoise / innerCount;
                double avgOuter = outerNoise / outerCount;
                // Ratio between face noise and surrounding noise
                double ratio = Math.max(avgInner, avgOuter) / (Math.min(avgInner, avgOuter) + 1.0);
                double score = Math.min(1.0, (ratio - 1.1) / 8.0); // Adjusted for better range
                if (score > maxInconsistency) maxInconsistency = score;
            }
        }

        return maxInconsistency;
    }

    private static double detectPortraitInconsistencyLegacy(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        
        // Deepfakes (especially face swaps) often have a 'halo' of inconsistency 
        // where the face area (center) doesn't match the background/hair boundary.
        
        int centerX = width / 2;
        int centerY = height / 2;
        int radius = width / 4;
        
        double innerNoise = 0;
        int innerCount = 0;
        double outerNoise = 0;
        int outerCount = 0;
        
        for (int y = 2; y < height - 2; y += 4) {
            for (int x = 2; x < width - 2; x += 4) {
                int p = Color.red(bitmap.getPixel(x, y));
                int p1 = Color.red(bitmap.getPixel(x - 1, y));
                int p2 = Color.red(bitmap.getPixel(x + 1, y));
                int noise = Math.abs(2 * p - p1 - p2);
                
                double dist = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
                if (dist < radius) {
                    innerNoise += noise;
                    innerCount++;
                } else if (dist > radius * 1.5) {
                    outerNoise += noise;
                    outerCount++;
                }
            }
        }
        
        if (innerCount == 0 || outerCount == 0) return 0;
        
        double avgInner = innerNoise / innerCount;
        double avgOuter = outerNoise / outerCount;
        
        // Only trigger for extreme ratios typical of face-swaps
        double ratio = Math.max(avgInner, avgOuter) / (Math.min(avgInner, avgOuter) + 1.0);
        return Math.min(1.0, (ratio - 1.0) / 6.0);
    }

    private static double calculateVariance(double[] array) {
        double sum = 0;
        int count = 0;
        for (double v : array) {
            if (v != 0) {
                sum += v;
                count++;
            }
        }
        if (count == 0) return 0;
        double mean = sum / count;
        double sqDiffSum = 0;
        for (double v : array) {
            if (v != 0) {
                sqDiffSum += Math.pow(v - mean, 2);
            }
        }
        return sqDiffSum / count;
    }
}