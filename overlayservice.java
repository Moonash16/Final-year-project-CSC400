package com.lisive.detector;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.DisplayMetrics;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.core.app.NotificationCompat;

import java.nio.ByteBuffer;
import java.util.Locale;

public class overlayservice extends Service {

    private WindowManager windowManager;
    private View overlayView;
    private MediaProjectionManager projectionManager;
    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;
    private deepfakedetector detector;
    private boolean isScanning = false;
    private Handler hideHandler = new Handler(Looper.getMainLooper());
    
    private static final String CHANNEL_ID = "OverlayServiceChannel";

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        createNotification();
        
        if (intent != null && intent.hasExtra("resultCode")) {
            int resultCode = intent.getIntExtra("resultCode", Activity.RESULT_CANCELED);
            Intent data = intent.getParcelableExtra("data");
            
            if (resultCode == Activity.RESULT_OK && data != null) {
                // To avoid "reuse resultData" errors, only create projection if not already active
                if (mediaProjection == null) {
                    projectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
                    mediaProjection = projectionManager.getMediaProjection(resultCode, data);
                    
                    if (mediaProjection != null) {
                        mediaProjection.registerCallback(new MediaProjection.Callback() {
                            @Override
                            public void onStop() {
                                cleanupCapture();
                                mediaProjection = null;
                            }
                        }, new Handler(Looper.getMainLooper()));
                        
                        // Setup the capture session once
                        new Handler(Looper.getMainLooper()).postDelayed(this::setupCapture, 500);
                    }
                }
            }
        }
        return START_STICKY;
    }

    private void setupCapture() {
        if (mediaProjection == null) return;
        
        try {
            DisplayMetrics metrics = getResources().getDisplayMetrics();
            int width = metrics.widthPixels;
            int height = metrics.heightPixels;
            int density = metrics.densityDpi;

            // Keep these alive for the duration of the service to avoid re-initialization errors
            if (imageReader == null) {
                imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2);
            }
            
            if (virtualDisplay == null) {
                virtualDisplay = mediaProjection.createVirtualDisplay("ScreenCapture",
                        width, height, density,
                        DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                        imageReader.getSurface(), null, null);
            }
        } catch (Exception e) {
            updateOverlayUI("Setup Error", "Capture failed: " + e.getMessage(), 5000);
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        detector = new deepfakedetector(this);
        detector.initialize(new deepfakedetector.InitializationCallback() {
            @Override
            public void onSuccess() {}
            @Override
            public void onError(String error) {}
        });

        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        overlayView = LayoutInflater.from(this).inflate(R.layout.overlay_layout, null);

        final WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ?
                        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY :
                        WindowManager.LayoutParams.TYPE_PHONE,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT);

        params.gravity = Gravity.TOP | Gravity.START;
        params.x = 0;
        params.y = 200;

        ImageView bubbleIcon = overlayView.findViewById(R.id.bubble_icon);
        View menuContainer = overlayView.findViewById(R.id.menu_container);
        Button scanBtn = overlayView.findViewById(R.id.btn_scan_now);
        Button closeBtn = overlayView.findViewById(R.id.btn_close_bubble);
        View resultContainer = overlayView.findViewById(R.id.result_container);

        bubbleIcon.setOnTouchListener(new View.OnTouchListener() {
            private int initialX;
            private int initialY;
            private float initialTouchX;
            private float initialTouchY;
            private long lastDownTime;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialX = params.x;
                        initialY = params.y;
                        initialTouchX = event.getRawX();
                        initialTouchY = event.getRawY();
                        lastDownTime = System.currentTimeMillis();
                        return true;
                    case MotionEvent.ACTION_UP:
                        if (System.currentTimeMillis() - lastDownTime < 200) {
                            int visibility = menuContainer.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE;
                            menuContainer.setVisibility(visibility);
                            resultContainer.setVisibility(View.GONE);
                        }
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        params.x = initialX + (int) (event.getRawX() - initialTouchX);
                        params.y = initialY + (int) (event.getRawY() - initialTouchY);
                        windowManager.updateViewLayout(overlayView, params);
                        return true;
                }
                return false;
            }
        });

        scanBtn.setOnClickListener(v -> {
            menuContainer.setVisibility(View.GONE);
            if (mediaProjection != null && virtualDisplay != null) {
                captureAndAnalyze();
            } else if (mediaProjection == null) {
                showToast("Please grant screen permission in app first");
            } else {
                // Try to recover if virtualDisplay was lost
                setupCapture();
                new Handler(Looper.getMainLooper()).postDelayed(this::captureAndAnalyze, 500);
            }
        });

        closeBtn.setOnClickListener(v -> stopSelf());

        windowManager.addView(overlayView, params);
    }

    private void createNotification() {
        createNotificationChannel();
        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("PreMune Bubble Active")
                .setContentText("Tap the bubble to scan screen")
                .setSmallIcon(R.drawable.shield_icon)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build();
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(1, notification, 
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION | 
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(1, notification);
        }
    }

    private void captureAndAnalyze() {
        if (isScanning) return;
        isScanning = true;
        
        updateOverlayUI("Scanning...", "", 0); // No hide timer while scanning
        
        // Wait for the latest frame
        new Handler(Looper.getMainLooper()).postDelayed(this::processScreenFrame, 300);
    }

    private void processScreenFrame() {
        Image image = null;
        try {
            if (imageReader == null) {
                isScanning = false;
                updateOverlayUI("Error", "No capture session", 5000);
                return;
            }

            image = imageReader.acquireLatestImage();
            if (image == null) {
                isScanning = false;
                updateOverlayUI("Failed", "Try scrolling slightly", 5000);
                return;
            }

            Image.Plane[] planes = image.getPlanes();
            ByteBuffer buffer = planes[0].getBuffer();
            int pixelStride = planes[0].getPixelStride();
            int rowStride = planes[0].getRowStride();
            int rowPadding = rowStride - pixelStride * image.getWidth();
            
            Bitmap bitmap = Bitmap.createBitmap(image.getWidth() + rowPadding / pixelStride, 
                    image.getHeight(), Bitmap.Config.ARGB_8888);
            bitmap.copyPixelsFromBuffer(buffer);
            
            int cropWidth = (int) (bitmap.getWidth() * 0.8);
            int cropHeight = (int) (bitmap.getHeight() * 0.6);
            Bitmap cropped = Bitmap.createBitmap(bitmap, 
                    (bitmap.getWidth() - cropWidth) / 2, 
                    (bitmap.getHeight() - cropHeight) / 2, 
                    cropWidth, cropHeight);
            
            detector.detectdeepfake(null, cropped, new deepfakedetector.DetectionCallback() {
                @Override
                public void onResult(boolean isFake, float confidence, forensicanalyzer.ForensicResult result) {
                    showFinalResult(isFake, confidence, result);
                    isScanning = false;
                    bitmap.recycle();
                    cropped.recycle();
                }

                @Override
                public void onProgress(String message) {}

                @Override
                public void onError(String error) {
                    isScanning = false;
                    updateOverlayUI("Error", error, 5000);
                }
            });
            
        } catch (Exception e) {
            isScanning = false;
            updateOverlayUI("Error", e.getMessage(), 5000);
        } finally {
            if (image != null) image.close();
        }
    }

    private void cleanupCapture() {
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
    }

    private void updateOverlayUI(String title, String sub, int hideDelayMillis) {
        new Handler(Looper.getMainLooper()).post(() -> {
            View resultContainer = overlayView.findViewById(R.id.result_container);
            TextView resText = overlayView.findViewById(R.id.overlay_result_text);
            TextView confText = overlayView.findViewById(R.id.overlay_confidence_text);

            resultContainer.setVisibility(View.VISIBLE);
            resText.setText(title);
            resText.setTextColor(0xFFFFFFFF);
            confText.setText(sub);
            
            hideHandler.removeCallbacksAndMessages(null);
            if (hideDelayMillis > 0) {
                hideHandler.postDelayed(() -> resultContainer.setVisibility(View.GONE), hideDelayMillis);
            }
        });
    }

    private void showFinalResult(boolean isFake, float confidence, forensicanalyzer.ForensicResult result) {
        new Handler(Looper.getMainLooper()).post(() -> {
            View resultContainer = overlayView.findViewById(R.id.result_container);
            TextView resText = overlayView.findViewById(R.id.overlay_result_text);
            TextView confText = overlayView.findViewById(R.id.overlay_confidence_text);

            resultContainer.setVisibility(View.VISIBLE);
            if (isFake) {
                if (result.aiArtifactScore > 0.4) {
                    resText.setText("🤖 AI GENERATED");
                } else if (result.portraitInconsistencyScore > 0.4) {
                    resText.setText("⚠️ FACE SWAP");
                } else {
                    resText.setText("⚠️ DEEPFAKE DETECTED");
                }
                resText.setTextColor(0xFFFF4444);
                confText.setText(String.format(Locale.getDefault(), "Deepfake Evidence: %.1f%%", confidence * 100));
            } else {
                resText.setText("✓ LIKELY AUTHENTIC");
                resText.setTextColor(0xFF44FF44);
                confText.setText(String.format(Locale.getDefault(), "Authenticity Score: %.1f%%", (1.0f - confidence) * 100));
            }
            
            hideHandler.removeCallbacksAndMessages(null);
            hideHandler.postDelayed(() -> resultContainer.setVisibility(View.GONE), 8000);
        });
    }

    private void showToast(String message) {
        new Handler(Looper.getMainLooper()).post(() -> 
            Toast.makeText(this, message, Toast.LENGTH_SHORT).show());
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel serviceChannel = new NotificationChannel(
                    CHANNEL_ID,
                    "Manual Scanner Service",
                    NotificationManager.IMPORTANCE_LOW
            );
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) manager.createNotificationChannel(serviceChannel);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        hideHandler.removeCallbacksAndMessages(null);
        if (overlayView != null) windowManager.removeView(overlayView);
        cleanupCapture();
        if (mediaProjection != null) mediaProjection.stop();
        if (detector != null) detector.close();
    }
}