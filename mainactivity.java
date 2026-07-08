package com.lisive.detector;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.android.material.card.MaterialCardView;

import java.util.Locale;

public class mainactivity extends AppCompatActivity {
    
    private Button selectimagebtn, scanbtn, enableoverlaybtn;
    private ImageView previewimage;
    private TextView resulttext, confidencetext;
    private MaterialCardView resultcard;
    private deepfakedetector detector;
    private Uri selectedimageuri;
    
    private final ActivityResultLauncher<String> imagepickerlauncher = 
        registerForActivityResult(new ActivityResultContracts.GetContent(),
            uri -> {
                if (uri != null) {
                    selectedimageuri = uri;
                    previewimage.setImageURI(uri);
                    previewimage.setVisibility(View.VISIBLE);
                    scanbtn.setEnabled(true);
                }
            });
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            setContentView(R.layout.activity_main);
            
            initializeviews();
            initializedetector();
            setupclicklisteners();
            checkpermissions();
        } catch (Exception e) {
            Toast.makeText(this, "Initialization Error: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }
    
    private void initializeviews() {
        selectimagebtn = findViewById(R.id.selectimagebtn);
        scanbtn = findViewById(R.id.scanbtn);
        enableoverlaybtn = findViewById(R.id.enableoverlaybtn);
        previewimage = findViewById(R.id.previewimage);
        resulttext = findViewById(R.id.resulttext);
        confidencetext = findViewById(R.id.confidencetext);
        resultcard = findViewById(R.id.resultcard);
        
        scanbtn.setEnabled(false);
    }
    
    private void initializedetector() {
        detector = new deepfakedetector(this);
        detector.initialize(new deepfakedetector.InitializationCallback() {
            @Override
            public void onSuccess() {
                runOnUiThread(() -> Toast.makeText(mainactivity.this, 
                    "PreMune AI Model Loaded!", Toast.LENGTH_SHORT).show());
            }
            
            @Override
            public void onError(String error) {
                runOnUiThread(() -> Toast.makeText(mainactivity.this, 
                    "Error: " + error, Toast.LENGTH_LONG).show());
            }
        });
    }
    
    private static final int SCREEN_CAPTURE_PERMISSION_REQUEST_CODE = 1001;

    private void setupclicklisteners() {
        selectimagebtn.setOnClickListener(v -> {
            if (checkstoragepermission()) {
                imagepickerlauncher.launch("image/*");
            }
        });
        
        scanbtn.setOnClickListener(v -> {
            if (selectedimageuri != null) {
                performdeepfakedetection();
            }
        });
        
        enableoverlaybtn.setOnClickListener(v -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (!Settings.canDrawOverlays(this)) {
                    Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                    startActivity(intent);
                } else {
                    startMediaProjectionRequest();
                }
            }
        });
    }

    private void startMediaProjectionRequest() {
        MediaProjectionManager manager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        startActivityForResult(manager.createScreenCaptureIntent(), SCREEN_CAPTURE_PERMISSION_REQUEST_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SCREEN_CAPTURE_PERMISSION_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                Intent serviceIntent = new Intent(this, overlayservice.class);
                serviceIntent.putExtra("resultCode", resultCode);
                serviceIntent.putExtra("data", data);
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent);
                } else {
                    startService(serviceIntent);
                }
            } else {
                Toast.makeText(this, "Screen capture permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
    
    private void performdeepfakedetection() {
        scanbtn.setEnabled(false);
        resultcard.setVisibility(android.view.View.GONE);
        
        detector.detectdeepfake(selectedimageuri, new deepfakedetector.DetectionCallback() {
            @Override
            public void onResult(boolean isFake, float confidence, forensicanalyzer.ForensicResult forensicResult) {
                runOnUiThread(() -> {
                    displayresults(isFake, confidence, forensicResult);
                    scanbtn.setEnabled(true);
                });
            }

            @Override
            public void onProgress(String message) {
                runOnUiThread(() -> {
                    // Optionally update a progress dialog or status text
                    Toast.makeText(mainactivity.this, message, Toast.LENGTH_SHORT).show();
                });
            }
            
            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    Toast.makeText(mainactivity.this, "Detection failed: " + error, Toast.LENGTH_LONG).show();
                    scanbtn.setEnabled(true);
                });
            }
        });
    }
    
    private void displayresults(boolean isFake, float confidence, forensicanalyzer.ForensicResult forensicResult) {
        resultcard.setVisibility(android.view.View.VISIBLE);
        
        float displayConfidence;
        if (isFake) {
            resultcard.setCardBackgroundColor(ContextCompat.getColor(this, R.color.fake_light));
            
            if (forensicResult.aiArtifactScore > 0.4) {
                resulttext.setText("🤖 AI GENERATED");
            } else if (forensicResult.portraitInconsistencyScore > 0.4) {
                resulttext.setText("⚠️ FACE SWAP");
            } else {
                resulttext.setText("⚠️ DEEPFAKE DETECTED");
            }

            resulttext.setTextColor(ContextCompat.getColor(this, R.color.fake_dark));
            displayConfidence = confidence;
        } else {
            resultcard.setCardBackgroundColor(ContextCompat.getColor(this, R.color.real_light));
            resulttext.setText("✓ LIKELY AUTHENTIC");
            resulttext.setTextColor(ContextCompat.getColor(this, R.color.real_dark));
            displayConfidence = 1.0f - confidence;
        }
        
        confidencetext.setText(String.format(Locale.getDefault(), "Confidence Score: %.1f%%", displayConfidence * 100));
        showforensicdetails(forensicResult);
    }
    
    private void showforensicdetails(forensicanalyzer.ForensicResult result) {
        String details = String.format(Locale.getDefault(),
            "Deepfake Forensic Report:\n\n" +
            "AI Generative Signature: %.1f%%\n" +
            "Face-Swap Inconsistency: %.1f%%\n" +
            "Metadata/ELA Continuity: %.1f%%\n" +
            "Sensor Noise Match: %.1f%%\n\n" +
            "Verdict: %s",
            result.aiArtifactScore * 100,
            result.portraitInconsistencyScore * 100,
            result.elaScore * 100,
            result.noiseInconsistencyScore * 100,
            result.isManipulated ? "DEEPFAKE DETECTED" : "AUTHENTIC"
        );
        
        new AlertDialog.Builder(this)
            .setTitle("🔬 Specialized Forensic Evidence")
            .setMessage(details)
            .setPositiveButton("OK", null)
            .show();
    }
    
    private void startoverlayservice() {
        Intent intent = new Intent(this, overlayservice.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
        Toast.makeText(this, "Overlay mode activated! Floating bubble will appear on other apps", Toast.LENGTH_LONG).show();
    }
    
    private boolean checkstoragepermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_MEDIA_IMAGES)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, 
                    new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 100);
                return false;
            }
        } else {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, 
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
                return false;
            }
        }
        return true;
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        checkpermissions();
    }

    private void checkpermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            enableoverlaybtn.setVisibility(View.VISIBLE);
            if (!Settings.canDrawOverlays(this)) {
                enableoverlaybtn.setText("🔓 Grant Overlay Permission");
            } else {
                enableoverlaybtn.setText("🚀 Start Floating Bubble");
            }
        }
    }
}