# PreMune Secure

**Developing a Lightweight Deepfake Detection Framework for Social Media in Eswatini**

Final Year Project — BSc Information Technology, Department of Computer Science, University of Eswatini (Kwaluseni Campus)

**Authors:** Precious T. Ndwandwe (202200294) & Munashe Matsanura (202203710)
**Supervisor:** Mr. E. L. Dube

---

## Overview
PreMune Secure is a lightweight, fully offline Android application designed to detect manipulated media (deepfakes) directly within the social media platforms where such content is most commonly encountered — Facebook, TikTok, and WhatsApp. It was built to address the lack of accessible, low-resource, on-device deepfake detection tools for regions like Eswatini, where computationally intensive detection solutions are impractical.

The system combines a custom-trained, quantized deepfake detection model with a multi-metric forensic analysis suite (Error Level Analysis, noise profiling, and structured pattern detection), fused through weighted averaging for stronger detection accuracy.

## Key Results
- **74.3%** validation accuracy (custom fine-tuned MobileNetV2 classifier) vs. **46.0%** for a generic pre-trained MobileNetV2 baseline — a 28.3 percentage point improvement
- **89.2%** model size reduction via INT8 post-training quantization — from 27.3 MB down to **2.96 MB**
- Trained on a balanced dataset of **6,000** real and synthetic face images
- Runs on **Android 5.0+**, via a persistent floating overlay service compatible with Facebook, TikTok, and WhatsApp

## Repository Structure

This repository contains the full working project — Android app source, model training scripts, trained models, and the final APK build — as a flat working directory rather than a separated package structure:

- **Android app source** — `mainactivity.java`, `overlayservice.java`, `deepfakedetector.java`, `forensicanalyzer.java`, `ForensicScoringTest.java`, and associated XML layouts (`activity_main.xml`, `AndroidManifest.xml`, `strings.xml`, `themes.xml`, etc.)
- **Model training scripts (Python)** — `data_preprocessing.py`, `model_training.py`, `model_export.py`, `train.py`, `retrain_model.py`, `evaluate.py`, `convert_to_tflite.py`, `utils.py`
- **Trained models** — `lisive_deepfake_model.h5` (Keras/HDF5), plus exported `.tflite` versions (float and INT8-quantized)
- **Final build** — `preMune Secure.apk`
- **requirements.txt** — Python dependencies for the training pipeline

## Methodology Summary

1. Reviewed existing deepfake detection techniques and their limitations for low-resource deployment
2. Designed a lightweight detection framework architecture based on MobileNetV2
3. Constructed a balanced deepfake dataset and trained a custom binary classifier
4. Benchmarked the custom model against a generic MobileNet baseline on identical data
5. Applied INT8 post-training quantization to minimize computational footprint
6. Integrated the model with a forensic-neural fusion engine and an in-app floating overlay for real-time, in-context scanning

## Running the Project

**Train / retrain the model:**
```bash
pip install -r requirements.txt
python data_preprocessing.py
python model_training.py
python model_export.py
```

**Android app:**
Open the project in Android Studio and build/run `mainactivity.java` as the entry point, or install the pre-built `preMune Secure.apk` directly on an Android 5.0+ device.

## Acknowledgements

We thank our supervisor, Mr. E. L. Dube, for his guidance throughout this project, the Department of Computer Science at the University of Eswatini, and the open-source communities behind TensorFlow, Android, FaceForensics++, and CelebA, whose publicly available resources supported this work.
