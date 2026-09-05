# Openvocab Detector

A high-density, real-time object detection Android application featuring a custom 270-class YOLO-World-S model. This app provides a "busy HUD" aesthetic while delivering professional-grade utility through instant video review and permanent HUD burn-in export.

## 🚀 Key Features

*   **270-Class Vocabulary:** Detects a wide array of objects (from "beard" and "glasses" to "cracks" and "wires") using an open-vocabulary YOLO-World model.
*   **Zero-Wait Review:** Detections are journaled in real-time during recording to a sidecar JSON file, allowing for instantaneous playback of recorded video with HUD overlays.
*   **Automatic HUD Burn-in:** While reviewing a capture, the app automatically generates a copy of the video with the bounding boxes permanently "burned" into the pixels, saved directly to the device Gallery.
*   **Eagle Eye Zoom:** Smooth, intuitive pinch-to-zoom gestures for investigating distant targets while maintaining real-time detection.
*   **Temporal Smoothing:** A 166ms persistence window ensures that the HUD feels solid and non-flickering, even when the model's inference rate fluctuates.

## 🛠 Tech Stack

*   **Language:** Kotlin / Jetpack Compose
*   **ML Engine:** TensorFlow Lite (TFLite) with XNNPACK acceleration.
*   **Camera Pipeline:** CameraX (Preview, VideoCapture, ImageAnalysis).
*   **Video Processing:** Media3 (ExoPlayer, Transformer, Effect).
*   **Architecture:** Multi-module Gradle setup (:app, :detection, :media, :overlay).

## 📊 Performance Optimization

*   **Calibrated Int8 Quantization:** The model uses full integer quantization calibrated with a representative dataset to preserve sensitivity for small/distant objects while maintaining CPU speed.
*   **Bulk Memory Reads:** Millions of output values are transferred from the model to the Kotlin layer in single native operations to eliminate JNI bridge overhead.
*   **XNNPACK Acceleration:** Optimized for ARM processors (like the Snapdragon 8 Gen 1 in the S22) to achieve ~80-120ms inference latency.

---

## ❓ FAQ

### Why does this implementation not use the GPU?
While the app supports the TFLite GPU delegate, we discovered during development on the S22 that the YOLO-World architecture triggers an "Adreno Penalty." Because certain custom operations in the model's head aren't natively optimized for mobile GPUs, the delegate falls back to a slow "safe mode," resulting in ~700ms latency. By pivoting to the **XNNPACK co-processor**, we achieved significantly faster real-time performance (~100ms) with the same model accuracy.

### What model does this implementation use, and why?
This project uses a custom-reparameterized **YOLO-World-S (Small)**. We chose YOLO-World because it is an "Open Vocabulary" model, allowing us to bake a massive 270-class dictionary directly into the detection head without needing 270 separate output layers. We opted for the "S" variant to strike the perfect balance between high-density detection (seeing small details) and mobile-friendly file size (~12MB).

### What were some major challenges in developing this project?
1.  **The Concurrency Lock:** Initial builds suffered from native crashes because the Camera analyzer and Video Processor were fighting for the same TFLite interpreter instance. We solved this with a strict atomic synchronization lock.
2.  **Temporal Persistence:** Real-time detections can "flicker" if the brain misses a frame. We implemented a temporal look-back window that keeps boxes "sticky" for 166ms, creating a professional, stable HUD.
3.  **The JNI Traffic Jam:** Moving data for 8,400 locations across 270 classes created a massive bottleneck in Kotlin. We overcame this by implementing bulk primitive array reads, reducing the "Decoding" time from several seconds to under 20ms.
4.  **Dependency Conflicts:** Merging standard TFLite with modern Media3 and older Task libraries caused significant manifest merger failures, which required a complete consolidation of the dependency tree into a unified `org.tensorflow` stack.
