A Eye Tracker: Smart Wildlife Detection Camera 🦌
An autonomous, edge-based AI trail camera for real-time wildlife species identification, operating completely offline.

🚀 The Problem
Traditional wildlife monitoring tools generate millions of images requiring slow, manual post-processing. Most existing solutions rely on stable internet connections for cloud-based analysis—a luxury rarely available in remote field locations—resulting in delayed data and high power consumption.

🎯 Our Solution
This project is an on-board AI monitoring system built on the OpenMV Cam N6 platform. It integrates motion/thermal sensing with a localized AI engine for an autonomous, real-time solution.

Detect: PIR motion and thermal sensors trigger the camera instantly upon movement.

Process: A custom-trained TinyML model, running directly on the device, analyzes the image to identify the species.

Alert: The system pushes immediate notifications to a mobile app, including the classification and GPS coordinates.

🧠 Model Training & Performance
To ensure high accuracy in field conditions, the classification model underwent a rigorous training process:

Dataset Size: The model was trained on a robust dataset of 1,000 verified images per species (House Sparrow & Feral Pigeon), sourced from high-quality, research-grade iNaturalist observations.

Validation: Performance metrics were validated using a dedicated test set of 150 images per species, ensuring the model generalizes well to new, unseen frames.

Optimization: The model architecture was optimized specifically for edge deployment using weight quantization and ONNX conversion, ensuring high inference speeds without sacrificing classification reliability.

✨ Key Features
Real-Time Species Identification: Instant classification upon detection.

True Edge AI: Zero dependency on internet or cloud infrastructure; all processing happens on-device.

High Accuracy: Validated performance using verified research-grade datasets.

Power Efficient: Optimized for long-term field deployment using low-power hardware components.

Modular Design: Built on the OpenMV platform for easy hardware integration.

🛠️ Technology Stack
Hardware: OpenMV Cam N6, PIR Motion Sensors, Thermal Sensors.

Software: MicroPython, TinyML (TensorFlow Lite), PyTorch (for training), ONNX (for model porting).



,,,
graph TD
    A[Detection: PIR & Thermal Sensors] --> B[Acquisition: Image Capture]
    B --> C[Analysis: TinyML Model Inference]
    C --> D[Notification: Mobile Alert + Data]

    subgraph "Edge Device (OpenMV Cam N6)"
    B
    C
    end

    style C fill:#f96,stroke:#333,stroke-width:2px
    style A fill:#bbf,stroke:#333
    style D fill:#dfd,stroke:#333
,,,
