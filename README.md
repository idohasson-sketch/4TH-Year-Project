# 🦅 A-EYE TRACKER: Autonomous Edge-AI Wildlife Monitoring Device

An autonomous, ultra-low-power smart trail camera built upon the OpenMV platform. The system performs real-time, on-device TinyML inference to identify wildlife species locally while strictly conserving energy via hardware-decoupled motion triggers and cloud data synchronization.

---

## Table of Contents
* [👥 The Team](#-the-team)
* [📚 Project Description](#-project-description)
* [⚡ Getting Started](#-getting-started)
* [🧱 Prerequisites](#-prerequisites)
* [🏗️ Installing](#️-installing)
* [🧪 Testing](#-testing)
* [🚀 Deployment & Cloud Portals](#-deployment--cloud-portals)
* [⚙️ Built With](#️-built-with)
* [🙏 Acknowledgments](#-acknowledgments)

---

## 👥 The Team

### Team Members
* **Ido Hasson** — ido.hasson@mail.huji.ac.il (The Hebrew University of Jerusalem)
* **Hodaya Hariri** — hodaya.hariri@mail.huji.ac.il (The Hebrew University of Jerusalem)

### Supervisors & Mentors
* **Academic Advisor:** Meir Eizenberg (mauroe@savion.huji.ac.il)
* **Project Mentor:** Daniella Har Shalom

---

## 📚 Project Description

### Overview
Real-time wildlife monitoring in remote habitats is often constrained by severe power limits, harsh environments, and the disruptive nature of manual human observation. Traditional camera traps either record continuous video (draining batteries rapidly) or stream raw footage to the cloud (wasting bandwidth on false positives).

**A-EYE TRACKER** addresses this by shifting computational vision directly to the edge. The system stays in a low-power deep sleep until an external PIR sensor detects movement via hardware interrupts. The OpenMV camera module snaps a frame, executes local INT8 quantized TinyML inference across 5 distinct categories (4 regional target species + "Other"), and asynchronously synchronizes verified observations to the cloud database and Telegram alerts.

### Key Features
* **Decoupled Power Architecture:** Passive Infrared (PIR) sensor wakes the microcontroller via hardware interrupts, maintaining an ultra-low standby power footprint.
* **On-Device TinyML Inference:** Local classification using a quantized INT8 MobileNetV2 architecture running on the OpenMV N6 microcontroller.
* **High-Accuracy Edge Model:** Achieves 87.8% field accuracy across 5 categories, matching the accuracy of large-scale baseline models (YOLOv8) on closed sets.
* **Automated Data Pipelines:** Asynchronous synchronization to Supabase SQL backend and real-time incident alerting via Telegram within <90 seconds.
* **Modular & Generic Architecture:** Centralized pipeline to mine regional data, balance datasets, and re-train models for any target ecosystem.

### Main Components
* **Hardware Unit:** OpenMV Cam N6, Low-Power PIR Motion Sensor, Wide-Angle Camera Lens, Integrated LED Lighting Module, Dedicated Wi-Fi Module, and Li-Po Battery Pack.
* **Backend & Analytics:** Supabase (PostgreSQL), Grafana Monitoring Dashboard, Telegram Bot API.

### Main Technologies
* **Edge Firmware:** MicroPython, TensorFlow Lite Micro principles.
* **ML & Preprocessing:** Python, PyTorch, Torchvision, YOLOv8 (yolov8n.pt), scikit-learn, ONNX.
* **Cloud & Monitoring:** Supabase, PostgreSQL, Grafana, Telegram API.

---

## ⚡ Getting Started

These instructions will guide you through setting up the offline training pipeline, preparing dataset inputs, and deploying firmware to the OpenMV hardware.

---

## 🧱 Prerequisites

Ensure the following tools and packages are installed:

* Python 3.10+
* OpenMV IDE (Latest version for OpenMV N6 support)
* PyTorch & CUDA (Optional, for GPU-accelerated local training)
* Python Dependencies:
  pip install torch torchvision ultralytics scikit-learn onnx numpy pillow requests
* Environment Configuration (.env file for secrets):
  SUPABASE_URL=your_supabase_project_url
  SUPABASE_KEY=your_supabase_anon_key
  TELEGRAM_BOT_TOKEN=your_telegram_bot_token
  TELEGRAM_CHAT_ID=your_telegram_chat_id

---

## 🏗️ Installing

### Step 1: Clone the Repository
  git clone https://github.com/idohasson-sketch/4TH-Year-Project.git
  cd 4TH-Year-Project

### Option A: End-to-End Centralized Pipeline (Automated Runner)
You can execute the entire sequential workflow via a single centralized management script that automatically executes the complete chain from data mining to INT8 compression:
  python scripts/run_full_pipeline.py --species_config configs/species_list.yaml

### Option B: Step-by-Step Manual Execution

#### Step 2: Prepare Dataset & Run Smart Cropping
Acquire high-resolution training images (e.g., from iNaturalist). Run the automated preprocessing pipeline to crop subjects using YOLOv8 and downsample them to 128x128 resolution:
  python scripts/preprocess_pipeline.py --input_dir ./data/raw --output_dir ./data/processed --img_size 128

#### Step 3: Train & Quantize the MobileNetV2 Model
Train the classification model with transfer learning and export the quantized INT8 .tflite model:
  python scripts/train_mobilenet.py --epochs 30 --batch_size 32 --quantize int8 --output ./model/model_int8.tflite

#### Step 4: Flash Firmware to OpenMV N6
1. Open the OpenMV IDE.
2. Connect the OpenMV Cam N6 via USB.
3. Copy model_int8.tflite and labels.txt to the root directory of the OpenMV flash storage.
4. Open src/main.py, configure your local Wi-Fi SSID, Password, and cloud endpoints.
5. Save main.py directly to the OpenMV device.

---

## 🧪 Testing

> **Important Workflow Note:** To evaluate model performance accurately, the full end-to-end pipeline must be executed in order: data mining, automated dataset preprocessing/cropping, multi-phase training iterations, and final INT8 quantization before running the evaluation benchmarks below.

The evaluation matrix tests model robustness across 4 incremental phases and verified field data.

### 1. Offline Model Evaluation & Confusion Matrix
Evaluate the quantized INT8 model against the test dataset:
  python scripts/evaluate_model.py --model ./model/model_int8.tflite --test_data ./data/test_group3
* Output: Generates confusion matrices, Precision/Recall scores, and overall test accuracy.

### 2. End-to-End Edge Pipeline Test
Run a live sanity test on the OpenMV board:
1. Trigger the PIR sensor by introducing movement in front of the lens.
2. Observe terminal output in OpenMV IDE:
   [INFO] PIR Interrupt Detected -> Waking up camera
   [INFO] Frame Captured (128x128)
   [INFO] TinyML Inference: House Sparrow (Confidence: 91.4%)
   [INFO] Wi-Fi Syncing to Supabase... OK
   [INFO] Telegram Alert Dispatched (<90s E2E latency)
   [INFO] Returning to Deep Sleep

---

## 🚀 Deployment & Cloud Portals

* **Field Housing:** Enclose the OpenMV board, PIR sensor, and battery inside a weather-resistant casing.
* **Camera & Sensor Placement:** Mount the PIR sensor directed toward the target area to ensure fast interrupt triggering before frame capture.
* **Live Telemetry & Databases:**
  * 📊 [Grafana Live Observation Dashboard](https://idohasson.grafana.net/d/idnnshv/observation-main-dashboard?orgId=1&from=now-90d&to=now&timezone=browser&var-query0=&var-observation_id=$__all&dtab=General-Info) — Real-time telemetry, bird detection trends, and system status logs.
  * 🗄️ [Supabase Cloud Project Portal](https://supabase.com/dashboard/project/pxkevqlcaiazhgqrxbsp/settings/general) — Centralized SQL database management and media storage.

---

## ⚙️ Built With

* [OpenMV](https://openmv.io/) — Edge machine vision hardware & MicroPython engine.
* [PyTorch](https://pytorch.org/) — Deep learning framework used for transfer learning.
* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection used for bounding-box Smart Cropping.
* [Supabase](https://supabase.com/) — Cloud backend and PostgreSQL storage.
* [Grafana](https://grafana.com/) — Real-time telemetry, dashboards, and alerting.
* [iNaturalist](https://www.inaturalist.org/) — Verified research-grade wildlife dataset sourcing.

---

## 🙏 Acknowledgments

* Faculty of Computer Science & Engineering, The Hebrew University of Jerusalem.
* Meir Eizenberg & Daniella Har Shalom for academic mentorship and project guidance.
* The iNaturalist Community for providing expert-verified biodiversity datasets.
* Open-source contributors of TensorFlow Lite Micro, OpenMV, and YOLOv8.
