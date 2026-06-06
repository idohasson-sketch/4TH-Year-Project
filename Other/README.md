Trail Cam: Smart Wildlife Detection Camera 🦌
A smart, edge-based AI trail camera for real-time wildlife species identification, operating completely offline.

🚀 The Problem
Traditional wildlife monitoring tools generate millions of images requiring slow manual processing. They fail to provide real-time alerts and often depend on a stable internet connection for cloud-based analysis, which is rarely available in remote field locations.

🎯 Our Solution
This project is an on-board AI monitoring system built on the OpenMV Cam N6 platform. It integrates multiple sensors with a local AI model for an autonomous, real-time solution.

Detect: Motion (PIR) and thermal sensors trigger the camera.

Process: A TinyML model running directly on the device captures and analyzes the image, instantly identifying the animal's species.

Alert: The system sends an immediate alert to a mobile application, complete with the image, species classification, and GPS coordinates.

This entire process happens on the edge, ensuring rapid response times, low power consumption, and reliable operation in isolated environments.

✨ Key Features
Real-Time Species Identification

On-Prem / Edge AI Processing (No internet or cloud dependency)

Instant Mobile Notifications

Low-Power & Optimized for long-term field deployment

Modular design based on the OpenMV platform

🛠️ Technology Stack
Hardware: OpenMV Cam N6, PIR Motion Sensors, Thermal Sensors

Software: TinyML, MicroPython
