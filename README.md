# Acrobatics Drone
---
Acrobatics Drone is an end-to-end deep learning-based flight controller system for quadrotor drones, designed to execute aggressive and dynamic maneuvers in simulation. This project showcases a robust approach for generating drone control commands based on input states, inspired by state-of-the-art research in UAV control and data-driven learning.

Contributors: [Sophie Liu](mailto:sophiezy.liu@mail.utoronto.ca), [George Xue](mailto:george.xue@mail.utoronto.ca), [Lucy Matchette](mailto:lucy.matchette@mail.utoronto.ca), [Ran Qi](lucy.matchette@mail.utoronto.ca)

# Overview
This project builds a drone controller that predicts the motor commands needed to transition between drone states in high-speed flight scenarios. The primary components include:
- **GRU-based Neural Network Controller**: Uses historical IMU data and trajectory states to predict 6-dimensional control outputs.
- **Data Pipeline**: Utilizes the Flightgoggles simulator and TOGT trajectory planner to generate training and testing data.
- **Noise Robustness**: Incorporates noise injection and sensor occlusion to improve controller reliability under real-world uncertainties.
The system leverages supervised learning rather than reinforcement learning, focusing on trajectory following and noise-robust adaptation.

## Architecture
The architecture consists of:
- **Input**: IMU sensor readings, current and next state data.
- **GRU Layer**: Captures temporal dependencies in IMU data.
- **Multi-Layer Perceptron (MLP)**: Outputs a 6-dimensional vector (4 motor commands, thrust, torque).
- **Loss**: Trained with Mean Squared Error on predicted vs. ground truth control commands.
- **Noise Handling**: Gaussian noise injection and simulated sensor occlusion during training.

# Data Source
- **Simulator**: Flightgoggles 3D simulation platform.
- **Trajectory Planner**: TOGT planner generates time-optimal acrobatic trajectories.
- **Dataset Generation**:
  - States include position, orientation, velocities, accelerations, jerk, snap, and control inputs.
  - Data is split into training (55%), validation (25%), and test (20%) sets.

# Performance Summary
- **Baseline**: MPC within Flightgoggles.
- Compared with the baseline, our model achieved:
  - MSE: 2.907840
  - RMSE: 1.705239
  - MAE: 0.905650
  - R² Score: -0.221797 (indicating underperformance relative to a mean predictor, likely due to aggressive noise injection and dropout).
  - **Robustness**: Despite noisy inputs, the model reliably predicts mean control commands, smoothing out extreme maneuvers and showing zero-shot generalization within the same trajectory domain.

# License
**Source Code Licensing**:
The source code for this project is licensed under the MIT License. This allows for free use, modification, and distribution, with conditions outlined in the license text.

# Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/acrobatics-drone.git
cd acrobatics-drone

# (Optional) Create a Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
