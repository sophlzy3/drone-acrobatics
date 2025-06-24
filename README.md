# Drone Acrobatics
Acrobatics Drone is an end-to-end deep learning-based flight controller system for quadrotor drones, designed to execute aggressive and dynamic maneuvers in simulation. This project showcases a robust approach for generating drone control commands based on input states, inspired by state-of-the-art research in UAV control and data-driven learning.

Contributors: [Sophie Liu](https://github.com/sophlzy3), [George Xue](https://github.com/sleepmastergx), [Lucy Matchette](https://github.com/LucyM3), [Ran Qi](https://github.com/fastwalker1118)

## Overview
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

## Data Source
We used the Flightgoggles 3D simulation platform as our simulation environment. The TOGT trajectory planner generated time-optimal acrobatic trajectories that served as the basis for our training and testing data. Each state in our dataset includes information such as position, orientation, velocities, accelerations, jerk, snap, and control inputs. The complete dataset is split into three parts: 55% for training, 25% for validation, and 20% for testing.

## Performance Summary
Our baseline for performance comparison is the Model Predictive Control (MPC) controller integrated within the Flightgoggles simulator. When compared to this baseline, our deep learning model achieved a Mean Squared Error (MSE) of 2.907840, a Root Mean Squared Error (RMSE) of 1.705239, and a Mean Absolute Error (MAE) of 0.905650. The R² Score was -0.221797, indicating underperformance relative to a mean predictor, which we attribute to the aggressive noise injection and dropout techniques used during training. Despite this, the model demonstrated strong robustness to noise, reliably predicting the mean control commands and smoothing out extreme maneuver behaviors. Additionally, it exhibited promising zero-shot generalization to unseen but similar trajectories within the same domain.

## License
The source code for this project is licensed under the MIT License. This allows for free use, modification, and distribution, with conditions outlined in the license text.

## Installation
```bash
# Clone the repository
git clone https://github.com/sophlzy3/acrobatics-drone.git
cd acrobatics-drone

# (Optional) Create a Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```
