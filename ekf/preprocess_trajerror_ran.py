import pandas as pd
import os, sys
import numpy as np
from train_preprocessing.ran.ekf_ran import ExtendedKalmanFilter, ekf_step

# Load CSV data (adjust the file path if needed)
data = pd.read_csv('training_circle.csv')

state_dim = 12  # 12-dimensional state
meas_dim = 3    # measurement is 3-dimensional (position)
ekf = ExtendedKalmanFilter(state_dim, meas_dim)

prev_timestamp = None
estimates = []

for index, row in data.iterrows():
     timestamp = row['timestamp']
     
     # Compute dt (time step); assume timestamps in seconds.
     if prev_timestamp is None:
          dt = 0.01
     else:
          dt = timestamp - prev_timestamp
          if dt <= 0:
               dt = 0.01
     prev_timestamp = timestamp
     
     # Simulate IMU measurements (in a real system, these come from sensors)
     a_sim = np.random.normal(0, 0.1, (3, 1))       # simulated acceleration (m/s^2)
     gyro_sim = np.random.normal(0, 0.01, (3, 1))     # simulated angular rates (rad/s)
     imu_measurement = np.vstack((a_sim, gyro_sim))   # 6x1 control input
     
     # Process a single row using our ekf_step function.
     current_state = ekf_step(ekf, row, dt, imu_measurement)
     
     # Save the current state (flattened for logging or further analysis)
     estimates.append(current_state.flatten())

estimates_array = np.array(estimates)
print("Final state estimate:")
print(ekf.x)
print("Estimation history shape:", estimates_array.shape)

# Optionally, save the estimates for further analysis.
estimates_df = pd.DataFrame(estimates_array, columns=[f"state_{i}" for i in range(state_dim)])
estimates_df.to_csv("ekf_drone_estimates.csv", index=False)