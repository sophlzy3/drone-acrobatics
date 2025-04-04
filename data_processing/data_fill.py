import pandas as pd
import numpy as np

def fill_trajectory_fields(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    time_col = 'timestamp'
    timestamps = df[time_col].values

    def finite_diff(data, t):
        result = np.zeros_like(data)
        result[1:-1] = (data[2:] - data[:-2]) / (t[2:] - t[:-2])[:, None]
        result[0] = (data[1] - data[0]) / (t[1] - t[0])
        result[-1] = (data[-1] - data[-2]) / (t[-1] - t[-2])
        return result

    def moving_avg(data, window=10):
        return pd.DataFrame(data).rolling(window=window, min_periods=1, center=True).mean().values

    # === Fields ===
    linear_vel_cols = ['velocity.linear.x', 'velocity.linear.y', 'velocity.linear.z']
    angular_vel_cols = ['velocity.angular.x', 'velocity.angular.y', 'velocity.angular.z']

    linear_acc_cols = ['acceleration.linear.x', 'acceleration.linear.y', 'acceleration.linear.z']
    angular_acc_cols = ['acceleration.angular.x', 'acceleration.angular.y', 'acceleration.angular.z']

    acc_bias_cols = ['acc_bias.x', 'acc_bias.y', 'acc_bias.z']
    gyr_bias_cols = ['gyr_bias.x', 'gyr_bias.y', 'gyr_bias.z']

    jerk_cols = ['jerk.x', 'jerk.y', 'jerk.z']
    snap_cols = ['snap.x', 'snap.y', 'snap.z']

    # === Step 1: Acceleration ===
    df[linear_acc_cols] = finite_diff(df[linear_vel_cols].values, timestamps)
    df[angular_acc_cols] = finite_diff(df[angular_vel_cols].values, timestamps)

    # === Step 2: Jerk ===
    jerk_data = finite_diff(df[linear_acc_cols].values, timestamps)
    df[jerk_cols] = jerk_data

    # === Step 3: Snap ===
    snap_data = finite_diff(jerk_data, timestamps)
    df[snap_cols] = snap_data

    # === Step 4: Biases (smoothed from existing signals) ===
    df[acc_bias_cols] = moving_avg(df[linear_acc_cols].values)
    df[gyr_bias_cols] = moving_avg(df[angular_acc_cols].values)

    # === Save ===
    df.to_csv(output_csv, index=False)
    print(f"Filled acceleration, jerk, snap, and bias fields to: {output_csv}")
