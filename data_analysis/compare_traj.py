import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS


# Define the relevant columns to compare
POSE_COLUMNS = [
    "pose.position.x", "pose.position.y", "pose.position.z",
    "pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"
]
VEL_COLUMNS = [
    "velocity.linear.x", "velocity.linear.y", "velocity.linear.z",
    "velocity.angular.x", "velocity.angular.y", "velocity.angular.z"
]
ACC_COLUMNS = [
    "acceleration.linear.x", "acceleration.linear.y", "acceleration.linear.z",
    "acceleration.angular.x", "acceleration.angular.y", "acceleration.angular.z"
]

ALL_COLUMNS = POSE_COLUMNS + VEL_COLUMNS + ACC_COLUMNS

def interpolate(df, time_col, target_time, columns):
    interpolated = {}
    for col in columns:
        f = interp1d(df[time_col], df[col], kind='linear', fill_value='extrapolate')
        interpolated[col] = f(target_time)
    return pd.DataFrame(interpolated, index=target_time)

def compute_trajectory_similarity(file1, file2, time_col='t'):
    # Load CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Clean and filter
    df1 = df1[[time_col] + ALL_COLUMNS].dropna()
    df2 = df2[[time_col] + ALL_COLUMNS].dropna()

    # Determine common time interval
    start_time = max(df1[time_col].min(), df2[time_col].min())
    end_time = min(df1[time_col].max(), df2[time_col].max())
    t_common = np.linspace(start_time, end_time, 1000)

    # Interpolate both datasets to the common time vector
    interp1 = interpolate(df1, time_col, t_common, ALL_COLUMNS)
    interp2 = interpolate(df2, time_col, t_common, ALL_COLUMNS)

    # Compute differences
    diffs = interp1.values - interp2.values
    mse = np.mean(diffs ** 2)
    mae = np.mean(np.abs(diffs))
    max_abs = np.max(np.abs(diffs))
    euc_dist = np.linalg.norm(diffs, axis=1)
    mean_euc_dist = np.mean(euc_dist)

    # # Print results
    # print("=== Trajectory Similarity Metrics ===")
    # print(f"MSE: {mse:.6f}")
    # print(f"MAE: {mae:.6f}")
    # print(f"Max Abs Difference: {max_abs:.6f}")
    # print(f"Mean Euclidean Distance: {mean_euc_dist:.6f}")

    # Optional plot
    # plt.plot(t_common, euc_dist)
    # plt.title("Euclidean Distance Between Trajectories Over Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Distance")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return {
        'MSE': mse,
        'MAE': mae,
        'Max Abs': max_abs,
        'Mean Euclidean Distance': mean_euc_dist,
        'Per Timestep Distances': euc_dist,
        'Time Vector': t_common
    }

# Example usage:
# result = compute_trajectory_similarity('traj_ref.csv', 'traj_test.csv')

# Example usage
file1 = 'data/train_unprocessed/' + BAGS_TRAINING[0] + SUBTOPICS[0] + '.csv'

comparison = {
    'MSE': [],
    'MAE': [],
    'Max Abs': [],
    'Mean Euclidean Distance': [],
    'Per Timestep Distances': [],
    'Time Vector': []
}
for name in BAGS_BASELINE:
    file2 = BASELINE_UNPROCESSED_PATH + name + SUBTOPICS[0] + '.csv'
    output = compute_trajectory_similarity(file1, file2)
    comparison['MSE'].append(output['MSE'])
    comparison['MAE'].append(output['MAE'])
    comparison['Max Abs'].append(output['Max Abs'])
    comparison['Mean Euclidean Distance'].append(output['Mean Euclidean Distance'])
    comparison['Per Timestep Distances'].append(output['Per Timestep Distances'])
    comparison['Time Vector'].append(output['Time Vector'])

# print(f"=== Comparison Results ===")
# print(f"Comparison with {name}:")
# print(f"MSE: {comparison['MSE']:.6f}")
# print(f"MAE: {comparison['MAE']:.6f}")
# print(f"Max Abs Difference: {comparison['Max Abs']:.6f}")
# print(f"Mean Euclidean Distance: {comparison['Mean Euclidean Distance']:.6f}")
# print()

for key in comparison.keys():
    plt.plot(list(range(len(comparison[key]))), comparison[key])
    plt.title(f"{key} Between Trajectories")
    plt.xlabel("Bag Number")
    plt.ylabel(key)
    plt.grid(True)
    plt.tight_layout()
    plt.show()