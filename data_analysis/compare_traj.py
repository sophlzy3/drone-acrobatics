import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH

def interpolate_trajectory(df, time_col, cols_to_interp, new_time):
    interp_data = {}
    for col in cols_to_interp:
        f = interp1d(df[time_col], df[col], kind='cubic', fill_value="extrapolate")
        interp_data[col] = f(new_time)
    return pd.DataFrame(interp_data, index=new_time)

def compute_similarity(traj1_df, traj2_df, time_col='time'):
    common_start = max(traj1_df[time_col].min(), traj2_df[time_col].min())
    common_end = min(traj1_df[time_col].max(), traj2_df[time_col].max())
    num_samples = 1000
    new_time = np.linspace(common_start, common_end, num_samples)

    cols = [col for col in traj1_df.columns if col != time_col]

    interp1 = interpolate_trajectory(traj1_df, time_col, cols, new_time)
    interp2 = interpolate_trajectory(traj2_df, time_col, cols, new_time)

    diffs = interp1.values - interp2.values
    mse = np.mean(diffs ** 2)
    mae = np.mean(np.abs(diffs))
    max_diff = np.max(np.abs(diffs))

    per_timestep_dist = np.linalg.norm(diffs, axis=1)
    mean_dist = np.mean(per_timestep_dist)

    return {
        'MSE': mse,
        'MAE': mae,
        'Max abs diff': max_diff,
        'Mean Euclidean distance': mean_dist,
        'Per timestep distances': per_timestep_dist,
        'Time vector': new_time
    }

def compare_trajectories(file1, file2, time_col='time'):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    result = compute_similarity(df1, df2, time_col)

    print("=== Trajectory Similarity Metrics ===")
    for k, v in result.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.6f}")

    # Optional: Plot per-timestep distance
    plt.plot(result['Time vector'], result['Per timestep distances'])
    plt.title("Per-timestep Euclidean Distance Between Trajectories")
    plt.xlabel("Time (s)")
    plt.ylabel("Euclidean Distance")
    plt.grid(True)
    plt.show()

    return result

# Example usage
file1 = TRAIN_UNPROCESSED_PATH + '2025-04-03-19-54-22_odom.csv'
file2 = '/Users/sophie/Downloads/Github/aps360-project/data/baseline_unprocessed/'
compare_trajectories(file1, file2)
