import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt
import os, sys

# Assuming other imports and path settings from your original code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS
from process_data.csv_operations import fill_zeros
from training.preprocess_readings import delete_cols

ideal_path = 'data/train_unprocessed/ratethrust/' + BAGS_TRAINING[1] + SUBTOPICS[1] + '.csv'
ideal_df = pd.read_csv(ideal_path)

mse_list = []
rmse_list = []
mae_list = []
r2_list = []
l = [BAGS_BASELINE[1], BAGS_BASELINE[4], BAGS_BASELINE[7], BAGS_BASELINE[8]]

for baseline in [BAGS_BASELINE[8]]:
    baseline_path = f'data/baseline_unprocessed/ratethrust/{baseline}{SUBTOPICS[1]}.csv'
    cols_to_delete = ['header.seq','header.stamp.secs','header.stamp.nsecs','header.frame_id','t']
    fill_zeros(baseline_path)
    
    baseline_df = pd.read_csv(baseline_path)

    # Ensure both dataframes have the same length for row-by-row comparison
    min_length = min(len(ideal_df), len(baseline_df))
    ideal_truncated = ideal_df.iloc[:min_length]
    baseline_truncated = baseline_df.iloc[:min_length]
    for col in cols_to_delete:
        baseline_truncated[col] = 0
        ideal_truncated[col] = 0

    window_size = 100  # Window size of 100 timesteps
    print('=====================================================================')

    for i in range(round(len(ideal_truncated) / window_size)):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size

        window_ideal = ideal_truncated.iloc[start_idx:end_idx]
        window_baseline = baseline_truncated.iloc[start_idx:end_idx]    
        
        # Calculate metrics for this window
        mse_list.append(mean_squared_error(window_ideal, window_baseline))
        rmse_list.append(math.sqrt(mse_list[-1]))
        mae_list.append(mean_absolute_error(window_ideal, window_baseline))
        
        # Handle potential division by zero or constant values in R²
        try:
            r2_val = r2_score(window_ideal, window_baseline)
            r2_list.append(r2_val)
        except:
            r2_list.append(0)  # Use 0 as fallback for invalid R² calculations

    x = list(range(round(len(ideal_truncated) / window_size)))
    
    print('='*10)
    print(f'x: {x}')
    print(f'mse_list: {len(mse_list)}')
    print(f'rmse_list: {len(rmse_list)}')
    print(f'mae_list: {len(mae_list)}')
    print(f'r2_list: {len(r2_list)}')

print(f'mse_list: {mse_list}')
print(f'rmse_list: {rmse_list}')
print(f'mae_list: {mae_list}')
print(f'r2_list: {r2_list}')

# Print final results with standard deviation
print('\navg values')
print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}") 
print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}") 
print(f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}") 
print(f"R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

plt.figure(figsize=(12, 8))
plt.plot(x, mse_list, label='MSE')
plt.plot(x, rmse_list, label='RMSE') 
plt.plot(x, mae_list, label='MAE')
plt.plot(x, r2_list, label='R²')
plt.xlabel('Timestep Index (start of window)')
plt.ylabel('Metric Value')
plt.title(f'Error Metrics Over Time - Baseline {baseline}')
plt.legend()
plt.grid(True)
# plt.savefig(f'plots/metrics_over_time_{baseline}.png')
plt.show()
plt.close()