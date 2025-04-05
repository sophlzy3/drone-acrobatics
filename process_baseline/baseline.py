import pandas as pd
import os, sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS
from process_data.csv_operations import delete_cols, add_noise_to_csv, random_dropout, combine_train, combine_csv_by_subtopic, combine_subtopic, fill_zeros
from process_data.data_fill import fill_trajectory_fields

# # ============= GENERATE TRAINING DATA ALL ==============

# ideal_path = 'data/train_unprocessed/' + BAGS_TRAINING[0] + SUBTOPICS[0] + '.csv' 

# for baseline in BAGS_BASELINE:
#     baseline_path = f'data/baseline_unprocessed/ratethrust/{baseline}{SUBTOPICS[1]}.csv'
#     fill_zeros(baseline_path)

#     # check if row number is the same 

#     # if not, interpolate 

#     # calculate MSE, rmse, mae, r2 between baseline and ground truth

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

ideal_path = 'data/train_unprocessed/ratethrust/' + BAGS_TRAINING[0] + SUBTOPICS[1] + '.csv'
ideal_df = pd.read_csv(ideal_path)

mse_list = []
rmse_list = []
mae_list = []
r2_list = []
for baseline in BAGS_BASELINE:
    baseline_path = f'data/baseline_unprocessed/ratethrust/{baseline}{SUBTOPICS[1]}.csv'
    fill_zeros(baseline_path)
    
    baseline_df = pd.read_csv(baseline_path)
    
    # Check if row number is the same
    if len(baseline_df) != len(ideal_df):
        # If not the same length, interpolate baseline to match ideal's index
        time_col = 'timestamp'  # Replace with your actual time/index column name
        
        # FIXED: Use proper interpolation approach instead of reindex with 'linear'
        # First, set the index properly
        baseline_df = baseline_df.set_index(time_col)
        
        # Create a new index based on the ideal data's time column
        new_index = pd.Index(ideal_df[time_col])
        
        # Reindex using valid methods (pad/ffill) 
        baseline_reindexed = baseline_df.reindex(
            new_index, 
            method='pad'  # Use 'pad' (forward fill) instead of 'linear'
        )
        
        # Now apply interpolation separately (this supports 'linear')
        baseline_reindexed = baseline_reindexed.interpolate(method='linear')
        
        # Reset the index to get back the timestamp column
        baseline_reindexed = baseline_reindexed.reset_index()
        
        # For any remaining NaN values after interpolation
        baseline_reindexed.fillna(method='ffill', inplace=True)
        baseline_reindexed.fillna(method='bfill', inplace=True)
        
        baseline_values = baseline_reindexed.iloc[:, 1].values  # Adjust column index as needed
    else:
        baseline_values = baseline_df.iloc[:, 1].values  # Adjust column index as needed
    
    ideal_values = ideal_df.iloc[:, 1].values  # Adjust column index as needed
    
    # Calculate error metrics
    mse = mean_squared_error(ideal_values, baseline_values)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(ideal_values, baseline_values)
    r2 = r2_score(ideal_values, baseline_values)
    
    # Print results
    # print(f"Baseline: {baseline}")
    # print(f"MSE: {mse:.4f}")
    # print(f"RMSE: {rmse:.4f}")
    # print(f"MAE: {mae:.4f}")
    # print(f"R²: {r2:.4f}")
    # print("-" * 40)

    mse_list.append(mse)
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)

print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}") 
print(f"RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}") 
print(f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}") 
print(f"R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}") 