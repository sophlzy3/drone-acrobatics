import pandas as pd
import glob
import numpy as np

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS


def combine_subtopic(input_folder, output_file):
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    print(f"Found {len(csv_files)} CSV files.")
    
    combined_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Save to output
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to: {output_file}")

import os
import glob
import pandas as pd
import re
from collections import defaultdict

def combine_csv_by_subtopic(input_folder):
    # Get paths to the subfolders
    ratethrust_folder = os.path.join(input_folder, 'ratethrust')
    state_folder = os.path.join(input_folder, 'state')
    print(f"Ratethrust folder: {ratethrust_folder}")
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Dictionary to store files by subtopic
    ratethrust_files_by_subtopic = defaultdict(list)
    state_files_by_subtopic = defaultdict(list)
    
    # Collect ratethrust files by subtopic
    for csv_file in glob.glob(os.path.join(ratethrust_folder, '*.csv')):
        filename = os.path.basename(csv_file)
        # Extract BAG_NAME and SUBTOPIC
        bag_name = filename[:19]
        match = re.search(r'_SUBTOPIC\[(\d+)\]', filename)
        if match and match.group(1) == '1':
            subtopic = match.group(0)
            ratethrust_files_by_subtopic[subtopic].append((bag_name, csv_file))
    
    # Collect state files by subtopic
    for csv_file in glob.glob(os.path.join(state_folder, '*.csv')):
        filename = os.path.basename(csv_file)
        # Extract BAG_NAME and SUBTOPIC
        bag_name = filename[:19]
        match = re.search(r'_SUBTOPIC\[(\d+)\]', filename)
        if match and match.group(1) == '0':
            subtopic = match.group(0).replace('0', '1')  # Match with corresponding ratethrust subtopic
            state_files_by_subtopic[subtopic].append((bag_name, csv_file))
    
    # Process and combine files for each subtopic
    combined_state_dfs = []
    combined_ratethrust_dfs = []
    
    for subtopic in ratethrust_files_by_subtopic.keys():
        for bag_name, ratethrust_file in ratethrust_files_by_subtopic[subtopic]:
            # Find corresponding state file
            state_file = None
            for state_bag_name, state_path in state_files_by_subtopic[subtopic.replace('1', '0')]:
                if state_bag_name == bag_name:
                    state_file = state_path
                    break
            
            if state_file:
                # Read both files
                ratethrust_df = pd.read_csv(ratethrust_file)
                state_df = pd.read_csv(state_file)
                
                # Ensure they have the same number of rows
                min_rows = min(len(ratethrust_df), len(state_df))
                ratethrust_df = ratethrust_df.iloc[:min_rows]
                state_df = state_df.iloc[:min_rows]
                
                # Add to combined dataframes
                combined_ratethrust_dfs.append(ratethrust_df)
                combined_state_dfs.append(state_df)
                
                print(f"Processed {bag_name} with subtopic {subtopic}")
            else:
                print(f"Warning: No matching state file found for {bag_name} with subtopic {subtopic}")
    
    # Combine all dataframes
    if combined_ratethrust_dfs and combined_state_dfs:
        final_ratethrust_df = pd.concat(combined_ratethrust_dfs, ignore_index=True)
        final_state_df = pd.concat(combined_state_dfs, ignore_index=True)
        
        # Save to output files
        final_ratethrust_df.to_csv('data/train_ratethrust.csv', index=False)
        final_state_df.to_csv('data/train_state.csv', index=False)
        
        print(f"Combined CSV files saved to: data/train_ratethrust.csv and data/train_state.csv")
        print(f"Total rows in train_ratethrust.csv: {len(final_ratethrust_df)}")
        print(f"Total rows in train_state.csv: {len(final_state_df)}")
    else:
        print("No matching files were found to combine.")

def delete_cols(input_csv, output_csv, columns_to_delete):
    df = pd.read_csv(input_csv)

    # Drop the specified columns
    df = df.drop(columns=columns_to_delete, errors='ignore')

    # Save to new CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved cleaned CSV to: {output_csv}")

def add_noise_to_csv(input_csv, output_csv, noise_std=0.1, fraction=0.1, skip_columns=None):
    df = pd.read_csv(input_csv)

    # Drop any non-numeric columns from noise addition
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric_df = df.drop(columns=numeric_df.columns)

    if skip_columns:
        numeric_df = numeric_df.drop(columns=skip_columns, errors='ignore')

    # Flatten to work with individual cells
    flat_values = numeric_df.values
    total_elements = flat_values.size
    num_noisy = int(fraction * total_elements)

    # Select random indices to add noise
    indices = np.unravel_index(
        np.random.choice(total_elements, num_noisy, replace=False),
        flat_values.shape
    )

    # Add Gaussian noise
    noise = np.random.normal(loc=0.0, scale=noise_std, size=num_noisy)
    flat_values[indices] += noise

    # Combine with non-numeric columns (if any)
    noisy_df = pd.DataFrame(flat_values, columns=numeric_df.columns)
    final_df = pd.concat([non_numeric_df.reset_index(drop=True), noisy_df], axis=1)

    # Reorder to match original column order
    final_df = final_df[df.columns]

    # Save result
    final_df.to_csv(output_csv, index=False)
    print(f"Noisy CSV saved to: {output_csv}")

def random_dropout(input_csv, output_csv, dropout_fraction=0.01, skip_columns=None):
    df = pd.read_csv(input_csv)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    non_numeric_df = df.drop(columns=numeric_df.columns)

    if skip_columns:
        numeric_df = numeric_df.drop(columns=skip_columns, errors='ignore')

    # Convert to numpy for manipulation
    data = numeric_df.values
    total_elements = data.size
    num_drop = int(total_elements * dropout_fraction)

    # Randomly select indices to zero out
    drop_indices = np.unravel_index(
        np.random.choice(total_elements, num_drop, replace=False),
        data.shape
    )
    data[drop_indices] = 0.0

    # Reconstruct DataFrame
    dropped_df = pd.DataFrame(data, columns=numeric_df.columns)
    final_df = pd.concat([non_numeric_df.reset_index(drop=True), dropped_df], axis=1)

    # Reorder columns to match original
    final_df = final_df[df.columns]

    # Save to new CSV
    final_df.to_csv(output_csv, index=False)
    print(f"Random dropout applied and saved to: {output_csv}")

def combine_train(key_column, in1, in2, out_x, out_y):
    """
    Combine two CSV files row by row based on a key column.
    If key_column values differ, keep the value from in1.
    For out_y, use values from the previous row for specific columns.
    Delete the last row of df1.
    
    Parameters:
    -----------
    in1 : str
        Path to the first input CSV file
    in2 : str
        Path to the second input CSV file
    key_column : str
        Name of the column to use as the key for merging
    out_x : str
        Path to save the output CSV file for features
    out_y : str
        Path to save the output CSV file for labels
    """
    # Read the CSV files
    df1 = pd.read_csv(in1)
    df2 = pd.read_csv(in2)
    
    # Ensure both dataframes have the key column
    if key_column not in df1.columns:
        raise ValueError(f"Key column '{key_column}' not found in {in1}")
    if key_column not in df2.columns:
        raise ValueError(f"Key column '{key_column}' not found in {in2}")
    
    # Delete the last row of df1
    df1 = df1.iloc[:-1]
    
    # Create a new dataframe for the output
    result_df = df1.copy()
    
    # Get columns from df2 that are not in df1 (excluding the key column)
    df2_cols = [col for col in df2.columns if col != key_column]
    
    # Add these columns to the result dataframe
    for col in df2_cols:
        result_df[col] = None
    
    # Create a dictionary for faster lookup of df2 values
    df2_dict = {row[key_column]: row for _, row in df2.iterrows()}
    
    # Iterate through each row in df1
    for idx, row in result_df.iterrows():
        key_value = row[key_column]
        
        # Check if key exists in df2_dict
        if key_value in df2_dict:
            # If a match is found, copy values from df2
            for col in df2_cols:
                result_df.at[idx, col] = df2_dict[key_value][col]
    
    # Save the result to out_x
    result_df.to_csv(out_x, index=False)
    print(f"Combined data saved to {out_x}")
    
    # For out_y, create a new dataframe with values from the previous row
    # for specific columns
    out_y_df = df2.copy()
    
    # Columns to use from the previous row
    prev_row_cols = ['angular_rates.x', 'angular_rates.y', 'angular_rates.z', 
                     'thrust.x', 'thrust.y', 'thrust.z']
    
    # Check if all required columns exist
    for col in prev_row_cols:
        if col not in out_y_df.columns:
            print(f"Warning: Column '{col}' not found in {in2}")
    
    # Shift values for the specified columns
    for col in prev_row_cols:
        if col in out_y_df.columns:
            out_y_df[col] = out_y_df[col].shift(1)
    
    # Delete the first row (which now has NaN values)
    out_y_df = out_y_df.iloc[1:]
    
    # Save to out_y
    out_y_df.to_csv(out_y, index=False)
    print(f"Labels with previous row values saved to {out_y}")

def fill_zeros(input_csv):
    """
    Load a CSV file, replace NaN values with 0, and save back to the same file.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values in {input_csv}")
    
    # Replace NaN values with 0
    df = df.fillna(0)
    
    # Save back to the same file
    df.to_csv(input_csv, index=False)
    print(f"NaN values replaced with zeros and saved to {input_csv}")