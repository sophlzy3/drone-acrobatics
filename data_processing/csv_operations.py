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

# Example usage
if __name__ == "__main__":
    input_folder = "path_to_your_input_folder"  # Replace with your actual input folder path
    combine_csv_by_subtopic(input_folder)

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

def combine_train(file1, file2, key_column, output_path):
    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Merge the dataframes on the key column
    merged_df = pd.merge(df1, df2, on=key_column, how='inner')
    merged_df.to_csv(output_path, index=False)
    print("Files merged successfully!")
