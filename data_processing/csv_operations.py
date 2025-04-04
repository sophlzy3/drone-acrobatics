import pandas as pd
import glob
import numpy as np

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS


def combine_csvs(input_folder, output_file):
    # Match all CSV files in the folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    print(f"Found {len(csv_files)} CSV files.")
    
    combined_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Save to output
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to: {output_file}")

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
