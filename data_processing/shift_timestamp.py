import pandas as pd

def normalize_timestamp(csv_path, output_path=None, time_col='time'):
    df = pd.read_csv(csv_path)

    # Subtract the first timestamp
    df[time_col] = df[time_col] - df[time_col].iloc[0]

    # Save it back
    if output_path is None:
        output_path = csv_path  # Overwrite original
    df.to_csv(output_path, index=False)
    print(f"Timestamps normalized. Saved to: {output_path}")
