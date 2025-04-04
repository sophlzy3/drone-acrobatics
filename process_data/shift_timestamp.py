import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS

def normalize_timestamp(csv_path, output_path=None, time_col='timestamp'):
    df = pd.read_csv(csv_path)

    # Subtract the first timestamp
    df[time_col] = df[time_col] - df[time_col].iloc[0]

    # Save it back
    if output_path is None:
        output_path = csv_path  # Overwrite original
    df.to_csv(output_path, index=False)
    print(f"Timestamps normalized. Saved to: {output_path}")

for bag_name in BAGS_TRAINING:
     for subtopic in SUBTOPICS:         
          csv_path = os.path.join('data/train_unprocessed/', f"{bag_name}{subtopic}.csv")
          normalize_timestamp(csv_path)

for bag_name in BAGS_BASELINE:
     for subtopic in SUBTOPICS:         
          csv_path = os.path.join(BASELINE_UNPROCESSED_PATH, f"{bag_name}{subtopic}.csv")
          normalize_timestamp(csv_path)
