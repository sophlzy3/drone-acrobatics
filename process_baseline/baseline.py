import pandas as pd
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS
from process_data.csv_operations import delete_cols, add_noise_to_csv, random_dropout, combine_train, combine_csv_by_subtopic, combine_subtopic, fill_zeros
from process_data.data_fill import fill_trajectory_fields

# ============= GENERATE TRAINING DATA ALL ==============
combine_subtopic('data/baseline_unprocessed/state', 'data/baseline_state.csv')
combine_subtopic('data/baseline_unprocessed/ratethrust', 'data/baseline_ratethrust.csv')
# combine_csv_by_subtopic('data/train_unprocessed')

cols_to_delete = ['header.seq','header.stamp.secs','header.stamp.nsecs','header.frame_id','t']
delete_cols('data/baseline_state.csv', 'data/baseline_state.csv', cols_to_delete+['motors'])
delete_cols('data/baseline_ratethrust.csv', 'data/baseline_ratethrust.csv', cols_to_delete)

# ============= FILL IN ACCELERATION ==============
fill_trajectory_fields('data/baseline_state.csv', 'data/baseline_state.csv')

# ============= ADD GAUSSIAN NOISE ==============
add_noise_to_csv('data/baseline_state.csv', 'data/baseline_state.csv', noise_std=0.1, fraction=0.1, skip_columns=None)
add_noise_to_csv('data/baseline_ratethrust.csv', 'data/baseline_ratethrust.csv', noise_std=0.1, fraction=0.1, skip_columns=None)

# ============= RANDOM DROPOUTS ==============
random_dropout('data/baseline_state.csv', 'data/baseline_state.csv', dropout_fraction=0.001, skip_columns=None)
random_dropout('data/baseline_ratethrust.csv', 'data/baseline_ratethrust.csv', dropout_fraction=0.01, skip_columns=None)

# ============= GENERATE FINAL TRAINING =========
key_column, in1, in2, out_x, out_y = 'timestamp', 'data/baseline_state.csv', 'data/baseline_ratethrust.csv', 'data/baseline_x.csv', 'data/baseline_y.csv'
combine_train(key_column, in1, in2, out_x, out_y)
delete_cols(out_y, out_y, ['timestamp'])
fill_zeros(out_x)
fill_zeros(out_y)

calculate_error('data/baseline_y.csv', 'data/baseline_x.csv')