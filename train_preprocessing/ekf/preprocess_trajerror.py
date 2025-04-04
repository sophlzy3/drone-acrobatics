import pandas as pd
import os, sys
import numpy as np
from train_preprocessing.ekf.ekf_pred import load_and_process_data

# ============= GENERATE TRAJECTORY ERROR DATA ==============
predictions = load_and_process_data('data/train_state.csv', 'data/train_ratethrust.csv')
predictions.to_csv('data/train_ekfpred.csv', index=False)
