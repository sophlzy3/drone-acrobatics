import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.vars import BAGS_BASELINE, BAGS_TRAINING, PROJECT_PATH, TRAIN_UNPROCESSED_PATH, BASELINE_UNPROCESSED_PATH, SUBTOPICS


def load_trajectory(filename):
    """
    Load trajectory data from a CSV file.
    Expected columns: time, x, y, z
    """
    df = pd.read_csv(filename)
    # Ensure the columns exist
    required_cols = ['timestamp', 'pose.position.x','pose.position.y','pose.position.z']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df

def calculate_dtw_distance(traj1, traj2):
    """
    Calculate the DTW distance between two trajectories.
    
    Args:
        traj1, traj2: DataFrames with columns x, y, z
        
    Returns:
        distance: A single number representing the difference between trajectories
        path: The optimal path found by DTW
    """
    # Extract the 3D coordinates
    series1 = traj1[['pose.position.x','pose.position.y','pose.position.z']].values
    series2 = traj2[['pose.position.x','pose.position.y','pose.position.z']].values
    
    # Calculate DTW distance
    distance, path = fastdtw(series1, series2, dist=euclidean)
    
    return distance, path

def normalize_distance(distance, traj1, traj2):
    """
    Normalize the DTW distance to account for trajectory lengths and scale
    
    Args:
        distance: The raw DTW distance
        traj1, traj2: The trajectory DataFrames
        
    Returns:
        normalized_distance: A normalized measure between 0 and 1
    """
    # Get lengths of trajectories
    len1 = len(traj1)
    len2 = len(traj2)
    
    # Calculate spatial extents
    extent1 = np.max(traj1[['pose.position.x','pose.position.y','pose.position.z']].max() - traj1[['pose.position.x','pose.position.y','pose.position.z']].min())
    extent2 = np.max(traj2[['pose.position.x','pose.position.y','pose.position.z']].max() - traj2[['pose.position.x','pose.position.y','pose.position.z']].min())
    
    # Average spatial extent
    avg_extent = (extent1 + extent2) / 2
    
    # Average length
    avg_length = (len1 + len2) / 2
    
    # Normalize by average length and spatial extent
    normalized_distance = distance / (avg_length * avg_extent)
    
    return normalized_distance

def visualize_trajectories(traj1, traj2, path=None, save_path=None):
    """
    Visualize two trajectories in 3D space
    
    Args:
        traj1, traj2: DataFrames with columns x, y, z
        path: Optional DTW path for visualization
        save_path: Optional file path to save the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
#     'pose.position.x','pose.position.y','pose.position.z'
    ax.plot(traj1['pose.position.x'], traj1['pose.position.y'], traj1['pose.position.z'], 'b-', label='Trajectory 1')
    ax.plot(traj2['pose.position.x'], traj2['pose.position.y'], traj2['pose.position.z'], 'r-', label='Trajectory 2')
    
    # If DTW path is provided, visualize the mapping
    if path is not None and len(path) <= 500:  # Only draw if not too dense
        for i, j in path:
            ax.plot([traj1.iloc[i]['pose.position.x'], traj2.iloc[j]['pose.position.x']],
                    [traj1.iloc[i]['pose.position.y'], traj2.iloc[j]['pose.position.y']],
                    [traj1.iloc[i]['pose.position.z'], traj2.iloc[j]['pose.position.z']], 'g-', alpha=0.1)
    
    ax.set_xlabel('pose.position.x')
    ax.set_ylabel('pose.position.y')
    ax.set_zlabel('pose.position.z')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def compare_trajectories(file1, file2, visualize=True):
    """
    Main function to compare two drone trajectories and return a similarity score.
    
    Args:
        file1, file2: Paths to CSV files containing trajectory data
        visualize: Whether to create a visualization
        
    Returns:
        A tuple containing (raw_distance, normalized_distance)
    """
    # Load trajectories
    traj1 = load_trajectory(file1)
    traj2 = load_trajectory(file2)
    
    # Calculate DTW distance
    distance, path = calculate_dtw_distance(traj1, traj2)
    
    # Normalize the distance
    norm_distance = normalize_distance(distance, traj1, traj2)
    
    # Print results
    print(f"Raw DTW distance: {distance:.2f}")
    print(f"Normalized distance: {norm_distance:.4f}")
    
    # Visualize if requested
    if visualize:
        visualize_trajectories(traj1, traj2, path)
    
    return distance, norm_distance



file1 = 'data/train_unprocessed/' + BAGS_TRAINING[0] + SUBTOPICS[0] + '.csv'

comparison = {
    'norm': [],
    'raw': []
}
for name in [BAGS_BASELINE[1], BAGS_BASELINE[4], BAGS_BASELINE[7], BAGS_BASELINE[8]]:
    file2 = BASELINE_UNPROCESSED_PATH + name + SUBTOPICS[0] + '.csv'
    print("Comparing to: ", file2)
    raw_dist, norm_dist = compare_trajectories(file1, file2, visualize=False)
    comparison['norm'].append(norm_dist)
    comparison['raw'].append(raw_dist)

for key in comparison.keys():
    plt.plot(list(range(len(comparison[key]))), comparison[key])
    plt.title(f"{key} Between Trajectories")
    plt.xlabel("Bag Number")
    plt.ylabel(key)
    plt.grid(True)
    plt.tight_layout()
    plt.show()