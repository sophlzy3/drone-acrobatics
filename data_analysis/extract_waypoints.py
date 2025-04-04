import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


# Load CSV
filepath = 'data/ground_truth/2025-03-07-23-28-38_flightgoggles_agiros_pilot_state.csv'
df = pd.read_csv(filepath)

# Example columns: ['time', 'pos_x', 'pos_y', 'pos_z', ...]
positions = df[['pose.position.x','pose.position.y','pose.position.z']].to_numpy()
quaternions = df[['pose.orientation.x','pose.orientation.y','pose.orientation.z','pose.orientation.w']].to_numpy()

# Compute direction vectors and angle changes
dirs = np.diff(positions, axis=0)
dirs_normed = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
angles = np.arccos(np.clip(np.sum(dirs_normed[1:] * dirs_normed[:-1], axis=1), -1.0, 1.0))

# Find sharpest 4 turns + start
turn_indices = np.where(angles > 0.3)[0]
key_indices = [0] + sorted(turn_indices[np.argsort(-angles[turn_indices])[:4]].tolist())
key_indices = sorted(set(key_indices))  # remove duplicates and sort

# Prepare YAML structure
waypoints_yaml = {"waypoints": {}}
orientations_euler = R.from_quat(quaternions[key_indices]).as_euler('xyz', degrees=True)

for i, idx in enumerate(key_indices):
    name = f"wp{i}"
    pos = positions[idx].tolist()
    ori = orientations_euler[i].tolist()
    waypoints_yaml["waypoints"][name] = {
        "position": [round(p, 4) for p in pos],
        "orientation": [round(o, 4) for o in ori]
    }

# Output YAML to console
print(yaml.dump(waypoints_yaml, sort_keys=False))

# Optional: save to file
with open("data/ground_truth/waypoints.yaml", "w") as f:
    yaml.dump(waypoints_yaml, f, sort_keys=False)
