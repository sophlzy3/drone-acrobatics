import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


def extract_waypoints(bag_names, bag_dir, save_dir):
    for bag_name in bag_names:
        filepath = bag_dir + bag_name + "_flightgoggles_agiros_pilot_state.csv"
        # Load CSV
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
                "type": "SingleBall",
                "position": [round(p, 4) for p in pos],
                "rpy": [round(o, 4) for o in ori],
                "radius": 0.001,
                "margin": 0.0,
                "stationary": True
            }

        # Output YAML to console
        print(yaml.dump(waypoints_yaml, sort_keys=False, default_flow_style=None))

        # Optional: save to file
        with open(f"{save_dir}/{bag_name}.yaml", "w") as f:
            yaml.dump(waypoints_yaml, f, sort_keys=False,default_flow_style=None)

bag_names = [
    '2025-04-03-19-54-22',
    '2025-04-03-19-55-56',
    '2025-04-03-19-56-41',
    '2025-04-03-19-57-11',
    '2025-04-03-19-57-44',
    '2025-04-03-19-58-12',
    '2025-04-03-19-58-53',
    '2025-04-03-19-59-26',
    '2025-04-03-20-00-38',
    '2025-04-03-20-01-06',
    '2025-03-07-23-28-38']
bag_dir = "data/unprocessed_train/"
save_dir = "data/baseline_waypoints/"

extract_waypoints(bag_names, bag_dir, save_dir)