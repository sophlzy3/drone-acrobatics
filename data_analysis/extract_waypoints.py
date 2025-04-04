import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


from sklearn.cluster import KMeans

def extract_waypoints(bag_names, bag_dir, save_dir):
    for bag_name in bag_names:
        filepath = f"{bag_dir}/{bag_name}_flightgoggles_agiros_pilot_state.csv"
        df = pd.read_csv(filepath)

        positions = df[['pose.position.x','pose.position.y','pose.position.z']].to_numpy()
        quaternions = df[['pose.orientation.x','pose.orientation.y','pose.orientation.z','pose.orientation.w']].to_numpy()

        # Cluster trajectory into 4 segments (you expect 4 corners)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(positions)
        centroids = kmeans.cluster_centers_

        # Convert orientations at cluster centers (by finding nearest index)
        waypoints_yaml = {"waypoints": {}}
        for i, center in enumerate(centroids):
            # Find closest original point
            distances = np.linalg.norm(positions - center, axis=1)
            idx = np.argmin(distances)
            euler = R.from_quat(quaternions[idx]).as_euler('xyz', degrees=True)

            waypoints_yaml["waypoints"][f"wp{i}"] = {
                "type": "SingleBall",
                "position": [round(p, 4) for p in positions[idx]],
                "rpy": [round(o, 4) for o in euler],
                "radius": 0.001,
                "margin": 0.0,
                "stationary": True
            }

        print(yaml.dump(waypoints_yaml, sort_keys=False, default_flow_style=None))
        with open(f"{save_dir}/{bag_name}.yaml", "w") as f:
            yaml.dump(waypoints_yaml, f, sort_keys=False, default_flow_style=None)

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