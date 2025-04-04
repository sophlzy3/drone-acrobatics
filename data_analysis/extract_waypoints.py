import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks

def extract_waypoints(bag_names, bag_dir, save_dir):
    for bag_name in bag_names:
        filepath = f"{bag_dir}/{bag_name}_flightgoggles_agiros_pilot_state.csv"
        df = pd.read_csv(filepath)

        # Extract positions and orientations
        positions = df[['pose.position.x','pose.position.y','pose.position.z']].to_numpy()
        quaternions = df[['pose.orientation.x','pose.orientation.y','pose.orientation.z','pose.orientation.w']].to_numpy()
        
        # Calculate velocity magnitudes (to detect stops) - using the correct column names
        velocities = df[['velocity.linear.x', 'velocity.linear.y', 'velocity.linear.z']].to_numpy()
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Method 1: Find points where velocity is minimal (drone slows down at waypoints)
        # Invert velocity for peak finding (we want minima, not maxima)
        inv_velocity = np.max(velocity_magnitudes) - velocity_magnitudes
        
        # Find peaks (local minima in original velocity) with minimum separation
        # Adjust distance parameter based on your expected waypoint separation
        peaks, _ = find_peaks(inv_velocity, distance=len(df)//10, prominence=0.1)
        
        # If we don't find exactly 6 points (start, 4 waypoints, end), adjust parameters
        if len(peaks) != 6:
            # Method 2: Extract local velocity minima with adaptive parameters
            distances = np.linspace(len(df)//20, len(df)//5, 10)
            prominences = np.linspace(0.05, 0.3, 10)
            
            for dist in distances:
                for prom in prominences:
                    peaks, _ = find_peaks(inv_velocity, distance=int(dist), prominence=prom)
                    if len(peaks) >= 6:
                        # Found enough peaks, sort by time and take the 6 most prominent
                        peak_values = inv_velocity[peaks]
                        sorted_indices = np.argsort(-peak_values)  # Descending order
                        peaks = peaks[sorted_indices[:6]]
                        peaks.sort()  # Resort by time
                        break
                if len(peaks) >= 6:
                    break
            
            # If we still don't have 6 points, try a different approach
            if len(peaks) != 6:
                # Method 3: Segment the trajectory evenly by time
                peaks = np.linspace(0, len(df)-1, 6).astype(int)
        
        # Generate the waypoints YAML
        waypoints_yaml = {"waypoints": {}}
        
        for i, idx in enumerate(peaks):
            # Only include the middle 4 points as actual waypoints
            if 0 < i < 5:  # Skip first and last (start and end points)
                euler = R.from_quat(quaternions[idx]).as_euler('xyz', degrees=True)
                
                waypoints_yaml["waypoints"][f"wp{i-1}"] = {
                    "type": "SingleBall",
                    "position": [round(float(p), 4) for p in positions[idx]],
                    "rpy": [round(float(o), 4) for o in euler],
                    "radius": 0.001,
                    "margin": 0.0,
                    "stationary": True
                }
        
        print(f"Extracted waypoints for {bag_name}:")
        print(yaml.dump(waypoints_yaml, sort_keys=False, default_flow_style=None))
        
        with open(f"{save_dir}/{bag_name}.yaml", "w") as f:
            yaml.dump(waypoints_yaml, f, sort_keys=False, default_flow_style=None)

# Example usage:
# bag_names = ["flight1", "flight2"]
# extract_waypoints(bag_names, "path/to/bags", "path/to/save")

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