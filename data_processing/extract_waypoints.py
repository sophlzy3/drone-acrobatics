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

import pandas as pd
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def extract_waypoints1(bag_names, bag_dir, save_dir, plot_detection=False):
    for bag_name in bag_names:
        filepath = f"{bag_dir}/{bag_name}_flightgoggles_agiros_pilot_state.csv"
        df = pd.read_csv(filepath)

        # Extract positions and orientations
        positions = df[['pose.position.x','pose.position.y','pose.position.z']].to_numpy()
        quaternions = df[['pose.orientation.x','pose.orientation.y','pose.orientation.z','pose.orientation.w']].to_numpy()
        timestamps = df['timestamp'].to_numpy()
        
        # Extract motion parameters
        velocities = df[['velocity.linear.x', 'velocity.linear.y', 'velocity.linear.z']].to_numpy()
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        ang_velocities = df[['velocity.angular.x', 'velocity.angular.y', 'velocity.angular.z']].to_numpy()
        ang_velocity_magnitudes = np.linalg.norm(ang_velocities, axis=1)
        
        accelerations = df[['acceleration.linear.x', 'acceleration.linear.y', 'acceleration.linear.z']].to_numpy()
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        jerks = df[['jerk.x', 'jerk.y', 'jerk.z']].to_numpy()
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        
        # Create a composite signal that emphasizes waypoints
        # Waypoints typically have:
        # 1. Low velocity (drone slows down)
        # 2. High angular velocity (drone is turning)
        # 3. Changes in acceleration direction (drone is changing speed)
        # 4. High jerk (rapid changes in acceleration)
        
        # Normalize each signal to [0,1] range
        norm_vel = (np.max(velocity_magnitudes) - velocity_magnitudes) / np.max(velocity_magnitudes) if np.max(velocity_magnitudes) > 0 else np.zeros_like(velocity_magnitudes)
        norm_ang_vel = ang_velocity_magnitudes / np.max(ang_velocity_magnitudes) if np.max(ang_velocity_magnitudes) > 0 else np.zeros_like(ang_velocity_magnitudes)
        norm_acc = acceleration_magnitudes / np.max(acceleration_magnitudes) if np.max(acceleration_magnitudes) > 0 else np.zeros_like(acceleration_magnitudes)
        norm_jerk = jerk_magnitudes / np.max(jerk_magnitudes) if np.max(jerk_magnitudes) > 0 else np.zeros_like(jerk_magnitudes)
        
        # Create a composite signal
        # Weight the signals - you may need to adjust these weights
        composite_signal = (0.5 * norm_vel + 
                            0.3 * norm_ang_vel + 
                            0.1 * norm_acc + 
                            0.1 * norm_jerk)
                            
        # Apply a smoothing filter to reduce noise
        window_size = min(25, len(composite_signal) // 20)  # Adaptive window size
        if window_size % 2 == 0:  # Ensure odd window size for centered smoothing
            window_size += 1
        
        # Simple moving average smoothing
        composite_signal_smooth = np.convolve(composite_signal, 
                                             np.ones(window_size)/window_size, 
                                             mode='same')
        
        # Find peaks in the composite signal
        # Adjust distance parameter based on trajectory length
        min_peak_distance = max(len(df) // 20, 10)  # At least 10 points between peaks
        
        # First attempt: find peaks with initial parameters
        peaks, peak_properties = find_peaks(composite_signal_smooth, 
                                          distance=min_peak_distance,
                                          prominence=0.05,
                                          width=3)
        
        # We need exactly 4 waypoints between start and end (so 6 total points)
        if len(peaks) != 6:
            # Adaptive peak finding with multiple parameter combinations
            best_peaks = None
            best_peak_score = 0
            
            # Try different parameter combinations
            for distance in [len(df)//30, len(df)//20, len(df)//15, len(df)//10]:
                for prominence in [0.01, 0.03, 0.05, 0.1, 0.15]:
                    for width in [1, 3, 5]:
                        try:
                            candidate_peaks, properties = find_peaks(composite_signal_smooth, 
                                                                  distance=distance,
                                                                  prominence=prominence,
                                                                  width=width)
                            
                            # Calculate a score based on how close to 6 peaks we get and their prominence
                            if len(candidate_peaks) >= 6:
                                # If we have too many peaks, sort by prominence and take the top 6
                                if len(candidate_peaks) > 6:
                                    sorted_indices = np.argsort(-properties['prominences'])
                                    candidate_peaks = candidate_peaks[sorted_indices[:6]]
                                    candidate_peaks.sort()  # Resort by time
                                    properties['prominences'] = properties['prominences'][sorted_indices[:6]]
                                
                                # Calculate peak score based on prominence and spacing
                                peak_score = np.sum(properties['prominences']) * (6 / len(candidate_peaks))
                                
                                # Check spacing - ideally, peaks should be somewhat evenly distributed
                                peak_intervals = np.diff(candidate_peaks)
                                interval_variation = np.std(peak_intervals) / np.mean(peak_intervals)
                                # Lower variation is better (more evenly spaced)
                                peak_score = peak_score * (1 / (1 + interval_variation))
                                
                                if peak_score > best_peak_score:
                                    best_peak_score = peak_score
                                    best_peaks = candidate_peaks
                        except:
                            # Skip errors in peak finding with certain parameters
                            pass
            
            if best_peaks is not None and len(best_peaks) >= 4:
                peaks = best_peaks
            else:
                # Fallback: Segment the trajectory evenly
                print(f"Warning: Could not reliably detect waypoints for {bag_name}. Using uniform spacing.")
                peaks = np.linspace(0, len(df)-1, 6).astype(int)
        
        # If we still have more than 6 points, take the most prominent 6
        if len(peaks) > 6:
            peak_values = composite_signal_smooth[peaks]
            sorted_indices = np.argsort(-peak_values)
            peaks = peaks[sorted_indices[:6]]
            peaks.sort()  # Resort by time
        
        # Ensure the first and last points are included
        if len(peaks) >= 2 and peaks[0] > len(df) // 10:
            peaks[0] = 0  # Set first peak to start
        if len(peaks) >= 2 and peaks[-1] < len(df) - len(df) // 10:
            peaks[-1] = len(df) - 1  # Set last peak to end
        
        # Make sure we have exactly 6 points (start, 4 waypoints, end)
        if len(peaks) < 6:
            # Fill missing points by segmenting evenly between detected points
            full_peaks = np.zeros(6, dtype=int)
            
            # Always keep detected points
            for i, peak in enumerate(peaks):
                full_peaks[i] = peak
            
            # Fill in missing points by interpolation
            for i in range(len(peaks), 6):
                # Simple linear interpolation between existing points
                full_peaks[i] = int(np.linspace(full_peaks[i-1], len(df)-1, 6-i+1)[1])
            
            peaks = np.sort(full_peaks)
        
        # Generate the waypoints YAML
        waypoints_yaml = {"waypoints": {}}
        
        # Visualize the detection process if requested
        if plot_detection:
            plt.figure(figsize=(15, 10))
            
            # Plot position over time
            plt.subplot(5, 1, 1)
            plt.plot(timestamps, positions[:, 0], 'r-', label='X')
            plt.plot(timestamps, positions[:, 1], 'g-', label='Y')
            plt.plot(timestamps, positions[:, 2], 'b-', label='Z')
            plt.plot(timestamps[peaks], positions[peaks, 0], 'ro', markersize=8)
            plt.plot(timestamps[peaks], positions[peaks, 1], 'go', markersize=8)
            plt.plot(timestamps[peaks], positions[peaks, 2], 'bo', markersize=8)
            plt.legend()
            plt.title('Position')
            
            # Plot velocity magnitude
            plt.subplot(5, 1, 2)
            plt.plot(timestamps, velocity_magnitudes)
            plt.plot(timestamps[peaks], velocity_magnitudes[peaks], 'ro', markersize=8)
            plt.title('Velocity Magnitude')
            
            # Plot angular velocity magnitude
            plt.subplot(5, 1, 3)
            plt.plot(timestamps, ang_velocity_magnitudes)
            plt.plot(timestamps[peaks], ang_velocity_magnitudes[peaks], 'ro', markersize=8)
            plt.title('Angular Velocity Magnitude')
            
            # Plot composite signal
            plt.subplot(5, 1, 4)
            plt.plot(timestamps, composite_signal, 'k--', alpha=0.5, label='Raw')
            plt.plot(timestamps, composite_signal_smooth, 'k-', label='Smoothed')
            plt.plot(timestamps[peaks], composite_signal_smooth[peaks], 'ro', markersize=8)
            plt.legend()
            plt.title('Composite Signal')
            
            # Plot trajectory in 3D
            ax = plt.subplot(5, 1, 5, projection='3d')
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k-', alpha=0.3)
            ax.plot(positions[peaks, 0], positions[peaks, 1], positions[peaks, 2], 'ro', markersize=8)
            
            # Add waypoint numbers
            for i, peak in enumerate(peaks):
                label = "Start" if i == 0 else "End" if i == len(peaks)-1 else f"WP{i}"
                ax.text(positions[peak, 0], positions[peak, 1], positions[peak, 2], label)
            
            ax.set_title('3D Trajectory')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{bag_name}_waypoint_detection.png")
            plt.close()
        
        # Create waypoint entries for the middle 4 points (excluding start and end)
        for i in range(1, 5):
            idx = peaks[i]
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
# extract_waypoints(bag_names, "path/to/bags", "path/to/save", plot_detection=True)
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

extract_waypoints1(bag_names, bag_dir, save_dir)