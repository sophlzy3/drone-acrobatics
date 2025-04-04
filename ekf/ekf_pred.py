import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

class DroneEKF:
    def __init__(self):
        # State vector: [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz, ax_bias, ay_bias, az_bias, gx_bias, gy_bias, gz_bias]
        # where (x,y,z) is position, (vx,vy,vz) is linear velocity,
        # (qx,qy,qz,qw) is orientation quaternion,
        # (wx,wy,wz) is angular velocity,
        # (ax_bias, ay_bias, az_bias) is accelerometer bias
        # (gx_bias, gy_bias, gz_bias) is gyroscope bias
        
        self.state_dim = 19
        self.measurement_dim = 6  # Angular rates (3) and thrust (3)
        
        # Initialize state vector
        self.x = np.zeros(self.state_dim)
        self.x[9] = 1.0  # Initialize quaternion to identity rotation
        
        # Initialize covariance matrix with reasonable uncertainties
        self.P = np.eye(self.state_dim)
        self.P[0:3, 0:3] *= 0.01  # Position uncertainty
        self.P[3:6, 3:6] *= 0.1   # Velocity uncertainty
        self.P[6:10, 6:10] *= 0.01  # Quaternion uncertainty
        self.P[10:13, 10:13] *= 0.1  # Angular velocity uncertainty
        self.P[13:16, 13:16] *= 0.01  # Accelerometer bias uncertainty
        self.P[16:19, 16:19] *= 0.01  # Gyroscope bias uncertainty
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise covariance
        self.R = np.eye(self.measurement_dim) * 0.1
        
        # Time step
        self.dt = 0.01  # Default 10ms, will be updated from timestamps
        
    def normalize_quaternion(self):
        """Normalize the quaternion in the state vector"""
        q = self.x[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            self.x[6:10] = q / q_norm
            
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([x, y, z, w])
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        
        rotation_matrix = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return rotation_matrix
    
    def state_transition(self, state, dt):
        """
        State transition function f(x, dt)
        Predict the next state given the current state and time step
        """
        next_state = np.copy(state)
        
        # Extract state components
        position = state[0:3]
        velocity = state[3:6]
        quaternion = state[6:10]  # [qx, qy, qz, qw]
        angular_velocity = state[10:13]
        acc_bias = state[13:16]
        gyro_bias = state[16:19]
        
        # Linear dynamics: position update based on velocity
        next_state[0:3] = position + velocity * dt
        
        # Angular dynamics: quaternion update based on angular velocity
        # Small angle approximation for quaternion increment
        angle = np.linalg.norm(angular_velocity) * dt
        if angle > 0:
            axis = angular_velocity / np.linalg.norm(angular_velocity)
            # Build quaternion increment: [sin(θ/2)*axis, cos(θ/2)]
            dq = np.zeros(4)
            dq[0:3] = np.sin(angle/2) * axis
            dq[3] = np.cos(angle/2)
            
            # Quaternion multiplication: q_new = q_old ⊗ dq
            next_state[6:10] = self.quaternion_multiply(quaternion, dq)
            
            # Normalize the quaternion
            q_norm = np.linalg.norm(next_state[6:10])
            if q_norm > 0:
                next_state[6:10] /= q_norm
        
        # Bias states are assumed to be relatively constant with slow drift
        # For simplicity, we keep them constant in the prediction step
        
        return next_state
    
    def state_transition_jacobian(self, state, dt):
        """
        Compute the Jacobian of the state transition function
        """
        F = np.eye(self.state_dim)
        
        # Position derivative with respect to velocity
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Quaternion derivative with respect to angular velocity
        # This is a simplified approximation
        angular_velocity = state[10:13]
        q = state[6:10]
        
        # For small angular velocities, linearize the quaternion update
        # This is a rough approximation and could be improved
        wx, wy, wz = angular_velocity
        qx, qy, qz, qw = q
        
        # Build the quaternion rate of change matrix (simplified)
        F[6, 10] = 0.5 * dt * qw
        F[6, 11] = -0.5 * dt * qz
        F[6, 12] = 0.5 * dt * qy
        
        F[7, 10] = 0.5 * dt * qz
        F[7, 11] = 0.5 * dt * qw
        F[7, 12] = -0.5 * dt * qx
        
        F[8, 10] = -0.5 * dt * qy
        F[8, 11] = 0.5 * dt * qx
        F[8, 12] = 0.5 * dt * qw
        
        F[9, 10] = -0.5 * dt * qx
        F[9, 11] = -0.5 * dt * qy
        F[9, 12] = -0.5 * dt * qz
        
        return F
    
    def predict(self, timestamp=None):
        """
        Predict step of the EKF
        """
        if timestamp is not None and hasattr(self, 'prev_timestamp'):
            self.dt = timestamp - self.prev_timestamp
            self.prev_timestamp = timestamp
        elif timestamp is not None:
            self.prev_timestamp = timestamp
        
        # Predict state
        self.x = self.state_transition(self.x, self.dt)
        
        # Normalize quaternion
        self.normalize_quaternion()
        
        # Predict covariance
        F = self.state_transition_jacobian(self.x, self.dt)
        self.P = F @ self.P @ F.T + self.Q
        
        return self.x
    
    def measurement_function(self, state):
        """
        Measurement function h(x)
        Maps the state to expected measurements
        """
        # For this implementation, we'll assume measurements are:
        # [angular_rates.x, angular_rates.y, angular_rates.z, thrust.x, thrust.y, thrust.z]
        
        # Extract relevant state components
        angular_velocity = state[10:13]
        gyro_bias = state[16:19]
        
        # Expected measurements
        z_pred = np.zeros(self.measurement_dim)
        
        # Angular rates are directly measured (with bias)
        z_pred[0:3] = angular_velocity - gyro_bias
        
        # Thrust is related to acceleration in body frame
        # We simplify by using the last 3 components
        quaternion = state[6:10]
        R_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # Convert acceleration to body frame (simplified)
        # In practice, you might need a more complex model relating acceleration to thrust
        acc_body = np.zeros(3)  # This would be derived from the drone's dynamics
        
        # For simplicity, just use all three components of thrust
        z_pred[3:6] = acc_body[0:3]
        
        return z_pred
    
    def measurement_jacobian(self, state):
        """
        Compute the Jacobian of the measurement function
        """
        H = np.zeros((self.measurement_dim, self.state_dim))
        
        # Angular rates with respect to angular velocity
        H[0:3, 10:13] = np.eye(3)
        
        # Angular rates with respect to gyro bias
        H[0:3, 16:19] = -np.eye(3)
        
        # Thrust measurements would have complex relationships with quaternion and acceleration
        # This is a simplified placeholder
        
        return H
    
    def update(self, measurement, timestamp=None):
        """
        Update step of the EKF
        """
        if timestamp is not None:
            # First predict using the new timestamp
            self.predict(timestamp)
        
        # Expected measurement based on current state
        z_pred = self.measurement_function(self.x)
        
        # Measurement Jacobian
        H = self.measurement_jacobian(self.x)
        
        # Innovation/residual
        y = measurement - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Normalize quaternion
        self.normalize_quaternion()
        
        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P
        
        return self.x
    
    def process_data(self, telemetry_df, control_df):
        """
        Process telemetry and control data frames to predict drone states
        
        Parameters:
        - telemetry_df: DataFrame with drone telemetry including position, orientation, etc.
        - control_df: DataFrame with control inputs including thrust and angular rates
        
        Returns:
        - predictions_df: DataFrame with predicted states
        """
        # Merge the dataframes on timestamp
        # Make sure timestamp columns have the same type before merging
        control_df['timestamp'] = control_df['timestamp'].astype(float)
        telemetry_df['timestamp'] = telemetry_df['timestamp'].astype(float)
        
        merged_df = pd.merge_asof(
            control_df.sort_values('timestamp'),
            telemetry_df.sort_values('timestamp'),
            on='timestamp'
        )
        
        # Initialize results dataframe
        predictions = []
        
        # Initialize state with first measurement
        if not merged_df.empty:
            first_row = merged_df.iloc[0]
            
            # Initialize position
            self.x[0] = first_row['pose.position.x']
            self.x[1] = first_row['pose.position.y']
            self.x[2] = first_row['pose.position.z']
            
            # Initialize velocity
            self.x[3] = first_row['velocity.linear.x']
            self.x[4] = first_row['velocity.linear.y']
            self.x[5] = first_row['velocity.linear.z']
            
            # Initialize quaternion
            self.x[6] = first_row['pose.orientation.x']
            self.x[7] = first_row['pose.orientation.y']
            self.x[8] = first_row['pose.orientation.z']
            self.x[9] = first_row['pose.orientation.w']
            
            # Initialize angular velocity
            self.x[10] = first_row['velocity.angular.x']
            self.x[11] = first_row['velocity.angular.y']
            self.x[12] = first_row['velocity.angular.z']
            
            # Initialize biases
            self.x[13] = first_row['acc_bias.x']
            self.x[14] = first_row['acc_bias.y']
            self.x[15] = first_row['acc_bias.z']
            self.x[16] = first_row['gyr_bias.x']
            self.x[17] = first_row['gyr_bias.y']
            self.x[18] = first_row['gyr_bias.z']
            
            # Save the initial timestamp
            self.prev_timestamp = float(first_row['timestamp'])
        
        # Process each row for prediction and update
        for i, row in merged_df.iterrows():
            timestamp = float(row['timestamp'])
            
            # Create measurement vector for update
            # [angular_rates.x, angular_rates.y, angular_rates.z, thrust.x, thrust.y, thrust.z]
            measurement = np.array([
                row['angular_rates.x'],
                row['angular_rates.y'],
                row['angular_rates.z'],
                row['thrust.x'],
                row['thrust.y'],
                row['thrust.z']
            ])
            
            # Update with measurement
            state = self.update(measurement, timestamp)
            
            # Save prediction
            prediction = {
                'timestamp': timestamp,
                'predicted_x': state[0],
                'predicted_y': state[1],
                'predicted_z': state[2],
                'predicted_vx': state[3],
                'predicted_vy': state[4],
                'predicted_vz': state[5],
                'predicted_qx': state[6],
                'predicted_qy': state[7],
                'predicted_qz': state[8],
                'predicted_qw': state[9],
                'predicted_wx': state[10],
                'predicted_wy': state[11],
                'predicted_wz': state[12]
            }
            predictions.append(prediction)
            
            # Also predict the next state without measurement
            next_state = self.predict()
            
            # Save next prediction
            next_prediction = {
                'timestamp': timestamp + self.dt,
                'predicted_x': next_state[0],
                'predicted_y': next_state[1],
                'predicted_z': next_state[2],
                'predicted_vx': next_state[3],
                'predicted_vy': next_state[4],
                'predicted_vz': next_state[5],
                'predicted_qx': next_state[6],
                'predicted_qy': next_state[7],
                'predicted_qz': next_state[8],
                'predicted_qw': next_state[9],
                'predicted_wx': next_state[10],
                'predicted_wy': next_state[11],
                'predicted_wz': next_state[12]
            }
            predictions.append(next_prediction)
        
        # Create DataFrame from predictions
        predictions_df = pd.DataFrame(predictions)
        return predictions_df

# Example usage:
def load_and_process_data(telemetry_path, control_path):
    # Load data
    telemetry_df = pd.read_csv(telemetry_path)
    control_df = pd.read_csv(control_path)
    
    # Convert timestamps to float if they aren't already
    telemetry_df['timestamp'] = telemetry_df['timestamp'].astype(float)
    control_df['timestamp'] = control_df['timestamp'].astype(float)
    
    # Initialize EKF
    ekf = DroneEKF()
    
    # Process data
    predictions_df = ekf.process_data(telemetry_df, control_df)
    
    return predictions_df

# Example of how to use the function:
# predictions = load_and_process_data('telemetry.csv', 'control.csv')
# predictions.to_csv('drone_state_predictions.csv', index=False)