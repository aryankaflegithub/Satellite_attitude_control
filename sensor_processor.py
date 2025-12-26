import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


class SensorProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.gyroscope_data = None
        self.orientation_data = None
        self.acceleration_data = None
        self.magnetometer_data = None
        self.dynamics_params = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        gyro_path = self.data_dir / "Gyroscope.csv"
        if gyro_path.exists():
            self.gyroscope_data = pd.read_csv(gyro_path, sep=';')
            self.gyroscope_data.columns = ['time', 'gyro_x', 'gyro_y', 'gyro_z']
            
        orient_path = self.data_dir / "Orientation.csv"
        if orient_path.exists():
            self.orientation_data = pd.read_csv(orient_path, sep=';')
            self.orientation_data.columns = [
                'time', 'qw', 'qx', 'qy', 'qz', 
                'direction', 'yaw', 'pitch', 'roll'
            ]
            
        accel_path = self.data_dir / "Linear Acceleration.csv"
        if accel_path.exists():
            self.acceleration_data = pd.read_csv(accel_path, sep=';')
            self.acceleration_data.columns = ['time', 'accel_x', 'accel_y', 'accel_z']
            
        mag_path = self.data_dir / "Magnetometer.csv"
        if mag_path.exists():
            self.magnetometer_data = pd.read_csv(mag_path, sep=';')
            self.magnetometer_data.columns = ['time', 'mag_x', 'mag_y', 'mag_z']
            
        return {
            'gyroscope': self.gyroscope_data,
            'orientation': self.orientation_data,
            'acceleration': self.acceleration_data,
            'magnetometer': self.magnetometer_data
        }
    
    def estimate_noise_characteristics(self) -> Dict[str, Dict[str, float]]:
        noise_stats = {}
        
        static_duration = 30.0
        
        if self.gyroscope_data is not None:
            static_gyro = self.gyroscope_data[
                self.gyroscope_data['time'] < static_duration
            ]
            noise_stats['gyroscope'] = {
                'std_x': static_gyro['gyro_x'].std(),
                'std_y': static_gyro['gyro_y'].std(),
                'std_z': static_gyro['gyro_z'].std(),
                'bias_x': static_gyro['gyro_x'].mean(),
                'bias_y': static_gyro['gyro_y'].mean(),
                'bias_z': static_gyro['gyro_z'].mean(),
            }
            
        if self.acceleration_data is not None:
            static_accel = self.acceleration_data[
                self.acceleration_data['time'] < static_duration
            ]
            noise_stats['acceleration'] = {
                'std_x': static_accel['accel_x'].std(),
                'std_y': static_accel['accel_y'].std(),
                'std_z': static_accel['accel_z'].std(),
            }
            
        if self.magnetometer_data is not None:
            static_mag = self.magnetometer_data[
                self.magnetometer_data['time'] < static_duration
            ]
            noise_stats['magnetometer'] = {
                'std_x': static_mag['mag_x'].std(),
                'std_y': static_mag['mag_y'].std(),
                'std_z': static_mag['mag_z'].std(),
            }
            
        return noise_stats
    
    def estimate_dynamics_parameters(self) -> Dict[str, float]:
        if self.gyroscope_data is None:
            raise ValueError("Gyroscope data not loaded")
            
        gyro = self.gyroscope_data
        
        dt = np.diff(gyro['time'].values)
        dt = np.where(dt > 0, dt, 0.01)
        
        angular_accel_x = np.diff(gyro['gyro_x'].values) / dt
        angular_accel_y = np.diff(gyro['gyro_y'].values) / dt
        angular_accel_z = np.diff(gyro['gyro_z'].values) / dt
        
        fast_period = gyro[(gyro['time'] > 210) & (gyro['time'] < 390)]
        
        self.dynamics_params = {
            'max_angular_vel_x': float(np.abs(gyro['gyro_x']).max()),
            'max_angular_vel_y': float(np.abs(gyro['gyro_y']).max()),
            'max_angular_vel_z': float(np.abs(gyro['gyro_z']).max()),
            
            'max_angular_accel_x': float(np.abs(angular_accel_x).max()),
            'max_angular_accel_y': float(np.abs(angular_accel_y).max()),
            'max_angular_accel_z': float(np.abs(angular_accel_z).max()),
            
            'sampling_rate': float(1.0 / np.mean(dt)),
            
            'inertia_x': 1.0,
            'inertia_y': 1.0,
            'inertia_z': 1.0,
        }
        
        return self.dynamics_params
    
    def get_sample_trajectory(
        self, 
        start_time: float = 0, 
        duration: float = 10
    ) -> Dict[str, np.ndarray]:
        end_time = start_time + duration
        
        trajectory = {}
        
        if self.orientation_data is not None:
            mask = (self.orientation_data['time'] >= start_time) & \
                   (self.orientation_data['time'] <= end_time)
            orient = self.orientation_data[mask]
            trajectory['quaternion'] = orient[['qw', 'qx', 'qy', 'qz']].values
            trajectory['euler'] = orient[['yaw', 'pitch', 'roll']].values
            trajectory['time'] = orient['time'].values
            
        if self.gyroscope_data is not None:
            mask = (self.gyroscope_data['time'] >= start_time) & \
                   (self.gyroscope_data['time'] <= end_time)
            gyro = self.gyroscope_data[mask]
            trajectory['angular_velocity'] = gyro[['gyro_x', 'gyro_y', 'gyro_z']].values
            
        return trajectory


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm > 0:
        return q / norm
    return np.array([1.0, 0.0, 0.0, 0.0])


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


if __name__ == "__main__":
    processor = SensorProcessor("data")
    data = processor.load_all_data()
    
    print("Loaded data shapes:")
    for name, df in data.items():
        if df is not None:
            print(f"  {name}: {df.shape}")
            
    noise = processor.estimate_noise_characteristics()
    print("\nNoise characteristics:")
    for sensor, stats in noise.items():
        print(f"  {sensor}: {stats}")
        
    dynamics = processor.estimate_dynamics_parameters()
    print("\nDynamics parameters:")
    for param, value in dynamics.items():
        print(f"  {param}: {value:.4f}")
