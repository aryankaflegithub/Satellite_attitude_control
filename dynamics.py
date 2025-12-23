import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SatelliteConfig:
    inertia_x: float = 1.0
    inertia_y: float = 1.0
    inertia_z: float = 1.0
    
    max_torque: float = 1.0
    
    damping: float = 0.001
    
    dt: float = 0.05
    
    @property
    def inertia_matrix(self) -> np.ndarray:
        return np.diag([self.inertia_x, self.inertia_y, self.inertia_z])
    
    @property
    def inertia_inverse(self) -> np.ndarray:
        return np.diag([1/self.inertia_x, 1/self.inertia_y, 1/self.inertia_z])


class SatelliteDynamics:
    def __init__(self, config: Optional[SatelliteConfig] = None):
        self.config = config or SatelliteConfig()
        self.reset()
        
    def reset(
        self, 
        quaternion: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if quaternion is None:
            self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            self.quaternion = self._normalize_quaternion(quaternion)
            
        if angular_velocity is None:
            self.angular_velocity = np.zeros(3)
        else:
            self.angular_velocity = np.array(angular_velocity)
            
        return self.quaternion.copy(), self.angular_velocity.copy()
    
    def step(self, torque: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        torque = np.clip(torque, -1, 1) * self.config.max_torque
        
        I = self.config.inertia_matrix
        I_inv = self.config.inertia_inverse
        w = self.angular_velocity
        
        gyroscopic = np.cross(w, I @ w)
        
        damping = self.config.damping * w
        
        angular_accel = I_inv @ (torque - gyroscopic - damping)
        
        self.angular_velocity = w + angular_accel * self.config.dt
        
        self.quaternion = self._integrate_quaternion(
            self.quaternion, 
            self.angular_velocity, 
            self.config.dt
        )
        
        return self.quaternion.copy(), self.angular_velocity.copy()
    
    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            return q / norm
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _integrate_quaternion(
        self, 
        q: np.ndarray, 
        omega: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        
        q_dot = 0.5 * self._quaternion_multiply(omega_quat, q)
        
        q_new = q + q_dot * dt
        
        return self._normalize_quaternion(q_new)
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def get_body_z_axis(self) -> np.ndarray:
        R = self.quaternion_to_rotation_matrix(self.quaternion)
        return R[:, 2]
    
    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])


def latlon_to_direction(lat: float, lon: float) -> np.ndarray:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return np.array([x, y, z])


def compute_angular_error(
    current_direction: np.ndarray, 
    target_direction: np.ndarray
) -> float:
    current_direction = current_direction / np.linalg.norm(current_direction)
    target_direction = target_direction / np.linalg.norm(target_direction)
    
    dot = np.clip(np.dot(current_direction, target_direction), -1.0, 1.0)
    
    return np.arccos(dot)


if __name__ == "__main__":
    config = SatelliteConfig()
    dynamics = SatelliteDynamics(config)
    
    print("Testing satellite dynamics...")
    print(f"Initial quaternion: {dynamics.quaternion}")
    print(f"Initial angular velocity: {dynamics.angular_velocity}")
    
    torque = np.array([0.5, 0.0, 0.0])
    for i in range(100):
        q, w = dynamics.step(torque)
        
    print(f"\nAfter 100 steps with torque {torque}:")
    print(f"Quaternion: {q}")
    print(f"Angular velocity: {w}")
    print(f"Euler angles (deg): {np.degrees(dynamics.quaternion_to_euler(q))}")
    
    target = latlon_to_direction(27.7, 85.3)
    print(f"\nTarget direction (Kathmandu): {target}")
    
    body_z = dynamics.get_body_z_axis()
    error = compute_angular_error(body_z, target)
    print(f"Angular error: {np.degrees(error):.2f} degrees")
