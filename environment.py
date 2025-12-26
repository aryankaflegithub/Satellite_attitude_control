import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from dynamics import (
    SatelliteDynamics, 
    SatelliteConfig, 
    latlon_to_direction, 
    compute_angular_error
)

class SatelliteAttitudeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        target_threshold_deg: float = 1.0,
        failure_threshold_deg: float = 120.0,
        random_initial_state: bool = True,
        config: Optional[SatelliteConfig] = None,
        sensor_noise: bool = False,
        noise_config: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.target_threshold = np.radians(target_threshold_deg)
        self.failure_threshold = np.radians(failure_threshold_deg)
        self.random_initial_state = random_initial_state
        
        self.sensor_noise = sensor_noise
        self.noise_config = noise_config or {
            'gyro_std': 0.01,
            'gyro_bias': 0.002,
            'quat_std': 0.001,
            'accel_std': 0.05,
        }
        
        self.config = config or SatelliteConfig()
        self.dynamics = SatelliteDynamics(self.config)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.target_direction = None
        self.previous_action = np.zeros(3)
        self.history = []
        
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        options = options or {}
        
        if 'target_lat' in options and 'target_lon' in options:
            self.target_direction = latlon_to_direction(
                options['target_lat'], 
                options['target_lon']
            )
        else:
            lat = self.np_random.uniform(-90, 90)
            lon = self.np_random.uniform(-180, 180)
            self.target_direction = latlon_to_direction(lat, lon)
        
        if 'initial_quaternion' in options:
            initial_q = np.array(options['initial_quaternion'])
            initial_q = initial_q / np.linalg.norm(initial_q)
        elif self.random_initial_state:
            initial_q = self._random_quaternion()
        else:
            initial_q = np.array([1.0, 0.0, 0.0, 0.0])
            
        if 'initial_angular_velocity' in options:
            initial_w = np.array(options['initial_angular_velocity'])
        elif self.random_initial_state:
            initial_w = self.np_random.uniform(-0.1, 0.1, size=3)
        else:
            initial_w = np.zeros(3)
        
        self.dynamics.reset(initial_q, initial_w)
        
        self.current_step = 0
        self.previous_action = np.zeros(3)
        self.history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        
        prev_error = self._get_angular_error()
        
        self.dynamics.step(action)
        
        self.current_step += 1
        
        new_error = self._get_angular_error()
        angular_vel = np.linalg.norm(self.dynamics.angular_velocity)
        
        reward = self._compute_reward(
            prev_error, 
            new_error, 
            angular_vel, 
            action
        )
        
        terminated = False
        truncated = False
        
        if new_error < self.target_threshold and angular_vel < 0.01:
            terminated = True
            reward += 10.0
            
        if new_error > self.failure_threshold:
            terminated = True
            reward -= 10.0
            
        if self.current_step >= self.max_steps:
            truncated = True
        
        self.previous_action = action.copy()
        self.history.append({
            'step': self.current_step,
            'error_deg': np.degrees(new_error),
            'angular_vel': angular_vel,
            'action_norm': np.linalg.norm(action)
        })
        
        observation = self._get_observation()
        info = self._get_info()
        info['success'] = terminated and new_error < self.target_threshold
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        quaternion = self.dynamics.quaternion.copy()
        angular_velocity = self.dynamics.angular_velocity.copy()
        
        if self.sensor_noise:
            gyro_noise = self.np.random.normal(0, self.noise_config['gyro_std'], size=3)
            gyro_bias = np.ones(3) * self.noise_config['gyro_bias']
            angular_velocity = angular_velocity + gyro_noise + gyro_bias
            
            quat_noise = self.np.random.normal(0, self.noise_config['quat_std'], size=4)
            quaternion = quaternion + quat_noise
            quaternion = quaternion / np.linalg.norm(quaternion)
        
        angular_error = np.array([self._get_angular_error()])
        
        observation = np.concatenate([
            quaternion,
            angular_velocity,
            self.target_direction,
            angular_error
        ]).astype(np.float32)
        
        return observation
    
    def _get_angular_error(self) -> float:
        body_z = self.dynamics.get_body_z_axis()
        return compute_angular_error(body_z, self.target_direction)
    
    def _compute_reward(
        self, 
        prev_error: float, 
        new_error: float,
        angular_vel: float,
        action: np.ndarray
    ) -> float:
        alignment_reward = np.exp(-5.5 * new_error) 
        
        stability_penalty = 0.001 * angular_vel
        
        action_penalty = 0.001 * np.sum(action ** 2)
        
        action_change = np.linalg.norm(action - self.previous_action)
        jerk_penalty = 0.005 * action_change
        reward = (
            alignment_reward  
            - stability_penalty  
            - action_penalty
            - jerk_penalty
        )
        
        
        return float(reward)
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            'step': self.current_step,
            'angular_error_deg': np.degrees(self._get_angular_error()),
            'angular_velocity': np.linalg.norm(self.dynamics.angular_velocity),
            'quaternion': self.dynamics.quaternion.tolist(),
        }
    
    def _random_quaternion(self) -> np.ndarray:
        u = self.np_random.uniform(0, 1, size=3)
        
        q = np.array([
            np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
            np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
            np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
            np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
        ])
        
        return np.array([q[3], q[0], q[1], q[2]])
    
    def render(self):
        if self.render_mode == "human":
            error = np.degrees(self._get_angular_error())
            print(f"Step {self.current_step}: Error = {error:.2f}°")
    
    def close(self):
        pass


gym.register(
    id='SatelliteAttitude-v0',
    entry_point='environment:SatelliteAttitudeEnv',
    max_episode_steps=1000,
)


if __name__ == "__main__":
    print("Testing SatelliteAttitudeEnv...")
    
    env = SatelliteAttitudeEnv(random_initial_state=True)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    print("Running 10 random steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.3f}, error={info['angular_error_deg']:.2f}°")
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward:.3f}")
    env.close()
