import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class PIDGains:
    kp_x: float = 2.0
    kp_y: float = 2.0
    kp_z: float = 2.0
    ki_x: float = 0.1
    ki_y: float = 0.1
    ki_z: float = 0.1
    kd_x: float = 1.0
    kd_y: float = 1.0
    kd_z: float = 1.0
    integral_limit: float = 1.0
    output_limit: float = 1.0

class PIDController:
    def __init__(self, gains: Optional[PIDGains] = None, dt: float = 0.01):
        self.gains = gains or PIDGains()
        self.dt = dt
        self.reset()
        
    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_angular_velocity = np.zeros(3)
        
    def compute_quaternion_error(self, current_q: np.ndarray, target_direction: np.ndarray) -> np.ndarray:
        R = self._quaternion_to_rotation_matrix(current_q)
        current_direction = R[:, 2]
        
        axis = np.cross(current_direction, target_direction)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-6:
            dot = np.dot(current_direction, target_direction)
            if dot > 0:
                return np.zeros(3)
            else:
                if abs(current_direction[0]) < 0.9:
                    axis = np.cross(current_direction, np.array([1, 0, 0]))
                else:
                    axis = np.cross(current_direction, np.array([0, 1, 0]))
                axis = axis / np.linalg.norm(axis)
                return axis * np.pi
        
        axis = axis / axis_norm
        dot = np.clip(np.dot(current_direction, target_direction), -1.0, 1.0)
        angle = np.arccos(dot)
        
        error_inertial = axis * angle
        error_body = R.T @ error_inertial
        
        return error_body
    
    def compute_control(self, current_q: np.ndarray, angular_velocity: np.ndarray, target_direction: np.ndarray) -> np.ndarray:
        error = self.compute_quaternion_error(current_q, target_direction)
        
        kp = np.array([self.gains.kp_x, self.gains.kp_y, self.gains.kp_z])
        P = kp * error
        
        ki = np.array([self.gains.ki_x, self.gains.ki_y, self.gains.ki_z])
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.gains.integral_limit, self.gains.integral_limit)
        I = ki * self.integral
        
        kd = np.array([self.gains.kd_x, self.gains.kd_y, self.gains.kd_z])
        D = -kd * angular_velocity
        
        control = P + I + D
        control = np.clip(control, -self.gains.output_limit, self.gains.output_limit)
        
        self.prev_error = error
        self.prev_angular_velocity = angular_velocity
        
        return control
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])

class CascadePIDController:
    def __init__(self, position_gains: Optional[PIDGains] = None, rate_gains: Optional[PIDGains] = None, dt: float = 0.01):
        if position_gains is None:
            position_gains = PIDGains(kp_x=5.0, kp_y=5.0, kp_z=5.0, ki_x=0.1, ki_y=0.1, ki_z=0.1, kd_x=0.0, kd_y=0.0, kd_z=0.0, integral_limit=0.5)
        
        if rate_gains is None:
            rate_gains = PIDGains(kp_x=1.5, kp_y=1.5, kp_z=1.5, ki_x=0.15, ki_y=0.15, ki_z=0.15, kd_x=0.1, kd_y=0.1, kd_z=0.1, integral_limit=1.0)
        
        self.position_controller = PIDController(position_gains, dt)
        self.rate_controller = PIDController(rate_gains, dt)
        self.dt = dt
        
    def reset(self):
        self.position_controller.reset()
        self.rate_controller.reset()
        
    def compute_control(self, current_q: np.ndarray, angular_velocity: np.ndarray, target_direction: np.ndarray) -> np.ndarray:
        error = self.position_controller.compute_quaternion_error(current_q, target_direction)
        kp = np.array([self.position_controller.gains.kp_x, self.position_controller.gains.kp_y, self.position_controller.gains.kp_z])
        rate_setpoint = kp * error
        rate_setpoint = np.clip(rate_setpoint, -0.5, 0.5)
        
        rate_error = rate_setpoint - angular_velocity
        kp_rate = np.array([self.rate_controller.gains.kp_x, self.rate_controller.gains.kp_y, self.rate_controller.gains.kp_z])
        kd_rate = np.array([self.rate_controller.gains.kd_x, self.rate_controller.gains.kd_y, self.rate_controller.gains.kd_z])
        
        self.rate_controller.integral += rate_error * self.dt
        self.rate_controller.integral = np.clip(self.rate_controller.integral, -1.0, 1.0)
        ki_rate = np.array([self.rate_controller.gains.ki_x, self.rate_controller.gains.ki_y, self.rate_controller.gains.ki_z])
        
        if self.dt > 0:
            derivative = (rate_error - self.rate_controller.prev_error) / self.dt
        else:
            derivative = np.zeros(3)
        control = kp_rate * rate_error + ki_rate * self.rate_controller.integral + kd_rate * derivative
        self.rate_controller.prev_error = rate_error
        
        return np.clip(control, -1.0, 1.0)

def run_pid_episode(env, controller, max_steps: int = 500, render: bool = False) -> dict:
    obs, info = env.reset()
    controller.reset()
    
    total_reward = 0
    errors = []
    actions = []
    angular_vels = []
    
    for step in range(max_steps):
        quaternion = obs[:4]
        angular_velocity = obs[4:7]
        target_direction = obs[7:10]
        
        action = controller.compute_control(quaternion, angular_velocity, target_direction)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        errors.append(info['angular_error_deg'])
        actions.append(action.tolist())
        angular_vels.append(info['angular_velocity'])
        
        if render:
            print(f"Step {step}: error={info['angular_error_deg']:.2f}\u00b0")
        
        if terminated or truncated:
            break
    
    return {
        'reward': total_reward,
        'length': step + 1,
        'final_error': errors[-1],
        'min_error': min(errors),
        'mean_error': np.mean(errors),
        'errors': errors,
        'actions': actions,
        'angular_vels': angular_vels,
        'success': info.get('success', False)
    }

if __name__ == "__main__":
    from environment import SatelliteAttitudeEnv
    env = SatelliteAttitudeEnv(random_initial_state=True, max_steps=500)
    pid = PIDController()
    result = run_pid_episode(env, pid)
    print(f"Total reward: {result['reward']:.2f}")
    print(f"Final error: {result['final_error']:.2f}\u00b0")
    print(f"Min error: {result['min_error']:.2f}\u00b0")
    print(f"Episode length: {result['length']}")
    
    cascade_pid = CascadePIDController()
    result = run_pid_episode(env, cascade_pid)
    print(f"Total reward: {result['reward']:.2f}")
    print(f"Final error: {result['final_error']:.2f}\u00b0")
    print(f"Min error: {result['min_error']:.2f}\u00b0")
    print(f"Episode length: {result['length']}")
    env.close()
