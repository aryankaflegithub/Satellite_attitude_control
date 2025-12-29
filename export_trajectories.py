import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from stable_baselines3 import PPO
from environment import SatelliteAttitudeEnv
from pid_controller import PIDController, CascadePIDController
from dynamics import latlon_to_direction


def run_controller_episode(
    env: SatelliteAttitudeEnv,
    controller,
    controller_type: str = "pid",
    model: Optional[PPO] = None,
    max_steps: int = 300
) -> Dict:

    obs, info = env.reset()
    
    if controller_type == "pid":
        controller.reset()
    
    trajectory = {
        'quaternions': [],
        'angular_velocities': [],
        'errors': [],
        'rewards': [],
        'actions': [],
        'cumulative_reward': [],
        'target_direction': env.target_direction.tolist(),
        'steps': 0
    }
    
    total_reward = 0
    done = False
    step = 0
    
    while not done and step < max_steps:

        if controller_type == "rl" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            
            quaternion = obs[:4]
            angular_velocity = obs[4:7]
            target_direction = obs[7:10]
            action = controller.compute_control(
                quaternion, angular_velocity, target_direction
            )
        
        trajectory['quaternions'].append(obs[:4].tolist())
        trajectory['angular_velocities'].append(obs[4:7].tolist())
        trajectory['errors'].append(float(np.degrees(obs[10])))
        trajectory['actions'].append(action.tolist() if hasattr(action, 'tolist') else list(action))
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        trajectory['rewards'].append(float(reward))
        trajectory['cumulative_reward'].append(float(total_reward))
        
        done = terminated or truncated
        step += 1
    
    trajectory['steps'] = step
    trajectory['final_error'] = trajectory['errors'][-1]
    trajectory['total_reward'] = float(total_reward)
    trajectory['success'] = info.get('success', False)
    
    return trajectory


def generate_trajectories_for_targets(
    targets: List[Dict],
    model_path: Optional[str] = None,
    output_dir: str = "./visualize/data",
    max_steps: int = 300,
    seed: int = 42
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = SatelliteAttitudeEnv(
        random_initial_state=False,
        max_steps=max_steps,
        sensor_noise=False
    )
    
    rl_model = None
    if model_path:
        model_path = Path(model_path)
        if model_path.exists() or model_path.with_suffix('.zip').exists():
            if not model_path.suffix:
                model_path = model_path.with_suffix('.zip')
            print(f"Loading RL model from {model_path}...")
            rl_model = PPO.load(str(model_path))
            print("Model loaded!")
    
    pid = CascadePIDController()
    
    all_data = {
        'trajectories': [],
        'metadata': {
            'max_steps': max_steps,
            'dt': 0.01,
            'has_rl': rl_model is not None
        }
    }
    
    for i, target in enumerate(targets):
        lat, lon = target['lat'], target['lon']
        print(f"Generating trajectory {i+1}/{len(targets)}: lat={lat}, lon={lon}")
        
        target_dir = latlon_to_direction(lat, lon)
        
        options = {
            'target_lat': lat,
            'target_lon': lon,
            'initial_quaternion': [1.0, 0.0, 0.0, 0.0],
            'initial_angular_velocity': [0.0, 0.0, 0.0]
        }
        
        env.reset(seed=seed + i, options=options)
        
        trajectory_data = {
            'target': {'lat': lat, 'lon': lon},
            'target_direction': target_dir.tolist(),
            'pid': None,
            'rl': None
        }
        
        env.reset(seed=seed + i, options=options)
        pid.reset()
        trajectory_data['pid'] = run_controller_episode(
            env, pid, controller_type="pid", max_steps=max_steps
        )
        
        if rl_model:
            env.reset(seed=seed + i, options=options)
            trajectory_data['rl'] = run_controller_episode(
                env, None, controller_type="rl", model=rl_model, max_steps=max_steps
            )
        
        all_data['trajectories'].append(trajectory_data)
    
    env.close()
    
    output_file = output_dir / "trajectories.json"
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"\nTrajectories saved to {output_file}")
    
    return all_data


def generate_grid_trajectories(
    lat_range: tuple = (-60, 60),
    lon_range: tuple = (-180, 180),
    lat_step: int = 30,
    lon_step: int = 45,
    model_path: Optional[str] = None,
    output_dir: str = "./visualize/data"
):

    targets = []
    for lat in range(lat_range[0], lat_range[1] + 1, lat_step):
        for lon in range(lon_range[0], lon_range[1] + 1, lon_step):
            targets.append({'lat': lat, 'lon': lon})
    
    print(f"Generating {len(targets)} trajectories...")
    return generate_trajectories_for_targets(targets, model_path, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Export RL/PID trajectories for web visualization'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./visualize/data',
    )
    parser.add_argument(
        '--grid',
        action='store_true',
    )
    parser.add_argument(
        '--lat',
        type=float,
        default=27.7172,
    )
    parser.add_argument(
        '--lon',
        type=float,
        default=85.3240,
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=300,
    )
    
    args = parser.parse_args()
    
    if args.grid:
        generate_grid_trajectories(
            model_path=args.model,
            output_dir=args.output_dir
        )
    else:
        targets = [
            {'lat': args.lat, 'lon': args.lon},
            {'lat': 0, 'lon': 0},
            {'lat': 40.7128, 'lon': -74.0060},
            {'lat': -33.8688, 'lon': 151.2093},
            {'lat': 35.6762, 'lon': 139.6503},
        ]
        generate_trajectories_for_targets(
            targets,
            model_path=args.model,
            output_dir=args.output_dir,
            max_steps=args.max_steps
        )


if __name__ == "__main__":
    main()
