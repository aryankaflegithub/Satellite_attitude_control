import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import SatelliteAttitudeEnv
from pid_controller import CascadePIDController
from generate_comparison import run_episode

def plot_pointing_error(model_path, stats_path=None):
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    env = DummyVecEnv([lambda: SatelliteAttitudeEnv(
        max_steps=500, 
        random_initial_state=True,
        failure_threshold_deg=180.0
    )])
    
    if stats_path:
        print(f"Loading stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: No stats path provided. Model might perform poorly.")

    print("Running Trained RL")
    seed = 42
    env.seed(seed)
    if hasattr(env, 'venv'):
        env.venv.envs[0].reset(seed=seed)
    else:
        env.envs[0].reset(seed=seed)
    env.reset()
    
    rl_result = run_episode(env, model=model, deterministic=True)
    
    print("Running PID")
    if hasattr(env, 'venv'):
        env.venv.envs[0].reset(seed=seed)
    else:
        env.envs[0].reset(seed=seed)
    env.reset()
    
    pid_ctrl = CascadePIDController()
    pid_result = run_episode(env, pid_controller=pid_ctrl)
    
    print("Running Untrained")
    if hasattr(env, 'venv'):
        env.venv.envs[0].reset(seed=seed)
    else:
        env.envs[0].reset(seed=seed)
    env.reset()
    
    untrained_result = run_episode(env, model=None)

    print(f"Untrained errors length: {len(untrained_result['errors'])}")
    print(f"Untrained mean error: {untrained_result['mean_error']}")
    print(f"PID errors length: {len(pid_result['errors'])}")
    print(f"PID mean error: {pid_result['mean_error']}")
    print(f"RL errors length: {len(rl_result['errors'])}")
    print(f"RL mean error: {rl_result['mean_error']}")

    plt.figure(figsize=(10, 6))
    
    steps = np.arange(len(rl_result['errors']))
    plt.plot(steps, rl_result['errors'], label='Trained RL', color='#32CD32', linewidth=2)
    
    steps_pid = np.arange(len(pid_result['errors']))
    plt.plot(steps_pid, pid_result['errors'], label='PID', color='#DC143C', alpha=0.9, linestyle='--')
    
    steps_untrained = np.arange(len(untrained_result['errors']))
    plt.plot(steps_untrained, untrained_result['errors'], label='Untrained', color='#4169E1', alpha=0.9, linestyle=':')
    
    plt.xlabel('Simulation Steps')
    plt.ylabel('Pointing Error (degrees)')
    plt.title('Pointing Error vs Time (Evaluation Episode)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'pointing_error_vs_time.png'
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .zip model')
    args = parser.parse_args()
    
    model_path = Path(args.model)
    stats_path = model_path.parent / f"{model_path.stem}_vecnormalize.pkl"
    if not stats_path.exists():
        pass
        
    plot_pointing_error(str(model_path), str(stats_path) if stats_path.exists() else None)
