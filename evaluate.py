import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Any
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import SatelliteAttitudeEnv
from dynamics import SatelliteDynamics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate trained PPO agent for satellite attitude control'
    )
    
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.zip file)')
    parser.add_argument('--n-episodes', type=int, default=200, help='Number of evaluation episodes (default: 200)')
    parser.add_argument('--max-steps', type=int, default=5000, help='Maximum steps per episode (default: 5000)')
    parser.add_argument('--render', action='store_true', help='Print step-by-step progress')
    parser.add_argument('--plot', action='store_true', help='Generate and save plots')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results/', help='Directory to save results (default: ./evaluation_results/)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--stochastic', action='store_true', default=False, help='Use stochastic actions ')
    
    return parser.parse_args()


def evaluate_episode(
    model: PPO, 
    env: Any, 
    deterministic: bool = True,
    render: bool = False
) -> Dict:
    obs = env.reset()
    
    episode_reward = 0
    episode_length = 0
    errors = []
    angular_vels = []
    actions = []
    quaternions = []
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        
        obs, rewards, dones, infos = env.step(action)
        
        reward = rewards[0]
        done = dones[0]
        info = infos[0]
        
        episode_reward += reward
        episode_length += 1
        
        errors.append(info['angular_error_deg'])
        angular_vels.append(info['angular_velocity'])
        
        actions.append(action[0].tolist())
        quaternions.append(info['quaternion'])
        
        if render:
            print(f"  Step {episode_length}: error={info['angular_error_deg']:.2f}°, reward={reward:.3f}")
    
    return {
        'reward': episode_reward,
        'length': episode_length,
        'final_error': errors[-1],
        'min_error': min(errors),
        'mean_error': np.mean(errors),
        'errors': errors,
        'angular_vels': angular_vels,
        'actions': actions,
        'quaternions': quaternions,
        'success': info.get('success', False)
    }


def evaluate_model(
    model: PPO,
    n_episodes: int = 20,
    max_steps: int = 1000,
    seed: int = 42,
    deterministic: bool = True,
    render: bool = False,
    stats_path: str = None
) -> Tuple[List[Dict], Dict]:
    def make_eval_env():
        return SatelliteAttitudeEnv(max_steps=max_steps, random_initial_state=True)
    
    env = DummyVecEnv([make_eval_env])
    
    if stats_path and Path(stats_path).exists():
        print(f"Loading normalization stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("WARNING: VecNormalize stats not found or not provided")
    
    results = []
    
    print(f"\nEvaluating over {n_episodes} episodes")
    for ep in tqdm(range(n_episodes), desc="Episodes"):
        if hasattr(env, 'venv'):
            base_env = env.venv.envs[0]
        else:
            base_env = env.envs[0]
            
        base_env.reset(seed=seed + ep)
        
        env.reset()
        
        if render:
            print(f"\nEpisode {ep + 1}")
        
        ep_result = evaluate_episode(model, env, deterministic, render)
        results.append(ep_result)
    
    env.close()
    
    rewards = [r['reward'] for r in results]
    lengths = [r['length'] for r in results]
    final_errors = [r['final_error'] for r in results]
    min_errors = [r['min_error'] for r in results]
    successes = [r['success'] for r in results]
    
    summary = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'mean_final_error': np.mean(final_errors),
        'mean_min_error': np.mean(min_errors),
        'success_rate': np.mean(successes) * 100,
        'total_episodes': n_episodes
    }
    
    return results, summary


def plot_results(results: List[Dict], output_dir: Path, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Evaluation Results: {model_name}', fontsize=14)
    
    ax = axes[0, 0]
    for i, result in enumerate(results[:10]):
        ax.plot(result['errors'], alpha=0.7, label=f'Ep {i+1}' if i < 5 else None)
    ax.axhline(y=1.0, color='g', linestyle='--', label='Target (1°)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title('Angular Error Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    final_errors = [r['final_error'] for r in results]
    ax.hist(final_errors, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(final_errors), color='r', linestyle='--', label=f'Mean: {np.mean(final_errors):.2f}°')
    ax.set_xlabel('Final Angular Error (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Final Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    rewards = [r['reward'] for r in results]
    ax.bar(range(len(rewards)), rewards, alpha=0.7)
    ax.axhline(y=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    sample_actions = np.array(results[0]['actions'])
    ax.plot(np.linalg.norm(sample_actions, axis=1), alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action Magnitude')
    ax.set_title('Action Magnitude (Episode 1)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_summary.png', dpi=150)
    print(f"\nPlot saved to: {output_dir / 'evaluation_summary.png'}")
    
    best_ep = min(range(len(results)), key=lambda i: results[i]['final_error'])
    best = results[best_ep]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Best Episode Details (Episode {best_ep + 1})', fontsize=14)
    
    ax = axes[0, 0]
    ax.plot(best['errors'], 'b-', linewidth=2)
    ax.axhline(y=1.0, color='g', linestyle='--', label='Target')
    ax.set_xlabel('Step')
    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title(f'Angular Error (Final: {best["final_error"]:.2f}°)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(best['angular_vels'], 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Angular Velocity Magnitude')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    actions = np.array(best['actions'])
    ax.plot(actions[:, 0], label='Torque X', alpha=0.7)
    ax.plot(actions[:, 1], label='Torque Y', alpha=0.7)
    ax.plot(actions[:, 2], label='Torque Z', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Normalized Torque')
    ax.set_title('Reaction Wheel Commands')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    quats = np.array(best['quaternions'])
    ax.plot(quats[:, 0], label='w', alpha=0.7)
    ax.plot(quats[:, 1], label='x', alpha=0.7)
    ax.plot(quats[:, 2], label='y', alpha=0.7)
    ax.plot(quats[:, 3], label='z', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Quaternion Component')
    ax.set_title('Orientation Quaternion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'best_episode_details.png', dpi=150)
    print(f"Plot saved to: {output_dir / 'best_episode_details.png'}")
    
    plt.close('all')


def main():
    args = parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        if not model_path.with_suffix('.zip').exists():
            print(f"Error: Model not found at {args.model}")
            return
        model_path = model_path.with_suffix('.zip')
    
    print("PPO Satellite Attitude Control - Evaluation")
    print(f"Model: {model_path}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Deterministic: {not args.stochastic}")
    
    print("\nLoading model...")
    model = PPO.load(str(model_path))
    print("Model loaded successfully!")
    
    stats_path = None
    possible_stats = model_path.parent / f"{model_path.stem}_vecnormalize.pkl"
    if possible_stats.exists():
        stats_path = str(possible_stats)
    else:
        print(f"WARNING: VecNormalize stats not found at {possible_stats}")
    
    results, summary = evaluate_model(
        model=model,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        deterministic=not args.stochastic,
        render=args.render,
        stats_path=stats_path
    )
    
    print("\nEVALUATION SUMMARY")
    print(f"Total Episodes: {summary['total_episodes']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
    print(f"Mean Episode Length: {summary['mean_length']:.1f}")
    print(f"Mean Final Error: {summary['mean_final_error']:.2f}°")
    print(f"Mean Best Error: {summary['mean_min_error']:.2f}°")
    
    if args.plot:
        output_dir = Path(args.output_dir)
        model_name = model_path.stem
        plot_results(results, output_dir, model_name)
    
    return results, summary


if __name__ == "__main__":
    main()
