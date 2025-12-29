import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import SatelliteAttitudeEnv
from pid_controller import CascadePIDController, run_pid_episode


def load_training_data(log_dir: str) -> Optional[tuple[List[int], List[float]]]:
    try:
        log_path = Path(log_dir)
        event_files = list(log_path.glob("**/events.out.tfevents.*"))
        if not event_files:
            return None
        
        event_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_event = str(event_files[0])
        
        ea = EventAccumulator(latest_event)
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        if 'custom/mean_angular_error_deg' in tags:
            events = ea.Scalars('custom/mean_angular_error_deg')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            return steps, values
        return None
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None


def run_episode(
    env,
    model: Optional[PPO] = None,
    pid_controller: Optional[CascadePIDController] = None,
    deterministic: bool = True
) -> Dict:
    obs = env.reset()
    
    if pid_controller:
        pid_controller.reset()
    
    total_reward = 0
    errors = []
    actions = []
    angular_vels = []
    wheel_speeds = []
    
    done = False
    step = 0
    
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        elif pid_controller is not None:
            obs_flat = obs[0] if len(obs.shape) > 1 else obs
            quaternion = obs_flat[:4]
            angular_velocity = obs_flat[4:7]
            target_direction = obs_flat[7:10]
            action = pid_controller.compute_control(
                quaternion, angular_velocity, target_direction
            )
            action = np.array([action])
        else:
            action = env.action_space.sample()
            if len(action.shape) == 1:
                action = np.array([action])
        
        obs, rewards, dones, infos = env.step(action)
        
        reward = rewards[0] if hasattr(rewards, '__len__') else rewards
        done = dones[0] if hasattr(dones, '__len__') else dones
        info = infos[0] if isinstance(infos, list) else infos
        
        total_reward += reward
        errors.append(info['angular_error_deg'])
        actions.append(action[0] if len(action.shape) > 1 else action)
        angular_vels.append(info['angular_velocity'])
        
        action_mag = np.linalg.norm(action)
        wheel_rpm = action_mag * 1500
        wheel_speeds.append(wheel_rpm)
        
        step += 1
    
    # Calculate final error and alignment score
    final_error = errors[-1]
    alignment_score = max(0.0, 100.0 * (1.0 - final_error / 180.0))
    
    # Calculate settling time
    settling_time = None
    for i, err in enumerate(errors):
        if err < 1.0:
            settling_time = i * 0.05
            break
    
    if settling_time is None:
        settling_time = len(errors) * 0.05
    
    return {
        'reward': total_reward,
        'length': step,
        'final_error': final_error,
        'min_error': min(errors),
        'mean_error': np.mean(errors),
        'errors': errors,
        'actions': [a.tolist() if hasattr(a, 'tolist') else list(a) for a in actions],
        'angular_vels': angular_vels,
        'wheel_speeds': wheel_speeds,
        'settling_time': settling_time,
        'success': final_error < 1.0,
        'alignment_score': alignment_score
    }


def run_multiple_episodes(
    make_env_fn,
    model: Optional[PPO] = None,
    pid_controller: Optional[CascadePIDController] = None,
    n_episodes: int = 10,
    seed: int = 42,
    stats_path: Optional[str] = None,
    deterministic: bool = True
) -> List[Dict]:
    env = DummyVecEnv([make_env_fn])
    
    try:
        if stats_path and Path(stats_path).exists():
            env = VecNormalize.load(stats_path, env)
            env.training = False
            env.norm_reward = False
        elif stats_path:
            print(f"WARNING: Stats file not found at {stats_path}")
            
        results = []
        for ep in tqdm(range(n_episodes), desc="Running episodes"):
            if hasattr(env, 'venv'):
                base_env = env.venv.envs[0]
            else:
                base_env = env.envs[0]
            base_env.reset(seed=seed + ep)
            env.reset()
            
            result = run_episode(env, model, pid_controller, deterministic)
            results.append(result)
        
        return results
    finally:
        env.close()


def generate_comparison_plots(
    untrained_results: List[Dict],
    trained_results: List[Dict],
    pid_results: List[Dict],
    training_errors: Optional[List[float]] = None,
    output_path: str = "comparison_visualization.png"
):
    fig = plt.figure(figsize=(15, 12))
    
    colors = {
        'untrained': '#4169E1',
        'trained': '#32CD32',
        'pid': '#DC143C'
    }
    
    ax1 = fig.add_subplot(3, 3, 1)
    
    if untrained_results and untrained_results[0]['errors']:
        steps = np.arange(len(untrained_results[0]['errors']))
        ax1.plot(steps, untrained_results[0]['errors'], 
                 color=colors['untrained'], label='Untrained RL', alpha=0.8)
                 
    if trained_results and trained_results[0]['errors']:
        steps = np.arange(len(trained_results[0]['errors']))
        ax1.plot(steps, trained_results[0]['errors'], 
                 color=colors['trained'], label='Trained RL', alpha=0.8)
                 
    if pid_results and pid_results[0]['errors']:
        steps = np.arange(len(pid_results[0]['errors']))
        ax1.plot(steps, pid_results[0]['errors'], 
                 color=colors['pid'], label='PID', alpha=0.8)
                 
    ax1.set_xlabel('Simulation Steps')
    ax1.set_ylabel('Pointing Error (°)')
    ax1.set_title('Pointing Error vs Time (Episode 1)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(3, 3, 2)
    controllers = ['Untrained RL', 'Trained RL', 'PID Controller']
    mean_errors = [
        np.mean([r['mean_error'] for r in untrained_results]),
        np.mean([r['mean_error'] for r in trained_results]) if trained_results else 0,
        np.mean([r['mean_error'] for r in pid_results])
    ]
    std_errors = [
        np.std([r['mean_error'] for r in untrained_results]),
        np.std([r['mean_error'] for r in trained_results]) if trained_results else 0,
        np.std([r['mean_error'] for r in pid_results])
    ]
    bar_colors = [colors['untrained'], colors['trained'], colors['pid']]
    bars = ax2.bar(controllers, mean_errors, color=bar_colors, alpha=0.8)
    ax2.errorbar(controllers, mean_errors, yerr=std_errors, fmt='none', 
                 color='black', capsize=5)
    ax2.set_ylabel('Mean Pointing Error (°)')
    ax2.set_title('Average Performance Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Fixed: Use alignment_score instead of success_score
    ax3 = fig.add_subplot(3, 3, 3)
    alignment_scores = [
        np.mean([r['alignment_score'] for r in untrained_results]),
        np.mean([r['alignment_score'] for r in trained_results]) if trained_results else 0,
        np.mean([r['alignment_score'] for r in pid_results])
    ]
    ax3.bar(controllers, alignment_scores, color=bar_colors, alpha=0.8)
    ax3.set_ylabel('Alignment Score (0-100)')
    ax3.set_title('Alignment Score (100 = 0° error)')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = fig.add_subplot(3, 3, 4)
    wheel_speeds = [
        np.mean([np.mean(r['wheel_speeds']) for r in untrained_results]),
        np.mean([np.mean(r['wheel_speeds']) for r in trained_results]) if trained_results else 0,
        np.mean([np.mean(r['wheel_speeds']) for r in pid_results])
    ]
    ax4.bar(controllers, wheel_speeds, color=bar_colors, alpha=0.8)
    ax4.set_ylabel('Control Effort (RPM)')
    ax4.set_title('Average Wheel Speed Usage')
    ax4.grid(True, alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(3, 3, 5)
    error_data = [
        [r['mean_error'] for r in untrained_results],
        [r['mean_error'] for r in trained_results] if trained_results else [0],
        [r['mean_error'] for r in pid_results]
    ]
    bp = ax5.boxplot(error_data, labels=['Untrained RL', 'Trained RL', 'PID'], 
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_ylabel('Pointing Error (°)')
    ax5.set_title('Error Distribution')
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(3, 3, 6)
    settling_times = [
        np.mean([r['settling_time'] for r in untrained_results]),
        np.mean([r['settling_time'] for r in trained_results]) if trained_results else 0,
        np.mean([r['settling_time'] for r in pid_results])
    ]
    ax6.bar(controllers, settling_times, color=bar_colors, alpha=0.8)
    ax6.set_ylabel('Settling Time (s)')
    ax6.set_title('Time to Achieve < 1° Error')
    ax6.grid(True, alpha=0.3, axis='y')
    
    ax7 = fig.add_subplot(3, 3, 7)
    
    if training_errors and isinstance(training_errors, tuple):
        steps, values = training_errors
        ax7.plot(steps, values, color=colors['trained'], linewidth=2)
        ax7.set_xlabel('Training Steps')
        ax7.set_ylabel('Validation Error (°)')
        ax7.set_title('RL Training Progress')
        ax7.grid(True, alpha=0.3)
        
        steps_np = np.array(steps)
        values_np = np.array(values)
        mask = steps_np > 500000
        if np.any(mask):
            mean_500k = np.mean(values_np[mask])
            ax7.axhline(mean_500k, color='red', linestyle='--', alpha=0.5, label=f'Mean (>500k): {mean_500k:.1f}°')
            ax7.legend()
    elif training_errors and isinstance(training_errors, list):
        steps = np.linspace(0, len(training_errors)*2048, len(training_errors))
        ax7.plot(steps, training_errors, color=colors['trained'], linewidth=2)
        ax7.set_title('RL Training Progress')
    else:
        ax7.text(0.5, 0.5, 'Training data\nnot available', 
                ha='center', va='center', fontsize=12)
        ax7.set_title('RL Training Progress')
    
    ax8 = fig.add_subplot(3, 3, 8)
    mean_rewards = [
        np.mean([r['reward'] for r in untrained_results]),
        np.mean([r['reward'] for r in trained_results]) if trained_results else 0,
        np.mean([r['reward'] for r in pid_results])
    ]
    ax8.bar(controllers, mean_rewards, color=bar_colors, alpha=0.8)
    ax8.set_ylabel('Mean Episode Reward')
    ax8.set_title('Average Reward')
    ax8.grid(True, alpha=0.3, axis='y')
    
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    # Fixed: Use alignment_score and add success rate
    table_data = [
        ['Controller', 'Mean Error', 'Alignment', 'Success', 'Effort'],
        ['Untrained RL', 
         f"{np.mean([r['mean_error'] for r in untrained_results]):.2f}°",
         f"{np.mean([r['alignment_score'] for r in untrained_results]):.1f}%",
         f"{np.mean([r['success'] for r in untrained_results])*100:.1f}%",
         f"{np.mean([np.mean(r['wheel_speeds']) for r in untrained_results]):.0f} RPM"],
        ['Trained RL',
         f"{np.mean([r['mean_error'] for r in trained_results]):.2f}°" if trained_results else "N/A",
         f"{np.mean([r['alignment_score'] for r in trained_results]):.1f}%" if trained_results else "N/A",
         f"{np.mean([r['success'] for r in trained_results])*100:.1f}%" if trained_results else "N/A",
         f"{np.mean([np.mean(r['wheel_speeds']) for r in trained_results]):.0f} RPM" if trained_results else "N/A"],
        ['PID',
         f"{np.mean([r['mean_error'] for r in pid_results]):.2f}°",
         f"{np.mean([r['alignment_score'] for r in pid_results]):.1f}%",
         f"{np.mean([r['success'] for r in pid_results])*100:.1f}%",
         f"{np.mean([np.mean(r['wheel_speeds']) for r in pid_results]):.0f} RPM"]
    ]
    
    table = ax9.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax9.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison visualization for satellite attitude control'
    )
    parser.add_argument('--model', type=str, default=None, help='Path to trained RL model (.zip file)')
    parser.add_argument('--stats', type=str, default=None, help='Path to VecNormalize stats (.pkl file)')
    parser.add_argument('--n-episodes', type=int, default=10, help='Number of episodes per controller')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--output', type=str, default='comparison_visualization.png', help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("Satellite Attitude Control - Comparison Visualization")
    
    def make_env():
        return SatelliteAttitudeEnv(max_steps=args.max_steps, random_initial_state=True)
    
    print("\n1. Running Untrained RL episodes")
    untrained_results = run_multiple_episodes(
        make_env,
        model=None,
        pid_controller=None,
        n_episodes=args.n_episodes,
        seed=args.seed,
        stats_path=None,
        deterministic=False
    )
    
    trained_results = []
    if args.model:
        model_path = Path(args.model)
        if not model_path.suffix:
            model_path = model_path.with_suffix('.zip')
        
        if model_path.exists():
            print(f"\n2. Loading trained model from {model_path}")
            model = PPO.load(str(model_path))
            
            stats_path = args.stats
            if not stats_path:
                possible_stats = model_path.parent / f"{model_path.stem}_vecnormalize.pkl"
                if possible_stats.exists():
                    stats_path = str(possible_stats)
            
            print(f"Running Trained RL episodes")
            trained_results = run_multiple_episodes(
                make_env,
                model=model,
                n_episodes=args.n_episodes,
                seed=args.seed,
                stats_path=stats_path,
                deterministic=True
            )
        else:
            print(f"WARNING: Model not found at {model_path}")
    else:
        print("\n2. No trained model provided, skipping Trained RL")

    print("\n3. Running PID Controller episodes")
    pid_controller = CascadePIDController()
    pid_results = run_multiple_episodes(
        make_env,
        pid_controller=pid_controller,
        n_episodes=args.n_episodes,
        seed=args.seed,
        stats_path=None,
        deterministic=True
    )
    
    print("\n4. Generating comparison visualization")
    generate_comparison_plots(
        untrained_results,
        trained_results,
        pid_results,
        training_errors=None,
        output_path=args.output
    )
    
    # Fixed: Display alignment scores in summary
    print("\nSUMMARY")
    print(f"{'Controller':<15} {'Mean Error':>12} {'Alignment Score':>15} {'Avg Reward':>12}")
    
    print(f"{'Untrained RL':<15} "
          f"{np.mean([r['mean_error'] for r in untrained_results]):>10.2f}° "
          f"{np.mean([r['alignment_score'] for r in untrained_results]):>13.1f}% "
          f"{np.mean([r['reward'] for r in untrained_results]):>12.1f}")
    
    if trained_results:
        print(f"{'Trained RL':<15} "
              f"{np.mean([r['mean_error'] for r in trained_results]):>10.2f}° "
              f"{np.mean([r['alignment_score'] for r in trained_results]):>13.1f}% "
              f"{np.mean([r['reward'] for r in trained_results]):>12.1f}")
    
    print(f"{'PID':<15} "
          f"{np.mean([r['mean_error'] for r in pid_results]):>10.2f}° "
          f"{np.mean([r['alignment_score'] for r in pid_results]):>13.1f}% "
          f"{np.mean([r['reward'] for r in pid_results]):>12.1f}")


if __name__ == "__main__":
    main()
    
    