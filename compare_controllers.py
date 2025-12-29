

import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import SatelliteAttitudeEnv
from pid_controller import CascadePIDController
from visualization import SatelliteVisualizer, visualize_comparison, plot_multiple_episodes


def run_rl_episode(model, env, deterministic=True):
    obs = env.reset()
    
    total_reward = 0
    errors = []
    actions = []
    angular_vels = []
    quaternions = []
    
    done = False
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = env.step(action)
        
        reward = rewards[0] if hasattr(rewards, '__len__') else rewards
        done = dones[0] if hasattr(dones, '__len__') else dones
        info = infos[0] if isinstance(infos, list) else infos
        
        total_reward += reward
        errors.append(info['angular_error_deg'])
        actions.append(action[0] if len(action.shape) > 1 else action)
        angular_vels.append(info['angular_velocity'])
        quaternions.append(info['quaternion'])
        
        step += 1
    
    final_error = errors[-1]
    alignment_score = max(0.0, 100.0 * (1.0 - final_error / 180.0))
    
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
        'final_error': errors[-1],
        'min_error': min(errors),
        'mean_error': np.mean(errors),
        'errors': errors,
        'actions': [a.tolist() if hasattr(a, 'tolist') else list(a) for a in actions],
        'angular_vels': angular_vels,
        'quaternions': quaternions,
        'success': info.get('success', False),
        'alignment_score': alignment_score,
        'settling_time': settling_time
    }


def run_pid_episode(controller, env):
    obs = env.reset()
    controller.reset()
    
    total_reward = 0
    errors = []
    actions = []
    angular_vels = []
    quaternions = []
    
    done = False
    step = 0
    
    while not done:
        obs_flat = obs[0] if len(obs.shape) > 1 else obs
        
        quaternion = obs_flat[:4]
        angular_velocity = obs_flat[4:7]
        target_direction = obs_flat[7:10]
        
        action = controller.compute_control(quaternion, angular_velocity, target_direction)
        action = np.array([action])
        
        obs, rewards, dones, infos = env.step(action)
        
        reward = rewards[0] if hasattr(rewards, '__len__') else rewards
        done = dones[0] if hasattr(dones, '__len__') else dones
        info = infos[0] if isinstance(infos, list) else infos
        
        total_reward += reward
        errors.append(info['angular_error_deg'])
        actions.append(action[0] if len(action.shape) > 1 else action)
        angular_vels.append(info['angular_velocity'])
        quaternions.append(info['quaternion'])
        
        step += 1
    
    final_error = errors[-1]
    alignment_score = max(0.0, 100.0 * (1.0 - final_error / 180.0))
    
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
        'final_error': errors[-1],
        'min_error': min(errors),
        'mean_error': np.mean(errors),
        'errors': errors,
        'actions': [a.tolist() if hasattr(a, 'tolist') else list(a) for a in actions],
        'angular_vels': angular_vels,
        'quaternions': quaternions,
        'success': info.get('success', False),
        'alignment_score': alignment_score,
        'settling_time': settling_time
    }


def compare_controllers(model_path=None, n_episodes=10, sensor_noise=False, save_dir="./comparison_results", seed=42):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Episodes: {n_episodes}")
    print(f"Sensor noise: {'Enabled' if sensor_noise else 'Disabled'}")
    if model_path:
        print(f"RL Model: {model_path}")
    
    def make_env():
        return SatelliteAttitudeEnv(
            random_initial_state=True,
            max_steps=500,
            sensor_noise=sensor_noise,
            failure_threshold_deg=180.0
        )
    
    pid_controller = CascadePIDController()
    
    rl_model = None
    stats_path = None
    if model_path:
        model_path_obj = Path(model_path)
        if not model_path_obj.suffix:
            model_path_obj = model_path_obj.with_suffix('.zip')
        
        if model_path_obj.exists():
            print(f"\nLoading RL model from {model_path_obj}")
            rl_model = PPO.load(str(model_path_obj))
            
            possible_stats = model_path_obj.parent / f"{model_path_obj.stem}_vecnormalize.pkl"
            if possible_stats.exists():
                stats_path = str(possible_stats)
                print(f"Found VecNormalize stats")
        else:
            print(f"Warning: Model not found at {model_path}")
    
    print("\nRunning comparison episodes")
    
    untrained_results = []
    trained_results = []
    pid_results = []
    
    for ep in range(n_episodes):
        env = DummyVecEnv([make_env])
        if hasattr(env, 'venv'):
            env.venv.envs[0].reset(seed=seed + ep)
        else:
            env.envs[0].reset(seed=seed + ep)
        env.reset()
        
        result = run_rl_episode(None, env, deterministic=False)
        untrained_results.append(result)
        env.close()
        
        env = DummyVecEnv([make_env])
        if hasattr(env, 'venv'):
            env.venv.envs[0].reset(seed=seed + ep)
        else:
            env.envs[0].reset(seed=seed + ep)
        env.reset()
        
        pid_controller.reset()
        result = run_pid_episode(pid_controller, env)
        pid_results.append(result)
        env.close()
        
        if rl_model:
            env = DummyVecEnv([make_env])
            
            if stats_path and Path(stats_path).exists():
                env = VecNormalize.load(stats_path, env)
                env.training = False
                env.norm_reward = False
            
            if hasattr(env, 'venv'):
                env.venv.envs[0].reset(seed=seed + ep)
            else:
                env.envs[0].reset(seed=seed + ep)
            env.reset()
            
            result = run_rl_episode(rl_model, env, deterministic=True)
            trained_results.append(result)
            env.close()
        
        print(f"  Episode {ep + 1}/{n_episodes} completed")
    
    def aggregate_results(results_list, name):
        rewards = [r['reward'] for r in results_list]
        final_errors = [r['final_error'] for r in results_list]
        min_errors = [r['min_error'] for r in results_list]
        mean_errors = [r['mean_error'] for r in results_list]
        lengths = [r['length'] for r in results_list]
        successes = [r['success'] for r in results_list]
        alignment_scores = [r['alignment_score'] for r in results_list]
        settling_times = [r['settling_time'] for r in results_list]
        
        return {
            'name': name,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_final_error': np.mean(final_errors),
            'std_final_error': np.std(final_errors),
            'mean_min_error': np.mean(min_errors),
            'mean_mean_error': np.mean(mean_errors),
            'mean_length': np.mean(lengths),
            'success_rate': np.mean(successes) * 100,
            'mean_alignment_score': np.mean(alignment_scores),
            'std_alignment_score': np.std(alignment_scores),
            'mean_settling_time': np.mean(settling_times),
            'all_results': results_list
        }
    
    summary = {
        'Untrained RL': aggregate_results(untrained_results, 'Untrained RL'),
        'PID Controller': aggregate_results(pid_results, 'PID Controller'),
    }
    if trained_results:
        summary['Trained RL'] = aggregate_results(trained_results, 'Trained RL')
    
    print("\nCOMPARISON SUMMARY")
    print(f"{'Controller':<15} {'Reward':>12} {'Final Err':>12} {'Mean Err':>12} {'Success':>10} {'Alignment':>12}")
    print("-" * 75)
    
    for name, stats in summary.items():
        print(f"{name:<15} {stats['mean_reward']:>8.1f}±{stats['std_reward']:<3.1f} "
              f"{stats['mean_final_error']:>8.1f}±{stats['std_final_error']:<3.1f}° "
              f"{stats['mean_mean_error']:>8.1f}° "
              f"{stats['success_rate']:>8.1f}% "
              f"{stats['mean_alignment_score']:>9.1f}%")
    
    print("\nGenerating comparison plots")
    
    results_to_plot = [
        untrained_results[0],
        pid_results[0],
    ]
    labels = ['Untrained RL', 'PID Controller']
    
    if trained_results:
        results_to_plot.append(trained_results[0])
        labels.append('Trained RL')
    
    fig = plot_multiple_episodes(results_to_plot, labels, save_path=str(save_dir / 'controller_comparison.png'))
    plt.close(fig)
    
    if trained_results:
        fig = visualize_comparison(pid_results[0], trained_results[0], save_path=str(save_dir / 'pid_vs_rl_detailed.png'))
        plt.close(fig)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    ax = axes[0, 0]
    data = [
        [r['final_error'] for r in untrained_results],
        [r['final_error'] for r in pid_results],
    ]
    labels_box = ['Untrained RL', 'PID']
    if trained_results:
        data.append([r['final_error'] for r in trained_results])
        labels_box.append('Trained RL')
    
    bp = ax.boxplot(data, labels=labels_box, patch_artist=True)
    colors = ['#4169E1', '#DC143C', '#32CD32']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Final Error (degrees)')
    ax.set_title('Final Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    data = [
        [r['mean_error'] for r in untrained_results],
        [r['mean_error'] for r in pid_results],
    ]
    if trained_results:
        data.append([r['mean_error'] for r in trained_results])
    
    bp = ax.boxplot(data, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Mean Error (degrees)')
    ax.set_title('Mean Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 2]
    data = [
        [r['alignment_score'] for r in untrained_results],
        [r['alignment_score'] for r in pid_results],
    ]
    if trained_results:
        data.append([r['alignment_score'] for r in trained_results])
    
    bp = ax.boxplot(data, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Alignment Score (0-100)')
    ax.set_title('Alignment Score Distribution')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 0]
    data = [
        [r['reward'] for r in untrained_results],
        [r['reward'] for r in pid_results],
    ]
    if trained_results:
        data.append([r['reward'] for r in trained_results])
    
    bp = ax.boxplot(data, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    settling_data = [
        [r['settling_time'] for r in untrained_results],
        [r['settling_time'] for r in pid_results],
    ]
    if trained_results:
        settling_data.append([r['settling_time'] for r in trained_results])
    
    bp = ax.boxplot(settling_data, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Settling Time (seconds)')
    ax.set_title('Time to Achieve <1° Error')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 2]
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Untrained', 'PID', 'Trained' if trained_results else ''],
        ['Success Rate', 
         f"{summary['Untrained RL']['success_rate']:.1f}%",
         f"{summary['PID Controller']['success_rate']:.1f}%",
         f"{summary['Trained RL']['success_rate']:.1f}%" if trained_results else ''],
        ['Alignment Score',
         f"{summary['Untrained RL']['mean_alignment_score']:.1f}",
         f"{summary['PID Controller']['mean_alignment_score']:.1f}",
         f"{summary['Trained RL']['mean_alignment_score']:.1f}" if trained_results else ''],
        ['Mean Error',
         f"{summary['Untrained RL']['mean_mean_error']:.2f}°",
         f"{summary['PID Controller']['mean_mean_error']:.2f}°",
         f"{summary['Trained RL']['mean_mean_error']:.2f}°" if trained_results else ''],
        ['Final Error',
         f"{summary['Untrained RL']['mean_final_error']:.2f}°",
         f"{summary['PID Controller']['mean_final_error']:.2f}°",
         f"{summary['Trained RL']['mean_final_error']:.2f}°" if trained_results else ''],
        ['Settling Time',
         f"{summary['Untrained RL']['mean_settling_time']:.2f}s",
         f"{summary['PID Controller']['mean_settling_time']:.2f}s",
         f"{summary['Trained RL']['mean_settling_time']:.2f}s" if trained_results else ''],
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)
    
    for i in range(4):
        table[(0, i)].set_facecolor('#2C3E50')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, 7):
        table[(i, 0)].set_facecolor('#ECF0F1')
        table[(i, 0)].set_text_props(fontweight='bold')
    
    ax.set_title('Performance Summary', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'distribution_comparison.png'}")
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    controllers = ['Untrained RL', 'PID']
    colors_bar = ['#4169E1', '#DC143C']
    if trained_results:
        controllers.append('Trained RL')
        colors_bar.append('#32CD32')
    
    ax = axes[0]
    success_rates = [summary[c]['success_rate'] for c in ['Untrained RL', 'PID Controller'] + (['Trained RL'] if trained_results else [])]
    bars = ax.bar(range(len(controllers)), success_rates, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(controllers)))
    ax.set_xticklabels(controllers, rotation=15, ha='right')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (Error < 1°)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax = axes[1]
    alignment = [summary[c]['mean_alignment_score'] for c in ['Untrained RL', 'PID Controller'] + (['Trained RL'] if trained_results else [])]
    bars = ax.bar(range(len(controllers)), alignment, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(controllers)))
    ax.set_xticklabels(controllers, rotation=15, ha='right')
    ax.set_ylabel('Alignment Score (0-100)')
    ax.set_title('Mean Alignment Score')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, alignment)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax = axes[2]
    mean_errs = [summary[c]['mean_mean_error'] for c in ['Untrained RL', 'PID Controller'] + (['Trained RL'] if trained_results else [])]
    bars = ax.bar(range(len(controllers)), mean_errs, color=colors_bar, alpha=0.8)
    ax.set_xticks(range(len(controllers)))
    ax.set_xticklabels(controllers, rotation=15, ha='right')
    ax.set_ylabel('Mean Error (degrees)')
    ax.set_title('Average Error During Episode')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, mean_errs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}°', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'bar_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir / 'bar_comparison.png'}")
    plt.close()
    
    viz = SatelliteVisualizer()
    
    best_pid = min(pid_results, key=lambda r: r['final_error'])
    best_idx = pid_results.index(best_pid)
    
    env = SatelliteAttitudeEnv(max_steps=500, random_initial_state=True)
    obs, info = env.reset(seed=seed + best_idx)
    pid_controller.reset()
    
    quaternions = [info['quaternion']]
    target_dirs = [env.target_direction.tolist()]
    errors = []
    
    done = False
    while not done:
        quaternion = obs[:4]
        angular_velocity = obs[4:7]
        target_direction = obs[7:10]
        
        action = pid_controller.compute_control(quaternion, angular_velocity, target_direction)
        obs, reward, terminated, truncated, info = env.step(action)
        
        quaternions.append(info['quaternion'])
        target_dirs.append(target_direction.tolist())
        errors.append(info['angular_error_deg'])
        done = terminated or truncated
    
    fig = viz.plot_static(
        quaternion=np.array(quaternions[-1]),
        target_direction=np.array(target_dirs[-1]),
        errors=errors,
        title="PID Controller Final State"
    )
    plt.savefig(save_dir / 'pid_final_state.png', dpi=150)
    print(f"Saved: {save_dir / 'pid_final_state.png'}")
    plt.close()
    
    env.close()
    
    print(f"\nAll results saved to: {save_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Compare PID and RL controllers for satellite attitude control')
    
    parser.add_argument('--model', type=str, default=None, help='Path to trained RL model (.zip file)')
    parser.add_argument('--n-episodes', type=int, default=10, help='Number of episodes per controller (default: 10)')
    parser.add_argument('--sensor-noise', action='store_true', help='Enable sensor noise')
    parser.add_argument('--save-dir', type=str, default='./comparison_results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    compare_controllers(
        model_path=args.model,
        n_episodes=args.n_episodes,
        sensor_noise=args.sensor_noise,
        save_dir=args.save_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
