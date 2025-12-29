import os
import argparse
import torch
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import SatelliteAttitudeEnv
from agent import (
    create_vectorized_env,
    create_ppo_agent,
    TensorboardCallback,
    create_env
)
from generate_comparison import (
    run_multiple_episodes, 
    generate_comparison_plots, 
    load_training_data
)
from pid_controller import CascadePIDController
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train PPO agent for satellite attitude control'
    )
    
    parser.add_argument('--timesteps', type=int, default=2000000, help='Total training timesteps (default: 2000000)')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--n-epochs', type=int, default=20, help='Number of epochs per update (default: 20)')
    parser.add_argument('--checkpoint-freq', type=int, default=50000, help='Checkpoint save frequency (default: 50000)')
    parser.add_argument('--eval-freq', type=int, default=10000, help='Evaluation frequency (default: 10000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--log-dir', type=str, default='./logs/', help='Tensorboard log directory (default: ./logs/)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/', help='Checkpoint save directory (default: ./checkpoints/)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cpu, cuda, or auto (default: cuda)')
    
    return parser.parse_args()


def train(args):
    log_dir = Path(args.log_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    run_name = f"ppo_satellite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("PPO Satellite Attitude Control Training")
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Number of environments: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("WARNING: CUDA not available, using CPU")
            args.device = 'cpu'

    print("Creating training environments")
    train_env = create_vectorized_env(n_envs=args.n_envs, seed=args.seed)
    
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    print("Creating evaluation environment")
    def make_eval_env():
        return create_env()
    eval_env = DummyVecEnv([make_eval_env])
    
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    print("Synced VecNormalize stats from train_env to eval_env")
    
    print("Creating PPO agent")
    agent = create_ppo_agent(
        env=train_env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        tensorboard_log=str(log_dir),
        device=args.device
    )
    
    print(f"Policy network:\n{agent.policy}")
    
    class SyncVecNormalizeCallback(BaseCallback):
        def __init__(self, train_env, eval_env, verbose=0):
            super().__init__(verbose)
            self.train_env = train_env
            self.eval_env = eval_env
        
        def _on_step(self):
            self.eval_env.obs_rms = self.train_env.obs_rms
            self.eval_env.ret_rms = self.train_env.ret_rms
            return True
    
    sync_callback = SyncVecNormalizeCallback(train_env, eval_env)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,
        save_path=str(checkpoint_dir),
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq // args.n_envs,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    tensorboard_callback = TensorboardCallback()
    
    callbacks = CallbackList([
        sync_callback,
        checkpoint_callback,
        eval_callback,
        tensorboard_callback
    ])
    
    print("Starting training")
    
    try:
        agent.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True, tb_log_name=run_name)
    except KeyboardInterrupt:
        print("Training interrupted by user!")
    
    final_model_path = checkpoint_dir / f"{run_name}_final"
    agent.save(str(final_model_path))
    
    train_env.save(str(checkpoint_dir / f"{run_name}_final_vecnormalize.pkl"))
    
    print(f"Final model saved to: {final_model_path}")
    
    train_env.close()
    eval_env.close()
    
    print("Training completed")
    print(f"To view tensorboard logs, run:")
    print(f"tensorboard --logdir {log_dir}")
    print(f"To evaluate the model, run:")
    print(f"python evaluate.py --model {final_model_path}.zip")
    
    print("Generating post-training comparison")
    
    def make_comparison_env():
        return SatelliteAttitudeEnv(
            max_steps=500, 
            random_initial_state=True,
            failure_threshold_deg=180.0
        )
    
    try:
        print("Running Untrained RL (100 episodes)")
        untrained_results = run_multiple_episodes(
            make_comparison_env, 
            model=None, 
            pid_controller=None, 
            n_episodes=100,
            deterministic=False
        )
        
        print("Running Trained RL (100 episodes)")
        stats_path = str(checkpoint_dir / f"{run_name}_final_vecnormalize.pkl")
        trained_results = run_multiple_episodes(
            make_comparison_env, 
            model=agent, 
            n_episodes=100, 
            stats_path=stats_path,
            deterministic=True
        )
        
        print("Running PID Controller (100 episodes)")
        pid_ctrl = CascadePIDController()
        pid_results = run_multiple_episodes(
            make_comparison_env, 
            pid_controller=pid_ctrl, 
            n_episodes=100,
            deterministic=True
        )
        
        output_plot = checkpoint_dir / f"{run_name}_comparison.png"
        
        training_errors = load_training_data(str(log_dir))
        
        generate_comparison_plots(
            untrained_results, 
            trained_results, 
            pid_results, 
            training_errors=training_errors,
            output_path=str(output_plot)
        )
        print(f"Comparison plot saved to: {output_plot}")
        
    except Exception as e:
        print(f"Error generating comparison plot: {e}")
        
    return agent


if __name__ == "__main__":
    args = parse_args()
    train(args)
