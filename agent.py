import os
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    EvalCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from environment import SatelliteAttitudeEnv


def create_env(random_initial_state: bool = True, max_steps: int = 500) -> SatelliteAttitudeEnv:
    return SatelliteAttitudeEnv(random_initial_state=random_initial_state, max_steps=max_steps)


def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        env = create_env()
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from typing import Union

def create_vectorized_env(n_envs: int = 4, seed: int = 0, use_subprocess: bool = False) -> VecEnv:
    if use_subprocess:
        return SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    else:
        return DummyVecEnv([make_env(i, seed) for i in range(n_envs)])


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_errors = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
            if 'angular_error_deg' in info:
                self.episode_errors.append(info['angular_error_deg'])
                
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_errors:
            mean_error = sum(self.episode_errors) / len(self.episode_errors)
            self.logger.record('custom/mean_angular_error_deg', mean_error)
            self.episode_errors = []


def create_ppo_agent(
    env,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    tensorboard_log: Optional[str] = "./logs/",
    verbose: int = 1,
    device: str = "auto"
) -> PPO:
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],
                "vf": [256, 256]
            },
            "activation_fn": torch.nn.Tanh,
        }
    
    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        device=device
    )
    
    return agent


def load_agent(model_path: str, env=None) -> PPO:
    return PPO.load(model_path, env=env)


def get_policy_network_summary(agent: PPO) -> str:
    policy = agent.policy
    return str(policy)


if __name__ == "__main__":
    print("Testing PPO agent creation...")
    
    env = create_vectorized_env(n_envs=1)
    
    agent = create_ppo_agent(env, tensorboard_log=None, verbose=1)
    
    print(f"Agent created successfully!")
    print(f"Policy: {agent.policy}")
    print(f"Device: {agent.device}")
    print(f"Learning rate: {agent.learning_rate}")
    
    print("Running 1000 training steps...")
    agent.learn(total_timesteps=1000, progress_bar=True)
    
    print("Training test completed!")
    env.close()
