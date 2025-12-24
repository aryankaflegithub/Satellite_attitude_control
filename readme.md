# Satellite Attitude Control using Reinforcement Learning

A comprehensive implementation of satellite attitude control using **Proximal Policy Optimization (PPO)** reinforcement learning, with classical PID controller baselines for comparison.

##  Overview

This project trains an AI agent to control a satellite's orientation in 3D space, learning to accurately point at target directions while minimizing control effort and maintaining stability. The system uses realistic physics simulation including:

- Quaternion-based orientation representation (no gimbal lock)
- Euler's rotation equations with gyroscopic coupling
- Reaction wheel actuators with torque limits
- Configurable sensor noise and disturbances

##  Key Features

- **Deep RL Training**: PPO agent with normalized observations and parallel environments
- **Classical Baselines**: Simple and Cascade PID controllers for comparison
- **Realistic Physics**: Full 3D rotational dynamics with inertia tensor
- **Comprehensive Evaluation**: Success rates, settling time, control effort metrics
- **Rich Visualizations**: 3D satellite rendering, error plots, training curves
- **Reproducible**: Seed control and checkpointing for consistent results

##  Requirements

```bash
pip install gymnasium
pip install stable-baselines3
pip install torch
pip install numpy
pip install matplotlib
pip install pandas
pip install tensorboard
pip install tqdm
```

##  Quick Start

### 1. Train an RL Agent

```bash
python train.py --timesteps 2000000 --n-envs 4
```

**Training Options:**
- `--timesteps`: Total training steps (default: 2,000,000)
- `--n-envs`: Parallel environments (default: 4)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--batch-size`: Batch size (default: 64)
- `--n-epochs`: Epochs per update (default: 20)
- `--checkpoint-freq`: Save frequency (default: 50,000)
- `--device`: `cuda`, `cpu`, or `auto`

**Output:**
- Trained models saved to `./checkpoints/`
- TensorBoard logs in `./logs/`
- Automatic comparison plot after training

### 2. Evaluate Trained Model

```bash
python evaluate.py --model checkpoints/ppo_satellite_final.zip --n-episodes 100 --plot
```

**Evaluation Options:**
- `--model`: Path to trained model (required)
- `--n-episodes`: Number of test episodes (default: 200)
- `--plot`: Generate evaluation plots
- `--stochastic`: Use stochastic policy instead of deterministic
- `--render`: Print step-by-step progress

**Output:**
- Success rate and error statistics
- Performance plots in `./evaluation_results/`

### 3. Compare Controllers

```bash
python compare_controllers.py --model checkpoints/ppo_satellite_final.zip --n-episodes 10
```

Tests all three controllers (Untrained RL, Trained RL, PID) on identical scenarios.

**Output:**
- Statistical comparison tables
- Box plots showing performance distributions
- Saved to `./comparison_results/`

### 4. Generate Comprehensive Analysis

```bash
python generate_comparison.py --model checkpoints/ppo_satellite_final.zip --n-episodes 100
```

Creates detailed 9-panel comparison visualization with training progress.

##  Project Structure

```
.
├── train.py                    # Main training script
├── evaluate.py                 # Model evaluation
├── compare_controllers.py      # Controller comparison
├── generate_comparison.py      # Comprehensive analysis
├── plot_error_vs_time.py      # Quick comparison plot
├── export_trajectories.py     # Export data for web viz
│
├── environment.py             # Gymnasium environment
├── dynamics.py                # Satellite physics engine
├── agent.py                   # PPO agent setup
├── pid_controller.py          # Classical controllers
├── visualization.py           # 3D plotting and animation
├── sensor_processor.py        # Real sensor data analysis
│
├── checkpoints/               # Saved models
├── logs/                      # TensorBoard logs
└── evaluation_results/        # Evaluation outputs
```

##  Core Components

### Environment (`environment.py`)

**State Space (11D):**
- Quaternion orientation (4D): `[w, x, y, z]`
- Angular velocity (3D): `[ωx, ωy, ωz]` rad/s
- Target direction (3D): `[x, y, z]` unit vector
- Angular error (1D): radians

**Action Space (3D):**
- Normalized torque commands: `[-1, 1]` for each axis
- Scaled by `max_torque` parameter

**Reward Function:**
```python
reward = exp(-5.5 × angular_error)     # Alignment reward
         - 0.001 × angular_velocity    # Stability penalty
         - 0.001 × action²              # Control effort penalty
         - 0.005 × Δaction              # Smoothness penalty
```

**Success Criteria:**
- Angular error < 1° (0.01745 rad)
- Angular velocity < 0.01 rad/s
- Both conditions must hold simultaneously

### Dynamics (`dynamics.py`)

Implements Euler's rotation equations:

```
I⁻¹(τ - ω × Iω - damping × ω) = α

q̇ = 0.5 × [0, ω] ⊗ q
```

Where:
- `I`: Inertia tensor (3×3 diagonal)
- `τ`: Applied torque
- `ω`: Angular velocity
- `q`: Quaternion orientation
- `α`: Angular acceleration

### Controllers

**1. Simple PID:**
```python
u = Kp × error + Ki × ∫error + Kd × (-angular_velocity)
```

**2. Untrained RL :**
This is the not fully trained RL with just the initial values.
It is to give a base line for the trained RL

**3. PPO RL Agent:**
- Neural network: 256×256 hidden layers with Tanh activation
- Learns optimal policy through trial and error
- Adapts to complex dynamics automatically

##  Performance Metrics

### Primary Metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **Success Rate** | % episodes achieving < 1° error | > 90% |
| **Mean Error** | Average error during episode | < 10° |
| **Final Error** | Error at episode end | < 1° |
| **Settling Time** | Time to reach < 1° error | < 5s |
| **Control Effort** | Average wheel speed (RPM) | Minimize |

### Success Score (Continuous)

```
Alignment Score = 100 × (1 - final_error/180°)
```

- 0° error → 100 points
- 90° error → 50 points
- 180° error → 0 points

##  Visualization

### 3D Satellite Rendering

```python
from visualization import SatelliteVisualizer

viz = SatelliteVisualizer()
viz.plot_static(quaternion, target_direction, errors)
```

Shows:
- Red/Green/Blue axes: Satellite body frame
- Orange arrow: Target direction
- Error plot over time

### Training Progress

```bash
tensorboard --logdir logs/
```

Monitor:
- Episode rewards
- Mean angular error
- Policy loss and entropy

##  Typical Results

After 2M training steps:

| Controller | Mean Error | Success Rate | Control Effort |
|-----------|------------|--------------|----------------|
| Untrained RL | 85° | 5% | High |
| **Trained RL** | **8°** | **92%** | **Medium** |
| Simple PID | 25° | 45% | High |

**Key Insight:** RL learns smooth, efficient control strategies that match or exceed classical PID controllers.

##  Advanced Usage

### Custom Satellite Configuration

```python
from dynamics import SatelliteConfig

config = SatelliteConfig(
    inertia_x=2.0,      # kg⋅m²
    inertia_y=1.5,
    inertia_z=1.0,
    max_torque=0.5,     # N⋅m
    damping=0.001
)
```

### Sensor Noise Simulation

```python
env = SatelliteAttitudeEnv(
    sensor_noise=True,
    noise_config={
        'gyro_std': 0.01,
        'gyro_bias': 0.002,
        'quat_std': 0.001
    }
)
```

### Export for Web Visualization

```bash
python export_trajectories.py --model checkpoints/model.zip --grid
```

Generates `trajectories.json` with full state history for interactive visualization.

##  Technical Details

### Why Quaternions?

- **No gimbal lock** (unlike Euler angles)
- **Continuous** (no discontinuities)
- **Compact** (4 parameters vs 9 for rotation matrix)
- **Efficient** (simple multiplication and normalization)

### PPO Hyperparameters

```python
learning_rate = 1e-4      # Step size for policy updates
n_steps = 2048            # Steps per environment before update
batch_size = 64           # Minibatch size
n_epochs = 20             # Optimization epochs per update
gamma = 0.99              # Discount factor
gae_lambda = 0.95         # GAE parameter
clip_range = 0.2          # PPO clipping parameter
```

### Observation Normalization

Critical for training stability:
```python
VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
```

Normalizes observations to zero mean, unit variance.

##  Troubleshooting

### Training Not Converging

1. **Check normalization**: Ensure VecNormalize stats are being used
2. **Reduce learning rate**: Try `--learning-rate 1e-5`
3. **Increase training time**: Try `--timesteps 5000000`
4. **Verify reward function**: Check TensorBoard for reward trends

### Poor Evaluation Performance

1. **Load VecNormalize stats**: Ensure `.pkl` file is in same directory as model
2. **Use deterministic policy**: `--stochastic` flag should be off
3. **Check initial conditions**: Verify environment randomization

### CUDA Out of Memory

1. **Reduce parallel environments**: `--n-envs 2`
2. **Reduce batch size**: `--batch-size 32`
3. **Use CPU**: `--device cpu`

##  Contributing

Potential improvements:

- [ ] Add disturbance torques (gravity gradient, solar pressure)
- [ ] Implement fuel constraints and optimization
- [ ] Multi-target sequential pointing missions

##  License

MIT License - Feel free to use for research and education.

##  References

### Reinforcement Learning
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
- Stable-Baselines3 Documentation

### Spacecraft Dynamics
- Wie, B. (1998). "Space Vehicle Dynamics and Control"
- Sidi, M. (1997). "Spacecraft Dynamics and Control"

### Quaternion Math
- Kuipers, J. (1999). "Quaternions and Rotation Sequences"

##  FAQ

**Q: How long does training take?**
A: 30-60 minutes for 2M steps on a modern GPU (RTX 3080), 3-4 hours on CPU.

**Q: Can this work on a real satellite?**
A: The physics are realistic, but real deployment requires:
- Hardware-in-the-loop testing
- Failure mode analysis
- Fuel/power constraints
- Attitude determination system integration

**Q: Why is RL better than PID?**
A: RL can:
- Learn optimal control for complex, coupled dynamics
- Adapt to changing conditions
- Optimize multiple objectives simultaneously
- Handle constraints naturally

**Q: What's the difference between success rate and alignment score?**
A: 
- **Success Rate**: Binary - did the episode achieve < 1° error? (%)
- **Alignment Score**: Continuous - how close to 0° error? (0-100 scale)

##  Citation

If you use this code in research, please cite:

```bibtex
@software{satellite_attitude_rl,
  title={Satellite Attitude Control using Reinforcement Learning},
  author={Aryan Kafle},
  year={2024},
  url={https://github.com/aryankaflegithub/Satellite_attitude_control}
}
```

---

**Built with:** Python 3.10+ | PyTorch | Stable-Baselines3 | Gymnasium

