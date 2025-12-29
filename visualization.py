import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Dict, Optional, Tuple
import matplotlib.patches as mpatches


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        raise ValueError("Quaternion norm is too small or zero")
    q = q / norm
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


class SatelliteVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        
        self.figsize = figsize
        self.fig = None
        self.ax_3d = None
        self.ax_error = None
        
    def create_figure(self):

        self.fig = plt.figure(figsize=self.figsize)
        
        # 3D satellite view
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_3d.set_xlim([-1.5, 1.5])
        self.ax_3d.set_ylim([-1.5, 1.5])
        self.ax_3d.set_zlim([-1.5, 1.5])
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('Satellite Orientation')
        
        # Error plot
        self.ax_error = self.fig.add_subplot(122)
        self.ax_error.set_xlabel('Step')
        self.ax_error.set_ylabel('Angular Error (degrees)')
        self.ax_error.set_title('Pointing Error Over Time')
        self.ax_error.grid(True, alpha=0.3)
        
        return self.fig
    
    def draw_satellite(
        self, 
        ax: Axes3D, 
        quaternion: np.ndarray,
        target_direction: Optional[np.ndarray] = None
    ):

        ax.cla()
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        R = quaternion_to_rotation_matrix(quaternion)
        
        self._draw_satellite_body(ax, R)
        
        origin = np.array([0, 0, 0])
        axis_length = 1.2
        
        x_axis = R @ np.array([axis_length, 0, 0])
        ax.quiver(*origin, *x_axis, color='red', arrow_length_ratio=0.1, linewidth=2)
        
        y_axis = R @ np.array([0, axis_length, 0])
        ax.quiver(*origin, *y_axis, color='green', arrow_length_ratio=0.1, linewidth=2)
        
        z_axis = R @ np.array([0, 0, axis_length])
        ax.quiver(*origin, *z_axis, color='blue', arrow_length_ratio=0.1, linewidth=3)
        
        if target_direction is not None:
            target = target_direction * axis_length
            ax.quiver(*origin, *target, color='orange', arrow_length_ratio=0.1, 
                     linewidth=2, linestyle='--')
        
        red_patch = mpatches.Patch(color='red', label='Body X')
        green_patch = mpatches.Patch(color='green', label='Body Y')
        blue_patch = mpatches.Patch(color='blue', label='Body Z (pointing)')
        orange_patch = mpatches.Patch(color='orange', label='Target')
        ax.legend(handles=[red_patch, green_patch, blue_patch, orange_patch],
                 loc='upper left', fontsize=8)
        
    def _draw_satellite_body(self, ax: Axes3D, R: np.ndarray):

        sx, sy, sz = 0.3, 0.2, 0.5  

        vertices = np.array([
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz]
        ])
        
        vertices = (R @ vertices.T).T
        
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
        ]
        
        colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#7f7f7f', '#ff7f0e']
        
        for face, color in zip(faces, colors):
            poly = Poly3DCollection([face], alpha=0.7)
            poly.set_facecolor(color)
            poly.set_edgecolor('black')
            ax.add_collection3d(poly)
    
    def plot_static(
        self,
        quaternion: np.ndarray,
        target_direction: np.ndarray,
        errors: Optional[List[float]] = None,
        title: str = "Satellite Attitude"
    ):
  
        self.create_figure()
        
        self.draw_satellite(self.ax_3d, quaternion, target_direction)
        self.ax_3d.set_title(title)
        
        if errors is not None:
            steps = range(len(errors))
            self.ax_error.plot(steps, errors, 'b-', linewidth=2)
            self.ax_error.axhline(y=1.0, color='g', linestyle='--', label='Target (1°)')
            self.ax_error.legend()
        
        plt.tight_layout()
        return self.fig
    
    def animate_episode(
        self,
        quaternions: List[np.ndarray],
        target_directions: List[np.ndarray],
        errors: List[float],
        interval: int = 50,
        save_path: Optional[str] = None
    ):

        self.create_figure()
        
        n_frames = len(quaternions)
        
        def update(frame):
            self.draw_satellite(
                self.ax_3d, 
                quaternions[frame], 
                target_directions[frame]
            )
            self.ax_3d.set_title(f'Step {frame} | Error: {errors[frame]:.1f}°')

            self.ax_error.cla()
            self.ax_error.plot(range(frame + 1), errors[:frame + 1], 'b-', linewidth=2)
            self.ax_error.axhline(y=1.0, color='g', linestyle='--', label='Target')
            self.ax_error.set_xlim([0, n_frames])
            self.ax_error.set_ylim([0, max(errors) * 1.1])
            self.ax_error.set_xlabel('Step')
            self.ax_error.set_ylabel('Angular Error (degrees)')
            self.ax_error.set_title('Pointing Error')
            self.ax_error.legend()
            self.ax_error.grid(True, alpha=0.3)
            
            return []
        
        anim = FuncAnimation(
            self.fig, update, frames=n_frames, 
            interval=interval, blit=False
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=20)
            print(f"Animation saved to {save_path}")
        
        return anim


def visualize_comparison(
    pid_results: Dict,
    rl_results: Dict,
    save_path: Optional[str] = None
):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PID vs RL Controller Comparison', fontsize=14)
    
    ax = axes[0, 0]
    ax.plot(pid_results['errors'], 'b-', label='PID', linewidth=2, alpha=0.8)
    ax.plot(rl_results['errors'], 'r-', label='RL', linewidth=2, alpha=0.8)
    ax.axhline(y=1.0, color='g', linestyle='--', label='Target (1°)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title('Angular Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    pid_actions = np.array(pid_results['actions'])
    rl_actions = np.array(rl_results['actions'])
    ax.plot(np.linalg.norm(pid_actions, axis=1), 'b-', label='PID', alpha=0.8)
    ax.plot(np.linalg.norm(rl_actions, axis=1), 'r-', label='RL', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action Magnitude')
    ax.set_title('Control Effort')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(pid_results['angular_vels'], 'b-', label='PID', alpha=0.8)
    ax.plot(rl_results['angular_vels'], 'r-', label='RL', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Angular Velocity Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Performance Comparison Summary

    Metric                 PID          RL
    Total Reward:     {pid_results['reward']:>8.1f}  {rl_results['reward']:>8.1f}
    Final Error:      {pid_results['final_error']:>8.1f}°  {rl_results['final_error']:>8.1f}°
    Min Error:        {pid_results['min_error']:>8.1f}°  {rl_results['min_error']:>8.1f}°
    Mean Error:       {pid_results['mean_error']:>8.1f}°  {rl_results['mean_error']:>8.1f}°
    Episode Length:   {pid_results['length']:>8d}  {rl_results['length']:>8d}
    Success:          {'Yes' if pid_results['success'] else 'No':>8}  {'Yes' if rl_results['success'] else 'No':>8}
    """
    
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
            fontsize=11, fontfamily='monospace', verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison saved to {save_path}")
    
    return fig


def plot_multiple_episodes(
    results_list: List[Dict],
    labels: List[str],
    save_path: Optional[str] = None
):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    ax = axes[0]
    for results, label, color in zip(results_list, labels, colors):
        ax.plot(results['errors'], label=label, color=color, alpha=0.8)
    ax.axhline(y=1.0, color='green', linestyle='--', label='Target')
    ax.set_xlabel('Step')
    ax.set_ylabel('Error (°)')
    ax.set_title('Angular Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    x = range(len(results_list))
    final_errors = [r['final_error'] for r in results_list]
    bars = ax.bar(x, final_errors, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Final Error (°)')
    ax.set_title('Final Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[2]
    rewards = [r['reward'] for r in results_list]
    bars = ax.bar(x, rewards, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Testing 3D Visualization")
    
    q = np.array([0.9, 0.2, 0.3, 0.1])
    q = q / np.linalg.norm(q)
    
    target = np.array([0, 0, 1])
    
    viz = SatelliteVisualizer()
    fig = viz.plot_static(
        quaternion=q,
        target_direction=target,
        errors=[90, 80, 60, 40, 20, 10, 5, 2, 1],
        title="Test Satellite Visualization"
    )
    
    plt.savefig('visualization_test.png', dpi=150)
    plt.close()
