#!/usr/bin/env python3
"""
Plot reference trajectories for both robot and parcel positions.
Shows the relationship between parcel trajectory (from tbi_Trajectory.json)
and the corresponding robot trajectory (offset 0.07m backward).
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import os

def load_reference_trajectory(file_path):
    """Load tbi trajectory data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['Trajectory'])

def convert_parcel_to_robot_trajectory(trajectory, offset_distance=0.07):
    """
    Convert parcel trajectory to robot trajectory by offsetting backwards along trajectory direction

    Args:
        trajectory: numpy array with columns [x, y, theta, v, omega]
        offset_distance: distance to offset backwards (default 0.07m)

    Returns:
        numpy array with robot positions [x_robot, y_robot, theta, v, omega]
    """
    robot_trajectory = trajectory.copy()

    if len(trajectory) < 2:
        return robot_trajectory

    # Extract parcel positions
    x_parcel = trajectory[:, 0]
    y_parcel = trajectory[:, 1]

    # Calculate trajectory direction (tangent) for smoother offset
    # Use forward differences for interior points, backward for endpoints
    dx = np.zeros_like(x_parcel)
    dy = np.zeros_like(y_parcel)

    # Forward difference for first point
    dx[0] = x_parcel[1] - x_parcel[0]
    dy[0] = y_parcel[1] - y_parcel[0]

    # Central difference for interior points (smoother)
    for i in range(1, len(x_parcel) - 1):
        dx[i] = (x_parcel[i+1] - x_parcel[i-1]) / 2.0
        dy[i] = (y_parcel[i+1] - y_parcel[i-1]) / 2.0

    # Backward difference for last point
    dx[-1] = x_parcel[-1] - x_parcel[-2]
    dy[-1] = y_parcel[-1] - y_parcel[-2]

    # Calculate trajectory direction angles
    traj_angles = np.arctan2(dy, dx)

    # Apply smoothing to the trajectory angles to reduce noise
    try:
        from scipy.ndimage import gaussian_filter1d
        # Apply gentle smoothing (sigma=1.0 for mild smoothing)
        traj_angles_smooth = gaussian_filter1d(traj_angles, sigma=1.0, mode='nearest')
    except ImportError:
        # Fallback: simple moving average if scipy not available
        window_size = 3
        traj_angles_smooth = np.convolve(traj_angles, np.ones(window_size)/window_size, mode='same')

    # Calculate robot position by moving backward along smoothed trajectory direction
    x_robot = x_parcel - offset_distance * np.cos(traj_angles_smooth)
    y_robot = y_parcel - offset_distance * np.sin(traj_angles_smooth)

    # Update trajectory with robot positions
    robot_trajectory[:, 0] = x_robot
    robot_trajectory[:, 1] = y_robot

    return robot_trajectory

def apply_speed_intensity(base_color, speed, min_speed, max_speed):
    """Apply speed-based intensity to base color (darker = faster, lighter = slower)"""
    if max_speed <= min_speed:
        intensity = 0.8  # Default intensity
    else:
        # Normalize speed to [0, 1]
        normalized_speed = np.clip((speed - min_speed) / (max_speed - min_speed), 0.0, 1.0)
        # Map to intensity range: 0.1 (very light) to 1.2 (very dark/saturated)
        intensity = 0.1 + (1.1 * normalized_speed)

    # Apply intensity with saturation boost for high speeds
    color = base_color * intensity
    # Cap values at 1.0 to prevent overflow, but allow higher saturation
    return np.clip(color, 0.0, 1.0)

def plot_reference_trajectories():
    """Plot reference trajectories for both robot and parcel positions"""
    base_path = "result/MAPS"

    # Find all map directories
    map_dirs = glob.glob(f"{base_path}/MAP*/MAP*/map*")

    for map_dir in map_dirs:
        map_path = Path(map_dir)
        map_name = map_path.name
        print(f"Processing {map_name}...")

        # Find all tbi_Trajectory files
        tbi_files = glob.glob(str(map_path / "tb*_Trajectory.json"))

        if not tbi_files:
            print(f"No tbi_Trajectory files found in {map_dir}")
            continue

        # Create figure for this map
        fig, ax = plt.subplots(figsize=(14, 10))

        # Process each robot's reference trajectory
        for tbi_file in sorted(tbi_files):
            filename = os.path.basename(tbi_file)
            robot_id = int(filename.replace('tb', '').replace('_Trajectory.json', ''))

            try:
                # Load parcel trajectory
                parcel_trajectory = load_reference_trajectory(tbi_file)

                if len(parcel_trajectory) < 2:
                    continue

                # Convert to robot trajectory
                robot_trajectory = convert_parcel_to_robot_trajectory(parcel_trajectory, 0.07)

                # Extract positions and velocities
                x_parcel = parcel_trajectory[:, 0]
                y_parcel = parcel_trajectory[:, 1]
                x_robot = robot_trajectory[:, 0]
                y_robot = robot_trajectory[:, 1]
                v_parcel = np.abs(parcel_trajectory[:, 3])

                # Calculate velocity range for color mapping
                min_speed = np.min(v_parcel) if len(v_parcel) > 0 else 0.0
                max_speed = np.max(v_parcel) if len(v_parcel) > 0 else 0.05

                # Define colors for this robot
                parcel_base_color = np.array(plt.cm.tab10(robot_id % 10)[:3])  # Get RGB only as numpy array
                robot_base_color = parcel_base_color * 0.7  # Darker version for robot

                # Plot parcel trajectory with speed-based coloring
                if len(x_parcel) > 1:
                    # Create line segments for parcel
                    parcel_points = np.array([x_parcel, y_parcel]).T.reshape(-1, 1, 2)
                    parcel_segments = np.concatenate([parcel_points[:-1], parcel_points[1:]], axis=1)

                    # Calculate colors for each parcel segment
                    parcel_colors = []
                    for j in range(len(parcel_segments)):
                        avg_speed = (v_parcel[j] + v_parcel[j+1]) / 2 if j+1 < len(v_parcel) else v_parcel[j]
                        speed_color = apply_speed_intensity(parcel_base_color, avg_speed, min_speed, max_speed)
                        parcel_colors.append(speed_color)

                    # Create LineCollection for parcel trajectory
                    parcel_lc = LineCollection(parcel_segments, colors=parcel_colors,
                                             linewidths=3, alpha=0.8, linestyles='solid')
                    ax.add_collection(parcel_lc)

                # Plot robot trajectory with speed-based coloring
                if len(x_robot) > 1:
                    # Create line segments for robot
                    robot_points = np.array([x_robot, y_robot]).T.reshape(-1, 1, 2)
                    robot_segments = np.concatenate([robot_points[:-1], robot_points[1:]], axis=1)

                    # Calculate colors for each robot segment (same velocity as parcel)
                    robot_colors = []
                    for j in range(len(robot_segments)):
                        avg_speed = (v_parcel[j] + v_parcel[j+1]) / 2 if j+1 < len(v_parcel) else v_parcel[j]
                        speed_color = apply_speed_intensity(robot_base_color, avg_speed, min_speed, max_speed)
                        robot_colors.append(speed_color)

                    # Create LineCollection for robot trajectory
                    robot_lc = LineCollection(robot_segments, colors=robot_colors,
                                            linewidths=2.5, alpha=0.8, linestyles='dashed')
                    ax.add_collection(robot_lc)

                # Add start and end markers for parcel
                ax.plot(x_parcel[0], y_parcel[0], 'o', color=parcel_base_color, markersize=10,
                       markeredgecolor='black', markeredgewidth=1, alpha=0.9, label=f'Parcel TB{robot_id} Start')
                ax.plot(x_parcel[-1], y_parcel[-1], 's', color=parcel_base_color, markersize=10,
                       markeredgecolor='black', markeredgewidth=1, alpha=0.9)

                # Add start and end markers for robot
                ax.plot(x_robot[0], y_robot[0], '^', color=robot_base_color, markersize=8,
                       markeredgecolor='black', markeredgewidth=1, alpha=0.9, label=f'Robot {robot_id} Start')
                ax.plot(x_robot[-1], y_robot[-1], 'v', color=robot_base_color, markersize=8,
                       markeredgecolor='black', markeredgewidth=1, alpha=0.9)

                # Add trajectory lines to legend
                ax.plot([], [], color=parcel_base_color, linewidth=3, linestyle='-',
                       alpha=0.8, label=f'Parcel TB{robot_id}')
                ax.plot([], [], color=robot_base_color, linewidth=2.5, linestyle='--',
                       alpha=0.8, label=f'Robot {robot_id}')

                # Add robot ID annotation at start
                ax.annotate(f'R{robot_id}',
                           (x_robot[0], y_robot[0]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

                print(f"  Processed robot {robot_id}: {len(parcel_trajectory)} points, "
                      f"speed range {min_speed:.4f}-{max_speed:.4f} m/s")

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue

        # Customize plot
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'Reference Trajectories - {map_name}\n'
                    f'Solid Lines: Parcel Trajectories, Dashed Lines: Robot Trajectories (0.07m offset)\n'
                    f'Color Intensity: Dark=Fast, Light=Slow', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')  # Keep same scale for both axes
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

        # Add explanation text
        explanation_text = ("Trajectory Guide:\n"
                          "• Solid lines = Parcel trajectories (tbi_Trajectory)\n"
                          "• Dashed lines = Robot trajectories (0.07m behind parcel)\n"
                          "• Circle/Triangle = Start, Square/Inverted Triangle = End\n"
                          "• Color intensity = Speed (Dark=Fast, Light=Slow)")
        ax.text(0.02, 0.02, explanation_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()

        # Save figure in the map directory
        output_path = map_path / f'reference_trajectories_{map_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reference trajectory plot: {output_path}")
        plt.close(fig)

if __name__ == "__main__":
    plot_reference_trajectories()