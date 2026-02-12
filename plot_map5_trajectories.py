#!/usr/bin/env python3
"""
Plot MAP5 trajectories: For each robot-parcel combination, plot:
- Reference trajectory (from tb*_Trajectory.json)
- Approach phase trajectory (from control_data/approach/)
- Push phase trajectory (from control_data/Push/)

Each robot-parcel combination gets its own figure.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_reference_trajectory(file_path):
    """Load reference trajectory (tb*_Trajectory.json)"""
    data = load_json(file_path)
    # Format: [x, y, theta, v, omega]
    return np.array(data['Trajectory'])


def load_real_trajectory(file_path):
    """Load real trajectory from control_data"""
    data = load_json(file_path)
    # Format: [timestamp, x, y, theta, v, omega]
    traj_data = np.array(data['data'])
    # Remove timestamp column
    return traj_data[:, 1:]


def convert_robot_to_parcel_trajectory(robot_trajectory, offset_distance=0.07, smooth_theta=True, sigma=2.0):
    """
    Convert robot trajectory to parcel trajectory by offsetting forward

    For pushing: parcel is in front of robot by offset_distance
    Uses robot's orientation (theta) with optional smoothing

    Args:
        robot_trajectory: numpy array [x, y, theta, v, omega]
        offset_distance: distance to offset forward (default 0.07m)
        smooth_theta: apply Gaussian smoothing to theta (default True)
        sigma: Gaussian filter sigma for theta smoothing (default 2.0)

    Returns:
        parcel_trajectory: numpy array [x, y, theta, v, omega]
    """
    parcel_trajectory = robot_trajectory.copy()

    if len(robot_trajectory) < 1:
        return parcel_trajectory

    x_robot = robot_trajectory[:, 0]
    y_robot = robot_trajectory[:, 1]
    theta_robot = robot_trajectory[:, 2]

    # Apply smoothing to theta to reduce noise
    if smooth_theta and len(theta_robot) > 3:
        try:
            from scipy.ndimage import gaussian_filter1d
            # Handle angle wrapping: convert to complex representation for smooth filtering
            theta_complex = np.exp(1j * theta_robot)
            theta_real_smooth = gaussian_filter1d(theta_complex.real, sigma=sigma, mode='nearest')
            theta_imag_smooth = gaussian_filter1d(theta_complex.imag, sigma=sigma, mode='nearest')
            theta_smooth = np.arctan2(theta_imag_smooth, theta_real_smooth)
        except ImportError:
            # Fallback: simple moving average
            window = min(5, len(theta_robot))
            theta_smooth = np.convolve(theta_robot, np.ones(window)/window, mode='same')
    else:
        theta_smooth = theta_robot

    # Calculate parcel position by moving forward along robot's heading
    x_parcel = x_robot + offset_distance * np.cos(theta_smooth)
    y_parcel = y_robot + offset_distance * np.sin(theta_smooth)

    parcel_trajectory[:, 0] = x_parcel
    parcel_trajectory[:, 1] = y_parcel

    return parcel_trajectory


def plot_robot_parcel_trajectories():
    """Plot trajectories for each robot-parcel combination"""

    base_path = Path("d:/Data_visualization_code/result/MAPS/MAP5")
    map5_data_path = base_path / "map5" / "map5"
    control_data_path = map5_data_path / "control_data"

    approach_path = control_data_path / "approach"
    push_path = control_data_path / "Push"

    # Find all approach trajectory files
    approach_pattern = str(approach_path / "robot*_parcel*_approach_robot_trajectory.json")
    approach_files = sorted(glob.glob(approach_pattern))

    print(f"Found {len(approach_files)} approach trajectory files")

    # Process each robot-parcel combination
    for approach_file in approach_files:
        filename = Path(approach_file).stem
        # Extract robot_id and parcel_id: robot{X}_parcel{Y}_approach_robot_trajectory
        robot_id = int(filename.split('_')[0].replace('robot', ''))
        parcel_id = int(filename.split('_')[1].replace('parcel', ''))

        print(f"  Processing Robot {robot_id} - Parcel {parcel_id}...")

        # Load reference trajectory for this robot (tb{robot_id}_Trajectory.json)
        ref_file = map5_data_path / f"tb{robot_id}_Trajectory.json"

        if not ref_file.exists():
            print(f"    Warning: Reference trajectory not found for robot {robot_id}")
            continue

        # Load reference trajectory (already in robot coordinates)
        ref_robot_traj = load_reference_trajectory(str(ref_file))

        # Find corresponding Push file
        push_file = push_path / f"robot{robot_id}_parcel{parcel_id}_PushObject_robot_trajectory.json"

        # Create figure for this robot-parcel combination
        fig, ax = plt.subplots(figsize=(4, 4))

        # Plot reference trajectory (robot only) - simple line
        ax.plot(ref_robot_traj[:, 0], ref_robot_traj[:, 1],
               'k-', linewidth=4.0, alpha=0.8, label='Reference', zorder=1)

        # Mark reference start and end - no edge
        ax.plot(ref_robot_traj[0, 0], ref_robot_traj[0, 1],
               'ko', markersize=8, markeredgewidth=0, zorder=5)
        ax.plot(ref_robot_traj[-1, 0], ref_robot_traj[-1, 1],
               'ks', markersize=8, markeredgewidth=0, zorder=5)

        # Load and plot approach trajectory - simple blue line
        try:
            approach_traj = load_real_trajectory(approach_file)
            if len(approach_traj) >= 2:
                ax.plot(approach_traj[:, 0], approach_traj[:, 1],
                       'b-', linewidth=3.5, alpha=0.8, label='Approach(robot)', zorder=3)
                ax.plot(approach_traj[0, 0], approach_traj[0, 1],
                       'bo', markersize=7, markeredgewidth=0, zorder=4)
                ax.plot(approach_traj[-1, 0], approach_traj[-1, 1],
                       'bs', markersize=7, markeredgewidth=0, zorder=4)
                print(f"    Approach: {len(approach_traj)} points")
        except Exception as e:
            print(f"    Error loading approach trajectory: {e}")

        # Load and plot Push trajectory - convert robot to parcel trajectory
        if push_file.exists():
            try:
                push_robot_traj = load_real_trajectory(str(push_file))
                if len(push_robot_traj) >= 2:
                    # Convert robot trajectory to parcel trajectory (offset forward 0.07m)
                    push_parcel_traj = convert_robot_to_parcel_trajectory(push_robot_traj, 0.07)

                    ax.plot(push_parcel_traj[:, 0], push_parcel_traj[:, 1],
                           'r-', linewidth=3.5, alpha=0.8, label='Push(parcel)', zorder=3)
                    ax.plot(push_parcel_traj[0, 0], push_parcel_traj[0, 1],
                           'ro', markersize=7, markeredgewidth=0, zorder=4)
                    ax.plot(push_parcel_traj[-1, 0], push_parcel_traj[-1, 1],
                           'rs', markersize=7, markeredgewidth=0, zorder=4)
                    print(f"    Push: {len(push_parcel_traj)} points (converted to parcel trajectory)")
            except Exception as e:
                print(f"    Error loading Push trajectory: {e}")
        else:
            print(f"    Warning: No Push trajectory found")

        # Customize plot - larger fonts
        ax.set_title(f'Robot {robot_id}',
                    fontsize=20, pad=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal')  # Equal aspect ratio

        # Expand axis limits for robot2 to show markers completely
        if robot_id == 2:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_margin = (xlim[1] - xlim[0]) * 0.05
            y_margin = (ylim[1] - ylim[0]) * 0.30
            ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
            ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)

            # Reduce y-axis tick density to avoid overlap
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=2))

        ax.tick_params(axis='both', which='major', labelsize=16)

        # Add legend outside the plot area
        # ax.legend(fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)

        # Save figure
        output_file = map5_data_path / f"robot{robot_id}_parcel{parcel_id}_approach_push.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file.name}")
        plt.close()

    print(f"\n{'='*60}")
    print("All trajectory plots completed!")


if __name__ == "__main__":
    plot_robot_parcel_trajectories()
