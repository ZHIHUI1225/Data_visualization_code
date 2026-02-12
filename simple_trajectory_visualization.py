import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_robot_colors():
    """Define distinct colors for different robots"""
    return {
        0: np.array([0.2, 0.4, 0.8]),   # Blue
        1: np.array([0.8, 0.2, 0.2]),   # Red
        2: np.array([0.2, 0.8, 0.2]),   # Green
        3: np.array([0.8, 0.6, 0.2]),   # Orange
        4: np.array([0.6, 0.2, 0.8]),   # Purple
    }

def get_parcel_colors():
    """Define distinct colors for different parcels"""
    return {
        0: np.array([0.1, 0.3, 0.7]),   # Dark Blue
        1: np.array([0.7, 0.1, 0.1]),   # Dark Red
        2: np.array([0.1, 0.7, 0.1]),   # Dark Green
        3: np.array([0.7, 0.5, 0.1]),   # Dark Orange
        4: np.array([0.5, 0.1, 0.7]),   # Dark Purple
    }

def load_reference_trajectory(file_path):
    """Load tb trajectory data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['Trajectory'])

def load_robot_trajectory(file_path):
    """Load robot trajectory data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['data'])

def convert_robot_to_parcel_trajectory(trajectory, offset_distance=0.07):
    """
    Convert robot trajectory to parcel trajectory by offsetting forward along robot direction
    The parcel is 0.07m in front of the robot (same direction as robot's orientation)

    Args:
        trajectory: numpy array with columns [x, y, theta, v, omega]
        offset_distance: distance to offset forward (default 0.07m)

    Returns:
        numpy array with parcel positions [x_parcel, y_parcel, theta, v, omega]
    """
    if len(trajectory) < 2:
        return trajectory.copy()

    parcel_trajectory = trajectory.copy()

    # Extract robot positions and orientations
    x_robot = trajectory[:, 0]
    y_robot = trajectory[:, 1]
    theta = trajectory[:, 2]  # Use orientation directly from the data

    # Calculate parcel position by moving forward along robot's orientation
    # Parcel is in front of the robot, so add the offset
    x_parcel = x_robot + offset_distance * np.cos(theta)
    y_parcel = y_robot + offset_distance * np.sin(theta)

    # Update trajectory with parcel positions
    parcel_trajectory[:, 0] = x_parcel
    parcel_trajectory[:, 1] = y_parcel

    return parcel_trajectory

def plot_trajectories_by_parcel():
    """Plot trajectories for each parcel in separate figures"""
    base_path = "result/MAPS"

    # Find all MAP directories
    map_dirs = glob.glob(f"{base_path}/MAP*")

    for map_dir in sorted(map_dirs):
        map_name = Path(map_dir).name
        print(f"Processing {map_name}...")

        # Look for trajectory files in the nested map structure
        nested_map_dir = Path(map_dir) / map_name / map_name.lower()
        if not nested_map_dir.exists():
            print(f"No nested map directory found: {nested_map_dir}")
            continue

        # Check if we have robot trajectory files in the Push directory
        push_dir = nested_map_dir / "control_data" / "Push"
        if not push_dir.exists():
            print(f"No Push directory found: {push_dir}")
            continue

        # Find all robot trajectory files
        robot_files = glob.glob(str(push_dir / "robot*_parcel*_PushObject_robot_trajectory.json"))

        if not robot_files:
            print(f"No robot trajectory files found in {push_dir}")
            continue

        # Find all tb trajectory files for reference
        tb_files = glob.glob(str(nested_map_dir / "tb*_Trajectory.json"))

        # Group robot files by parcel_id
        parcel_data = {}
        for robot_file in robot_files:
            filename = os.path.basename(robot_file)
            parts = filename.split('_')
            robot_id = int(parts[0].replace('robot', ''))
            parcel_id = int(parts[1].replace('parcel', ''))

            if parcel_id not in parcel_data:
                parcel_data[parcel_id] = []
            parcel_data[parcel_id].append((robot_id, robot_file))

        # Get color maps
        robot_colors = get_robot_colors()
        parcel_colors = get_parcel_colors()

        # Create separate figure for each parcel
        for parcel_id, robot_list in parcel_data.items():
            fig, ax = plt.subplots(figsize=(12, 10))

            print(f"Creating figure for Parcel {parcel_id} in {map_name}")

            # Plot reference trajectories (tb files)
            for tb_file in sorted(tb_files):
                try:
                    filename = os.path.basename(tb_file)
                    robot_id = int(filename.replace('tb', '').replace('_Trajectory.json', ''))

                    # Only plot reference if this robot handles this parcel
                    if any(r_id == robot_id for r_id, _ in robot_list):
                        ref_trajectory = load_reference_trajectory(tb_file)
                        if len(ref_trajectory) > 0:
                            x_ref = ref_trajectory[:, 0]
                            y_ref = ref_trajectory[:, 1]
                            color = robot_colors.get(robot_id, np.array([0.5, 0.5, 0.5]))

                            ax.plot(x_ref, y_ref, color=color, linewidth=2, alpha=0.7,
                                   linestyle='--', label=f'Reference Robot {robot_id}')

                except Exception as e:
                    continue

            # Plot actual parcel trajectories only
            for robot_id, robot_file in robot_list:
                try:
                    # Load actual robot trajectory data
                    robot_trajectory = load_robot_trajectory(robot_file)
                    if len(robot_trajectory) == 0:
                        continue

                    # Remove timestamp column and get [x, y, theta, v, omega]
                    robot_pos = robot_trajectory[:, 1:]

                    # Calculate parcel trajectory (0.07m forward from robot)
                    parcel_trajectory = convert_robot_to_parcel_trajectory(robot_pos, 0.07)
                    x_parcel = parcel_trajectory[:, 0]
                    y_parcel = parcel_trajectory[:, 1]
                    parcel_color = parcel_colors.get(parcel_id, np.array([0.3, 0.3, 0.3]))

                    # Plot parcel trajectory (bold line)
                    ax.plot(x_parcel, y_parcel, color=parcel_color, linewidth=3, alpha=0.9,
                           label=f'Parcel {parcel_id} by Robot {robot_id}')

                    print(f"Plotted Parcel {parcel_id} trajectory from Robot {robot_id}")

                except Exception as e:
                    print(f"Error processing {robot_file}: {e}")
                    continue

            # Plot relay points (waypoints)
            waypoints_file = nested_map_dir / "Waypoints.json"
            if waypoints_file.exists():
                try:
                    with open(waypoints_file, 'r') as f:
                        waypoints_data = json.load(f)

                    # Plot waypoints as relay points
                    for i, waypoint in enumerate(waypoints_data):
                        x_wp = waypoint[0]
                        y_wp = waypoint[1]
                        ax.plot(x_wp, y_wp, 'ko', markersize=8, alpha=0.8,
                               markerfacecolor='yellow', markeredgewidth=2)
                        ax.text(x_wp, y_wp + 0.02, f'WP{i}', fontsize=8, ha='center',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

                    # Add to legend
                    ax.plot([], [], 'ko', markersize=8, markerfacecolor='yellow',
                           markeredgewidth=2, label='Waypoints')

                    print(f"Plotted {len(waypoints_data)} waypoints")

                except Exception as e:
                    print(f"Error loading waypoints: {e}")

            # Customize plot
            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)
            ax.set_title(f'Parcel {parcel_id} Trajectories - {map_name}\n'
                        f'Dashed: Reference, Solid: Actual Parcel Path (0.07m forward from robot)\n'
                        f'Yellow dots: Relay Points (Waypoints)',
                        fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

            plt.tight_layout()

            # Save figure for this parcel
            output_path = nested_map_dir / f'parcel_{parcel_id}_trajectories_{map_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved parcel {parcel_id} plot: {output_path}")
            plt.close(fig)

if __name__ == "__main__":
    plot_trajectories_by_parcel()