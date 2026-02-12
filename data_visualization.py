import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def get_parcel_base_colors():
    """Define distinct base colors for different parcels"""
    return {
        0: np.array([0.2, 0.4, 0.8]),   # Blue
        1: np.array([0.8, 0.2, 0.2]),   # Red  
        2: np.array([0.2, 0.8, 0.2]),   # Green
        3: np.array([0.8, 0.6, 0.2]),   # Orange
        4: np.array([0.6, 0.2, 0.8]),   # Purple
        5: np.array([0.8, 0.8, 0.2]),   # Yellow
        6: np.array([0.2, 0.8, 0.8]),   # Cyan
        7: np.array([0.8, 0.2, 0.6]),   # Magenta
        8: np.array([0.4, 0.8, 0.4]),   # Light Green
    }


def load_robot_trajectory(file_path):
    """Load robot trajectory data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['data'])

def load_control_commands(file_path):
    """Load control commands data from JSON file to get actual velocities"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # Format: [timestamp, velocity, angular_velocity]
        return np.array(data['data'])
    except (FileNotFoundError, KeyError):
        return None

def load_reference_trajectory(file_path):
    """Load tbi trajectory data from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['Trajectory'])

def load_environment(file_path):
    """Load environment data including obstacles from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def convert_obstacles_to_meters(polygons, pixel_to_meter_scale=0.0023):
    """Convert obstacle polygons from pixel coordinates to meters"""
    converted_polygons = []
    for polygon in polygons:
        vertices_meters = []
        for vertex in polygon['vertices']:
            x_meters = vertex[0] * pixel_to_meter_scale
            y_meters = vertex[1] * pixel_to_meter_scale
            vertices_meters.append([x_meters, y_meters])
        converted_polygons.append({'vertices': vertices_meters})
    return converted_polygons

def smooth_angle(angles, filter_type='gaussian', **kwargs):
    """
    Smooth angles while handling angle wrapping properly

    Args:
        angles: array of angles in radians
        filter_type: smoothing filter type
        **kwargs: filter parameters

    Returns:
        smoothed angles array
    """
    if len(angles) < 3:
        return angles.copy()

    # Convert to complex numbers to handle angle wrapping
    complex_angles = np.exp(1j * angles)

    if filter_type == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        # Apply Gaussian filter to real and imaginary parts separately
        real_smooth = gaussian_filter1d(complex_angles.real, sigma=sigma)
        imag_smooth = gaussian_filter1d(complex_angles.imag, sigma=sigma)
        complex_smooth = real_smooth + 1j * imag_smooth

    elif filter_type == 'moving_average':
        window = kwargs.get('window', 5)
        if window >= len(angles):
            window = len(angles) // 2 + 1

        # Apply moving average to real and imaginary parts
        kernel = np.ones(window) / window
        real_smooth = np.convolve(complex_angles.real, kernel, mode='same')
        imag_smooth = np.convolve(complex_angles.imag, kernel, mode='same')
        complex_smooth = real_smooth + 1j * imag_smooth

    elif filter_type == 'savgol':
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 3)

        # Ensure valid parameters
        if window_length >= len(angles):
            window_length = len(angles) // 2 * 2 - 1
        if window_length < 3:
            window_length = 3
        if window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1

        # Apply Savitzky-Golay filter to real and imaginary parts
        real_smooth = savgol_filter(complex_angles.real, window_length, polyorder)
        imag_smooth = savgol_filter(complex_angles.imag, window_length, polyorder)
        complex_smooth = real_smooth + 1j * imag_smooth

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Convert back to angles
    smoothed_angles = np.angle(complex_smooth)
    return smoothed_angles

def smooth_trajectory(trajectory, filter_type='gaussian', **kwargs):
    """
    Apply smoothing filter to trajectory data

    Args:
        trajectory: numpy array with columns [x, y, theta, v, omega]
        filter_type: 'gaussian', 'moving_average', or 'savgol'
        **kwargs: filter-specific parameters
                 - gaussian: sigma (default=1.0)
                 - moving_average: window (default=5)
                 - savgol: window_length (default=5), polyorder (default=3)

    Returns:
        smoothed trajectory with same shape as input
    """
    if len(trajectory) < 3:
        return trajectory.copy()

    smoothed = trajectory.copy()

    if filter_type == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        # Apply Gaussian filter to position
        smoothed[:, 0] = gaussian_filter1d(trajectory[:, 0], sigma=sigma)  # x
        smoothed[:, 1] = gaussian_filter1d(trajectory[:, 1], sigma=sigma)  # y
        # Use special angle smoothing for theta
        smoothed[:, 2] = smooth_angle(trajectory[:, 2], filter_type, **kwargs)

    elif filter_type == 'moving_average':
        window = kwargs.get('window', 5)
        if window >= len(trajectory):
            window = len(trajectory) // 2 + 1

        # Apply moving average to position
        smoothed[:, 0] = np.convolve(trajectory[:, 0], np.ones(window)/window, mode='same')  # x
        smoothed[:, 1] = np.convolve(trajectory[:, 1], np.ones(window)/window, mode='same')  # y
        # Use special angle smoothing for theta
        smoothed[:, 2] = smooth_angle(trajectory[:, 2], filter_type, **kwargs)

    elif filter_type == 'savgol':
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 3)

        # Ensure window_length is odd and valid
        if window_length >= len(trajectory):
            window_length = len(trajectory) // 2 * 2 - 1  # Make it odd and smaller
        if window_length < 3:
            window_length = 3
        if window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1

        # Apply Savitzky-Golay filter to position
        smoothed[:, 0] = savgol_filter(trajectory[:, 0], window_length, polyorder)  # x
        smoothed[:, 1] = savgol_filter(trajectory[:, 1], window_length, polyorder)  # y
        # Use special angle smoothing for theta
        smoothed[:, 2] = smooth_angle(trajectory[:, 2], filter_type, **kwargs)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Recalculate velocity and angular velocity from smoothed positions
    if len(smoothed) > 1:
        dt = 0.1  # Assume 0.1s timestep

        # Linear velocity from position differences
        dx = np.diff(smoothed[:, 0])
        dy = np.diff(smoothed[:, 1])
        v = np.sqrt(dx**2 + dy**2) / dt
        smoothed[1:, 3] = v  # Keep first velocity unchanged

        # Angular velocity from orientation differences
        dtheta = np.diff(smoothed[:, 2])
        # Handle angle wrapping
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        omega = dtheta / dt
        smoothed[1:, 4] = omega  # Keep first angular velocity unchanged

    return smoothed

def convert_robot_to_parcel_trajectory(trajectory, offset_distance=0.07,
                                      smooth_filter=None, **filter_kwargs):
    """
    Convert robot trajectory to parcel trajectory by offsetting forwards along trajectory direction
    Optionally applies smoothing filter before conversion

    Args:
        trajectory: numpy array with columns [x, y, theta, v, omega]
        offset_distance: distance to offset forwards (default 0.07m)
        smooth_filter: smoothing filter type ('gaussian', 'moving_average', 'savgol', or None)
        **filter_kwargs: parameters for the smoothing filter

    Returns:
        numpy array with parcel positions [x_parcel, y_parcel, theta, v, omega]
    """
    if len(trajectory) < 2:
        return trajectory.copy()

    # Apply smoothing filter if specified
    if smooth_filter is not None:
        smoothed_trajectory = smooth_trajectory(trajectory, smooth_filter, **filter_kwargs)
    else:
        smoothed_trajectory = trajectory.copy()

    parcel_trajectory = smoothed_trajectory.copy()

    # Extract robot positions and orientations
    x_robot = smoothed_trajectory[:, 0]
    y_robot = smoothed_trajectory[:, 1]
    theta = smoothed_trajectory[:, 2]  # Use orientation directly from the data

    # Calculate parcel position by moving forward along robot's orientation
    x_parcel = x_robot + offset_distance * np.cos(theta)
    y_parcel = y_robot + offset_distance * np.sin(theta)

    # Update trajectory with parcel positions
    parcel_trajectory[:, 0] = x_parcel
    parcel_trajectory[:, 1] = y_parcel

    return parcel_trajectory

# Trajectory smoothing configuration
SMOOTH_CONFIG = {
    'enable_smoothing': True,
    'filter_type': 'gaussian',  # 'gaussian', 'moving_average', 'savgol'
    'gaussian': {'sigma': 1.0},
    'moving_average': {'window': 5},
    'savgol': {'window_length': 5, 'polyorder': 3}
}

def plot_robot_trajectories():
    """Plot each robot's trajectories in separate figures"""
    base_path = "result/MAPS"
    
    # Find all success directories
    success_dirs = glob.glob(f"{base_path}/MAP*/MAP*/map*/control_data/Push")
    
    for success_dir in success_dirs:
        success_name = Path(success_dir).parents[2].name
        print(f"Processing {success_name}...")
        
        # Find all robot trajectory files
        robot_files = glob.glob(f"{success_dir}/robot*_parcel*_PushObject_robot_trajectory.json")
        
        # Group files by robot_id
        robot_data = {}
        for robot_file in sorted(robot_files):
            filename = os.path.basename(robot_file)
            parts = filename.split('_')
            robot_id = int(parts[0].replace('robot', ''))
            parcel_id = int(parts[1].replace('parcel', ''))
            
            if robot_id not in robot_data:
                robot_data[robot_id] = []
            robot_data[robot_id].append((parcel_id, robot_file))
        
        # Load reference trajectories from experi directory
        # Use Path to properly handle directory navigation
        push_path = Path(success_dir)
        experi_dir = push_path.parent.parent  # Go up from Push -> control_data -> experi
        print(f"Looking for tbi files in: {experi_dir}")
        tbi_files = glob.glob(str(experi_dir / "tb*_Trajectory.json"))
        print(f"Found tbi files: {tbi_files}")
        tbi_data = {}
        
        for tbi_file in sorted(tbi_files):
            filename = os.path.basename(tbi_file)
            robot_id = int(filename.replace('tb', '').replace('_Trajectory.json', ''))
            tbi_data[robot_id] = tbi_file
        
        # Get the map directory for saving figures
        map_dir = push_path.parent.parent  # Go up from Push -> control_data -> map directory

        # Create separate figure for each robot
        for robot_id, parcel_files in robot_data.items():
            fig, ax = plt.subplots(figsize=(12, 10))

            # Get parcel color map
            parcel_colors = get_parcel_base_colors()
            
            # Plot all parcels for this robot
            for i, (parcel_id, robot_file) in enumerate(parcel_files):
                # Load robot trajectory data [timestamp, x, y, theta, v, omega]
                robot_trajectory = load_robot_trajectory(robot_file)

                if len(robot_trajectory) == 0:
                    continue

                # Convert robot trajectory to parcel trajectory with optional smoothing
                if SMOOTH_CONFIG['enable_smoothing']:
                    filter_type = SMOOTH_CONFIG['filter_type']
                    filter_params = SMOOTH_CONFIG[filter_type]
                    parcel_trajectory = convert_robot_to_parcel_trajectory(
                        robot_trajectory[:, 1:], 0.07,
                        smooth_filter=filter_type, **filter_params
                    )
                    print(f"Applied {filter_type} smoothing to {os.path.basename(robot_file)}")
                else:
                    parcel_trajectory = convert_robot_to_parcel_trajectory(robot_trajectory[:, 1:], 0.07)

                x = parcel_trajectory[:, 0]  # parcel x coordinates
                y = parcel_trajectory[:, 1]  # parcel y coordinates
                
                # Get base color for this parcel
                base_color = parcel_colors.get(parcel_id, np.array([0.5, 0.5, 0.5]))
                
                # Plot trajectory with consistent color
                if len(x) > 1:
                    ax.plot(x, y, color=base_color, linewidth=3, alpha=0.8, label=f'Parcel {parcel_id}')
            
            # Plot reference trajectory for this robot
            if robot_id in tbi_data:
                try:
                    tbi_trajectory = load_reference_trajectory(tbi_data[robot_id])

                    # Use tbi_Trajectory directly as it contains parcel positions
                    x_ref = tbi_trajectory[:, 0]  # parcel x coordinates from reference
                    y_ref = tbi_trajectory[:, 1]  # parcel y coordinates from reference
                    
                    # Plot reference trajectory with consistent color
                    if len(x_ref) > 1:
                        ref_base_color = np.array([0.2, 0.8, 0.8])  # Cyan base color for reference

                        # Plot reference trajectory with dashed line
                        ax.plot(x_ref, y_ref, color=ref_base_color, linewidth=2.5,
                               linestyle='--', alpha=0.8, label=f'Reference TB{robot_id}')
                        
                        # Add reference start/end markers
                        ax.plot(x_ref[0], y_ref[0], '*', color=ref_base_color, markersize=10, 
                               markeredgecolor='black', markeredgewidth=1, alpha=0.9)
                        ax.plot(x_ref[-1], y_ref[-1], '*', color=ref_base_color, markersize=10, 
                               markeredgecolor='black', markeredgewidth=1, alpha=0.9)
                    
                except Exception as e:
                    print(f"Could not load reference for robot {robot_id}: {e}")
            
            # Customize plot
            ax.set_xlabel('X Position (m)', fontsize=12)
            ax.set_ylabel('Y Position (m)', fontsize=12)
            ax.set_title(f'Robot {robot_id} Parcel Trajectories - {success_name}\nSolid Lines: Actual Parcel Paths (from robot+0.07m), Dashed: Reference Parcel Paths',
                        fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')  # Keep same scale for both axes
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
            
            
            # # Add speed intensity explanation
            # speed_text = ("Line Color Guide:\n"
            #              "• Each parcel = Different base color\n"
            #              "• Darker lines = Higher speeds\n" 
            #              "• Lighter lines = Lower speeds\n"
            #              "• Circle = Start, Square = End\n"
            #              "• Dashed = Reference trajectory")
            # ax.text(0.02, 0.02, speed_text, transform=ax.transAxes, 
            #        fontsize=9, verticalalignment='bottom',
            #        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()

            # Save figure in the map directory
            output_path = map_dir / f'robot_{robot_id}_trajectories_{success_name}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved trajectory plot: {output_path}")
            plt.close(fig)  # Close figure to free memory

def plot_all_trajectories_combined():
    """Plot all robot trajectories in a single combined figure"""
    base_path = "result/MAPS"

    # Find all success directories
    success_dirs = glob.glob(f"{base_path}/MAP*/MAP*/map*/control_data/Push")

    for success_dir in success_dirs:
        success_name = Path(success_dir).parents[2].name
        print(f"Processing combined plot for {success_name}...")

        # Find all robot trajectory files
        robot_files = glob.glob(f"{success_dir}/robot*_parcel*_PushObject_robot_trajectory.json")

        # Group files by robot_id
        robot_data = {}
        for robot_file in sorted(robot_files):
            filename = os.path.basename(robot_file)
            parts = filename.split('_')
            robot_id = int(parts[0].replace('robot', ''))
            parcel_id = int(parts[1].replace('parcel', ''))

            if robot_id not in robot_data:
                robot_data[robot_id] = []
            robot_data[robot_id].append((parcel_id, robot_file))

        # Load reference trajectories from experi directory
        push_path = Path(success_dir)
        experi_dir = push_path.parent.parent
        tbi_files = glob.glob(str(experi_dir / "tb*_Trajectory.json"))
        tbi_data = {}

        for tbi_file in sorted(tbi_files):
            filename = os.path.basename(tbi_file)
            robot_id = int(filename.replace('tb', '').replace('_Trajectory.json', ''))
            tbi_data[robot_id] = tbi_file

        # Get the map directory for saving figures
        map_dir = push_path.parent.parent

        # Load environment data (obstacles)
        map_name = success_name.lower()  # Convert to lowercase for file matching
        env_file = f"result/MAPS/{success_name.upper()}/environment_{map_name}.json"
        try:
            env_data = load_environment(env_file)
            obstacles = convert_obstacles_to_meters(env_data['polygons'])
            print(f"Loaded {len(obstacles)} obstacles from {env_file}")
        except FileNotFoundError:
            print(f"Environment file not found: {env_file}")
            obstacles = []

        # Create single combined figure
        fig, ax = plt.subplots(figsize=(16, 12))

        # Get parcel color map
        parcel_colors = get_parcel_base_colors()

        # Track which labels we've already added to avoid duplicates
        added_labels = set()

        # Plot all robots and parcels on the same figure
        for robot_id, parcel_files in robot_data.items():
            # Plot all parcels for this robot
            for i, (parcel_id, robot_file) in enumerate(parcel_files):
                # Load robot trajectory data
                robot_trajectory = load_robot_trajectory(robot_file)

                if len(robot_trajectory) == 0:
                    continue

                # Convert robot trajectory to parcel trajectory with optional smoothing
                if SMOOTH_CONFIG['enable_smoothing']:
                    filter_type = SMOOTH_CONFIG['filter_type']
                    filter_params = SMOOTH_CONFIG[filter_type]
                    parcel_trajectory = convert_robot_to_parcel_trajectory(
                        robot_trajectory[:, 1:], 0.07,
                        smooth_filter=filter_type, **filter_params
                    )
                else:
                    parcel_trajectory = convert_robot_to_parcel_trajectory(robot_trajectory[:, 1:], 0.07)

                x = parcel_trajectory[:, 0]
                y = parcel_trajectory[:, 1]

                # Get base color for this parcel
                base_color = parcel_colors.get(parcel_id, np.array([0.5, 0.5, 0.5]))

                # Plot trajectory with consistent color
                if len(x) > 1:
                    # Only add label if we haven't seen this parcel ID before
                    label = f'Parcel {parcel_id}' if f'Parcel {parcel_id}' not in added_labels else None
                    if label:
                        added_labels.add(label)

                    ax.plot(x, y, color=base_color, linewidth=2.5, alpha=0.8, label=label)

            # Plot reference trajectory for this robot
            if robot_id in tbi_data:
                try:
                    tbi_trajectory = load_reference_trajectory(tbi_data[robot_id])

                    x_ref = tbi_trajectory[:, 0]
                    y_ref = tbi_trajectory[:, 1]

                    # Plot reference trajectory with consistent color
                    if len(x_ref) > 1:
                        ref_base_color = np.array([0.2, 0.8, 0.8])  # Cyan base color for reference

                        # Plot reference trajectory with dashed line
                        # Only add label for first reference trajectory
                        ref_label = 'Reference' if 'Reference' not in added_labels else None
                        if ref_label:
                            added_labels.add(ref_label)

                        ax.plot(x_ref, y_ref, color=ref_base_color, linewidth=2,
                               linestyle='--', alpha=0.6, label=ref_label)

                        # Add reference start/end markers
                        ax.plot(x_ref[0], y_ref[0], '*', color=ref_base_color, markersize=8,
                               markeredgecolor='black', markeredgewidth=1, alpha=0.7)
                        ax.plot(x_ref[-1], y_ref[-1], '*', color=ref_base_color, markersize=8,
                               markeredgecolor='black', markeredgewidth=1, alpha=0.7)

                except Exception as e:
                    print(f"Could not load reference for robot {robot_id}: {e}")

        # Plot obstacles
        for i, obstacle in enumerate(obstacles):
            vertices = np.array(obstacle['vertices'])

            # Create polygon patch
            poly = Polygon(vertices, closed=True, fill=True,
                          facecolor='gray', alpha=0.3,
                          edgecolor='black', linewidth=2)
            ax.add_patch(poly)

            # Add label only for first obstacle to avoid legend clutter
            if i == 0:
                ax.plot([], [], 'k-', linewidth=2, alpha=0.8, label='Obstacles')

        # Customize plot
        ax.set_xlabel('X Position (m)', fontsize=14)
        ax.set_ylabel('Y Position (m)', fontsize=14)
        ax.set_title(f'All Robot Trajectories Combined - {success_name}\nSolid Lines: Actual Parcel Paths, Dashed: Reference Paths',
                    fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

        plt.tight_layout()

        # Save combined figure
        output_path = map_dir / f'all_trajectories_combined_{success_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined trajectory plot: {output_path}")
        plt.close(fig)

def plot_reeb_graph_with_environment():
    """
    Plot Reeb Graph overlaid on the environment map with start and goal positions
    Similar to the visualization in Graph_map1_visualization.png
    """
    # File paths
    graph_file = "result/MAPS/MAP1/Graph_map1.json"
    env_file = "result/MAPS/MAP1/environment_map1.json"
    output_file = "result/MAPS/MAP1/reeb_graph_environment.png"

    # Load data
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)

    with open(env_file, 'r') as f:
        env_data = json.load(f)

    # Extract data
    nodes = graph_data['nodes']  # [[id, [x, y], null, false], ...]
    out_neighbors = graph_data['out_neighbors']  # {'0': [2], '1': [2], ...}
    start_pose = graph_data['start_pose']  # [x, y, theta]
    goal_pose = graph_data['goal_pose']  # [x, y, theta]
    polygons = env_data['polygons']  # List of obstacle polygons
    width = env_data['width']
    height = env_data['height']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot obstacles (environment)
    for polygon in polygons:
        vertices = np.array(polygon['vertices'])
        poly = Polygon(vertices, closed=True, fill=True,
                      facecolor='lightgray', alpha=0.5,
                      edgecolor='black', linewidth=2)
        ax.add_patch(poly)

    # Plot Reeb Graph edges
    for node_id_str, neighbors in out_neighbors.items():
        node_id = int(node_id_str)
        node_pos = nodes[node_id][1]  # [x, y]

        for neighbor_id in neighbors:
            neighbor_pos = nodes[neighbor_id][1]

            # Draw edge as purple line
            ax.plot([node_pos[0], neighbor_pos[0]],
                   [node_pos[1], neighbor_pos[1]],
                   color='purple', linewidth=2.5, alpha=0.8, zorder=2)

    # Plot Reeb Graph nodes
    node_positions = np.array([node[1] for node in nodes])
    ax.scatter(node_positions[:, 0], node_positions[:, 1],
              c='purple', s=100, zorder=3, edgecolors='black', linewidths=1.5,
              label='Graph Nodes')

    # Add node labels
    for node in nodes:
        node_id = node[0]
        x, y = node[1]
        ax.text(x, y + 15, str(node_id), fontsize=9, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot start position
    ax.scatter(start_pose[0], start_pose[1], c='green', s=200, marker='o',
              edgecolors='black', linewidths=2, zorder=4, label='Start')

    # Plot goal position
    ax.scatter(goal_pose[0], goal_pose[1], c='red', s=200, marker='o',
              edgecolors='black', linewidths=2, zorder=4, label='Goal')

    # Add start/goal orientation arrows
    arrow_length = 40
    ax.arrow(start_pose[0], start_pose[1],
            arrow_length * np.cos(start_pose[2]),
            arrow_length * np.sin(start_pose[2]),
            head_width=15, head_length=20, fc='green', ec='black', linewidth=1.5, zorder=5)

    ax.arrow(goal_pose[0], goal_pose[1],
            arrow_length * np.cos(goal_pose[2]),
            arrow_length * np.sin(goal_pose[2]),
            head_width=15, head_length=20, fc='red', ec='black', linewidth=1.5, zorder=5)

    # Customize plot
    ax.set_xlim(-50, width + 50)
    ax.set_ylim(-50, height + 50)
    ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)
    ax.set_title('Reeb Graph with Environment (MAP1)\nPurple: Graph Structure | Green: Start | Red: Goal',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # Add coordinate frame info
    info_text = (f"Coordinate Frame: {graph_data['coordinate_frame']}\n"
                f"Graph Type: {graph_data['generation_info']['graph_type']}\n"
                f"Nodes: {len(nodes)} | Obstacles: {len(polygons)}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved Reeb Graph visualization: {output_file}")
    plt.show()

if __name__ == "__main__":
    # Uncomment the functions you want to run:

    # Plot individual robot trajectories
    # plot_robot_trajectories()

    # Plot all trajectories combined
    # plot_all_trajectories_combined()

    # Plot Reeb Graph with environment
    plot_reeb_graph_with_environment()