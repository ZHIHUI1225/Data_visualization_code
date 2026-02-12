#!/usr/bin/env python3
"""
Plot pickup phase trajectories: For each robot, plot:
- Reference trajectory (from tb*_Trajectory.json)
- Real pickup trajectory (from control_data/pickup/)

Each robot gets its own figure.
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_reference_trajectory(file_path):
    """Load reference trajectory (tb*_Trajectory.json)"""
    data = load_json(file_path)
    # Format: [x, y, theta, v, omega]
    dt = data.get('dt', 0.1)  # Time step, default 0.1s
    total_time = data.get('total_time', None)  # Total time from file
    return np.array(data['Trajectory']), dt, total_time


def load_real_trajectory(file_path):
    """Load real trajectory from control_data"""
    data = load_json(file_path)
    # Format: [timestamp, x, y, theta, v, omega]
    traj_data = np.array(data['data'])
    # Return full data including timestamps
    return traj_data


def calculate_velocity_from_position(trajectory, dt=None, use_timestamp=False):
    """
    Calculate velocity from position data

    Args:
        trajectory: numpy array with positions [x, y, ...] or [timestamp, x, y, ...]
        dt: time step (required if use_timestamp=False)
        use_timestamp: if True, calculate dt from timestamps in column 0

    Returns:
        velocities: numpy array of linear velocities (m/s)
    """
    if use_timestamp:
        # Extract timestamps and positions
        timestamps = trajectory[:, 0]
        x = trajectory[:, 1]
        y = trajectory[:, 2]
    else:
        # Positions are in first two columns
        x = trajectory[:, 0]
        y = trajectory[:, 1]

    velocities = []

    for i in range(len(x)):
        if i == 0:
            # First point: use velocity to next point
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            if use_timestamp:
                dt_actual = timestamps[1] - timestamps[0]
            else:
                dt_actual = dt
        elif i == len(x) - 1:
            # Last point: use velocity from previous point
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            if use_timestamp:
                dt_actual = timestamps[i] - timestamps[i-1]
            else:
                dt_actual = dt
        else:
            # Middle points: use central difference
            dx = x[i+1] - x[i-1]
            dy = y[i+1] - y[i-1]
            if use_timestamp:
                dt_actual = timestamps[i+1] - timestamps[i-1]
            else:
                dt_actual = 2 * dt

        # Calculate velocity magnitude
        distance = np.sqrt(dx**2 + dy**2)
        velocity = distance / dt_actual if dt_actual > 0 else 0.0
        velocities.append(velocity)

    return np.array(velocities)


def plot_pickup_trajectories():
    """Plot pickup trajectories for each robot with velocity color coding"""

    base_path = Path("d:/Data_visualization_code/result/MAPS/MAP5")
    map5_data_path = base_path / "map5" / "map5"
    pickup_path = map5_data_path / "control_data" / "pickup"

    # Find all pickup trajectory files
    pickup_pattern = str(pickup_path / "robot*_Pickup_robot_trajectory.json")
    pickup_files = sorted(glob.glob(pickup_pattern))

    print(f"Found {len(pickup_files)} pickup trajectory files")

    # Velocity parameters
    v_max = 0.03  # max linear velocity (m/s)

    # Create colormap (viridis: blue to yellow)
    try:
        cmap = plt.colormaps['viridis']  # New API (matplotlib >= 3.7)
    except AttributeError:
        cmap = plt.cm.get_cmap('viridis')  # Fallback for older versions

    # ========================================================================
    # FIRST PASS: Collect all velocities to determine global color bar range
    # ========================================================================
    print("\n[Pass 1/2] Collecting velocity data from all robots...")
    all_velocities_global = []

    for pickup_file in pickup_files:
        filename = Path(pickup_file).stem
        robot_id = int(filename.split('_')[0].replace('robot', ''))

        # Load reference trajectory
        ref_file = map5_data_path / f"tb{robot_id}_Trajectory_replanned.json"
        if not ref_file.exists():
            continue

        ref_traj, ref_dt, _ = load_reference_trajectory(str(ref_file))
        ref_velocities = ref_traj[:, 3]
        all_velocities_global.extend(ref_velocities)

        # Load pickup trajectory
        try:
            pickup_traj = load_real_trajectory(pickup_file)
            if len(pickup_traj) >= 2:
                pickup_velocities = calculate_velocity_from_position(pickup_traj, use_timestamp=True)
                all_velocities_global.extend(pickup_velocities)
        except Exception:
            pass  # Skip if pickup data not available

    # Calculate global velocity range for unified color bar
    if len(all_velocities_global) > 0:
        global_min_velocity = max(0.0, np.min(all_velocities_global))
        global_max_velocity = max(global_min_velocity + 0.001, np.max(all_velocities_global))
    else:
        global_min_velocity = 0.0
        global_max_velocity = v_max

    print(f"Global velocity range: [{global_min_velocity:.4f}, {global_max_velocity:.4f}] m/s")
    print(f"{'='*60}")

    # Create unified normalization for all plots
    global_norm = Normalize(vmin=global_min_velocity, vmax=global_max_velocity)

    # ========================================================================
    # SECOND PASS: Plot each robot with unified color bar
    # ========================================================================
    print("\n[Pass 2/2] Plotting trajectories with unified color bar...")
    # Process each robot
    for pickup_file in pickup_files:
        filename = Path(pickup_file).stem
        # Extract robot_id: robot{X}_Pickup_robot_trajectory
        robot_id = int(filename.split('_')[0].replace('robot', ''))

        print(f"  Processing Robot {robot_id}...")

        # Load reference trajectory for this robot (replanned version)
        ref_file = map5_data_path / f"tb{robot_id}_Trajectory_replanned.json"

        if not ref_file.exists():
            print(f"    Warning: Reference trajectory not found for robot {robot_id}")
            print(f"    Looking for: {ref_file}")
            continue

        # Load reference trajectory with dt and total_time
        ref_traj, ref_dt, ref_total_time_from_file = load_reference_trajectory(str(ref_file))

        # Use total_time from file if available, otherwise calculate
        if ref_total_time_from_file is not None:
            ref_total_time = ref_total_time_from_file
        else:
            ref_total_time = (len(ref_traj) - 1) * ref_dt

        # Extract velocity from file directly (column 3: v - linear velocity)
        ref_velocities = ref_traj[:, 3]

        # Print reference velocity range
        ref_v_min = np.min(ref_velocities)
        ref_v_max = np.max(ref_velocities)
        ref_v_mean = np.mean(ref_velocities)
        print(f"    Reference: v_min={ref_v_min:.4f} m/s, v_max={ref_v_max:.4f} m/s, v_mean={ref_v_mean:.4f} m/s, time={ref_total_time:.2f}s")

        # Create figure for this robot
        fig, ax = plt.subplots(figsize=(4, 4))

        # Load and plot pickup trajectory with velocity color coding
        pickup_velocities = None
        pickup_total_time = 0.0
        try:
            pickup_traj = load_real_trajectory(pickup_file)
            if len(pickup_traj) >= 2:
                # Calculate pickup trajectory total time from timestamps
                pickup_total_time = pickup_traj[-1, 0] - pickup_traj[0, 0]

                # Calculate velocity from position for pickup trajectory
                pickup_velocities = calculate_velocity_from_position(pickup_traj, use_timestamp=True)

                # Print pickup velocity range
                pickup_v_min = np.min(pickup_velocities)
                pickup_v_max = np.max(pickup_velocities)
                pickup_v_mean = np.mean(pickup_velocities)
                print(f"    Pickup:    v_min={pickup_v_min:.4f} m/s, v_max={pickup_v_max:.4f} m/s, v_mean={pickup_v_mean:.4f} m/s, time={pickup_total_time:.2f}s")

        except Exception as e:
            print(f"    Error loading pickup trajectory: {e}")

        # Use global normalization for unified color bar across all robots
        norm = global_norm

        # Plot reference trajectory with velocity color coding (dashed line, thinner)
        # Reversed: from end to start
        for i in range(len(ref_traj) - 1, 0, -1):
            current_velocity = ref_velocities[i]
            color = cmap(norm(current_velocity))
            ax.plot([ref_traj[i, 0], ref_traj[i-1, 0]],
                   [ref_traj[i, 1], ref_traj[i-1, 1]],
                   color=color, linewidth=2.0, alpha=0.6, linestyle='--', zorder=1)

        # Mark reference start and end (reversed markers with distinct colors)
        ax.plot(ref_traj[-1, 0], ref_traj[-1, 1],
               'o', color='blue', markersize=8, markeredgewidth=0, zorder=5, label='Reference')  # End becomes start (blue)
        ax.plot(ref_traj[0, 0], ref_traj[0, 1],
               's', color='navy', markersize=8, markeredgewidth=0, zorder=5)  # Start becomes end (navy)

        # Store reference end point for text placement (above the node to avoid overlap)
        ref_end_x, ref_end_y = ref_traj[0, 0], ref_traj[0, 1]
        ref_text_x = ref_end_x + 0.05  # Slight right offset
        ref_text_y = ref_end_y + 0.05  # Above the node

        # Plot pickup trajectory with velocity color coding (if loaded successfully)
        if pickup_velocities is not None and len(pickup_traj) >= 2:
            for i in range(len(pickup_traj) - 1):
                current_velocity = pickup_velocities[i]
                color = cmap(norm(current_velocity))
                ax.plot([pickup_traj[i, 1], pickup_traj[i+1, 1]],  # Column 1,2 are x,y (after timestamp)
                       [pickup_traj[i, 2], pickup_traj[i+1, 2]],
                       color=color, linewidth=3.5, alpha=0.8, zorder=3)

            # Mark pickup start and end with distinct colors
            ax.plot(pickup_traj[0, 1], pickup_traj[0, 2],
                   'o', color='red', markersize=8,
                   markeredgewidth=0, zorder=6, label='Pickup')  # Start (red circle)
            ax.plot(pickup_traj[-1, 1], pickup_traj[-1, 2],
                   's', color='darkred', markersize=8,
                   markeredgewidth=0, zorder=6)  # End (dark red square)

            # Store pickup end point for text placement (below the node to avoid overlap)
            pickup_end_x, pickup_end_y = pickup_traj[-1, 1], pickup_traj[-1, 2]
            pickup_text_x = pickup_end_x + 0.05  # Slight right offset
            pickup_text_y = pickup_end_y - 0.05  # Below the node
        else:
            pickup_text_x = None
            pickup_text_y = None

        # Add horizontal colorbar at bottom
        cbar = plt.colorbar(ScalarMappable(norm=global_norm, cmap=cmap), ax=ax,
                           orientation='horizontal', pad=0.15, aspect=30)
        cbar.set_label('Velocity (m/s)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        # Add legend for start/end markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=8, label='Reference Start', markeredgewidth=0),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='navy',
                   markersize=8, label='Reference End', markeredgewidth=0),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='Pickup Start', markeredgewidth=0),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='darkred',
                   markersize=8, label='Pickup End', markeredgewidth=0),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                 fontsize=9, framealpha=0.9, borderaxespad=0, ncol=2)

        # Customize plot
        ax.set_title(f'Robot {robot_id}',
                    fontsize=18, pad=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal')  # Equal aspect ratio

        # Adjust axis limits to include text labels
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Collect all text positions to ensure they're included in the plot
        text_positions_x = [ref_text_x]
        text_positions_y = [ref_text_y]
        if pickup_text_x is not None:
            text_positions_x.append(pickup_text_x)
            text_positions_y.append(pickup_text_y)

        # Expand limits to include text (with margin for text size)
        text_margin = 0.1  # Extra margin for text rendering
        x_min = min(xlim[0], min(text_positions_x) - text_margin)
        x_max = max(xlim[1], max(text_positions_x) + text_margin)
        y_min = min(ylim[0], min(text_positions_y) - text_margin)
        y_max = max(ylim[1], max(text_positions_y) + text_margin)

        # Add standard margin
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_margin = x_range * 0.05
        y_margin = y_range * 0.05
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.tick_params(axis='both', which='major', labelsize=16)

        # Add text labels AFTER setting axis limits
        # Reference: above and to the right, Pickup: below and to the right (avoid overlap)
        ax.text(ref_text_x, ref_text_y, f'{ref_total_time:.2f}s',
               fontsize=10, ha='left', va='bottom', color='navy', fontweight='bold')
        if pickup_text_x is not None:
            ax.text(pickup_text_x, pickup_text_y, f'{pickup_total_time:.2f}s',
                   fontsize=10, ha='left', va='top', color='darkred', fontweight='bold')

        # Save figure
        output_file = map5_data_path / f"robot{robot_id}_pickup_trajectory.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file.name}")
        plt.close()

    print(f"\n{'='*60}")
    print("All pickup trajectory plots completed!")


if __name__ == "__main__":
    plot_pickup_trajectories()
