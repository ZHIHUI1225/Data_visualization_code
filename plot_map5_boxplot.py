#!/usr/bin/env python3
"""
Box plot visualization for MAP5 robot timing data.
Shows distribution of target time, pushing time, and picking up time across all robots.
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


def parse_time_string(time_str):
    """Convert time string to datetime object."""
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")


def calculate_duration_seconds(start_str, end_str):
    """Calculate duration between start and end time in seconds."""
    start = parse_time_string(start_str)
    end = parse_time_string(end_str)
    return (end - start).total_seconds()


def extract_timing_data(json_file_path):
    """
    Extract pushing and picking up durations from a robot's timing file.
    Returns dict with robot_id and lists of durations.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    pushing_durations = []
    picking_durations = []

    for event_data in data.values():
        node_name = event_data.get("node_name", "")

        if node_name == "Pushing":
            duration = calculate_duration_seconds(
                event_data["start_time"],
                event_data["end_time"]
            )
            pushing_durations.append(duration)

        elif node_name == "PickingUp":
            duration = calculate_duration_seconds(
                event_data["start_time"],
                event_data["end_time"]
            )
            picking_durations.append(duration)

    robot_id = data[list(data.keys())[0]]["robot_id"]

    return {
        "robot_id": robot_id,
        "pushing": pushing_durations,
        "picking": picking_durations
    }


def load_all_robots_data(data_dir):
    """Load timing data from all robot files, organized by robot."""
    robots_data = []

    for i in range(4):
        robot_file = os.path.join(data_dir, f"Time_robot{i}.json")
        trajectory_file = os.path.join(data_dir, f"tb{i}_Trajectory.json")

        if not os.path.exists(robot_file):
            print(f"Warning: {robot_file} not found, skipping robot {i}")
            continue

        # Get pushing and picking times
        robot_timing = extract_timing_data(robot_file)

        # Get target time from trajectory file
        target_time = 0
        if os.path.exists(trajectory_file):
            with open(trajectory_file, 'r') as f:
                traj_data = json.load(f)
                target_time = traj_data.get("total_time", 0)
        else:
            print(f"Warning: {trajectory_file} not found for robot {i}")

        robots_data.append({
            "robot_id": i,
            "target_time": target_time,
            "pushing": robot_timing["pushing"],
            "picking": robot_timing["picking"]
        })

        print(f"Robot {i}: target={target_time:.2f}s, "
              f"{len(robot_timing['pushing'])} pushing, "
              f"{len(robot_timing['picking'])} picking events")

    return robots_data


def create_boxplot(robots_data, output_file):
    """Create box plot with 4 columns (one per robot), each column at same X position."""
    fig, ax = plt.subplots(figsize=(5.5, 4))

    num_robots = len(robots_data)

    color_map = {
        'target': '#FF6B6B',
        'pushing': '#2CA02C',  # Green - same as Push in timing visualization
        'picking': '#FF7F0E'   # Orange - same as Pickup in timing visualization
    }

    # Collect all data for box plots (pushing and picking only)
    box_positions = []
    box_data = []
    box_colors = []

    # X positions for each robot
    robot_x_positions = list(range(1, num_robots + 1))

    for robot_idx, robot in enumerate(robots_data):
        x_pos = robot_idx + 1

        # Pushing time - box plot at robot's X position
        box_positions.append(x_pos)
        box_data.append(robot["pushing"])
        box_colors.append(color_map['pushing'])

        # Picking time - box plot at robot's X position
        box_positions.append(x_pos)
        box_data.append(robot["picking"])
        box_colors.append(color_map['picking'])

    # Create box plots for pushing and picking
    bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True,
                    showmeans=False, widths=0.25, showfliers=False)

    # Color boxes
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Style median lines
    for median in bp['medians']:
        median.set_color('darkred')
        median.set_linewidth(2)

    # Plot target times as horizontal lines
    for robot_idx, robot in enumerate(robots_data):
        x_pos = robot_idx + 1
        target_time = robot["target_time"]
        # Draw horizontal line at target time position
        ax.hlines(target_time, x_pos - 0.3, x_pos + 0.3,
                  color=color_map['target'], linewidth=3,
                  alpha=0.9, zorder=5)

    # Set x-axis labels
    ax.set_xticks(robot_x_positions)
    ax.set_xticklabels([f" {r['robot_id']}" for r in robots_data],
                       fontsize=14)

    # Set y-axis tick label size
    ax.tick_params(axis='y', labelsize=14)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_ylabel("Time (s)", fontsize=14, loc='top')
    ax.set_xlabel("Robot ID", fontsize=14, loc='right')
    ax.set_title("Timing Distribution", fontsize=14)

    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_map['target'], linewidth=3, label='Target'),
        Patch(facecolor=color_map['pushing'], alpha=0.7, label='Pushing'),
        Patch(facecolor=color_map['picking'], alpha=0.7, label='Picking Up')
    ]
    ax.legend(handles=legend_elements, loc='center left', fontsize=14,
              bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False,
              handletextpad=0.5)

    plt.tight_layout()
    # plt.show()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Box plot saved to: {output_file}")

    return fig


if __name__ == "__main__":
    # Data directory for MAP5
    data_dir = os.path.join(
        os.path.dirname(__file__),
        "result", "MAPS", "MAP5", "map5", "map5"
    )

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        exit(1)

    print(f"Loading data from: {data_dir}\n")

    # Load and process data
    timing_data = load_all_robots_data(data_dir)

    # Create output
    output_file = os.path.join(data_dir, "map5_timing_boxplot.png")
    create_boxplot(timing_data, output_file)

    print("\nDone!")
