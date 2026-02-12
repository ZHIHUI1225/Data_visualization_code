#!/usr/bin/env python3
"""
Script to plot robot trajectories with parcel-based colors and speed-based intensity.
Each parcel gets a distinct color family, and speed variations are shown through
color intensity (darker = faster, lighter = slower).
"""

import sys
import os
import glob

# Add the current directory to the path to import the visualization module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualize_behavior_timing import visualize_trajectory_with_speed

if __name__ == "__main__":
    # Define the data directory and automatically find all robot trajectory files
    data_dir = os.path.join(os.path.dirname(__file__), "result", "romi", "success0", "experi", "control_data")
    
    # Find all robot*_parcel*_approach_robot_trajectory.json files in the directory
    trajectory_files = []
    pattern = os.path.join(data_dir, "robot*_parcel*_approach_robot_trajectory.json")
    trajectory_files = sorted(glob.glob(pattern))  # Sort to ensure consistent order
    
    if not trajectory_files:
        print(f"No robot trajectory files found in {data_dir}")
        print("Looking for files matching: robot*_parcel*_approach_robot_trajectory.json")
        exit(1)
    
    print(f"Found {len(trajectory_files)} robot trajectory files:")
    for file in trajectory_files:
        print(f"  - {os.path.basename(file)}")
    
    output_file = os.path.join(data_dir, "robot_trajectories_speed_visualization.png")
    
    # Plot all robot trajectory files with parcel-based colors and speed intensity
    visualize_trajectory_with_speed(trajectory_files, output_file, show_figure=True)
    print(f"Trajectory visualization saved to: {output_file}")