#!/usr/bin/env python3
"""Test script to verify timing plot changes"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualize_behavior_timing import visualize_combined_timing_data
import glob

# Test with MAP5
data_dir = os.path.join(os.path.dirname(__file__), "result", "MAPS", "MAP5", "MAP5", "map5")
pattern = os.path.join(data_dir, "Time_robot*.json")
robot_files = sorted(glob.glob(pattern))

if robot_files:
    print(f"Found {len(robot_files)} robot files:")
    for f in robot_files:
        print(f"  - {os.path.basename(f)}")

    output_file = os.path.join(data_dir, "all_robots_timing_NEW.png")
    print(f"\nGenerating: {output_file}")

    visualize_combined_timing_data(robot_files, output_file, show_figure=False)
    print(f"✓ Generated: {output_file}")
    print("\nPlease check the NEW file to see the changes!")
else:
    print(f"ERROR: No robot files found in {data_dir}")
