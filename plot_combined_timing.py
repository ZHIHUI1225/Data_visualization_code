#!/usr/bin/env python3
"""
Simple script to plot combined timing data for both robots.
"""

import sys
import os

# Add the current directory to the path to import the visualization module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualize_behavior_timing import visualize_combined_timing_data, visualize_trajectory_with_speed
import os
import argparse

def find_data_files_in_case(case_dir, file_pattern):
    """Recursively find data files in case directory."""
    import glob
    found_files = []

    # Search recursively in all subdirectories
    for root, _, _ in os.walk(case_dir):
        pattern = os.path.join(root, file_pattern)
        matches = glob.glob(pattern)
        if matches:
            found_files.extend(matches)

    return sorted(found_files)


def process_single_case(case_dir, case_name, mode, show_figure):
    """Process a single case directory."""
    print(f"\n{'='*60}")
    print(f"Processing case: {case_name}")
    print(f"{'='*60}")

    if mode == "timing":
        # Find all Time_robot*.json files recursively
        robot_files = find_data_files_in_case(case_dir, "Time_robot*.json")

        if not robot_files:
            print(f"  [SKIP] No timing files found in {case_name}")
            return False

        print(f"  Found {len(robot_files)} robot timing files:")
        for file in robot_files:
            print(f"    - {os.path.relpath(file, case_dir)}")

        # Save output in the same directory where data files are found
        output_dir = os.path.dirname(robot_files[0])
        output_file = os.path.join(output_dir, "all_robots_timing.png")

        try:
            visualize_combined_timing_data(robot_files, output_file, show_figure=show_figure)
            print(f"  [SUCCESS] Timing visualization saved to: {os.path.relpath(output_file, case_dir)}")
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to generate timing plot: {e}")
            return False

    elif mode == "trajectory":
        # Find all robot*_parcel*_approach_robot_trajectory.json files recursively
        trajectory_files = find_data_files_in_case(case_dir, "robot*_parcel*_approach_robot_trajectory.json")

        if not trajectory_files:
            print(f"  [SKIP] No trajectory files found in {case_name}")
            return False

        print(f"  Found {len(trajectory_files)} robot trajectory files:")
        for file in trajectory_files:
            print(f"    - {os.path.relpath(file, case_dir)}")

        # Save output in the same directory where data files are found
        output_dir = os.path.dirname(trajectory_files[0])
        output_file = os.path.join(output_dir, "robot_trajectories_speed_visualization.png")

        try:
            visualize_trajectory_with_speed(trajectory_files, output_file, show_figure=show_figure)
            print(f"  [SUCCESS] Trajectory visualization saved to: {os.path.relpath(output_file, case_dir)}")
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to generate trajectory plot: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize robot timing or trajectory data for all MAPS cases")
    parser.add_argument("--mode", choices=["timing", "trajectory"], default="timing",
                        help="Choose visualization mode: timing (default) or trajectory")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show figures interactively (default: False for batch processing)")
    parser.add_argument("--case", type=str, default=None,
                        help="Process only specific case (e.g., MAP1, MAP5, LOOP1). If not specified, process all cases.")
    args = parser.parse_args()

    # Define the MAPS directory
    maps_dir = os.path.join(os.path.dirname(__file__), "result", "MAPS")

    if not os.path.exists(maps_dir):
        print(f"ERROR: MAPS directory not found: {maps_dir}")
        exit(1)

    # Get all case directories (MAP*, LOOP*)
    all_cases = []
    for item in os.listdir(maps_dir):
        item_path = os.path.join(maps_dir, item)
        if os.path.isdir(item_path) and (item.startswith("MAP") or item.startswith("LOOP")):
            all_cases.append((item, item_path))

    all_cases.sort()  # Sort for consistent processing order

    if not all_cases:
        print(f"ERROR: No MAP* or LOOP* directories found in {maps_dir}")
        exit(1)

    # Filter to specific case if requested
    if args.case:
        all_cases = [(name, path) for name, path in all_cases if name == args.case]
        if not all_cases:
            print(f"ERROR: Case '{args.case}' not found in {maps_dir}")
            print(f"Available cases: {', '.join([name for name, _ in all_cases])}")
            exit(1)

    # Process all cases
    print(f"Found {len(all_cases)} case(s) to process: {', '.join([name for name, _ in all_cases])}")

    success_count = 0
    skip_count = 0
    error_count = 0

    for case_name, case_path in all_cases:
        result = process_single_case(case_path, case_name, args.mode, args.show)
        if result:
            success_count += 1
        else:
            skip_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases: {len(all_cases)}")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}")
