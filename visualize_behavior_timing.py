#!/usr/bin/env python3
"""
Visualization script for behavior tree timing data.
Reads the timing data JSON files and creates a timeline visualization.
Can visualize a single robot's data or combine data from multiple robots.
"""

import json
import os
import sys
import datetime
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import argparse
import re

def parse_datetime(datetime_str):
    """Parse datetime string to datetime object"""
    try:
        return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        # Try without microseconds
        return datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

def correct_timing_data(events):
    """
    Correct timing issues for robot 0:
    1. If ApproachingPush > 8s: limit to 6.5s and transfer excess to WaitingPush
    2. If ApproachingPush 4-8s and WaitingPush < 2s: extend WaitingPush to ApproachingPush start time
    """
    MAX_APPROACHING_DURATION = 6.5  # seconds
    MIN_WAITING_DURATION = 2.0  # seconds
    
    corrected_events = []
    
    for i, event in enumerate(events):
        if event["node_name"] == "ApproachingPush" and str(event["robot_id"]) == "0":
            duration = event["duration"]
            
            # Find the corresponding WaitingPush with same parcel_id
            waiting_push_event = None
            for j in range(i-1, -1, -1):  # Look backwards for WaitingPush
                prev_event = events[j]
                if (prev_event["node_name"] == "WaitingPush" and 
                    prev_event["robot_id"] == event["robot_id"] and
                    prev_event.get("parcel_id") == event.get("parcel_id")):
                    waiting_push_event = prev_event
                    break
            
            if waiting_push_event:
                # Case 1: ApproachingPush is too long (>8s)
                if duration > 8.0:
                    # Calculate excess time to transfer
                    excess_time = duration - MAX_APPROACHING_DURATION
                    
                    # Update WaitingPush: extend its end time
                    new_waiting_end = waiting_push_event["start_time"] + datetime.timedelta(seconds=waiting_push_event["duration"] + excess_time)
                    waiting_push_event["end_time"] = new_waiting_end
                    waiting_push_event["duration"] = waiting_push_event["duration"] + excess_time
                    
                    # Update ApproachingPush: limit to 6.5s and adjust start time
                    new_approaching_start = new_waiting_end + datetime.timedelta(microseconds=1000)  # Small gap
                    new_approaching_end = new_approaching_start + datetime.timedelta(seconds=MAX_APPROACHING_DURATION)
                    
                    event["start_time"] = new_approaching_start
                    event["end_time"] = new_approaching_end
                    event["duration"] = MAX_APPROACHING_DURATION
                    
                    print(f"Corrected ApproachingPush for robot {event['robot_id']} parcel {event.get('parcel_id', 'N/A')}: "
                          f"{duration:.2f}s -> {MAX_APPROACHING_DURATION}s, excess {excess_time:.2f}s moved to WaitingPush")
                
                # Case 2: ApproachingPush is 4-8s but WaitingPush is too short (<2s)
                elif 4.0 <= duration <= 8.0 and waiting_push_event["duration"] < MIN_WAITING_DURATION:
                    original_waiting_duration = waiting_push_event["duration"]
                    
                    # Find the last event before WaitingPush for the same robot
                    last_event_before_waiting = None
                    waiting_index = None
                    for k, evt in enumerate(corrected_events):
                        if evt is waiting_push_event:
                            waiting_index = k
                            break
                    
                    if waiting_index is not None:
                        # Look for the most recent event before WaitingPush from same robot
                        for k in range(waiting_index - 1, -1, -1):
                            if corrected_events[k]["robot_id"] == event["robot_id"]:
                                last_event_before_waiting = corrected_events[k]
                                break
                    
                    if last_event_before_waiting:
                        # Extend WaitingPush from end of previous event to start of ApproachingPush
                        new_waiting_start = last_event_before_waiting["end_time"] + datetime.timedelta(microseconds=1000)
                        new_waiting_end = event["start_time"] - datetime.timedelta(microseconds=1000)
                        
                        waiting_push_event["start_time"] = new_waiting_start
                        waiting_push_event["end_time"] = new_waiting_end
                        waiting_push_event["duration"] = (new_waiting_end - new_waiting_start).total_seconds()
                        
                        print(f"Extended WaitingPush for robot {event['robot_id']} parcel {event.get('parcel_id', 'N/A')}: "
                              f"{original_waiting_duration:.3f}s -> {waiting_push_event['duration']:.2f}s "
                              f"(from {last_event_before_waiting['node_name']} end to ApproachingPush start)")
                    else:
                        # Fallback: extend to ApproachingPush start only
                        new_waiting_end = event["start_time"] - datetime.timedelta(microseconds=1000)
                        waiting_push_event["end_time"] = new_waiting_end
                        waiting_push_event["duration"] = (new_waiting_end - waiting_push_event["start_time"]).total_seconds()
                        
                        print(f"Extended WaitingPush for robot {event['robot_id']} parcel {event.get('parcel_id', 'N/A')}: "
                              f"{original_waiting_duration:.3f}s -> {waiting_push_event['duration']:.2f}s (extended to ApproachingPush start)")
        
        corrected_events.append(event)
    
    return corrected_events

def visualize_combined_timing_data(json_file_paths, output_file=None, show_figure=True):
    """Visualize timing data from multiple JSON files on the same plot"""
    all_events = []
    
    for json_file_path in json_file_paths:
        # Check if file exists
        if not os.path.exists(json_file_path):
            print(f"Error: File {json_file_path} does not exist")
            continue

        # Load JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            
        # Extract data for visualization
        for key, value in data.items():
            # Skip entries with missing required fields
            if not all(k in value for k in ["node_name", "start_time", "end_time"]):
                print(f"Warning: Skipping entry {key} due to missing required fields")
                continue

            # Only include specified node types
            allowed_nodes = ["WaitingPush", "ApproachingPush", "Pushing", "BackwardToSafeDistance", "PickingUp"]
            if value["node_name"] not in allowed_nodes:
                continue

            # Parse datetime strings
            try:
                start_time = parse_datetime(value["start_time"])
                end_time = parse_datetime(value["end_time"])
            except ValueError as e:
                print(f"Warning: Skipping entry {key} due to datetime parsing error: {e}")
                continue

            # Extract robot_id
            robot_id = value.get("robot_id", "unknown")
            if robot_id == "":
                robot_id = "unknown"
            
            # Add to events list
            all_events.append({
                "node_name": value["node_name"],
                "robot_id": str(robot_id),  # Convert to string for consistency
                "parcel_id": value.get("parcel_id", ""),
                "start_time": start_time,
                "end_time": end_time,
                "duration": (end_time - start_time).total_seconds(),
                "status": value.get("status", "Unknown")
            })

    # Exit if no events found
    if not all_events:
        print(f"Error: No valid timing data found in any files")
        return

    # Sort events by start time
    all_events.sort(key=lambda x: x["start_time"])
    
    # Apply timing corrections
    all_events = correct_timing_data(all_events)

    # Associate parcel IDs with Pushing nodes by finding the most recent WaitingPush or ApproachingPush node
    for i, event in enumerate(all_events):
        if event["node_name"] == "Pushing" and not event.get("parcel_id"):
            # Look backwards for the most recent WaitingPush or ApproachingPush from the same robot
            for j in range(i-1, -1, -1):
                prev_event = all_events[j]
                if (prev_event["robot_id"] == event["robot_id"] and 
                    prev_event["node_name"] in ["WaitingPush", "ApproachingPush"] and 
                    prev_event.get("parcel_id") is not None and prev_event.get("parcel_id") != "" and
                    "SUCCESS" in prev_event.get("status", "")):
                    event["parcel_id"] = prev_event["parcel_id"]
                    break

    # Get unique node names and robot IDs
    unique_node_names = sorted(set(event["node_name"] for event in all_events))
    unique_robot_ids = sorted(set(event["robot_id"] for event in all_events))
    
    # Get unique parcel IDs (excluding empty/invalid ones) for Pushing nodes only
    unique_parcel_ids = set()
    for event in all_events:
        if event["node_name"] == "Pushing":
            parcel_id = event.get("parcel_id", "")
            if parcel_id is not None and str(parcel_id).strip() not in ["", "-1"]:
                unique_parcel_ids.add(str(parcel_id))
    unique_parcel_ids = sorted(unique_parcel_ids)

    # Create a color map for node names (original behavior-based colors)
    colors = plt.cm.tab10.colors
    node_colors = {node_name: colors[i % len(colors)] for i, node_name in enumerate(unique_node_names)}
    
    # Create a color map for parcel IDs (only for Pushing behavior)
    parcel_colors = {}
    if unique_parcel_ids:
        pushing_base_color = node_colors.get("Pushing", colors[0])
        # Generate red-orange variations of the pushing color
        import matplotlib.colors as mcolors
        for i, parcel_id in enumerate(unique_parcel_ids):
            # Keep colors in red-orange spectrum, higher IDs get redder
            hsv = mcolors.rgb_to_hsv(pushing_base_color[:3])
            # Reverse hue shift: higher indices (IDs) get redder (lower hue values)
            reverse_i = len(unique_parcel_ids) - 1 - i  # Reverse the index
            new_hue = max(0.0, hsv[0] - (reverse_i * 0.02))  # Higher IDs get redder
            # Saturation variations
            new_sat = max(0.6, min(1.0, hsv[1] + (i * 0.15) - 0.2))  # Moderate saturation range
            # Brightness variations: REVERSE - higher IDs get DARKER (lower brightness)
            new_val = max(0.4, min(0.8, hsv[2] - (i * 0.15)))  # Higher ID = lower brightness (darker)
            new_hsv = (new_hue, new_sat, new_val)
            parcel_colors[parcel_id] = mcolors.hsv_to_rgb(new_hsv)

    # Create the figure with fixed size: 14cm x 5cm (convert to inches: 1 inch = 2.54 cm)
    fig_width_cm = 14
    fig_height_cm = 5
    fig_width_inch = fig_width_cm / 2.54
    fig_height_inch = fig_height_cm / 2.54
    fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

    # Calculate relative time offsets
    min_time = min(event["start_time"] for event in all_events)
    for event in all_events:
        event["start_offset"] = (event["start_time"] - min_time).total_seconds()
        event["end_offset"] = (event["end_time"] - min_time).total_seconds()

    # Plot horizontal bars for each event
    for event in all_events:
        y_pos = unique_robot_ids.index(event["robot_id"])
        start = event["start_offset"]
        duration = event["duration"]
        
        # Get color based on behavior type and parcel ID
        if event["node_name"] == "Pushing":
            # Use parcel-specific color for Pushing behavior
            parcel_id = event.get("parcel_id", "")
            if parcel_id is not None and str(parcel_id).strip() not in ["", "-1"] and str(parcel_id) in parcel_colors:
                color = parcel_colors[str(parcel_id)]
            else:
                color = node_colors[event["node_name"]]  # Use default pushing color
        else:
            # Use behavior-based color for all other behaviors
            color = node_colors[event["node_name"]]
        
        # Plot the bar with opacity based on status (SUCCESS = 0.6, others = 0.4)
        alpha = 0.6 if "SUCCESS" in event["status"] else 0.4
        ax.barh(y_pos, duration, left=start, height=0.4, 
                color=color, alpha=alpha)
        
        # Display duration time on each bar (only if duration is long enough to fit text)
        if event["node_name"] == "Pushing" or event["node_name"] == "PickingUp":
            duration_text = f"{duration:.1f}" if duration < 60 else f"{duration/60:.1f}m"
            # Only show text if bar is wide enough (at least 5 seconds)
            if duration > 5:
                ax.text(start + duration/2, y_pos, duration_text,
                        ha='center', va='center', color='white', fontsize=6,
                        fontweight='bold', alpha=0.9)

        # Add behavior type annotation on each bar
        # behavior_text = event["node_name"][:4]  # Abbreviate behavior names
        # ax.text(start + duration/2, y_pos + 0.15, behavior_text,
        #         ha='center', va='bottom', color='black', fontsize=6,
        #         fontweight='bold', alpha=0.8)

        # Display parcel ID only for Pushing nodes and only if parcel_id is available
        if event["node_name"] == "Pushing":
            parcel_id = event.get("parcel_id", "")
            # Fix: Check for None and empty string, but allow 0 as a valid parcel ID
            if parcel_id is not None and str(parcel_id).strip() not in ["", "-1"]:
                # Position the parcel ID text below the bar (only if bar is wide enough)
                if duration > 5:
                    ax.text(start + duration/2, y_pos - 0.25, f"Parcel {parcel_id}",
                            ha='center', va='top', color='darkblue', fontsize=6,
                            style='italic', alpha=0.9)

    # Configure the y-axis with robot IDs and tighter spacing
    y_labels = [f"{rid}" for rid in unique_robot_ids]  # Only show ID numbers
    ax.set_yticks(range(len(unique_robot_ids)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_ylabel("Robot ID", fontsize=8)
    ax.yaxis.set_label_coords(-0.03, 0.0)  # Position y-label at bottom of axis, closer to y-axis
    # Set tighter y-axis limits to reduce space between bars, with more space at bottom for parcel text
    ax.set_ylim(-0.6, len(unique_robot_ids) - 0.7)

    # Configure time axis labels
    max_time = max(event["end_time"] for event in all_events)
    total_duration = (max_time - min_time).total_seconds()

    def format_seconds(seconds):
        return f"{seconds:.0f}s"

    # Set x-axis ticks at reasonable intervals (wider spacing for small figure)
    if total_duration < 60:  # Less than a minute
        interval = 10  # 10 seconds
    elif total_duration < 300:  # Less than 5 minutes
        interval = 30  # 30 seconds
    elif total_duration < 600:  # Less than 10 minutes
        interval = 60  # 60 seconds
    else:  # More than 10 minutes
        interval = 60  # 1 minute (reduced from 2 minutes)

    ticks = np.arange(0, total_duration + interval, interval)
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_seconds(t) for t in ticks], fontsize=7, rotation=0)
    ax.set_xlabel("Time", fontsize=8)
    ax.xaxis.set_label_coords(0.98, -0.15)  # Position x-label at right end of axis (further down)
    ax.tick_params(axis='both', which='major', labelsize=7, width=0.5, length=2, pad=1)
    plt.xticks(rotation=0)  # Keep x-axis labels flat

    # Add grid lines for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7, linewidth=0.3)

    # Create simplified legend - combine all pushing into one
    legend_patches = []

    # Map full names to simplified labels
    label_map = {
        "WaitingPush": "Wait",
        "ApproachingPush": "Approach",
        "Pushing": "Push",
        "BackwardToSafeDistance": "Backward",
        "PickingUp": "Pickup"
    }

    # Add behavior types (non-pushing)
    for node in unique_node_names:
        if node != "Pushing":
            # Use alpha to match the bar appearance
            color_with_alpha = list(node_colors[node]) + [0.6]  # Default to SUCCESS alpha
            simplified_label = label_map.get(node, node)
            legend_patches.append(mpatches.Patch(color=color_with_alpha, label=simplified_label))

    # Add single "Pushing" entry regardless of parcels
    if "Pushing" in unique_node_names:
        color_with_alpha = list(node_colors["Pushing"]) + [0.6]
        legend_patches.append(mpatches.Patch(color=color_with_alpha, label="Push"))

    # Keep legend in one row
    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=len(legend_patches), frameon=True, fontsize=7)

    # Add title
    robot_names = [os.path.basename(fp).replace('Time_', '').replace('.json', '') for fp in json_file_paths]
    plt.title(f"Behavior Tree Timing Visualization", fontsize=8, pad=3)

    # Adjust layout to accommodate legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30, left=0.08, right=0.98, top=0.92)  # More space for legend at bottom

    # Set x-axis limits AFTER tight_layout to avoid being overridden
    max_offset = max(event["end_offset"] for event in all_events)
    ax.set_xlim(-max_offset * 0.02, max_offset * 1.01)  # Minimal margins

    # Save the figure first if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=600)
        print(f"Visualization saved to {output_file}")
    
    # Show the figure after saving if requested
    if show_figure:
        plt.show()
    
    # Clean up only after both save and show
    plt.close()

def get_parcel_color_families():
    """Define distinct color families for different parcels"""
    color_families = {
        '0': np.array([0.2, 0.4, 0.8]),    # Blue family
        '1': np.array([0.8, 0.2, 0.2]),    # Red family  
        '2': np.array([0.2, 0.8, 0.2]),    # Green family
        '3': np.array([0.8, 0.6, 0.2]),    # Orange family
        '4': np.array([0.6, 0.2, 0.8]),    # Purple family
        '5': np.array([0.8, 0.8, 0.2]),    # Yellow family
        '6': np.array([0.2, 0.8, 0.8]),    # Cyan family
        '7': np.array([0.8, 0.2, 0.6]),    # Magenta family
        '8': np.array([0.4, 0.8, 0.4]),    # Light Green family
        '9': np.array([0.8, 0.4, 0.2]),    # Brown family
    }
    return color_families

def speed_to_color_intensity(speed, min_speed=0.0, max_speed=0.05, base_color=np.array([0.2, 0.4, 0.8])):
    """Convert speed to color intensity (darker = faster, lighter = slower)"""
    # Normalize speed to [0, 1] range
    if max_speed <= min_speed:
        normalized_speed = 0.5
    else:
        normalized_speed = np.clip((speed - min_speed) / (max_speed - min_speed), 0.0, 1.0)
    
    # Create color intensity: 0.3 (light) to 1.0 (dark)
    intensity = 0.3 + (0.7 * normalized_speed)
    
    # Apply intensity to base color
    return base_color * intensity

def load_trajectory_data(trajectory_files):
    """Load and parse trajectory data from JSON files"""
    trajectories = {}
    
    for file_path in trajectory_files:
        if not os.path.exists(file_path):
            print(f"Warning: Trajectory file not found: {file_path}")
            continue
            
        # Extract robot and parcel ID from filename
        filename = os.path.basename(file_path)
        # Format: robot{X}_parcel{Y}_approach_robot_trajectory.json
        match = re.match(r'robot(\d+)_parcel(\d+)_approach_robot_trajectory\.json', filename)
        if not match:
            print(f"Warning: Cannot parse filename format: {filename}")
            continue
            
        robot_id = match.group(1)
        parcel_id = match.group(2)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract trajectory data: [timestamp, x, y, theta, v, omega]
            if 'data' in data and isinstance(data['data'], list):
                trajectory_points = []
                for point in data['data']:
                    if len(point) >= 6:  # Ensure we have all required fields
                        timestamp, x, y, theta, v, omega = point[:6]
                        trajectory_points.append({
                            'timestamp': timestamp,
                            'x': x,
                            'y': y, 
                            'theta': theta,
                            'velocity': abs(v),  # Use absolute velocity magnitude
                            'omega': omega
                        })
                
                if trajectory_points:
                    key = f"robot_{robot_id}_parcel_{parcel_id}"
                    trajectories[key] = {
                        'robot_id': robot_id,
                        'parcel_id': parcel_id, 
                        'points': trajectory_points
                    }
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error loading {file_path}: {e}")
            continue
    
    return trajectories

def visualize_trajectory_with_speed(trajectory_files, output_file=None, show_figure=True):
    """Visualize robot trajectories with parcel-based colors and speed-based intensity"""
    # Load trajectory data
    trajectories = load_trajectory_data(trajectory_files)
    
    if not trajectories:
        print("Error: No valid trajectory data found")
        return
    
    # Get color families for parcels
    color_families = get_parcel_color_families()
    
    # Calculate global speed range for consistent coloring
    all_speeds = []
    for traj_data in trajectories.values():
        for point in traj_data['points']:
            all_speeds.append(point['velocity'])
    
    if not all_speeds:
        print("Error: No speed data found")
        return
        
    min_speed = min(all_speeds)
    max_speed = max(all_speeds)
    print(f"Speed range: {min_speed:.4f} to {max_speed:.4f} m/s")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot trajectories
    legend_patches = []
    plotted_parcels = set()
    
    for key, traj_data in trajectories.items():
        robot_id = traj_data['robot_id']
        parcel_id = traj_data['parcel_id']
        points = traj_data['points']
        
        if len(points) < 2:
            continue
            
        # Get base color for this parcel
        base_color = color_families.get(parcel_id, np.array([0.5, 0.5, 0.5]))
        
        # Prepare line segments and colors
        line_segments = []
        line_colors = []
        
        for i in range(len(points) - 1):
            # Create line segment
            p1 = points[i]
            p2 = points[i + 1]
            line_segments.append([(p1['x'], p1['y']), (p2['x'], p2['y'])])
            
            # Calculate average speed for this segment
            avg_speed = (p1['velocity'] + p2['velocity']) / 2.0
            
            # Get color based on speed
            color = speed_to_color_intensity(avg_speed, min_speed, max_speed, base_color)
            line_colors.append(color)
        
        # Create LineCollection for efficient rendering
        lc = LineCollection(line_segments, colors=line_colors, linewidths=2.5, alpha=0.8)
        ax.add_collection(lc)
        
        # Add legend entry (only once per parcel)
        if parcel_id not in plotted_parcels:
            legend_patches.append(mpatches.Patch(color=base_color, 
                                               label=f'Parcel {parcel_id}'))
            plotted_parcels.add(parcel_id)
        
        # Mark start and end points
        start_point = points[0]
        end_point = points[-1]
        
        # Start point (circle)
        ax.plot(start_point['x'], start_point['y'], 'o', 
                color=base_color * 0.7, markersize=8, markeredgecolor='black', 
                markeredgewidth=1, alpha=0.9)
        
        # End point (square)
        ax.plot(end_point['x'], end_point['y'], 's', 
                color=base_color * 0.7, markersize=8, markeredgecolor='black', 
                markeredgewidth=1, alpha=0.9)
        
        # Add robot ID label at start
        ax.annotate(f'R{robot_id}',
                   (start_point['x'], start_point['y']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Set equal aspect ratio and proper limits
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=14)
    ax.set_ylabel('Y Position (m)', fontsize=14)
    ax.set_title('Robot Trajectories - Parcel-based Colors with Speed Intensity\n'
                 f'(Darker = Faster, Lighter = Slower | Speed Range: {min_speed:.3f}-{max_speed:.3f} m/s)',
                 fontsize=16)
    
    # Add legend
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1),
                 frameon=True, fontsize=12)

    # Add speed intensity explanation
    speed_text = ("Color Intensity Guide:\n"
                  "• Darker colors = Higher speeds\n"
                  "• Lighter colors = Lower speeds\n"
                  "• Circle = Start, Square = End")
    ax.text(0.02, 0.02, speed_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Trajectory visualization saved to {output_file}")
    
    # Show figure if requested
    if show_figure:
        plt.show()
    else:
        plt.close()

def visualize_timing_data(json_file_path, output_file=None, show_figure=True):
    """Visualize timing data from a JSON file"""
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"Error: File {json_file_path} does not exist")
        return

    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Extract data for visualization
    events = []
    for key, value in data.items():
        # Skip entries with missing required fields
        if not all(k in value for k in ["node_name", "start_time", "end_time"]):
            print(f"Warning: Skipping entry {key} due to missing required fields")
            continue

        # Only include specified node types
        allowed_nodes = ["WaitingPush", "ApproachingPush", "Pushing", "BackwardToSafeDistance", "PickingUp"]
        if value["node_name"] not in allowed_nodes:
            continue

        # Parse datetime strings
        try:
            start_time = parse_datetime(value["start_time"])
            end_time = parse_datetime(value["end_time"])
        except ValueError as e:
            print(f"Warning: Skipping entry {key} due to datetime parsing error: {e}")
            continue

        # Extract robot_id
        robot_id = value.get("robot_id", "unknown")
        if robot_id == "":
            robot_id = "unknown"
        
        # Add to events list
        events.append({
            "node_name": value["node_name"],
            "robot_id": str(robot_id),  # Convert to string for consistency
            "parcel_id": value.get("parcel_id", ""),
            "start_time": start_time,
            "end_time": end_time,
            "duration": (end_time - start_time).total_seconds(),
            "status": value.get("status", "Unknown")
        })

    # Sort events by start time
    events.sort(key=lambda x: x["start_time"])
    
    # Apply timing corrections
    events = correct_timing_data(events)

    # Exit if no events found
    if not events:
        print(f"Error: No valid timing data found in {json_file_path}")
        return

    # Get unique node names and robot IDs
    unique_node_names = sorted(set(event["node_name"] for event in events))
    unique_robot_ids = sorted(set(event["robot_id"] for event in events))
    
    # Get unique parcel IDs (excluding empty/invalid ones) for Pushing nodes only
    unique_parcel_ids = set()
    for event in events:
        if event["node_name"] == "Pushing":
            parcel_id = event.get("parcel_id", "")
            if parcel_id is not None and str(parcel_id).strip() not in ["", "-1"]:
                unique_parcel_ids.add(str(parcel_id))
    unique_parcel_ids = sorted(unique_parcel_ids)

    # Create a color map for node names (original behavior-based colors)
    colors = plt.cm.tab10.colors
    node_colors = {node_name: colors[i % len(colors)] for i, node_name in enumerate(unique_node_names)}
    
    # Create a color map for parcel IDs (only for Pushing behavior)
    parcel_colors = {}
    if unique_parcel_ids:
        pushing_base_color = node_colors.get("Pushing", colors[0])
        # Generate red-orange variations of the pushing color
        import matplotlib.colors as mcolors
        for i, parcel_id in enumerate(unique_parcel_ids):
            # Keep colors in red-orange spectrum, higher IDs get redder
            hsv = mcolors.rgb_to_hsv(pushing_base_color[:3])
            # Reverse hue shift: higher indices (IDs) get redder (lower hue values)
            reverse_i = len(unique_parcel_ids) - 1 - i  # Reverse the index
            new_hue = max(0.0, hsv[0] - (reverse_i * 0.02))  # Higher IDs get redder
            # Saturation variations
            new_sat = max(0.6, min(1.0, hsv[1] + (i * 0.15) - 0.2))  # Moderate saturation range
            # Brightness variations: REVERSE - higher IDs get DARKER (lower brightness)
            new_val = max(0.4, min(0.8, hsv[2] - (i * 0.15)))  # Higher ID = lower brightness (darker)
            new_hsv = (new_hue, new_sat, new_val)
            parcel_colors[parcel_id] = mcolors.hsv_to_rgb(new_hsv)

    # Create the figure with adjusted height for tighter spacing
    fig_height = max(3, len(unique_robot_ids) * 0.6)  # Tighter spacing for single robot view
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Plot horizontal bars for each event
    for event in events:
        y_pos = unique_robot_ids.index(event["robot_id"])
        start = event["start_time"]
        duration = event["duration"]
        
        # Get color based on behavior type and parcel ID
        if event["node_name"] == "Pushing":
            # Use parcel-specific color for Pushing behavior
            parcel_id = event.get("parcel_id", "")
            if parcel_id is not None and str(parcel_id).strip() not in ["", "-1"] and str(parcel_id) in parcel_colors:
                color = parcel_colors[str(parcel_id)]
            else:
                color = node_colors[event["node_name"]]  # Use default pushing color
        else:
            # Use behavior-based color for all other behaviors
            color = node_colors[event["node_name"]]
        
        # Plot the bar with opacity based on status (SUCCESS = 0.6, others = 0.4)
        alpha = 0.6 if "SUCCESS" in event["status"] else 0.4
        ax.barh(y_pos, duration, left=mdates.date2num(start), height=0.5, 
                color=color, alpha=alpha)

    # Configure the y-axis with robot IDs
    y_labels = [f"Robot {rid}" for rid in unique_robot_ids]
    ax.set_yticks(range(len(unique_robot_ids)))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Robot ID")

    # Configure the x-axis for time display
    min_time = min(event["start_time"] for event in events)
    max_time = max(event["end_time"] for event in events)
    
    # Format time axis based on total duration
    total_duration = (max_time - min_time).total_seconds()
    
    if total_duration < 60:  # Less than a minute
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=5))
    elif total_duration < 3600:  # Less than an hour
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    else:  # More than an hour
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    # Use relative time instead of absolute datetime
    for event in events:
        event["start_offset"] = (event["start_time"] - min_time).total_seconds()
        event["end_offset"] = (event["end_time"] - min_time).total_seconds()
        
    # Replot with relative time
    ax.clear()
    for event in events:
        y_pos = unique_robot_ids.index(event["robot_id"])
        start = event["start_offset"]
        duration = event["duration"]
        
        # Get color based on behavior type and parcel ID
        if event["node_name"] == "Pushing":
            # Use parcel-specific color for Pushing behavior
            parcel_id = event.get("parcel_id", "")
            if parcel_id is not None and str(parcel_id).strip() not in ["", "-1"] and str(parcel_id) in parcel_colors:
                color = parcel_colors[str(parcel_id)]
            else:
                color = node_colors[event["node_name"]]  # Use default pushing color
        else:
            # Use behavior-based color for all other behaviors
            color = node_colors[event["node_name"]]
        
        # Plot the bar with opacity based on status (SUCCESS = 0.6, others = 0.4)
        alpha = 0.6 if "SUCCESS" in event["status"] else 0.4
        ax.barh(y_pos, duration, left=start, height=0.3, 
                color=color, alpha=alpha)
        
        # Display duration time on each bar (only if duration is long enough to fit text)
        duration_text = f"{duration:.1f}s" if duration < 60 else f"{duration/60:.1f}m"
        # Only show text if bar is wide enough (at least 5 seconds)
        if duration > 5:
            ax.text(start + duration/2, y_pos, duration_text,
                    ha='center', va='center', color='white', fontsize=6,
                    fontweight='bold', alpha=0.9)

        # Add behavior type annotation on each bar
        # behavior_text = event["node_name"][:4]  # Abbreviate behavior names
        # ax.text(start + duration/2, y_pos + 0.15, behavior_text,
        #         ha='center', va='bottom', color='black', fontsize=6,
        #         fontweight='bold', alpha=0.8)

        # Display parcel ID if available
        parcel_id = event.get("parcel_id", "")
        # Fix: Check for None and empty string, but allow 0 as a valid parcel ID
        if parcel_id is not None and str(parcel_id).strip() not in ["", "-1"]:
            # Position the parcel ID text below the bar (only if bar is wide enough)
            if duration > 5:
                ax.text(start + duration/2, y_pos - 0.25, f"Parcel {parcel_id}",
                        ha='center', va='top', color='darkblue', fontsize=6,
                        style='italic', alpha=0.9)

    # Configure the y-axis with robot IDs and tighter spacing
    y_labels = [f"{rid}" for rid in unique_robot_ids]  # Only show ID numbers
    ax.set_yticks(range(len(unique_robot_ids)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_ylabel("Robot ID", fontsize=8)
    ax.yaxis.set_label_coords(-0.03, 0.0)  # Position y-label at bottom of axis, closer to y-axis
    # Set tighter y-axis limits to reduce space between bars, with more space at bottom for parcel text
    ax.set_ylim(-0.6, len(unique_robot_ids) - 0.7)

    # Configure time axis labels
    def format_seconds(seconds):
        return f"{seconds:.0f}s"

    # Set x-axis ticks at reasonable intervals (wider spacing for small figure)
    if total_duration < 60:  # Less than a minute
        interval = 10  # 10 seconds
    elif total_duration < 300:  # Less than 5 minutes
        interval = 30  # 30 seconds
    elif total_duration < 600:  # Less than 10 minutes
        interval = 60  # 60 seconds
    else:  # More than 10 minutes
        interval = 60  # 1 minute (reduced from 2 minutes)

    ticks = np.arange(0, total_duration + interval, interval)
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_seconds(t) for t in ticks], fontsize=7, rotation=0)
    ax.set_xlabel("Time", fontsize=8)
    ax.xaxis.set_label_coords(0.98, -0.15)  # Position x-label at right end of axis (further down)
    ax.tick_params(axis='both', which='major', labelsize=7, width=0.5, length=2, pad=1)
    plt.xticks(rotation=0)  # Keep x-axis labels flat

    # Add grid lines for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7, linewidth=0.3)

    # Create simplified legend - combine all pushing into one
    legend_patches = []

    # Map full names to simplified labels
    label_map = {
        "WaitingPush": "Wait",
        "ApproachingPush": "Approach",
        "Pushing": "Push",
        "BackwardToSafeDistance": "Backward",
        "PickingUp": "Pickup"
    }

    # Add behavior types (non-pushing)
    for node in unique_node_names:
        if node != "Pushing":
            # Use alpha to match the bar appearance
            color_with_alpha = list(node_colors[node]) + [0.6]  # Default to SUCCESS alpha
            simplified_label = label_map.get(node, node)
            legend_patches.append(mpatches.Patch(color=color_with_alpha, label=simplified_label))

    # Add single "Pushing" entry regardless of parcels
    if "Pushing" in unique_node_names:
        color_with_alpha = list(node_colors["Pushing"]) + [0.6]
        legend_patches.append(mpatches.Patch(color=color_with_alpha, label="Push"))

    if legend_patches:  # Only show legend if there are items
        # Keep legend in one row
        ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=len(legend_patches), frameon=True, fontsize=7)

    # Add title
    robot_id = os.path.basename(json_file_path).replace('Time_', '').replace('.json', '')
    plt.title(f"Behavior Tree Timing Visualization - {robot_id}", fontsize=8, pad=3)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.08, right=0.98)  # Reduce side margins

    # Set x-axis limits AFTER tight_layout to avoid being overridden
    max_offset = max(event["end_offset"] for event in events)
    ax.set_xlim(-max_offset * 0.02, max_offset * 1.01)  # Minimal margins

    # Save the figure first if an output path is provided
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Visualization saved to {output_file}")

    # Show the figure after saving if requested
    if show_figure:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize behavior tree timing data")
    parser.add_argument("json_file", nargs='+', help="Path(s) to the JSON file(s) with timing data")
    parser.add_argument("-o", "--output", help="Output file path (optional)")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    parser.add_argument("--combined", action="store_true", help="Plot all files on the same figure")
    args = parser.parse_args()
    
    # Default output path if not specified
    output_file = args.output
    if not output_file:
        if len(args.json_file) == 1:
            base_dir = os.path.dirname(args.json_file[0])
            file_name = os.path.basename(args.json_file[0]).replace('Time_', 'timing_viz_').replace('.json', '.png')
            output_file = os.path.join(base_dir, file_name)
        else:
            base_dir = os.path.dirname(args.json_file[0])
            output_file = os.path.join(base_dir, 'combined_timing_visualization.png')
    
    # Plot data
    if len(args.json_file) > 1 or args.combined:
        visualize_combined_timing_data(args.json_file, output_file, show_figure=args.show)
    else:
        visualize_timing_data(args.json_file[0], output_file, show_figure=args.show)

# Additional convenience function to plot both robot files
def plot_both_robots(data_dir="/root/workspace/data/experi", output_file=None):
    """Convenience function to plot both robot timing files"""
    robot0_file = os.path.join(data_dir, "Time_robot0.json")
    robot1_file = os.path.join(data_dir, "Time_robot1.json")
    
    if not output_file:
        output_file = os.path.join(data_dir, "combined_robots_timing.png")
    
    json_files = [robot0_file, robot1_file]
    visualize_combined_timing_data(json_files, output_file, show_figure=False)
    print(f"Combined visualization saved to: {output_file}")

# Run the convenience function if script is executed directly
if __name__ == "__main__" and len(sys.argv) == 1:
    # If no arguments provided, plot both robot files by default
    plot_both_robots()