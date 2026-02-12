"""
Batch generate 2D trajectory with velocity color coding for different MAP cases.
Replicates the exact visualization from trajectory_visualization.py (ax2 subplot).
Uses the SAME data loading and trajectory generation logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
import json
import os

# ========== HARDCODED CASES ==========
# Note: MAP5_1 excluded - no trajectory data found
CASES = ["MAP1", "MAP2", "MAP3", "MAP4", "MAP5", "MAP6", "warehouse"]
BASE_PATH = "result/MAPS"
OUTPUT_PATH = "result/MAPS/separate_plots"

# ========== ROBOT PARAMETERS ==========
PIXEL_TO_METER_SCALE = 0.0023
v_max = 0.03  # max linear velocity (m/s)


def load_reeb_graph(case_name, use_rebuilt=False):
    """
    Load Reeb graph for waypoint positions.

    Args:
        case_name: MAP case name (e.g., "MAP1")
        use_rebuilt: If True, load Graph_new_{case}.json (for assignment visualization)
                     If False, load Graph_{case}.json (for trajectory data lookup)
    """
    case_lower = case_name.lower()

    # Determine which graph file to load
    if use_rebuilt:
        # Use rebuilt graph for assignment visualization
        graph_file = os.path.join(BASE_PATH, case_name, f"Graph_new_{case_lower}.json")
    else:
        # Use original graph for trajectory data
        # Special case: MAP5_1 uses Graph_map5.json not Graph_map5_1.json
        if case_name == "MAP5_1":
            graph_file = os.path.join(BASE_PATH, case_name, "Graph_map5.json")
        else:
            graph_file = os.path.join(BASE_PATH, case_name, f"Graph_{case_lower}.json")

    if not os.path.exists(graph_file):
        print(f"[X] Graph file not found: {graph_file}")
        return None

    try:
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)

        # Create simple graph object with nodes
        # Graph format: {"nodes": [[node_id, [x, y], ...], ...]}
        class SimpleGraph:
            def __init__(self, nodes_list, in_neighbors, out_neighbors):
                self.nodes = {}
                for node_entry in nodes_list:
                    node_id = node_entry[0]
                    node_position = node_entry[1]  # [x, y] in pixels
                    self.nodes[node_id] = type('Node', (), {
                        'configuration': node_position
                    })()
                # Convert string keys to int for neighbors
                self.in_neighbors = {int(k): v for k, v in in_neighbors.items()}
                self.out_neighbors = {int(k): v for k, v in out_neighbors.items()}

        return SimpleGraph(graph_data['nodes'],
                          graph_data.get('in_neighbors', {}),
                          graph_data.get('out_neighbors', {}))

    except Exception as e:
        print(f"[X] Failed to load graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_robot_data(case_name, robot_id):
    """Load single robot trajectory data."""
    case_lower = case_name.lower()

    # warehouse case: files directly in warehouse/ directory
    if case_name.lower() == 'warehouse':
        robot_file = os.path.join(BASE_PATH, case_name,
                                  f"robot_{robot_id}_trajectory_parameters_{case_lower}.json")
    else:
        # MAP cases have nested structure: MAP*/{MAP*}/{case_lower}/robot_...
        robot_file = os.path.join(BASE_PATH, case_name, case_name, case_lower,
                                  f"robot_{robot_id}_trajectory_parameters_{case_lower}.json")

    if not os.path.exists(robot_file):
        return None

    try:
        with open(robot_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def convert_pixel_to_meter(pos):
    """Convert pixel coordinates to meters."""
    return (pos[0] * PIXEL_TO_METER_SCALE, pos[1] * PIXEL_TO_METER_SCALE)


def generate_robot_trajectory(robot_data, reeb_graph):
    """
    Generate trajectory points using EXACT same logic as trajectory_visualization.py.
    Returns: all_points_x, all_points_y, all_velocities, wp_x, wp_y, relay_indices, arc_annotations

    arc_annotations: List of (arc_mid_x, arc_mid_y, radius_value, annotation_offset_x, annotation_offset_y)
    """
    waypoints = robot_data['waypoints']
    phi = robot_data['phi']
    r0 = robot_data['r0']
    l = robot_data['l']
    phi_new = robot_data.get('phi_new', phi)
    time_segments = robot_data['time_segments']
    Flagb = robot_data.get('Flagb', [0] * len(waypoints))

    # Extract waypoint positions
    # Use waypoint_positions if available (already in meters!), otherwise lookup from graph
    if 'waypoint_positions' in robot_data and robot_data['waypoint_positions']:
        wp_x = [pos[0] for pos in robot_data['waypoint_positions']]
        wp_y = [pos[1] for pos in robot_data['waypoint_positions']]
    else:
        # Fallback: lookup from reeb graph (in pixels, need conversion)
        wp_x = []
        wp_y = []
        for wp_idx in waypoints:
            if wp_idx in reeb_graph.nodes:
                node_pos_pixel = reeb_graph.nodes[wp_idx].configuration
                world_pos = convert_pixel_to_meter(node_pos_pixel)
                wp_x.append(world_pos[0])
                wp_y.append(world_pos[1])
            else:
                print(f"[!] Warning: waypoint {wp_idx} not in graph")
                # Return empty trajectory
                return [], [], [], [], [], []

    # Initialize trajectory arrays
    all_points_x = []
    all_points_y = []
    all_velocities = []
    cumulative_time = 0.0
    arc_annotations = []  # Store arc radius annotations (mid_x, mid_y, radius, offset_x, offset_y)

    # Process each segment (EXACT COPY from trajectory_visualization.py lines 188-364)
    for i in range(len(waypoints) - 1):
        # Get starting position and angle
        flagb_i = Flagb[i] if i < len(Flagb) else 0
        phi1 = phi[i] + flagb_i * np.pi / 2
        angle_start = phi1

        x_start = wp_x[i]
        y_start = wp_y[i]

        # Calculate arc center
        r_x = x_start - r0[i] * np.cos(phi1 + np.pi / 2)
        r_y = y_start - r0[i] * np.sin(phi1 + np.pi / 2)

        # Add starting point if first segment
        if i == 0:
            all_points_x.append(x_start)
            all_points_y.append(y_start)
            all_velocities.append(0.0)

        # Process arc segment
        if i < len(time_segments) and 'arc' in time_segments[i] and len(time_segments[i]['arc']) > 0:
            arc_times = time_segments[i]['arc']
            delta_phi = phi[i+1] - phi_new[i]
            arc_radius = r0[i]

            if len(arc_times) > 0:
                N_arc = len(arc_times)

                for j in range(1, N_arc + 1):
                    angle_fraction = j / N_arc
                    current_angle = phi1 + delta_phi * angle_fraction

                    point_x = r0[i] * np.cos(current_angle + np.pi / 2) + r_x
                    point_y = r0[i] * np.sin(current_angle + np.pi / 2) + r_y

                    cumulative_time += arc_times[j-1]

                    # Calculate velocity
                    total_arc_length = abs(arc_radius * delta_phi)
                    arc_segment_length = total_arc_length / N_arc
                    velocity = arc_segment_length / arc_times[j-1] if arc_times[j-1] > 0 else 0

                    all_points_x.append(point_x)
                    all_points_y.append(point_y)
                    all_velocities.append(velocity)

                    # Store arc annotation at midpoint
                    if j == (N_arc // 2):  # Midpoint of arc
                        # Place text outside the arc (beyond the arc point)
                        # Calculate annotation offset direction (perpendicular to arc, pointing outward from center)
                        # For positive radius: offset away from center
                        # For negative radius: offset toward center (which is actually away from curve)
                        offset_angle = current_angle + np.pi / 2
                        offset_distance = 0.12  # 12cm offset outside the arc
                        offset_direction = np.sign(arc_radius) if abs(arc_radius) > 0.001 else 1.0

                        annotation_x = point_x + offset_direction * offset_distance * np.cos(offset_angle)
                        annotation_y = point_y + offset_direction * offset_distance * np.sin(offset_angle)

                        # Store: (text_x, text_y, radius_value)
                        arc_annotations.append((annotation_x, annotation_y, abs(arc_radius)))

        # Get current position after arc
        if len(all_points_x) > 0:
            x_after_arc = all_points_x[-1]
            y_after_arc = all_points_y[-1]
        else:
            x_after_arc = x_start
            y_after_arc = y_start

        # Process line segment
        if i < len(time_segments) and 'line' in time_segments[i] and len(time_segments[i]['line']) > 0:
            line_times = time_segments[i]['line']
            line_length = l[i]

            if line_length > 0.001:
                N_line = len(line_times)
                is_straight_line = abs(r0[i]) < 0.01

                if is_straight_line:
                    l_x = x_start
                    l_y = y_start
                    phi1_line = phi_new[i] if i < len(phi_new) else phi[i]
                else:
                    l_x = r0[i] * np.cos(phi[i+1] + np.pi / 2) + r_x
                    l_y = r0[i] * np.sin(phi[i+1] + np.pi / 2) + r_y
                    phi1_line = phi[i+1]

                for j in range(N_line + 1):
                    if j == 0:
                        point_x = l_x
                        point_y = l_y
                    else:
                        segment_length = line_length / N_line
                        l_delta = segment_length * j
                        point_x = l_x + l_delta * np.cos(phi1_line)
                        point_y = l_y + l_delta * np.sin(phi1_line)

                    if j > 0:
                        cumulative_time += line_times[j-1]
                        line_segment_length = line_length / N_line
                        velocity = line_segment_length / line_times[j-1] if line_times[j-1] > 0 else 0
                        all_velocities.append(velocity)
                    else:
                        if len(all_velocities) > 0:
                            all_velocities.append(all_velocities[-1])
                        else:
                            all_velocities.append(0.0)

                    all_points_x.append(point_x)
                    all_points_y.append(point_y)

    # Get relay point indices
    relay_indices = [i for i in range(len(Flagb)) if i < len(Flagb) and Flagb[i] != 0]

    return all_points_x, all_points_y, all_velocities, wp_x, wp_y, relay_indices, arc_annotations


def load_environment_data(case_name):
    """Load environment data (obstacles)."""
    case_lower = case_name.lower()
    env_file = os.path.join(BASE_PATH, case_name, f"environment_{case_lower}.json")

    if not os.path.exists(env_file):
        print(f"[!] Environment file not found: {env_file}")
        return None

    try:
        with open(env_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Failed to load environment: {e}")
        return None


def load_assignment_data(case_name):
    """Load assignment result data (waypoints and relay points)."""
    case_lower = case_name.lower()

    # Determine N from first robot data
    # Try to load from result/MAPS/{CASE}/AssignmentResult{N}{case}.json
    for N in [4, 5, 3, 6]:  # Common robot counts
        assignment_file = os.path.join(BASE_PATH, case_name, f"AssignmentResult{N}{case_lower}.json")
        if os.path.exists(assignment_file):
            try:
                with open(assignment_file, 'r') as f:
                    data = json.load(f)
                return data.get('Waypoints', []), data.get('RelayPoints', [])
            except Exception as e:
                print(f"[!] Failed to load assignment: {e}")
                return [], []

    return [], []


def plot_case_velocities(case_name, robot_data_list, reeb_graph):
    """
    Plot velocity-colored trajectories for all robots in two subfigures:
    Left: Assignment result with arrows
    Right: Velocity-colored trajectories
    """
    # Manual subplot positioning for perfect alignment
    # Convert 10cm x 4cm to inches (1 inch = 2.54 cm)
    fig = plt.figure(figsize=(10/2.54, 4/2.54))

    # Define exact subplot positions [left, bottom, width, height] in figure coordinates
    # Leave space at bottom for legend/colorbar (0.18 from bottom)
    subplot_height = 0.70  # 70% of figure height for plots
    subplot_width = 0.38   # 38% of figure width for each subplot
    bottom_margin = 0.25   # Start plots at 25% from bottom (leave space for legend/colorbar)

    # Left subplot (Assignment)
    ax_assignment = fig.add_axes([0.08, bottom_margin, subplot_width, subplot_height])
    # Right subplot (Velocity) - same bottom and height!
    ax_velocity = fig.add_axes([0.54, bottom_margin, subplot_width, subplot_height])

    # Create colormap (viridis: blue to yellow - EXACT same as reference)
    try:
        cmap = plt.colormaps['viridis']  # New API (matplotlib >= 3.7)
    except AttributeError:
        cmap = plt.cm.get_cmap('viridis')  # Fallback for older versions

    # Collect all velocities for normalization
    all_case_velocities = []
    trajectory_data = []

    for robot_data in robot_data_list:
        if robot_data is None:
            continue

        all_points_x, all_points_y, all_velocities, wp_x, wp_y, relay_indices, arc_annotations = \
            generate_robot_trajectory(robot_data, reeb_graph)

        trajectory_data.append({
            'x': all_points_x,
            'y': all_points_y,
            'v': all_velocities,
            'wp_x': wp_x,
            'wp_y': wp_y,
            'relay': relay_indices,
            'arcs': arc_annotations
        })

        all_case_velocities.extend(all_velocities)

    # Set velocity normalization (EXACT same as reference)
    min_velocity = 0.0
    max_velocity = v_max
    if all_case_velocities:
        actual_min_vel = min(v for v in all_case_velocities if v is not None)
        actual_max_vel = max(v for v in all_case_velocities if v is not None)
        min_velocity = max(0.0, actual_min_vel)
        max_velocity = max(min_velocity + 0.001, actual_max_vel)

    norm = Normalize(vmin=min_velocity, vmax=max_velocity)

    # ========== RIGHT SUBPLOT: Velocity-colored trajectories ==========
    for traj in trajectory_data:
        all_points_x = traj['x']
        all_points_y = traj['y']
        all_velocities = traj['v']
        wp_x = traj['wp_x']
        wp_y = traj['wp_y']
        relay_indices = traj['relay']
        arc_annotations = traj['arcs']

        # Plot trajectory with color based on velocity
        if len(all_points_x) > 1:
            for i in range(len(all_points_x) - 1):
                current_velocity = all_velocities[i] if all_velocities[i] is not None else 0.0
                color = cmap(norm(current_velocity))
                ax_velocity.plot([all_points_x[i], all_points_x[i+1]],
                                [all_points_y[i], all_points_y[i+1]],
                                color=color, linewidth=0.8, alpha=0.8)

        # Plot waypoints (green circles) - NO LABEL for legend
        ax_velocity.scatter(wp_x, wp_y, color='green', s=8, marker='o', zorder=5)

        # Plot relay points (red triangles) - NO LABEL for legend
        if relay_indices:
            relay_x = [wp_x[i] for i in relay_indices]
            relay_y = [wp_y[i] for i in relay_indices]
            ax_velocity.scatter(relay_x, relay_y, color='red', s=9, marker='^', zorder=5)

        # Add arc radius annotations
        for annotation_x, annotation_y, radius_value in arc_annotations:
            # Format radius: show 2 decimal places for small values, 1 for large
            if abs(radius_value) < 1.0:
                radius_text = f'r={radius_value:.2f}'
            else:
                radius_text = f'r={radius_value:.1f}'

            ax_velocity.text(annotation_x, annotation_y, radius_text,
                           fontsize=5, color='darkblue', ha='center', va='center',
                           zorder=10)

    # Add colorbar manually at specific position to align with legend
    # Create colorbar axes: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.54, 0.08, 0.38, 0.04])  # Match right subplot horizontal position
    cbar_velocity = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax,
                                  orientation='horizontal')
    # cbar_velocity.set_label('Velocity (m/s)', fontsize=5)
    cbar_velocity.ax.tick_params(labelsize=4)

    # Set axis properties for velocity subplot
    ax_velocity.set_xlabel('X (m)', fontsize=5, labelpad=1, loc='right')
    ax_velocity.set_ylabel('Y (m)', fontsize=5, labelpad=1, loc='top')
    ax_velocity.set_title('Planning Trajectory', fontsize=6, pad=2)
    ax_velocity.set_aspect('equal', adjustable='datalim')
    ax_velocity.grid(True, linewidth=0.3)
    # NO legend on velocity subplot - unified legend will be on assignment subplot
    ax_velocity.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2, pad=1)

    # ========== Calculate UNIFIED axis limits BEFORE plotting ==========
    # Collect all coordinates from trajectory data ONLY (not obstacles - they may extend beyond visible area)
    all_x_coords = []
    all_y_coords = []
    for traj in trajectory_data:
        all_x_coords.extend(traj['x'])
        all_y_coords.extend(traj['y'])

    # Load environment data for obstacles (will be plotted later)
    env_data = load_environment_data(case_name)

    # Calculate bounds based on trajectory data with padding
    if all_x_coords and all_y_coords:
        x_min, x_max = min(all_x_coords), max(all_x_coords)
        y_min, y_max = min(all_y_coords), max(all_y_coords)
        padding = 0.1  # 10% padding around trajectory
        x_range = x_max - x_min
        y_range = y_max - y_min
        unified_xlim = (max(0, x_min - padding * x_range), x_max + padding * x_range)
        unified_ylim = (max(0, y_min - padding * y_range), y_max + padding * y_range)
    else:
        unified_xlim = (0, 2.5)
        unified_ylim = (0, 1.5)

    # ========== LEFT SUBPLOT: Assignment result with arrows (METER COORDINATES) ==========

    # Load REBUILT graph for assignment visualization (Graph_new_{case}.json)
    reeb_graph_rebuilt = load_reeb_graph(case_name, use_rebuilt=True)
    if reeb_graph_rebuilt is None:
        print(f"[!] Warning: Rebuilt graph not found for {case_name}, using original graph")
        reeb_graph_rebuilt = reeb_graph

    # Draw environment obstacles (black polygons) - CONVERT TO METERS
    # env_data already loaded above for axis limits calculation
    if env_data and 'polygons' in env_data:
        for polygon_data in env_data['polygons']:
            vertices_pixel = np.array(polygon_data['vertices'])
            # Convert to meters
            vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
            # Close the polygon by adding first vertex at the end
            vertices_closed = np.vstack([vertices_meter, vertices_meter[0]])
            ax_assignment.fill(vertices_closed[:, 0], vertices_closed[:, 1],
                              color='black', alpha=1.0, zorder=1)

    # Draw graph nodes (bigger gray circles) - CONVERT TO METERS
    for node_id in reeb_graph_rebuilt.nodes:
        pos_pixel = reeb_graph_rebuilt.nodes[node_id].configuration
        pos_meter = convert_pixel_to_meter(pos_pixel)
        ax_assignment.scatter(pos_meter[0], pos_meter[1], color='grey', s=8, marker='o',
                             alpha=0.6, zorder=2, edgecolors='darkgrey', linewidths=0.3)

    # Draw graph structure (grey lines connecting nodes) - CONVERT TO METERS
    for node_id, out_neighbors in reeb_graph_rebuilt.out_neighbors.items():
        if node_id in reeb_graph_rebuilt.nodes:
            start_pixel = reeb_graph_rebuilt.nodes[node_id].configuration
            start_meter = convert_pixel_to_meter(start_pixel)
            for neighbor_id in out_neighbors:
                if neighbor_id in reeb_graph_rebuilt.nodes:
                    end_pixel = reeb_graph_rebuilt.nodes[neighbor_id].configuration
                    end_meter = convert_pixel_to_meter(end_pixel)
                    ax_assignment.plot([start_meter[0], end_meter[0]],
                                      [start_meter[1], end_meter[1]],
                                      'grey', linewidth=0.5, alpha=0.5, zorder=2)

    # Load assignment data
    waypoints_arcs, relay_arcs = load_assignment_data(case_name)

    # Track if we've added legend labels (only label first occurrence)
    waypoint_label_added = False
    relay_label_added = False

    if waypoints_arcs:
        # Draw waypoint arrows (green) - CONVERT TO METERS
        for i, j, _ in waypoints_arcs:
            if i in reeb_graph_rebuilt.nodes and j in reeb_graph_rebuilt.nodes:
                start_pixel = reeb_graph_rebuilt.nodes[i].configuration
                end_pixel = reeb_graph_rebuilt.nodes[j].configuration
                start_meter = convert_pixel_to_meter(start_pixel)
                end_meter = convert_pixel_to_meter(end_pixel)

                # Only add label for first waypoint arrow
                label = 'Waypoints' if not waypoint_label_added else ''
                waypoint_label_added = True

                ax_assignment.arrow(start_meter[0], start_meter[1],
                                   end_meter[0] - start_meter[0], end_meter[1] - start_meter[1],
                                   width=0.002, head_width=0.015, head_length=0.012,
                                   fc='green', ec='green', alpha=0.8, zorder=5,
                                   label=label)

    if relay_arcs:
        # Draw relay point arrows (red) - CONVERT TO METERS
        for i, j, _ in relay_arcs:
            if i in reeb_graph_rebuilt.nodes and j in reeb_graph_rebuilt.nodes:
                start_pixel = reeb_graph_rebuilt.nodes[i].configuration
                end_pixel = reeb_graph_rebuilt.nodes[j].configuration
                start_meter = convert_pixel_to_meter(start_pixel)
                end_meter = convert_pixel_to_meter(end_pixel)

                # Only add label for first relay point arrow
                label = 'Relay Points' if not relay_label_added else ''
                relay_label_added = True

                ax_assignment.arrow(start_meter[0], start_meter[1],
                                   end_meter[0] - start_meter[0], end_meter[1] - start_meter[1],
                                   width=0.002, head_width=0.015, head_length=0.012,
                                   fc='red', ec='red', alpha=0.8, zorder=6,
                                   label=label)

    # Set axis properties for assignment subplot (METER COORDINATES - UNIFIED)
    ax_assignment.set_xlabel('X (m)', fontsize=5, labelpad=1, loc='right')
    ax_assignment.set_ylabel('Y (m)', fontsize=5, labelpad=1, loc='top')
    ax_assignment.set_title('Assignment Result', fontsize=6, pad=2)
    ax_assignment.set_aspect('equal', adjustable='datalim')
    ax_assignment.grid(True, linewidth=0.3)  # Enable grid to match velocity subplot
    ax_assignment.tick_params(axis='both', which='major', labelsize=6, width=0.5, length=2, pad=1)

    # Add legend manually at specific position (aligned with colorbar height)
    # Use figure legend with bbox in figure coordinates
    handles, labels = ax_assignment.get_legend_handles_labels()
    if handles:
        # Place legend at same vertical position as colorbar (0.08 to 0.12)
        fig.legend(handles, labels, fontsize=4, loc='center',
                  bbox_to_anchor=(0.27, 0.10), ncol=2, frameon=False,
                  columnspacing=1.0, handletextpad=0.5)

    # ========== Apply UNIFIED axis limits to BOTH subplots ==========
    # Always apply unified limits (calculated earlier)
    ax_assignment.set_xlim(unified_xlim)
    ax_assignment.set_ylim(unified_ylim)
    ax_velocity.set_xlim(unified_xlim)
    ax_velocity.set_ylim(unified_ylim)

    # ========== Set UNIFORM tick intervals for both axes ==========
    # Use fixed 0.5m spacing for both X and Y axes
    unified_tick_spacing = 0.5

    # Apply same spacing to both axes
    ax_assignment.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax_assignment.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax_velocity.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax_velocity.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    # Save figure
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'{case_name}_velocity_trajectories.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SAVE] {output_file}")


def main():
    """Main batch processing loop."""
    print("=" * 60)
    print("BATCH VELOCITY VISUALIZATION - Exact Replication")
    print("=" * 60)

    success_count = 0
    failed_cases = []

    for case_name in CASES:
        print(f"\n[*] Processing {case_name}...")

        # Load Reeb graph
        reeb_graph = load_reeb_graph(case_name)
        if reeb_graph is None:
            failed_cases.append(case_name)
            continue

        # Load all robot data
        robot_data_list = []
        robot_id = 0
        while True:
            robot_data = load_robot_data(case_name, robot_id)
            if robot_data is None:
                break
            robot_data_list.append(robot_data)
            robot_id += 1

        if len(robot_data_list) == 0:
            print(f"[X] No robot data found")
            failed_cases.append(case_name)
            continue

        print(f"[OK] Loaded {len(robot_data_list)} robots")

        try:
            plot_case_velocities(case_name, robot_data_list, reeb_graph)
            success_count += 1
        except Exception as e:
            print(f"[X] Plot failed: {e}")
            import traceback
            traceback.print_exc()
            failed_cases.append(case_name)

    # Summary
    print("\n" + "=" * 60)
    print(f"[OK] Success: {success_count}/{len(CASES)}")
    if failed_cases:
        print(f"[X] Failed: {', '.join(failed_cases)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
