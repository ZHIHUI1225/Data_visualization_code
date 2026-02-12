"""
Generate separate plots for the two subfigures from velocity_visualization.py:
1. Assignment Result (arrows showing waypoints and relay points)
2. Velocity-Colored Trajectories

Each subfigure is saved as a separate file.
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
CASES = ["MAP1", "MAP2", "MAP3", "MAP4", "MAP5", "MAP6", "LOOP1", "LOOP2", "warehouse", "maze", "bottleneck_expansion", "large_expansion"]
BASE_PATH = "result/MAPS"
OUTPUT_PATH = "result/MAPS/separate_plots"

# Robot count limits (prevent loading wrong data from mixed directories)
ROBOT_COUNT = {
    "MAP1": 4,
    "MAP2": 5,
    "MAP3": 4,
    "MAP4": 4,
    "MAP5": 4,  # MAP5 has only 4 robots, ignore robot_4 (belongs to MAP5_2)
    "MAP6": 4,
    "LOOP1": 4,  # LOOP1 has 4 robots (robot_0 to robot_3)
    "LOOP2": 5,  # LOOP2 has 5 robots (robot_0 to robot_4)
    "warehouse": 4,  # warehouse has 4 robots (robot_0 to robot_3)
    "maze": 8,
    "bottleneck_expansion": 6,
    "large_expansion": 3
}

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

    # Cases with flat structure: files directly in case directory
    flat_structure_cases = ['warehouse', 'maze', 'bottleneck_expansion', 'large_expansion']

    if case_name.lower() in flat_structure_cases:
        robot_file = os.path.join(BASE_PATH, case_name,
                                  f"robot_{robot_id}_trajectory_parameters_{case_lower}.json")
    else:
        # MAP cases have nested structure: MAP*/{case_lower}/{case_lower}/robot_...
        robot_file = os.path.join(BASE_PATH, case_name, case_lower, case_lower,
                                  f"robot_{robot_id}_trajectory_parameters_{case_lower}.json")

    if not os.path.exists(robot_file):
        print(f"    [DEBUG] File not found: {robot_file}")
        return None

    try:
        with open(robot_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"    [DEBUG] Failed to load: {robot_file}, error: {e}")
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
        delta_phi = phi[i+1] - phi_new[i]
        arc_radius = r0[i]
        has_arc_times = i < len(time_segments) and 'arc' in time_segments[i] and len(time_segments[i]['arc']) > 0

        # Only draw arc if there's actual rotation (delta_phi != 0) and radius is significant
        if abs(delta_phi) > 0.001 and abs(arc_radius) > 0.001:
            if has_arc_times:
                # Draw arc with velocity data from time segments
                arc_times = time_segments[i]['arc']
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
                        # Calculate radial direction from arc center to arc midpoint
                        # This gives us the direction to place text (away from center)
                        dx_from_center = point_x - r_x
                        dy_from_center = point_y - r_y
                        radial_distance = np.sqrt(dx_from_center**2 + dy_from_center**2)

                        if radial_distance > 0.001:
                            # Normalize direction
                            dx_norm = dx_from_center / radial_distance
                            dy_norm = dy_from_center / radial_distance

                            # Place text further away (25cm from arc point)
                            text_offset = 0.25  # 25cm offset
                            annotation_x = point_x + text_offset * dx_norm
                            annotation_y = point_y + text_offset * dy_norm

                            # Store: (arc_point_x, arc_point_y, text_x, text_y, radius_value)
                            arc_annotations.append((point_x, point_y, annotation_x, annotation_y, abs(arc_radius)))
            else:
                # No time data - draw arc directly to endpoint with last known velocity
                N_arc_default = 20  # Default number of arc segments
                for j in range(1, N_arc_default + 1):
                    angle_fraction = j / N_arc_default
                    current_angle = phi1 + delta_phi * angle_fraction

                    point_x = r0[i] * np.cos(current_angle + np.pi / 2) + r_x
                    point_y = r0[i] * np.sin(current_angle + np.pi / 2) + r_y

                    all_points_x.append(point_x)
                    all_points_y.append(point_y)
                    all_velocities.append(all_velocities[-1] if all_velocities else 0.0)

        # Get current position after arc
        if len(all_points_x) > 0:
            x_after_arc = all_points_x[-1]
            y_after_arc = all_points_y[-1]
        else:
            x_after_arc = x_start
            y_after_arc = y_start

        # Process line segment
        line_length = l[i] if i < len(l) else 0
        has_line_times = i < len(time_segments) and 'line' in time_segments[i] and len(time_segments[i]['line']) > 0

        if line_length > 0.001:
            is_straight_line = abs(r0[i]) < 0.01

            if is_straight_line:
                l_x = x_start
                l_y = y_start
                phi1_line = phi_new[i] if i < len(phi_new) else phi[i]
            else:
                l_x = r0[i] * np.cos(phi[i+1] + np.pi / 2) + r_x
                l_y = r0[i] * np.sin(phi[i+1] + np.pi / 2) + r_y
                phi1_line = phi[i+1]

            if has_line_times:
                # Draw line with velocity data from time segments
                line_times = time_segments[i]['line']
                N_line = len(line_times)

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
            else:
                # No time data - draw line directly to endpoint with last known velocity
                line_end_x = l_x + line_length * np.cos(phi1_line)
                line_end_y = l_y + line_length * np.sin(phi1_line)

                # Add start point if not already the last point
                if len(all_points_x) == 0 or (abs(all_points_x[-1] - l_x) > 0.001 or abs(all_points_y[-1] - l_y) > 0.001):
                    all_points_x.append(l_x)
                    all_points_y.append(l_y)
                    all_velocities.append(all_velocities[-1] if all_velocities else 0.0)

                # Add end point
                all_points_x.append(line_end_x)
                all_points_y.append(line_end_y)
                all_velocities.append(all_velocities[-1] if all_velocities else 0.0)

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


def adjust_annotation_positions(annotations, min_distance=0.12, max_iterations=200,
                               xlim=None, ylim=None, margin=0.08):
    """
    Adjust annotation text positions to avoid overlaps using grid-based placement.

    Args:
        annotations: List of annotation dicts with 'arc_x', 'arc_y', 'text_x', 'text_y', 'radius'
        min_distance: Minimum allowed distance between annotation centers (meters)
        max_iterations: Maximum iterations for force-directed adjustment
        xlim: Tuple (x_min, x_max) for plot bounds
        ylim: Tuple (y_min, y_max) for plot bounds
        margin: Safety margin from plot edges (meters)

    Returns:
        List of adjusted annotation dicts
    """
    if len(annotations) <= 1:
        return annotations

    # Make a copy to avoid modifying input
    adjusted = [anno.copy() for anno in annotations]

    # Set default bounds if not provided
    if xlim is None:
        xlim = (0, 2.5)
    if ylim is None:
        ylim = (0, 1.5)

    # Text bounding box approximation (in meters, for fontsize=7)
    # Increase width to prevent horizontal overlap
    text_width = 0.25   # "r=X.XX" needs more horizontal clearance
    text_height = 0.10  # Single line height with padding

    # Sort by arc_x position for consistent processing
    indices = list(range(len(adjusted)))
    indices.sort(key=lambda i: (adjusted[i]['arc_x'], adjusted[i]['arc_y']))

    # Place each annotation, avoiding collisions with already-placed ones
    placed_boxes = []  # List of (x_center, y_center, half_width, half_height)

    for idx in indices:
        anno = adjusted[idx]
        arc_x, arc_y = anno['arc_x'], anno['arc_y']

        # Try different positions around the arc point
        # Start with original position, then try alternatives
        base_offset = 0.20  # Base distance from arc point

        # Candidate positions: original, then 8 directions at increasing distances
        candidates = []

        # Original position
        candidates.append((anno['text_x'], anno['text_y']))

        # Try positions in 12 directions at multiple distances
        # Prioritize vertical offsets (up/down) to separate horizontally-close labels
        angles = [
            np.pi/2, -np.pi/2,  # Up, Down (priority)
            np.pi/3, -np.pi/3, 2*np.pi/3, -2*np.pi/3,  # Diagonal up/down
            0, np.pi,  # Right, Left
            np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4,  # 45-degree diagonals
        ]
        for dist_mult in [1.0, 1.3, 1.6, 2.0, 2.5]:
            offset = base_offset * dist_mult
            for angle in angles:
                cx = arc_x + offset * np.cos(angle)
                cy = arc_y + offset * np.sin(angle)
                candidates.append((cx, cy))

        # Find first non-colliding position
        best_pos = None
        for cx, cy in candidates:
            # Check boundary constraints
            if cx - text_width/2 < xlim[0] + margin:
                cx = xlim[0] + margin + text_width/2
            if cx + text_width/2 > xlim[1] - margin:
                cx = xlim[1] - margin - text_width/2
            if cy - text_height/2 < ylim[0] + margin:
                cy = ylim[0] + margin + text_height/2
            if cy + text_height/2 > ylim[1] - margin:
                cy = ylim[1] - margin - text_height/2

            # Check collision with placed boxes
            collision = False
            for (px, py, pw, ph) in placed_boxes:
                # Rectangle overlap check
                if (abs(cx - px) < (text_width/2 + pw) and
                    abs(cy - py) < (text_height/2 + ph)):
                    collision = True
                    break

            if not collision:
                best_pos = (cx, cy)
                break

        # If all positions collide, use the one furthest from others
        if best_pos is None:
            max_min_dist = -1
            for cx, cy in candidates:
                # Apply boundary constraints
                if cx - text_width/2 < xlim[0] + margin:
                    cx = xlim[0] + margin + text_width/2
                if cx + text_width/2 > xlim[1] - margin:
                    cx = xlim[1] - margin - text_width/2
                if cy - text_height/2 < ylim[0] + margin:
                    cy = ylim[0] + margin + text_height/2
                if cy + text_height/2 > ylim[1] - margin:
                    cy = ylim[1] - margin - text_height/2

                # Find minimum distance to any placed box
                if not placed_boxes:
                    min_dist = float('inf')
                else:
                    min_dist = min(np.sqrt((cx-px)**2 + (cy-py)**2)
                                   for (px, py, pw, ph) in placed_boxes)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_pos = (cx, cy)

        # Update annotation position
        adjusted[idx]['text_x'] = best_pos[0]
        adjusted[idx]['text_y'] = best_pos[1]

        # Add to placed boxes
        placed_boxes.append((best_pos[0], best_pos[1], text_width/2, text_height/2))

    return adjusted


def load_assignment_data(case_name):
    """Load assignment result data (waypoints and relay points)."""
    case_lower = case_name.lower()
    N = ROBOT_COUNT.get(case_name, 4)

    # LOOP1/LOOP2 use special naming: AssignmentResultLoop{N}{case}.json
    if case_name in ["LOOP1", "LOOP2"]:
        assignment_file = os.path.join(BASE_PATH, case_name, f"AssignmentResultLoop{N}{case_lower}.json")
    else:
        assignment_file = os.path.join(BASE_PATH, case_name, f"AssignmentResult{N}{case_lower}.json")

    if os.path.exists(assignment_file):
        try:
            with open(assignment_file, 'r') as f:
                data = json.load(f)
            return data.get('Waypoints', []), data.get('RelayPoints', [])
        except Exception as e:
            print(f"[!] Failed to load assignment: {e}")

    return [], []


def get_unified_limits(robot_data_list, reeb_graph):
    """Calculate unified axis limits based on trajectory data."""
    all_x_coords = []
    all_y_coords = []
    
    for robot_data in robot_data_list:
        if robot_data is None:
            continue
        all_points_x, all_points_y, _, _, _, _, _ = generate_robot_trajectory(robot_data, reeb_graph)
        all_x_coords.extend(all_points_x)
        all_y_coords.extend(all_points_y)
    
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
    
    return unified_xlim, unified_ylim


def plot_assignment_result(case_name, reeb_graph, unified_xlim, unified_ylim):
    """
    Plot Assignment Result with arrows showing waypoints and relay points.
    """
    # Create single figure for assignment result
    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4/2.54))  # 5cm x 4cm

    # Load REBUILT graph for assignment visualization (Graph_new_{case}.json)
    reeb_graph_rebuilt = load_reeb_graph(case_name, use_rebuilt=True)
    if reeb_graph_rebuilt is None:
        print(f"[!] Warning: Rebuilt graph not found for {case_name}, using original graph")
        reeb_graph_rebuilt = reeb_graph

    # Load environment data for obstacles
    env_data = load_environment_data(case_name)

    # Draw environment boundary - CONVERT TO METERS
    if env_data:
        coord_bounds = env_data.get('coord_bounds', [0, 1100, 0, 600])
        x_min, x_max, y_min, y_max = coord_bounds
        boundary_vertices_pixel = np.array([
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]
        ])
        boundary_vertices_meter = np.array([convert_pixel_to_meter(v) for v in boundary_vertices_pixel])
        ax.plot(boundary_vertices_meter[:, 0], boundary_vertices_meter[:, 1],
               color='black', linewidth=1.5, zorder=1)

    # Draw environment obstacles (black polygons) - CONVERT TO METERS
    if env_data and 'polygons' in env_data:
        for polygon_data in env_data['polygons']:
            vertices_pixel = np.array(polygon_data['vertices'])
            # Convert to meters
            vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
            # Close the polygon by adding first vertex at the end
            vertices_closed = np.vstack([vertices_meter, vertices_meter[0]])
            ax.fill(vertices_closed[:, 0], vertices_closed[:, 1],
                   color='black', alpha=1.0, zorder=1)

    # Draw graph nodes (bigger gray circles) - CONVERT TO METERS
    for node_id in reeb_graph_rebuilt.nodes:
        pos_pixel = reeb_graph_rebuilt.nodes[node_id].configuration
        pos_meter = convert_pixel_to_meter(pos_pixel)
        ax.scatter(pos_meter[0], pos_meter[1], color='grey', s=16, marker='o',
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
                    ax.plot([start_meter[0], end_meter[0]],
                           [start_meter[1], end_meter[1]],
                           'grey', linewidth=0.5, alpha=0.5, zorder=2)

    # Load assignment data
    waypoints_arcs, relay_arcs = load_assignment_data(case_name)

    # For LOOP1/LOOP2, load waypoint positions from complete_trajectory file
    # because Graph file has incorrect coordinates
    waypoint_positions_map = {}
    if case_name in ["LOOP1", "LOOP2"]:
        case_lower = case_name.lower()
        complete_traj_file = os.path.join(BASE_PATH, case_name, case_lower, case_lower,
                                          f"complete_trajectory_parameters_{case_lower}.json")
        if os.path.exists(complete_traj_file):
            try:
                with open(complete_traj_file, 'r') as f:
                    complete_data = json.load(f)
                waypoints_list = complete_data.get('waypoints', [])
                positions_list = complete_data.get('waypoint_positions', [])
                for wp_id, pos in zip(waypoints_list, positions_list):
                    waypoint_positions_map[wp_id] = pos  # Already in meters
            except Exception as e:
                print(f"[!] Failed to load complete trajectory for {case_name}: {e}")

    # Helper function to get node position
    def get_node_position(node_id):
        if node_id in waypoint_positions_map:
            return waypoint_positions_map[node_id]  # Already in meters
        elif node_id in reeb_graph_rebuilt.nodes:
            pos_pixel = reeb_graph_rebuilt.nodes[node_id].configuration
            return convert_pixel_to_meter(pos_pixel)
        return None

    # Track if we've added legend labels (only label first occurrence)
    waypoint_label_added = False
    relay_label_added = False

    if waypoints_arcs:
        # Draw waypoint arrows (green)
        for i, j, _ in waypoints_arcs:
            start_meter = get_node_position(i)
            end_meter = get_node_position(j)
            if start_meter and end_meter:
                # Only add label for first waypoint arrow
                label = 'Waypoints' if not waypoint_label_added else ''
                waypoint_label_added = True

                ax.arrow(start_meter[0], start_meter[1],
                        end_meter[0] - start_meter[0], end_meter[1] - start_meter[1],
                        width=0.008, head_width=0.050, head_length=0.040,
                        fc='green', ec='green', alpha=0.8, zorder=5,
                        label=label)

    if relay_arcs:
        # Draw relay point arrows (red)
        for i, j, _ in relay_arcs:
            start_meter = get_node_position(i)
            end_meter = get_node_position(j)
            if start_meter and end_meter:
                # Only add label for first relay point arrow
                label = 'Relay Points' if not relay_label_added else ''
                relay_label_added = True

                ax.arrow(start_meter[0], start_meter[1],
                        end_meter[0] - start_meter[0], end_meter[1] - start_meter[1],
                        width=0.008, head_width=0.050, head_length=0.040,
                        fc='red', ec='red', alpha=0.8, zorder=6,
                        label=label)

    # Set axis properties
    ax.set_title('Assignment Result', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    # ax.grid(True, linewidth=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)

    # Apply unified limits
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    # Set uniform tick intervals
    unified_tick_spacing = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    # Add legend below the plot area
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=2, frameon=True)

    # Save figure with fixed layout (NO bbox_inches='tight' to ensure consistent size)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'{case_name}_assignment_result.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")


def plot_velocity_trajectories(case_name, robot_data_list, reeb_graph, unified_xlim, unified_ylim):
    """
    Plot velocity-colored trajectories for all robots.
    """
    # Create single figure for velocity trajectories - TALLER to accommodate colorbar
    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4.5/2.54))  # 5cm x 4.5cm (taller)

    # Load and draw obstacles with very light gray
    env_data = load_environment_data(case_name)
    if env_data and 'polygons' in env_data:
        from matplotlib.patches import Polygon as PolygonPatch
        for polygon_data in env_data['polygons']:
            vertices_pixel = np.array(polygon_data['vertices'])
            vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
            poly = PolygonPatch(vertices_meter, closed=True, facecolor='#C0C0C0',
                               edgecolor='#909090', alpha=0.7, linewidth=0.5, zorder=1)
            ax.add_patch(poly)

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

    # Collect all arc annotations from all robots FIRST
    all_annotations = []
    for traj in trajectory_data:
        for arc_x, arc_y, text_x, text_y, radius_value in traj['arcs']:
            all_annotations.append({
                'arc_x': arc_x,
                'arc_y': arc_y,
                'text_x': text_x,
                'text_y': text_y,
                'radius': radius_value
            })

    # Adjust annotation positions to avoid overlaps and axis boundaries
    # Note: unified_xlim and unified_ylim are passed from the parent function
    adjusted_annotations = adjust_annotation_positions(all_annotations,
                                                       xlim=unified_xlim,
                                                       ylim=unified_ylim,
                                                       margin=0.15)

    # Plot trajectories with velocity coloring
    for traj in trajectory_data:
        all_points_x = traj['x']
        all_points_y = traj['y']
        all_velocities = traj['v']
        wp_x = traj['wp_x']
        wp_y = traj['wp_y']
        relay_indices = traj['relay']

        # Plot trajectory with color based on velocity
        if len(all_points_x) > 1:
            for i in range(len(all_points_x) - 1):
                current_velocity = all_velocities[i] if all_velocities[i] is not None else 0.0
                color = cmap(norm(current_velocity))
                ax.plot([all_points_x[i], all_points_x[i+1]],
                       [all_points_y[i], all_points_y[i+1]],
                       color=color, linewidth=0.8, alpha=0.8)

        # Plot waypoints (green circles) - NO LABEL for legend
        ax.scatter(wp_x, wp_y, color='green', s=16, marker='o', zorder=5)

        # Plot relay points (red triangles) - NO LABEL for legend
        if relay_indices:
            relay_x = [wp_x[i] for i in relay_indices]
            relay_y = [wp_y[i] for i in relay_indices]
            ax.scatter(relay_x, relay_y, color='red', s=16, marker='^', zorder=5)

    # Draw all annotations with adjusted positions (no overlap)
    for anno in adjusted_annotations:
        radius_text = f"r={anno['radius']:.2f}"

        # Draw annotation text WITHOUT background box (to avoid blocking arcs)
        ax.text(anno['text_x'], anno['text_y'], radius_text,
               fontsize=7, ha='center', va='center',
               color='black', weight='bold',
               zorder=10)

        # Draw arrow from text to arc point
        ax.annotate('', xy=(anno['arc_x'], anno['arc_y']),
                   xytext=(anno['text_x'], anno['text_y']),
                   arrowprops=dict(arrowstyle='-', color='gray',
                                 lw=0.5, alpha=0.6),
                   zorder=9)

    # Set axis properties
    ax.set_title('Planning Trajectory', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    # ax.grid(True, linewidth=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)

    # Apply unified limits
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    # Set uniform tick intervals
    unified_tick_spacing = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    # Save figure - move plot area UP to make room for colorbar below
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.28)

    # Add colorbar with manual positioning (place it at absolute position to control height)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbar_ax = inset_axes(ax, width="80%", height="5%", loc='lower center',
                        bbox_to_anchor=(0, -0.20, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=6)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'{case_name}_velocity_trajectories.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")


def check_line_polygon_collision(start, end, polygon_vertices):
    """
    Check if line segment intersects with polygon.
    Uses shapely for robust geometric intersection.
    Obstacles are expanded by 0.01m for minimal safety clearance.
    """
    from shapely.geometry import LineString, Polygon

    line = LineString([start, end])
    poly = Polygon(polygon_vertices)
    expanded_poly = poly.buffer(0.01)  # Expand obstacle by 0.01m (smaller margin)

    return line.intersects(expanded_poly)


def check_line_boundary_collision(start, end, coord_bounds):
    """
    Check if line segment goes outside workspace boundaries.
    """
    if coord_bounds is None:
        return False

    x_min, x_max, y_min, y_max = coord_bounds

    # Check if any point is outside boundaries
    for point in [start, end]:
        if (point[0] < x_min or point[0] > x_max or
            point[1] < y_min or point[1] > y_max):
            return True

    return False


def check_corridor_collision(start, end, polygons, coord_bounds):
    """
    Check if corridor edge collides with any obstacle or boundary.
    """
    # Check boundary collision
    if check_line_boundary_collision(start, end, coord_bounds):
        return True

    # Check polygon collisions
    for polygon in polygons:
        vertices = polygon.get('vertices', [])
        if vertices and check_line_polygon_collision(start, end, vertices):
            return True

    return False


def compute_safe_corridors(waypoints, nodes, environment):
    """
    Compute safe corridors for each waypoint segment with collision detection.
    Uses EXACT same logic as Planing_functions.py get_safe_corridor():
    1. Calculate signed perpendicular distance from obstacle vertices to path line
    2. Only consider vertices that project onto the segment (with margin)
    3. Combine with step-based edge collision checking

    Returns list of [slope, y_min, y_max] for each segment IN METERS.
    """
    N = len(waypoints)
    safe_corridors = []
    max_distance = 100 * PIXEL_TO_METER_SCALE  # Maximum corridor width in meters
    step_size = 0.2 * PIXEL_TO_METER_SCALE     # Step size in meters (smaller for precision)
    margin = 5.0 * PIXEL_TO_METER_SCALE        # Margin beyond segment endpoints (smaller)

    coord_bounds = environment.get('coord_bounds', None)
    # Add safety margin and convert to meters
    boundary_margin = 5.0 * PIXEL_TO_METER_SCALE
    if coord_bounds:
        x_min, x_max, y_min, y_max = coord_bounds
        coord_bounds_m = [
            (x_min + 5.0) * PIXEL_TO_METER_SCALE,
            (x_max - 5.0) * PIXEL_TO_METER_SCALE,
            (y_min + 5.0) * PIXEL_TO_METER_SCALE,
            (y_max - 5.0) * PIXEL_TO_METER_SCALE
        ]
    else:
        coord_bounds_m = None

    # Collect ALL obstacle vertices and convert to meters
    all_obstacle_vertices = []
    for poly in environment.get('polygons', []):
        vertices = poly.get('vertices', [])
        for v in vertices:
            all_obstacle_vertices.append(convert_pixel_to_meter(v))

    # Convert polygons to meters for collision detection
    polygons_m = []
    for poly in environment.get('polygons', []):
        vertices = poly.get('vertices', [])
        if vertices:
            vertices_m = [convert_pixel_to_meter(v) for v in vertices]
            polygons_m.append({'vertices': vertices_m})

    for i in range(N-1):
        # Get waypoint positions (already in meters)
        start_pos = np.array(nodes[waypoints[i]])
        end_pos = np.array(nodes[waypoints[i+1]])

        # Calculate line direction and length
        line_vec = end_pos - start_pos
        line_length = np.linalg.norm(line_vec)

        if line_length < 1e-9:
            safe_corridors.append([0, 0, 0])
            continue

        # Unit vectors
        line_unit = line_vec / line_length
        # Perpendicular vector (rotate 90 degrees counterclockwise) - SAME as reference
        perp_unit = np.array([-line_unit[1], line_unit[0]])

        # Find minimum SIGNED perpendicular distance from obstacle vertices to the line
        # Positive distance = in perp_unit direction
        # Negative distance = opposite to perp_unit direction
        min_dist_positive = max_distance  # Closest obstacle in positive normal direction
        min_dist_negative = max_distance  # Closest obstacle in negative normal direction

        for obs_vertex in all_obstacle_vertices:
            obs_vertex = np.array(obs_vertex)
            # Vector from start_pos to obstacle vertex
            vec_to_vertex = obs_vertex - start_pos

            # Project onto line direction to check if vertex is "alongside" the segment
            proj_along_line = np.dot(vec_to_vertex, line_unit)

            # Only consider vertices that project onto the segment (with margin)
            # This is the KEY difference from before!
            if proj_along_line >= -margin and proj_along_line <= line_length + margin:
                # Signed perpendicular distance (positive = in perp_unit direction)
                signed_perp_dist = np.dot(vec_to_vertex, perp_unit)

                if signed_perp_dist > 0:
                    # Vertex is in positive normal direction
                    min_dist_positive = min(min_dist_positive, signed_perp_dist)
                else:
                    # Vertex is in negative normal direction
                    min_dist_negative = min(min_dist_negative, abs(signed_perp_dist))

        # Also use step-based collision checking (same as reference)
        # Check positive side (in perp_unit direction)
        db_max_step = 0
        collision_found = False
        while not collision_found and db_max_step < max_distance:
            db_max_step += step_size
            p_start_up = list(start_pos + db_max_step * perp_unit)
            p_end_up = list(end_pos + db_max_step * perp_unit)
            if check_corridor_collision(p_start_up, p_end_up, polygons_m, coord_bounds_m):
                collision_found = True
                break

        # Check negative side (opposite to perp_unit direction)
        db_min_step = 0
        collision_found = False
        while not collision_found and db_min_step < max_distance:
            db_min_step += step_size
            p_start_low = list(start_pos - db_min_step * perp_unit)
            p_end_low = list(end_pos - db_min_step * perp_unit)
            if check_corridor_collision(p_start_low, p_end_low, polygons_m, coord_bounds_m):
                collision_found = True
                break

        # Use minimum of step-based and vertex-based distances (SAME as reference)
        db_min = min(db_min_step, min_dist_negative)
        db_max = min(db_max_step, min_dist_positive)

        # Handle vertical lines for slope
        if abs(end_pos[0] - start_pos[0]) < 1e-6:
            slope = 100000000  # Large number for vertical
        else:
            slope = (end_pos[1] - start_pos[1]) / (end_pos[0] - start_pos[0])

        # y_min and y_max in perpendicular distance
        y_min = -db_min
        y_max = db_max

        safe_corridors.append([slope, min(y_min, y_max), max(y_min, y_max)])

    return safe_corridors


def plot_planning_result(case_name, unified_xlim, unified_ylim):
    """
    Plot planning result with safe corridors and optimized trajectory.
    Uses SAME trajectory calculation logic as generate_robot_trajectory() for consistency.
    """
    case_lower = case_name.lower()

    # Find optimization file first, then match waypoint file by robot count
    import glob
    import re

    # LOOP1/LOOP2 use special naming: Optimization_withSC_loop_path{N}{case}.json
    if case_name in ["LOOP1", "LOOP2"]:
        optimization_files = glob.glob(os.path.join(BASE_PATH, case_name, f"Optimization_withSC_loop_path*{case_lower}.json"))
    else:
        optimization_files = glob.glob(os.path.join(BASE_PATH, case_name, f"Optimization_withSC_path*{case_lower}.json"))

    if not optimization_files:
        print(f"[!] Optimization file not found for {case_name}")
        return False

    # Extract robot count from optimization filename
    # LOOP1/LOOP2: "Optimization_withSC_loop_path4loop1.json" -> 4
    # Others: "Optimization_withSC_path8maze.json" -> 8
    opt_filename = os.path.basename(optimization_files[0])
    if case_name in ["LOOP1", "LOOP2"]:
        match = re.search(r'Optimization_withSC_loop_path(\d+)', opt_filename)
    else:
        match = re.search(r'Optimization_withSC_path(\d+)', opt_filename)
    if not match:
        print(f"[!] Cannot extract robot count from {opt_filename}")
        return False
    robot_count = match.group(1)

    # Find matching WayPointFlag file with same robot count
    # LOOP1 uses WayPointFlagLoop{N}{case}.json, others use WayPointFlag{N}{case}.json
    if case_name == "LOOP1":
        waypoint_file = os.path.join(BASE_PATH, case_name, f"WayPointFlagLoop{robot_count}{case_lower}.json")
    else:
        waypoint_file = os.path.join(BASE_PATH, case_name, f"WayPointFlag{robot_count}{case_lower}.json")
    if not os.path.exists(waypoint_file):
        print(f"[!] Waypoint file not found: {waypoint_file}")
        return False

    # Load data
    try:
        with open(waypoint_file, 'r') as f:
            waypoint_data = json.load(f)
        with open(optimization_files[0], 'r') as f:
            optimization = json.load(f)
    except Exception as e:
        print(f"[!] Failed to load planning data: {e}")
        return False

    waypoints = waypoint_data['Waypoints']
    flagb = waypoint_data['FlagB']

    # Load graph and environment
    reeb_graph = load_reeb_graph(case_name, use_rebuilt=True)
    if reeb_graph is None:
        reeb_graph = load_reeb_graph(case_name, use_rebuilt=False)
    if reeb_graph is None:
        return False

    env_data = load_environment_data(case_name)
    if env_data is None:
        return False

    # Convert graph nodes to meters for corridor computation
    nodes_meter = {}
    for node_id in reeb_graph.nodes:
        pos_pixel = reeb_graph.nodes[node_id].configuration
        nodes_meter[node_id] = convert_pixel_to_meter(pos_pixel)

    # For LOOP1/LOOP2, override with correct positions from complete_trajectory file
    if case_name in ["LOOP1", "LOOP2"]:
        complete_traj_file = os.path.join(BASE_PATH, case_name, case_lower, case_lower,
                                          f"complete_trajectory_parameters_{case_lower}.json")
        if os.path.exists(complete_traj_file):
            try:
                with open(complete_traj_file, 'r') as f:
                    complete_data = json.load(f)
                waypoints_list = complete_data.get('waypoints', [])
                positions_list = complete_data.get('waypoint_positions', [])
                for wp_id, pos in zip(waypoints_list, positions_list):
                    nodes_meter[wp_id] = pos  # Already in meters
            except Exception as e:
                print(f"[!] Failed to load complete trajectory for {case_name}: {e}")

    # Compute safe corridors
    safe_corridors = compute_safe_corridors(waypoints, nodes_meter, env_data)

    # Create figure with SAME size as other plots (5cm x 4cm)
    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4/2.54))

    # Plot obstacles with expanded area (light gray)
    from shapely.geometry import Polygon as ShapelyPolygon
    for polygon in env_data['polygons']:
        vertices_pixel = np.array(polygon['vertices'])
        vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])

        # Draw expanded obstacle area (0.04m buffer)
        shapely_poly = ShapelyPolygon(vertices_meter)
        expanded_poly = shapely_poly.buffer(0.04)

        if hasattr(expanded_poly, 'exterior'):
            exp_coords = list(expanded_poly.exterior.coords)
            from matplotlib.patches import Polygon as PolygonPatch
            expanded_patch = PolygonPatch(exp_coords, closed=True,
                                         facecolor='lightgray', edgecolor='black',
                                         alpha=0.4, linewidth=0.5)
            ax.add_patch(expanded_patch)

        # Draw original obstacle (darker)
        poly = PolygonPatch(vertices_meter, closed=True, facecolor='#404040',
                           edgecolor='black', alpha=0.9, linewidth=0.8)
        ax.add_patch(poly)

    # Plot trajectory using SAME logic as generate_robot_trajectory()
    N = len(waypoints)
    phi = np.array(optimization['Optimization_phi'])
    l = np.array(optimization['Optimization_l']) * PIXEL_TO_METER_SCALE

    # Handle sentinel value for straight lines (100000000 or similar large values indicate straight line)
    r0_raw = np.array(optimization['Optimization_r'])
    STRAIGHT_LINE_SENTINEL = 1e7  # Values >= this are treated as straight line (r=0)
    r0 = np.where(np.abs(r0_raw) >= STRAIGHT_LINE_SENTINEL, 0.0, r0_raw * PIXEL_TO_METER_SCALE)

    # Get waypoint positions in meters
    wp_x = [nodes_meter[wp][0] for wp in waypoints]
    wp_y = [nodes_meter[wp][1] for wp in waypoints]

    # Calculate phi_new (same as generate_robot_trajectory)
    phi_new = []
    for i in range(len(waypoints) - 1):
        flagb_i = flagb[i] if i < len(flagb) else 0
        phi_new.append(phi[i] + flagb_i * np.pi / 2)

    # Process each segment (EXACT same logic as generate_robot_trajectory)
    all_points_x = []
    all_points_y = []

    for i in range(N - 1):
        flagb_i = flagb[i] if i < len(flagb) else 0
        phi1 = phi[i] + flagb_i * np.pi / 2

        x_start = wp_x[i]
        y_start = wp_y[i]

        # Calculate arc center (EXACT same as generate_robot_trajectory line 184-185)
        r_x = x_start - r0[i] * np.cos(phi1 + np.pi / 2)
        r_y = y_start - r0[i] * np.sin(phi1 + np.pi / 2)

        # Add starting point if first segment
        if i == 0:
            all_points_x.append(x_start)
            all_points_y.append(y_start)

        # Plot waypoint marker
        if flagb_i != 0:
            ax.plot(x_start, y_start, 'ro', markersize=4)
        else:
            ax.plot(x_start, y_start, 'go', markersize=4)

        # Process arc segment
        delta_phi = phi[i+1] - phi_new[i]
        arc_radius = r0[i]

        # Only draw arc if there's actual rotation AND radius is significant (not straight line)
        if abs(delta_phi) > 0.001 and abs(arc_radius) > 0.001:
            N_arc = 50  # Number of arc points
            arc_x = []
            arc_y = []

            for j in range(N_arc + 1):
                angle_fraction = j / N_arc
                current_angle = phi1 + delta_phi * angle_fraction
                point_x = r0[i] * np.cos(current_angle + np.pi / 2) + r_x
                point_y = r0[i] * np.sin(current_angle + np.pi / 2) + r_y
                arc_x.append(point_x)
                arc_y.append(point_y)

            ax.plot(arc_x, arc_y, 'b-', linewidth=0.8)
            all_points_x.extend(arc_x)
            all_points_y.extend(arc_y)

            # Get position after arc
            x_after_arc = arc_x[-1]
            y_after_arc = arc_y[-1]
        else:
            x_after_arc = x_start
            y_after_arc = y_start

        # Process line segment
        line_length = l[i] if i < len(l) else 0

        if line_length > 0.001:
            # Calculate line start position (same as generate_robot_trajectory line 258-265)
            is_straight_line = abs(r0[i]) < 0.01

            if is_straight_line:
                l_x = x_start
                l_y = y_start
                phi1_line = phi_new[i]
            else:
                l_x = r0[i] * np.cos(phi[i+1] + np.pi / 2) + r_x
                l_y = r0[i] * np.sin(phi[i+1] + np.pi / 2) + r_y
                phi1_line = phi[i+1]

            # Draw line segment
            line_end_x = l_x + line_length * np.cos(phi1_line)
            line_end_y = l_y + line_length * np.sin(phi1_line)

            ax.plot([l_x, line_end_x], [l_y, line_end_y], 'b-', linewidth=0.8)
            all_points_x.extend([l_x, line_end_x])
            all_points_y.extend([l_y, line_end_y])

    # Plot final waypoint
    final_flagb = flagb[-1] if len(flagb) > 0 else 0
    if final_flagb != 0:
        ax.plot(wp_x[-1], wp_y[-1], 'ro', markersize=4)
    else:
        ax.plot(wp_x[-1], wp_y[-1], 'go', markersize=4)

    # Plot safe corridors
    for i in range(N-1):
        start_pos = np.array(nodes_meter[waypoints[i]])
        end_pos = np.array(nodes_meter[waypoints[i+1]])

        slope = safe_corridors[i][0]
        y_min = safe_corridors[i][1]
        y_max = safe_corridors[i][2]

        # Vertical line case
        if abs(slope) > 100000:
            x_coords = [start_pos[0] + y_min, start_pos[0] + y_max,
                       end_pos[0] + y_max, end_pos[0] + y_min, start_pos[0] + y_min]
            y_coords = [start_pos[1], start_pos[1], end_pos[1], end_pos[1], start_pos[1]]
        else:
            # Non-vertical line - calculate perpendicular corridor
            length = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            dx = (end_pos[0] - start_pos[0]) / length
            dy = (end_pos[1] - start_pos[1]) / length

            perp_dx = -dy
            perp_dy = dx

            p1 = [start_pos[0] + y_min * perp_dx, start_pos[1] + y_min * perp_dy]
            p2 = [start_pos[0] + y_max * perp_dx, start_pos[1] + y_max * perp_dy]
            p3 = [end_pos[0] + y_max * perp_dx, end_pos[1] + y_max * perp_dy]
            p4 = [end_pos[0] + y_min * perp_dx, end_pos[1] + y_min * perp_dy]

            x_coords = [p1[0], p2[0], p3[0], p4[0], p1[0]]
            y_coords = [p1[1], p2[1], p3[1], p4[1], p1[1]]

        ax.plot(x_coords, y_coords, 'g--', alpha=0.5, linewidth=1,
               label='Safe Corridor' if i == 0 else "")

    # Set axis properties (SAME style as other plots)
    ax.set_title('Planning Result', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    # ax.grid(True, linewidth=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)

    # Apply UNIFIED limits (same as other plots)
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    # Set uniform tick intervals
    unified_tick_spacing = 0.5
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    # Add legend below the plot area
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=2, frameon=True)

    # Save figure with fixed layout (NO bbox_inches='tight' to ensure consistent size)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'{case_name}_planning_result.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")
    return True


def plot_reeb_graph_with_environment(case_name, unified_xlim, unified_ylim):
    """
    Plot Reeb Graph overlaid on environment with start and goal positions.
    Uses yellow star for start and red flag for goal.
    """
    case_lower = case_name.lower()

    # Load graph data
    graph_file = os.path.join(BASE_PATH, case_name, f"Graph_{case_lower}.json")
    if not os.path.exists(graph_file):
        print(f"[!] Graph file not found: {graph_file}")
        return False

    with open(graph_file, 'r') as f:
        graph_data = json.load(f)

    # Load environment data
    env_data = load_environment_data(case_name)
    if env_data is None:
        return False

    # Extract data from graph
    nodes = graph_data['nodes']  # [[id, [x, y], null, false], ...]
    out_neighbors = graph_data['out_neighbors']  # {'0': [2], '1': [2], ...}
    start_pose = graph_data.get('start_pose', [120, 150, 0])  # [x, y, theta] in pixels
    goal_pose = graph_data.get('goal_pose', [900, 400, 0])  # [x, y, theta] in pixels

    polygons = env_data['polygons']

    # Create figure with SAME size as other plots (5cm x 4cm)
    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4/2.54))

    # Plot environment boundary from coord_bounds - CONVERT TO METERS
    # coord_bounds format: [x_min, x_max, y_min, y_max]
    coord_bounds = env_data.get('coord_bounds', [0, 1100, 0, 600])
    x_min, x_max, y_min, y_max = coord_bounds

    boundary_vertices_pixel = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min]
    ])
    boundary_vertices_meter = np.array([convert_pixel_to_meter(v) for v in boundary_vertices_pixel])
    ax.plot(boundary_vertices_meter[:, 0], boundary_vertices_meter[:, 1],
           color='black', linewidth=1.5, zorder=1)

    # Plot obstacles (environment polygons) - CONVERT TO METERS
    for polygon in polygons:
        vertices_pixel = np.array(polygon['vertices'])
        vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
        from matplotlib.patches import Polygon as PolygonPatch
        poly = PolygonPatch(vertices_meter, closed=True, fill=True,
                           facecolor='lightgray', alpha=0.5,
                           edgecolor='black', linewidth=1.0)
        ax.add_patch(poly)

    # Plot Reeb Graph edges (BLUE lines) - CONVERT TO METERS
    for node_id_str, neighbors in out_neighbors.items():
        node_id = int(node_id_str)
        if node_id < len(nodes):
            node_pos_pixel = nodes[node_id][1]  # [x, y] in pixels
            node_pos_meter = convert_pixel_to_meter(node_pos_pixel)

            for neighbor_id in neighbors:
                if neighbor_id < len(nodes):
                    neighbor_pos_pixel = nodes[neighbor_id][1]
                    neighbor_pos_meter = convert_pixel_to_meter(neighbor_pos_pixel)

                    # Draw edge in BLUE
                    ax.plot([node_pos_meter[0], neighbor_pos_meter[0]],
                           [node_pos_meter[1], neighbor_pos_meter[1]],
                           color='blue', linewidth=0.8, alpha=0.7, zorder=2)

    # Plot Reeb Graph nodes (blue circles) - CONVERT TO METERS
    node_positions_pixel = np.array([node[1] for node in nodes])
    node_positions_meter = np.array([convert_pixel_to_meter(pos) for pos in node_positions_pixel])
    ax.scatter(node_positions_meter[:, 0], node_positions_meter[:, 1],
              c='blue', s=6, zorder=3, edgecolors='black', linewidths=0.3,
              alpha=0.7)

    # NO node ID labels (removed as requested)

    # Convert start and goal positions to meters
    start_meter = convert_pixel_to_meter(start_pose[:2])
    goal_meter = convert_pixel_to_meter(goal_pose[:2])

    # Plot start position - YELLOW STAR (smaller size)
    ax.scatter(start_meter[0], start_meter[1], c='yellow', s=40, marker='*',
              edgecolors='black', linewidths=0.5, zorder=5, label='Start')

    # Plot goal position - RED FLAG (smaller size)
    ax.scatter(goal_meter[0], goal_meter[1], c='red', s=30, marker='P',
              edgecolors='black', linewidths=0.5, zorder=5, label='Goal')

    # Set axis properties (SAME style as other plots)
    ax.set_title('Reeb Graph', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    # ax.grid(True, linewidth=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)

    # Apply UNIFIED limits (same as other two plots)
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    # Set uniform tick intervals
    unified_tick_spacing = 0.5
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    # Add legend below the plot area
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=2, frameon=True)

    # Save figure with fixed layout (NO bbox_inches='tight' to ensure consistent size)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'{case_name}_reeb_graph_environment.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")
    return True


def process_case(case_name):
    """Process a single case to generate both separate plots."""
    print(f"\n[*] Processing {case_name}...")

    # Load Reeb graph
    reeb_graph = load_reeb_graph(case_name)
    if reeb_graph is None:
        return False

    # Load all robot data (with robot count limit)
    robot_data_list = []
    max_robots = ROBOT_COUNT.get(case_name, 10)  # Default to 10 if not specified

    for robot_id in range(max_robots):
        robot_data = load_robot_data(case_name, robot_id)
        if robot_data is None:
            break  # Stop if robot file not found
        robot_data_list.append(robot_data)

    if len(robot_data_list) == 0:
        print(f"[X] No robot data found")
        return False

    print(f"[OK] Loaded {len(robot_data_list)} robots")

    try:
        # Calculate unified limits once
        unified_xlim, unified_ylim = get_unified_limits(robot_data_list, reeb_graph)

        # Generate all four separate plots with unified limits
        plot_assignment_result(case_name, reeb_graph, unified_xlim, unified_ylim)
        plot_velocity_trajectories(case_name, robot_data_list, reeb_graph, unified_xlim, unified_ylim)
        plot_reeb_graph_with_environment(case_name, unified_xlim, unified_ylim)
        plot_planning_result(case_name, unified_xlim, unified_ylim)

        return True
    except Exception as e:
        print(f"[X] Plot failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main batch processing loop."""
    print("=" * 60)
    print("SEPARATE SUBFIGURES VISUALIZATION")
    print("Generating 4 separate plots for each MAP:")
    print("  1. Assignment Result")
    print("  2. Velocity Trajectory")
    print("  3. Reeb Graph with Environment")
    print("  4. Planning Result with Safe Corridors")
    print("=" * 60)

    success_count = 0
    failed_cases = []

    for case_name in CASES:
        if process_case(case_name):
            success_count += 1
        else:
            failed_cases.append(case_name)

    # Summary
    print("\n" + "=" * 60)
    print(f"[OK] Success: {success_count}/{len(CASES)}")
    if failed_cases:
        print(f"[X] Failed: {', '.join(failed_cases)}")
    print("=" * 60)


if __name__ == "__main__":
    main()