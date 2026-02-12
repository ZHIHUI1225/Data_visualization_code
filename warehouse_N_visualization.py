"""
Generate separate plots for warehouse cases with different robot counts (N).
Based on separate_subfigures_visualization.py, adapted for warehouse_N directory structure.

Generates 4 plots for each N:
1. Assignment Result (arrows showing waypoints and relay points)
2. Velocity-Colored Trajectories
3. Reeb Graph with Environment
4. Planning Result with Safe Corridors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Polygon as PolygonPatch
import json
import os
import glob
import re

# ========== CONFIGURATION ==========
BASE_PATH = "result/MAPS/warehouse_N"
OUTPUT_PATH = "result/MAPS/warehouse_N/separate_plots"

# Robot counts available in warehouse_N
ROBOT_COUNTS = [2, 3, 4, 5, 6]

# ========== ROBOT PARAMETERS ==========
PIXEL_TO_METER_SCALE = 0.0023
v_max = 0.03  # max linear velocity (m/s)


def convert_pixel_to_meter(pos):
    """Convert pixel coordinates to meters."""
    return (pos[0] * PIXEL_TO_METER_SCALE, pos[1] * PIXEL_TO_METER_SCALE)


def load_reeb_graph(use_rebuilt=False):
    """
    Load Reeb graph for waypoint positions.

    Args:
        use_rebuilt: If True, load Graph_new_warehouse.json (for assignment visualization)
                     If False, load Graph_warehouse.json (for trajectory data lookup)
    """
    if use_rebuilt:
        graph_file = os.path.join(BASE_PATH, "Graph_new_warehouse.json")
    else:
        graph_file = os.path.join(BASE_PATH, "Graph_warehouse.json")

    if not os.path.exists(graph_file):
        print(f"[X] Graph file not found: {graph_file}")
        return None

    try:
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)

        class SimpleGraph:
            def __init__(self, nodes_list, in_neighbors, out_neighbors):
                self.nodes = {}
                for node_entry in nodes_list:
                    node_id = node_entry[0]
                    node_position = node_entry[1]
                    self.nodes[node_id] = type('Node', (), {
                        'configuration': node_position
                    })()
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


def load_robot_data(robot_id, N):
    """
    Load single robot trajectory data for a specific N.
    Tries flat structure first, then nested structure.
    """
    # Try flat structure first (files directly in warehouse_N directory)
    robot_file = os.path.join(BASE_PATH, f"robot_{robot_id}_trajectory_parameters_warehouse.json")

    if os.path.exists(robot_file):
        try:
            with open(robot_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"    [DEBUG] Failed to load: {robot_file}, error: {e}")

    # Try nested structure (warehouse_N/warehouse/robot_...json)
    robot_file_nested = os.path.join(BASE_PATH, "warehouse",
                                      f"robot_{robot_id}_trajectory_parameters_warehouse.json")
    if os.path.exists(robot_file_nested):
        try:
            with open(robot_file_nested, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"    [DEBUG] Failed to load: {robot_file_nested}, error: {e}")

    return None


def load_environment_data():
    """Load environment data (obstacles)."""
    env_file = os.path.join(BASE_PATH, "environment_warehouse.json")

    if not os.path.exists(env_file):
        print(f"[!] Environment file not found: {env_file}")
        return None

    try:
        with open(env_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Failed to load environment: {e}")
        return None


def load_assignment_data(N):
    """Load assignment result data for specific N."""
    assignment_file = os.path.join(BASE_PATH, f"AssignmentResult{N}warehouse.json")

    if os.path.exists(assignment_file):
        try:
            with open(assignment_file, 'r') as f:
                data = json.load(f)
            return data.get('Waypoints', []), data.get('RelayPoints', [])
        except Exception as e:
            print(f"[!] Failed to load assignment: {e}")

    return [], []


def load_waypoint_flag_data(N):
    """Load WayPointFlag data for specific N."""
    waypoint_file = os.path.join(BASE_PATH, f"WayPointFlag{N}warehouse.json")

    if os.path.exists(waypoint_file):
        try:
            with open(waypoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Failed to load waypoint flag: {e}")

    return None


def load_optimization_data(N):
    """Load optimization result data for specific N.
    For N=2: Use GA result (InitialGuess)
    For N=3,4,5,6: Use Optimization result
    """
    if N == 2:
        # N=2 uses GA result
        opt_file = os.path.join(BASE_PATH, f"InitialGuess{N}warehouse.json")
    else:
        # Other N use Optimization result
        opt_file = os.path.join(BASE_PATH, f"Optimization_withSC_path{N}warehouse.json")

    if os.path.exists(opt_file):
        try:
            with open(opt_file, 'r') as f:
                data = json.load(f)
            # For GA result, rename 'Initial_guess_phi' to 'Optimization_phi' for consistency
            if N == 2 and 'Initial_guess_phi' in data:
                data['Optimization_phi'] = data['Initial_guess_phi']
            return data
        except Exception as e:
            print(f"[!] Failed to load optimization: {e}")

    return None


def generate_robot_trajectory(robot_data, reeb_graph):
    """
    Generate trajectory points using same logic as separate_subfigures_visualization.py.
    Returns: all_points_x, all_points_y, all_velocities, wp_x, wp_y, relay_indices, arc_annotations
    """
    waypoints = robot_data['waypoints']
    phi = robot_data['phi']
    r0 = robot_data['r0']
    l = robot_data['l']
    phi_new = robot_data.get('phi_new', phi)
    time_segments = robot_data['time_segments']
    Flagb = robot_data.get('Flagb', [0] * len(waypoints))

    # Extract waypoint positions
    if 'waypoint_positions' in robot_data and robot_data['waypoint_positions']:
        wp_x = [pos[0] for pos in robot_data['waypoint_positions']]
        wp_y = [pos[1] for pos in robot_data['waypoint_positions']]
    else:
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
                return [], [], [], [], [], [], []

    all_points_x = []
    all_points_y = []
    all_velocities = []
    cumulative_time = 0.0
    arc_annotations = []

    for i in range(len(waypoints) - 1):
        flagb_i = Flagb[i] if i < len(Flagb) else 0
        phi1 = phi[i] + flagb_i * np.pi / 2
        angle_start = phi1

        x_start = wp_x[i]
        y_start = wp_y[i]

        r_x = x_start - r0[i] * np.cos(phi1 + np.pi / 2)
        r_y = y_start - r0[i] * np.sin(phi1 + np.pi / 2)

        if i == 0:
            all_points_x.append(x_start)
            all_points_y.append(y_start)
            all_velocities.append(0.0)

        delta_phi = phi[i+1] - phi_new[i]
        arc_radius = r0[i]
        has_arc_times = i < len(time_segments) and 'arc' in time_segments[i] and len(time_segments[i]['arc']) > 0

        if abs(delta_phi) > 0.001 and abs(arc_radius) > 0.001:
            if has_arc_times:
                arc_times = time_segments[i]['arc']
                N_arc = len(arc_times)

                for j in range(1, N_arc + 1):
                    angle_fraction = j / N_arc
                    current_angle = phi1 + delta_phi * angle_fraction

                    point_x = r0[i] * np.cos(current_angle + np.pi / 2) + r_x
                    point_y = r0[i] * np.sin(current_angle + np.pi / 2) + r_y

                    cumulative_time += arc_times[j-1]

                    total_arc_length = abs(arc_radius * delta_phi)
                    arc_segment_length = total_arc_length / N_arc
                    velocity = arc_segment_length / arc_times[j-1] if arc_times[j-1] > 0 else 0

                    all_points_x.append(point_x)
                    all_points_y.append(point_y)
                    all_velocities.append(velocity)

                    if j == (N_arc // 2):
                        dx_from_center = point_x - r_x
                        dy_from_center = point_y - r_y
                        radial_distance = np.sqrt(dx_from_center**2 + dy_from_center**2)

                        if radial_distance > 0.001:
                            dx_norm = dx_from_center / radial_distance
                            dy_norm = dy_from_center / radial_distance

                            text_offset = 0.25
                            annotation_x = point_x + text_offset * dx_norm
                            annotation_y = point_y + text_offset * dy_norm

                            arc_annotations.append((point_x, point_y, annotation_x, annotation_y, abs(arc_radius)))
            else:
                N_arc_default = 20
                for j in range(1, N_arc_default + 1):
                    angle_fraction = j / N_arc_default
                    current_angle = phi1 + delta_phi * angle_fraction

                    point_x = r0[i] * np.cos(current_angle + np.pi / 2) + r_x
                    point_y = r0[i] * np.sin(current_angle + np.pi / 2) + r_y

                    all_points_x.append(point_x)
                    all_points_y.append(point_y)
                    all_velocities.append(all_velocities[-1] if all_velocities else 0.0)

        if len(all_points_x) > 0:
            x_after_arc = all_points_x[-1]
            y_after_arc = all_points_y[-1]
        else:
            x_after_arc = x_start
            y_after_arc = y_start

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
                line_end_x = l_x + line_length * np.cos(phi1_line)
                line_end_y = l_y + line_length * np.sin(phi1_line)

                if len(all_points_x) == 0 or (abs(all_points_x[-1] - l_x) > 0.001 or abs(all_points_y[-1] - l_y) > 0.001):
                    all_points_x.append(l_x)
                    all_points_y.append(l_y)
                    all_velocities.append(all_velocities[-1] if all_velocities else 0.0)

                all_points_x.append(line_end_x)
                all_points_y.append(line_end_y)
                all_velocities.append(all_velocities[-1] if all_velocities else 0.0)

    relay_indices = [i for i in range(len(Flagb)) if i < len(Flagb) and Flagb[i] != 0]

    return all_points_x, all_points_y, all_velocities, wp_x, wp_y, relay_indices, arc_annotations


def get_unified_limits_from_environment():
    """Calculate unified axis limits based on environment bounds."""
    env_data = load_environment_data()
    if env_data:
        coord_bounds = env_data.get('coord_bounds', [0, 1100, 0, 600])
        x_min, x_max, y_min, y_max = coord_bounds
        # Convert to meters and add padding
        unified_xlim = (x_min * PIXEL_TO_METER_SCALE - 0.05,
                        x_max * PIXEL_TO_METER_SCALE + 0.05)
        unified_ylim = (y_min * PIXEL_TO_METER_SCALE - 0.05,
                        y_max * PIXEL_TO_METER_SCALE + 0.05)
    else:
        unified_xlim = (0, 2.5)
        unified_ylim = (0, 1.5)

    return unified_xlim, unified_ylim


def adjust_annotation_positions(annotations, min_distance=0.12, xlim=None, ylim=None, margin=0.08):
    """Adjust annotation text positions to avoid overlaps."""
    if len(annotations) <= 1:
        return annotations

    adjusted = [anno.copy() for anno in annotations]

    if xlim is None:
        xlim = (0, 2.5)
    if ylim is None:
        ylim = (0, 1.5)

    text_width = 0.25
    text_height = 0.10

    indices = list(range(len(adjusted)))
    indices.sort(key=lambda i: (adjusted[i]['arc_x'], adjusted[i]['arc_y']))

    placed_boxes = []

    for idx in indices:
        anno = adjusted[idx]
        arc_x, arc_y = anno['arc_x'], anno['arc_y']

        base_offset = 0.20
        candidates = []
        candidates.append((anno['text_x'], anno['text_y']))

        angles = [
            np.pi/2, -np.pi/2,
            np.pi/3, -np.pi/3, 2*np.pi/3, -2*np.pi/3,
            0, np.pi,
            np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4,
        ]
        for dist_mult in [1.0, 1.3, 1.6, 2.0, 2.5]:
            offset = base_offset * dist_mult
            for angle in angles:
                cx = arc_x + offset * np.cos(angle)
                cy = arc_y + offset * np.sin(angle)
                candidates.append((cx, cy))

        best_pos = None
        for cx, cy in candidates:
            if cx - text_width/2 < xlim[0] + margin:
                cx = xlim[0] + margin + text_width/2
            if cx + text_width/2 > xlim[1] - margin:
                cx = xlim[1] - margin - text_width/2
            if cy - text_height/2 < ylim[0] + margin:
                cy = ylim[0] + margin + text_height/2
            if cy + text_height/2 > ylim[1] - margin:
                cy = ylim[1] - margin - text_height/2

            collision = False
            for (px, py, pw, ph) in placed_boxes:
                if (abs(cx - px) < (text_width/2 + pw) and
                    abs(cy - py) < (text_height/2 + ph)):
                    collision = True
                    break

            if not collision:
                best_pos = (cx, cy)
                break

        if best_pos is None:
            max_min_dist = -1
            for cx, cy in candidates:
                if cx - text_width/2 < xlim[0] + margin:
                    cx = xlim[0] + margin + text_width/2
                if cx + text_width/2 > xlim[1] - margin:
                    cx = xlim[1] - margin - text_width/2
                if cy - text_height/2 < ylim[0] + margin:
                    cy = ylim[0] + margin + text_height/2
                if cy + text_height/2 > ylim[1] - margin:
                    cy = ylim[1] - margin - text_height/2

                if not placed_boxes:
                    min_dist = float('inf')
                else:
                    min_dist = min(np.sqrt((cx-px)**2 + (cy-py)**2)
                                   for (px, py, pw, ph) in placed_boxes)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_pos = (cx, cy)

        adjusted[idx]['text_x'] = best_pos[0]
        adjusted[idx]['text_y'] = best_pos[1]
        placed_boxes.append((best_pos[0], best_pos[1], text_width/2, text_height/2))

    return adjusted


def plot_assignment_result(N, reeb_graph, unified_xlim, unified_ylim):
    """Plot Assignment Result with arrows showing waypoints and relay points for N robots."""
    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4/2.54))

    reeb_graph_rebuilt = load_reeb_graph(use_rebuilt=True)
    if reeb_graph_rebuilt is None:
        reeb_graph_rebuilt = reeb_graph

    env_data = load_environment_data()

    # Draw environment boundary
    if env_data:
        coord_bounds = env_data.get('coord_bounds', [0, 1100, 0, 600])
        x_min, x_max, y_min, y_max = coord_bounds
        boundary_vertices_pixel = np.array([
            [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]
        ])
        boundary_vertices_meter = np.array([convert_pixel_to_meter(v) for v in boundary_vertices_pixel])
        ax.plot(boundary_vertices_meter[:, 0], boundary_vertices_meter[:, 1],
               color='black', linewidth=1.5, zorder=1)

    # Draw obstacles
    if env_data and 'polygons' in env_data:
        for polygon_data in env_data['polygons']:
            vertices_pixel = np.array(polygon_data['vertices'])
            vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
            vertices_closed = np.vstack([vertices_meter, vertices_meter[0]])
            ax.fill(vertices_closed[:, 0], vertices_closed[:, 1],
                   color='black', alpha=1.0, zorder=1)

    # Draw graph nodes
    for node_id in reeb_graph_rebuilt.nodes:
        pos_pixel = reeb_graph_rebuilt.nodes[node_id].configuration
        pos_meter = convert_pixel_to_meter(pos_pixel)
        ax.scatter(pos_meter[0], pos_meter[1], color='grey', s=16, marker='o',
                  alpha=0.6, zorder=2, edgecolors='darkgrey', linewidths=0.3)

    # Draw graph edges
    for node_id, out_neighbors in reeb_graph_rebuilt.out_neighbors.items():
        if node_id in reeb_graph_rebuilt.nodes:
            start_pixel = reeb_graph_rebuilt.nodes[node_id].configuration
            start_meter = convert_pixel_to_meter(start_pixel)
            for neighbor_id in out_neighbors:
                if neighbor_id in reeb_graph_rebuilt.nodes:
                    end_pixel = reeb_graph_rebuilt.nodes[neighbor_id].configuration
                    end_meter = convert_pixel_to_meter(end_pixel)
                    ax.plot([start_meter[0], end_meter[0]], [start_meter[1], end_meter[1]],
                           'grey', linewidth=0.5, alpha=0.5, zorder=2)

    waypoints_arcs, relay_arcs = load_assignment_data(N)

    def get_node_position(node_id):
        if node_id in reeb_graph_rebuilt.nodes:
            pos_pixel = reeb_graph_rebuilt.nodes[node_id].configuration
            return convert_pixel_to_meter(pos_pixel)
        return None

    waypoint_label_added = False
    relay_label_added = False

    if waypoints_arcs:
        for i, j, _ in waypoints_arcs:
            start_meter = get_node_position(i)
            end_meter = get_node_position(j)
            if start_meter and end_meter:
                label = 'Waypoints' if not waypoint_label_added else ''
                waypoint_label_added = True
                ax.arrow(start_meter[0], start_meter[1],
                        end_meter[0] - start_meter[0], end_meter[1] - start_meter[1],
                        width=0.008, head_width=0.050, head_length=0.040,
                        fc='green', ec='green', alpha=0.8, zorder=5, label=label)

    if relay_arcs:
        for i, j, _ in relay_arcs:
            start_meter = get_node_position(i)
            end_meter = get_node_position(j)
            if start_meter and end_meter:
                label = 'Relay Points' if not relay_label_added else ''
                relay_label_added = True
                ax.arrow(start_meter[0], start_meter[1],
                        end_meter[0] - start_meter[0], end_meter[1] - start_meter[1],
                        width=0.008, head_width=0.050, head_length=0.040,
                        fc='red', ec='red', alpha=0.8, zorder=6, label=label)

    ax.set_title(f'Assignment Result (N={N})', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    unified_tick_spacing = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=2, frameon=True)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'warehouse_N{N}_assignment_result.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")


def plot_reeb_graph_with_environment(unified_xlim, unified_ylim):
    """Plot Directed Skeleton Graph overlaid on environment (plotted once, not per N).
    Uses Graph_new_warehouse.json (rebuilt graph) to match nodes in Assignment Result.
    """
    # Use rebuilt graph to match the gray points in Assignment Result
    graph_file = os.path.join(BASE_PATH, "Graph_new_warehouse.json")
    if not os.path.exists(graph_file):
        # Fallback to original graph
        graph_file = os.path.join(BASE_PATH, "Graph_warehouse.json")
    if not os.path.exists(graph_file):
        print(f"[!] Graph file not found: {graph_file}")
        return False

    with open(graph_file, 'r') as f:
        graph_data = json.load(f)

    env_data = load_environment_data()
    if env_data is None:
        return False

    nodes = graph_data['nodes']
    out_neighbors = graph_data['out_neighbors']

    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4/2.54))

    # Draw boundary
    coord_bounds = env_data.get('coord_bounds', [0, 800, 0, 600])
    x_min, x_max, y_min, y_max = coord_bounds
    boundary_vertices_pixel = np.array([
        [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]
    ])
    boundary_vertices_meter = np.array([convert_pixel_to_meter(v) for v in boundary_vertices_pixel])
    ax.plot(boundary_vertices_meter[:, 0], boundary_vertices_meter[:, 1],
           color='black', linewidth=1.5, zorder=1)

    # Draw obstacles (solid black)
    for polygon in env_data['polygons']:
        vertices_pixel = np.array(polygon['vertices'])
        vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
        vertices_closed = np.vstack([vertices_meter, vertices_meter[0]])
        ax.fill(vertices_closed[:, 0], vertices_closed[:, 1],
               color='black', alpha=1.0, zorder=1)

    # Draw graph edges (blue)
    for node_id_str, neighbors in out_neighbors.items():
        node_id = int(node_id_str)
        if node_id < len(nodes):
            node_pos_pixel = nodes[node_id][1]
            node_pos_meter = convert_pixel_to_meter(node_pos_pixel)
            for neighbor_id in neighbors:
                if neighbor_id < len(nodes):
                    neighbor_pos_pixel = nodes[neighbor_id][1]
                    neighbor_pos_meter = convert_pixel_to_meter(neighbor_pos_pixel)
                    ax.plot([node_pos_meter[0], neighbor_pos_meter[0]],
                           [node_pos_meter[1], neighbor_pos_meter[1]],
                           color='steelblue', linewidth=0.8, alpha=0.7, zorder=2)

    # Draw nodes (blue with dark edge)
    node_positions_pixel = np.array([node[1] for node in nodes])
    node_positions_meter = np.array([convert_pixel_to_meter(pos) for pos in node_positions_pixel])
    ax.scatter(node_positions_meter[:, 0], node_positions_meter[:, 1],
              c='steelblue', s=16, zorder=3,
              edgecolors='darkblue', linewidths=0.5, alpha=0.9)

    # Find start and end using start_pose and goal_pose from graph data
    start_pose = graph_data.get('start_pose', None)
    goal_pose = graph_data.get('goal_pose', None)

    # Mark start node
    if start_pose is not None:
        start_pos_meter = convert_pixel_to_meter(start_pose)
        ax.scatter(start_pos_meter[0], start_pos_meter[1], c='orange', s=30, zorder=5,
                  marker='o', edgecolors='darkorange', linewidths=0.8, label='Start')

    # Mark end node
    if goal_pose is not None:
        end_pos_meter = convert_pixel_to_meter(goal_pose)
        ax.scatter(end_pos_meter[0], end_pos_meter[1], c='orange', s=30, zorder=5,
                  marker='s', edgecolors='darkorange', linewidths=0.8, label='End')

    ax.set_title('Directed Skeleton Graph', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    unified_tick_spacing = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    # Add legend for start/end
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=2, frameon=True)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, 'warehouse_directed_skeleton_graph.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")
    return True


def check_line_polygon_collision(start, end, polygon_vertices):
    """Check if line segment intersects with polygon."""
    from shapely.geometry import LineString, Polygon
    line = LineString([start, end])
    poly = Polygon(polygon_vertices)
    expanded_poly = poly.buffer(0.01)
    return line.intersects(expanded_poly)


def check_line_boundary_collision(start, end, coord_bounds):
    """Check if line segment goes outside workspace boundaries."""
    if coord_bounds is None:
        return False
    x_min, x_max, y_min, y_max = coord_bounds
    for point in [start, end]:
        if (point[0] < x_min or point[0] > x_max or
            point[1] < y_min or point[1] > y_max):
            return True
    return False


def check_corridor_collision(start, end, polygons, coord_bounds):
    """Check if corridor edge collides with any obstacle or boundary."""
    if check_line_boundary_collision(start, end, coord_bounds):
        return True
    for polygon in polygons:
        vertices = polygon.get('vertices', [])
        if vertices and check_line_polygon_collision(start, end, vertices):
            return True
    return False


def compute_safe_corridors(waypoints, nodes, environment):
    """Compute safe corridors for each waypoint segment with collision detection.
    Uses expanded obstacles (buffered by 0.04m) for robot safety margin.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    N = len(waypoints)
    safe_corridors = []
    max_distance = 100 * PIXEL_TO_METER_SCALE
    step_size = 0.2 * PIXEL_TO_METER_SCALE
    margin = 5.0 * PIXEL_TO_METER_SCALE
    obstacle_buffer = 0.04  # Same buffer as used in plot_planning_result

    coord_bounds = environment.get('coord_bounds', None)
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

    # Use expanded obstacle vertices for distance calculation
    all_obstacle_vertices = []
    for poly in environment.get('polygons', []):
        vertices = poly.get('vertices', [])
        if vertices:
            vertices_m = [convert_pixel_to_meter(v) for v in vertices]
            # Create expanded polygon
            shapely_poly = ShapelyPolygon(vertices_m)
            expanded_poly = shapely_poly.buffer(obstacle_buffer)
            if hasattr(expanded_poly, 'exterior'):
                # Extract vertices from expanded polygon
                for coord in expanded_poly.exterior.coords:
                    all_obstacle_vertices.append(coord)

    # Use expanded polygons for collision detection
    polygons_m = []
    for poly in environment.get('polygons', []):
        vertices = poly.get('vertices', [])
        if vertices:
            vertices_m = [convert_pixel_to_meter(v) for v in vertices]
            # Create expanded polygon and extract its vertices
            shapely_poly = ShapelyPolygon(vertices_m)
            expanded_poly = shapely_poly.buffer(obstacle_buffer)
            if hasattr(expanded_poly, 'exterior'):
                expanded_vertices = list(expanded_poly.exterior.coords)
                polygons_m.append({'vertices': expanded_vertices})

    for i in range(N-1):
        start_pos = np.array(nodes[waypoints[i]])
        end_pos = np.array(nodes[waypoints[i+1]])

        line_vec = end_pos - start_pos
        line_length = np.linalg.norm(line_vec)

        if line_length < 1e-9:
            safe_corridors.append([0, 0, 0])
            continue

        line_unit = line_vec / line_length
        perp_unit = np.array([-line_unit[1], line_unit[0]])

        min_dist_positive = max_distance
        min_dist_negative = max_distance

        for obs_vertex in all_obstacle_vertices:
            obs_vertex = np.array(obs_vertex)
            vec_to_vertex = obs_vertex - start_pos
            proj_along_line = np.dot(vec_to_vertex, line_unit)

            if proj_along_line >= -margin and proj_along_line <= line_length + margin:
                signed_perp_dist = np.dot(vec_to_vertex, perp_unit)
                if signed_perp_dist > 0:
                    min_dist_positive = min(min_dist_positive, signed_perp_dist)
                else:
                    min_dist_negative = min(min_dist_negative, abs(signed_perp_dist))

        db_max_step = 0
        collision_found = False
        while not collision_found and db_max_step < max_distance:
            db_max_step += step_size
            p_start_up = list(start_pos + db_max_step * perp_unit)
            p_end_up = list(end_pos + db_max_step * perp_unit)
            if check_corridor_collision(p_start_up, p_end_up, polygons_m, coord_bounds_m):
                collision_found = True
                break

        db_min_step = 0
        collision_found = False
        while not collision_found and db_min_step < max_distance:
            db_min_step += step_size
            p_start_low = list(start_pos - db_min_step * perp_unit)
            p_end_low = list(end_pos - db_min_step * perp_unit)
            if check_corridor_collision(p_start_low, p_end_low, polygons_m, coord_bounds_m):
                collision_found = True
                break

        db_min = min(db_min_step, min_dist_negative)
        db_max = min(db_max_step, min_dist_positive)

        if abs(end_pos[0] - start_pos[0]) < 1e-6:
            slope = 100000000
        else:
            slope = (end_pos[1] - start_pos[1]) / (end_pos[0] - start_pos[0])

        y_min = -db_min
        y_max = db_max

        safe_corridors.append([slope, min(y_min, y_max), max(y_min, y_max)])

    return safe_corridors


def plot_planning_result(N, unified_xlim, unified_ylim):
    """Plot planning result with safe corridors and optimized trajectory."""
    waypoint_data = load_waypoint_flag_data(N)
    optimization = load_optimization_data(N)

    if waypoint_data is None or optimization is None:
        print(f"[!] Missing data for N={N}")
        return False

    waypoints = waypoint_data['Waypoints']
    flagb = waypoint_data['FlagB']

    reeb_graph = load_reeb_graph(use_rebuilt=True)
    if reeb_graph is None:
        reeb_graph = load_reeb_graph(use_rebuilt=False)
    if reeb_graph is None:
        return False

    env_data = load_environment_data()
    if env_data is None:
        return False

    # Convert graph nodes to meters
    nodes_meter = {}
    for node_id in reeb_graph.nodes:
        pos_pixel = reeb_graph.nodes[node_id].configuration
        nodes_meter[node_id] = convert_pixel_to_meter(pos_pixel)

    safe_corridors = compute_safe_corridors(waypoints, nodes_meter, env_data)

    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4/2.54))

    # Plot obstacles
    from shapely.geometry import Polygon as ShapelyPolygon
    for polygon in env_data['polygons']:
        vertices_pixel = np.array(polygon['vertices'])
        vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])

        shapely_poly = ShapelyPolygon(vertices_meter)
        expanded_poly = shapely_poly.buffer(0.04)

        if hasattr(expanded_poly, 'exterior'):
            exp_coords = list(expanded_poly.exterior.coords)
            expanded_patch = PolygonPatch(exp_coords, closed=True,
                                         facecolor='lightgray', edgecolor='black',
                                         alpha=0.4, linewidth=0.5)
            ax.add_patch(expanded_patch)

        poly = PolygonPatch(vertices_meter, closed=True, facecolor='#404040',
                           edgecolor='black', alpha=0.9, linewidth=0.8)
        ax.add_patch(poly)

    # Plot trajectory
    num_waypoints = len(waypoints)
    phi = np.array(optimization['Optimization_phi'])
    l = np.array(optimization['Optimization_l']) * PIXEL_TO_METER_SCALE

    r0_raw = np.array(optimization['Optimization_r'])
    STRAIGHT_LINE_SENTINEL = 1e7
    r0 = np.where(np.abs(r0_raw) >= STRAIGHT_LINE_SENTINEL, 0.0, r0_raw * PIXEL_TO_METER_SCALE)

    wp_x = [nodes_meter[wp][0] for wp in waypoints]
    wp_y = [nodes_meter[wp][1] for wp in waypoints]

    phi_new = []
    for i in range(len(waypoints) - 1):
        flagb_i = flagb[i] if i < len(flagb) else 0
        phi_new.append(phi[i] + flagb_i * np.pi / 2)

    for i in range(num_waypoints - 1):
        flagb_i = flagb[i] if i < len(flagb) else 0
        phi1 = phi[i] + flagb_i * np.pi / 2

        x_start = wp_x[i]
        y_start = wp_y[i]

        r_x = x_start - r0[i] * np.cos(phi1 + np.pi / 2)
        r_y = y_start - r0[i] * np.sin(phi1 + np.pi / 2)

        if flagb_i != 0:
            ax.plot(x_start, y_start, 'ro', markersize=4)
        else:
            ax.plot(x_start, y_start, 'go', markersize=4)

        delta_phi = phi[i+1] - phi_new[i]
        arc_radius = r0[i]

        if abs(delta_phi) > 0.001 and abs(arc_radius) > 0.001:
            N_arc = 50
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

        line_length = l[i] if i < len(l) else 0

        if line_length > 0.001:
            is_straight_line = abs(r0[i]) < 0.01

            if is_straight_line:
                l_x = x_start
                l_y = y_start
                phi1_line = phi_new[i]
            else:
                l_x = r0[i] * np.cos(phi[i+1] + np.pi / 2) + r_x
                l_y = r0[i] * np.sin(phi[i+1] + np.pi / 2) + r_y
                phi1_line = phi[i+1]

            line_end_x = l_x + line_length * np.cos(phi1_line)
            line_end_y = l_y + line_length * np.sin(phi1_line)

            ax.plot([l_x, line_end_x], [l_y, line_end_y], 'b-', linewidth=0.8)

    # Plot final waypoint
    final_flagb = flagb[-1] if len(flagb) > 0 else 0
    if final_flagb != 0:
        ax.plot(wp_x[-1], wp_y[-1], 'ro', markersize=4)
    else:
        ax.plot(wp_x[-1], wp_y[-1], 'go', markersize=4)

    # Plot safe corridors
    for i in range(num_waypoints - 1):
        start_pos = np.array(nodes_meter[waypoints[i]])
        end_pos = np.array(nodes_meter[waypoints[i+1]])

        slope = safe_corridors[i][0]
        y_min = safe_corridors[i][1]
        y_max = safe_corridors[i][2]

        if abs(slope) > 100000:
            x_coords = [start_pos[0] + y_min, start_pos[0] + y_max,
                       end_pos[0] + y_max, end_pos[0] + y_min, start_pos[0] + y_min]
            y_coords = [start_pos[1], start_pos[1], end_pos[1], end_pos[1], start_pos[1]]
        else:
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

    ax.set_title(f'Planning Result (N={N})', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    unified_tick_spacing = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=2, frameon=True)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'warehouse_N{N}_planning_result.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")
    return True


def plot_velocity_trajectories(N, reeb_graph, unified_xlim, unified_ylim):
    """Plot velocity-colored trajectories for N robots."""
    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4.5/2.54))

    env_data = load_environment_data()
    if env_data and 'polygons' in env_data:
        for polygon_data in env_data['polygons']:
            vertices_pixel = np.array(polygon_data['vertices'])
            vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
            poly = PolygonPatch(vertices_meter, closed=True, facecolor='#C0C0C0',
                               edgecolor='#909090', alpha=0.7, linewidth=0.5, zorder=1)
            ax.add_patch(poly)

    try:
        cmap = plt.colormaps['viridis']
    except AttributeError:
        cmap = plt.cm.get_cmap('viridis')

    # Load robot data for this N
    robot_data_list = []
    for robot_id in range(N):
        robot_data = load_robot_data(robot_id, N)
        if robot_data is not None:
            robot_data_list.append(robot_data)

    if len(robot_data_list) == 0:
        print(f"[!] No robot data found for N={N}")
        plt.close()
        return

    all_case_velocities = []
    trajectory_data = []

    for robot_data in robot_data_list:
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

    min_velocity = 0.0
    max_velocity = v_max
    if all_case_velocities:
        actual_min_vel = min(v for v in all_case_velocities if v is not None)
        actual_max_vel = max(v for v in all_case_velocities if v is not None)
        min_velocity = max(0.0, actual_min_vel)
        max_velocity = max(min_velocity + 0.001, actual_max_vel)

    norm = Normalize(vmin=min_velocity, vmax=max_velocity)

    all_annotations = []
    for traj in trajectory_data:
        for arc_x, arc_y, text_x, text_y, radius_value in traj['arcs']:
            all_annotations.append({
                'arc_x': arc_x, 'arc_y': arc_y,
                'text_x': text_x, 'text_y': text_y,
                'radius': radius_value
            })

    adjusted_annotations = adjust_annotation_positions(all_annotations, xlim=unified_xlim, ylim=unified_ylim, margin=0.15)

    for traj in trajectory_data:
        all_points_x = traj['x']
        all_points_y = traj['y']
        all_velocities = traj['v']
        wp_x = traj['wp_x']
        wp_y = traj['wp_y']
        relay_indices = traj['relay']

        if len(all_points_x) > 1:
            for i in range(len(all_points_x) - 1):
                current_velocity = all_velocities[i] if all_velocities[i] is not None else 0.0
                color = cmap(norm(current_velocity))
                ax.plot([all_points_x[i], all_points_x[i+1]],
                       [all_points_y[i], all_points_y[i+1]],
                       color=color, linewidth=0.8, alpha=0.8)

        ax.scatter(wp_x, wp_y, color='green', s=16, marker='o', zorder=5)

        if relay_indices:
            relay_x = [wp_x[i] for i in relay_indices]
            relay_y = [wp_y[i] for i in relay_indices]
            ax.scatter(relay_x, relay_y, color='red', s=16, marker='^', zorder=5)

    for anno in adjusted_annotations:
        radius_text = f"r={anno['radius']:.2f}"
        ax.text(anno['text_x'], anno['text_y'], radius_text,
               fontsize=7, ha='center', va='center',
               color='black', weight='bold', zorder=10)
        ax.annotate('', xy=(anno['arc_x'], anno['arc_y']),
                   xytext=(anno['text_x'], anno['text_y']),
                   arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.6),
                   zorder=9)

    ax.set_title(f'Planning Trajectory (N={N})', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    unified_tick_spacing = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.28)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbar_ax = inset_axes(ax, width="80%", height="5%", loc='lower center',
                        bbox_to_anchor=(0, -0.20, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=6)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, f'warehouse_N{N}_velocity_trajectories.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")


def plot_all_waypoints(unified_xlim, unified_ylim):
    """Plot all waypoints from all N values on a single figure."""
    fig, ax = plt.subplots(1, 1, figsize=(5/2.54, 4/2.54))

    reeb_graph = load_reeb_graph(use_rebuilt=True)
    if reeb_graph is None:
        reeb_graph = load_reeb_graph(use_rebuilt=False)
    if reeb_graph is None:
        print("[!] Cannot load reeb graph for all waypoints plot")
        return False

    env_data = load_environment_data()

    # Draw environment boundary
    if env_data:
        coord_bounds = env_data.get('coord_bounds', [0, 1100, 0, 600])
        x_min, x_max, y_min, y_max = coord_bounds
        boundary_vertices_pixel = np.array([
            [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]
        ])
        boundary_vertices_meter = np.array([convert_pixel_to_meter(v) for v in boundary_vertices_pixel])
        ax.plot(boundary_vertices_meter[:, 0], boundary_vertices_meter[:, 1],
               color='black', linewidth=1.5, zorder=1)

    # Draw obstacles
    if env_data and 'polygons' in env_data:
        for polygon_data in env_data['polygons']:
            vertices_pixel = np.array(polygon_data['vertices'])
            vertices_meter = np.array([convert_pixel_to_meter(v) for v in vertices_pixel])
            vertices_closed = np.vstack([vertices_meter, vertices_meter[0]])
            ax.fill(vertices_closed[:, 0], vertices_closed[:, 1],
                   color='black', alpha=1.0, zorder=1)

    # Draw graph nodes (light gray background)
    for node_id in reeb_graph.nodes:
        pos_pixel = reeb_graph.nodes[node_id].configuration
        pos_meter = convert_pixel_to_meter(pos_pixel)
        ax.scatter(pos_meter[0], pos_meter[1], color='lightgrey', s=8, marker='o',
                  alpha=0.4, zorder=2)

    # Collect all waypoints from all N values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each N

    for idx, N in enumerate(ROBOT_COUNTS):
        waypoint_data = load_waypoint_flag_data(N)
        if waypoint_data is None:
            continue

        waypoints = waypoint_data['Waypoints']
        flagb = waypoint_data.get('FlagB', [0] * len(waypoints))

        # Get waypoint positions
        wp_positions = []
        for wp_id in waypoints:
            if wp_id in reeb_graph.nodes:
                pos_pixel = reeb_graph.nodes[wp_id].configuration
                pos_meter = convert_pixel_to_meter(pos_pixel)
                wp_positions.append(pos_meter)

        if wp_positions:
            wp_x = [p[0] for p in wp_positions]
            wp_y = [p[1] for p in wp_positions]

            # Plot waypoints with different colors for each N
            ax.scatter(wp_x, wp_y, color=colors[idx % len(colors)], s=20, marker='o',
                      zorder=5, label=f'N={N}', alpha=0.8, edgecolors='black', linewidths=0.3)

            # Connect waypoints with lines
            ax.plot(wp_x, wp_y, color=colors[idx % len(colors)], linewidth=0.8, alpha=0.5, zorder=4)

    ax.set_title('All Waypoints', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)
    ax.set_xlim(unified_xlim)
    ax.set_ylim(unified_ylim)

    unified_tick_spacing = 0.5
    ax.xaxis.set_major_locator(MultipleLocator(unified_tick_spacing))
    ax.yaxis.set_major_locator(MultipleLocator(unified_tick_spacing))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=5, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=5, frameon=True)

    plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.20)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, 'warehouse_all_waypoints.png')
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"[SAVE] {output_file}")
    return True


def process_warehouse_N(N, reeb_graph, unified_xlim, unified_ylim):
    """Process warehouse case with N robots to generate plots (excluding Reeb graph)."""
    print(f"\n[*] Processing warehouse with N={N} robots...")

    try:
        plot_assignment_result(N, reeb_graph, unified_xlim, unified_ylim)
        plot_planning_result(N, unified_xlim, unified_ylim)
        # Only N=3,4,5 have velocity trajectory data
        if N in [3, 4, 5]:
            plot_velocity_trajectories(N, reeb_graph, unified_xlim, unified_ylim)
        return True
    except Exception as e:
        print(f"[X] Plot failed for N={N}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main batch processing loop."""
    print("=" * 60)
    print("WAREHOUSE_N VISUALIZATION")
    print("Generating plots:")
    print("  - Reeb Graph (once)")
    print("  - All Waypoints (once)")
    print("  - Per N: Assignment Result, Planning Result, Velocity Trajectory")
    print("=" * 60)

    # Load shared resources
    reeb_graph = load_reeb_graph()
    if reeb_graph is None:
        print("[X] Failed to load Reeb graph")
        return

    unified_xlim, unified_ylim = get_unified_limits_from_environment()

    # Plot Directed Skeleton Graph only once
    print("\n[*] Plotting Directed Skeleton Graph (once)...")
    plot_reeb_graph_with_environment(unified_xlim, unified_ylim)

    # Plot all waypoints figure (once)
    print("\n[*] Plotting All Waypoints (once)...")
    plot_all_waypoints(unified_xlim, unified_ylim)

    # Process each N value
    success_count = 0
    failed_N = []

    for N in ROBOT_COUNTS:
        if process_warehouse_N(N, reeb_graph, unified_xlim, unified_ylim):
            success_count += 1
        else:
            failed_N.append(N)

    print("\n" + "=" * 60)
    print(f"[OK] Success: {success_count}/{len(ROBOT_COUNTS)}")
    if failed_N:
        print(f"[X] Failed N values: {failed_N}")
    print("=" * 60)


if __name__ == "__main__":
    main()