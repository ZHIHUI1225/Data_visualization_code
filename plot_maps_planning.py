#!/usr/bin/env python3
"""
Visualize planning results from MAPS folder.
Matches the style of Optimization_winthSC_path4map5.png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon

# Coordinate conversion factor: pixel to meter
PIXEL_TO_METER = 0.0023


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_graph(graph_file):
    """
    Load graph nodes from Graph_*.json
    Returns dict mapping node_id -> configuration [x, y] in METERS
    Graph format: nodes are arrays [id, [x, y], null, false]
    """
    data = load_json(graph_file)
    nodes = {}
    for node in data.get('nodes', []):
        # Node format: [id, [x, y], null, false]
        if isinstance(node, list) and len(node) >= 2:
            node_id = node[0]
            config = node[1]  # in pixels
            if node_id is not None and config is not None:
                # Convert to meters
                nodes[node_id] = [config[0] * PIXEL_TO_METER, config[1] * PIXEL_TO_METER]
    return nodes


def plot_environment(ax, environment_data):
    """Draw environment obstacles and boundaries"""
    from shapely.geometry import Polygon as ShapelyPolygon

    polygons = environment_data.get('polygons', [])
    coord_bounds = environment_data.get('coord_bounds', None)

    # Draw boundary (convert to meters)
    if coord_bounds:
        x_min, x_max, y_min, y_max = coord_bounds
        x_min_m = x_min * PIXEL_TO_METER
        x_max_m = x_max * PIXEL_TO_METER
        y_min_m = y_min * PIXEL_TO_METER
        y_max_m = y_max * PIXEL_TO_METER

        boundary_width = 1.0
        ax.plot([x_min_m, x_max_m], [y_min_m, y_min_m], 'k-', linewidth=boundary_width)
        ax.plot([x_min_m, x_max_m], [y_max_m, y_max_m], 'k-', linewidth=boundary_width)
        ax.plot([x_min_m, x_min_m], [y_min_m, y_max_m], 'k-', linewidth=boundary_width)
        ax.plot([x_max_m, x_max_m], [y_min_m, y_max_m], 'k-', linewidth=boundary_width)

    # Draw obstacles (convert to meters)
    expanded_labeled = False
    for polygon in polygons:
        vertices = polygon.get('vertices', [])
        if vertices:
            vertices_m = [[v[0] * PIXEL_TO_METER, v[1] * PIXEL_TO_METER] for v in vertices]

            # Draw expanded obstacle area (0.04m buffer) with light gray fill
            shapely_poly = ShapelyPolygon(vertices_m)
            expanded_poly = shapely_poly.buffer(0.04)

            # Extract exterior coordinates and fill expanded area
            if hasattr(expanded_poly, 'exterior'):
                exp_coords = list(expanded_poly.exterior.coords)

                # Fill expanded area with light gray
                if not expanded_labeled:
                    expanded_patch = MPLPolygon(exp_coords, closed=True,
                                               facecolor='lightgray', edgecolor='black',
                                               alpha=0.4, linewidth=0.5,
                                               label='Expanded area (0.04m)')
                    expanded_labeled = True
                else:
                    expanded_patch = MPLPolygon(exp_coords, closed=True,
                                               facecolor='lightgray', edgecolor='black',
                                               alpha=0.4, linewidth=0.5)
                ax.add_patch(expanded_patch)

            # Draw original obstacle (darker, on top)
            poly = MPLPolygon(vertices_m, closed=True, facecolor='#404040',
                            edgecolor='black', alpha=0.9, linewidth=0.8)
            ax.add_patch(poly)


def check_line_polygon_collision(start, end, polygon_vertices):
    """
    Check if line segment intersects with polygon.
    Uses shapely for robust geometric intersection.
    Obstacles are expanded by 0.04m for safety clearance.
    """
    from shapely.geometry import LineString, Polygon

    line = LineString([start, end])
    poly = Polygon(polygon_vertices)
    expanded_poly = poly.buffer(0.04)  # Expand obstacle by 0.04m

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
    Returns list of [slope, y_min, y_max] for each segment IN METERS.
    Matches the algorithm from Planing_functions.py get_safe_corridor().
    """
    N = len(waypoints)
    safe_corridors = []
    max_distance = 100 * PIXEL_TO_METER  # Maximum corridor width in meters
    step_size = 0.5 * PIXEL_TO_METER     # Step size in meters

    coord_bounds = environment.get('coord_bounds', None)
    # Add safety margin and convert to meters
    safety_margin = 5.0 * PIXEL_TO_METER
    if coord_bounds:
        x_min, x_max, y_min, y_max = coord_bounds
        coord_bounds = [(x_min + safety_margin / PIXEL_TO_METER) * PIXEL_TO_METER,
                       (x_max - safety_margin / PIXEL_TO_METER) * PIXEL_TO_METER,
                       (y_min + safety_margin / PIXEL_TO_METER) * PIXEL_TO_METER,
                       (y_max - safety_margin / PIXEL_TO_METER) * PIXEL_TO_METER]

    # Convert polygons to meters for collision detection
    polygons_m = []
    for poly in environment.get('polygons', []):
        vertices = poly.get('vertices', [])
        if vertices:
            vertices_m = [[v[0] * PIXEL_TO_METER, v[1] * PIXEL_TO_METER] for v in vertices]
            polygons_m.append({'vertices': vertices_m})

    for i in range(N-1):
        start_pos = np.array(nodes[waypoints[i]])  # Already in meters
        end_pos = np.array(nodes[waypoints[i+1]])  # Already in meters

        # Handle vertical lines
        if abs(end_pos[0] - start_pos[0]) < 1e-6:
            slope = 100000000  # Large number for vertical

            # Find lower bound (left side)
            db_min = 0
            collision_found = False

            while not collision_found and db_min < max_distance:
                db_min += step_size
                p_start_low = [start_pos[0] - db_min, start_pos[1]]
                p_end_low = [end_pos[0] - db_min, end_pos[1]]

                if check_corridor_collision(p_start_low, p_end_low, polygons_m, coord_bounds):
                    collision_found = True
                    break

            # Find upper bound (right side)
            db_max = 0
            collision_found = False

            while not collision_found and db_max < max_distance:
                db_max += step_size
                p_start_up = [start_pos[0] + db_max, start_pos[1]]
                p_end_up = [end_pos[0] + db_max, end_pos[1]]

                if check_corridor_collision(p_start_up, p_end_up, polygons_m, coord_bounds):
                    collision_found = True
                    break

            y_min = -db_min
            y_max = db_max

        else:
            # Non-vertical line
            slope = (end_pos[1] - start_pos[1]) / (end_pos[0] - start_pos[0])

            # Find lower bound (offset in negative normal direction)
            db_min = 0
            collision_found = False

            while not collision_found and db_min < max_distance:
                db_min += step_size
                # Offset perpendicular to the line
                p_start_low = [start_pos[0] + db_min * slope, start_pos[1] - db_min]
                p_end_low = [end_pos[0] + db_min * slope, end_pos[1] - db_min]

                if check_corridor_collision(p_start_low, p_end_low, polygons_m, coord_bounds):
                    collision_found = True
                    break

            # Find upper bound (offset in positive normal direction)
            db_max = 0
            collision_found = False

            while not collision_found and db_max < max_distance:
                db_max += step_size
                # Offset perpendicular to the line
                p_start_up = [start_pos[0] - db_max * slope, start_pos[1] + db_max]
                p_end_up = [end_pos[0] - db_max * slope, end_pos[1] + db_max]

                if check_corridor_collision(p_start_up, p_end_up, polygons_m, coord_bounds):
                    collision_found = True
                    break

            # Calculate y_min and y_max in local coordinate system
            P = start_pos
            p_A_low = np.array([start_pos[0] + db_min * slope, start_pos[1] - db_min])
            p_A_up = np.array([start_pos[0] - db_max * slope, start_pos[1] + db_max])

            y_min = -np.linalg.norm(P - p_A_low)
            y_max = np.linalg.norm(P - p_A_up)

        safe_corridors.append([slope, min(y_min, y_max), max(y_min, y_max)])

    return safe_corridors


def plot_trajectory(ax, waypoints, nodes, optimization, flagb, safe_corridors=None):
    """
    Plot trajectory using optimization parameters.
    This matches the algorithm from Planning_error_withinSC.
    All coordinates in METERS.
    """
    N = len(waypoints)
    phi_opt = np.array(optimization['Optimization_phi'])
    l_opt = np.array(optimization['Optimization_l']) * PIXEL_TO_METER  # Convert to meters
    r_opt = np.array(optimization['Optimization_r']) * PIXEL_TO_METER  # Convert to meters

    # Calculate Distance and Angle between waypoints
    Distance = np.zeros(N-1)
    Angle = np.zeros(N-1)

    for i in range(N-1):
        pos_i = np.array(nodes[waypoints[i]])
        pos_i1 = np.array(nodes[waypoints[i+1]])
        Distance[i] = np.linalg.norm(pos_i1 - pos_i)
        Angle[i] = np.arctan2(pos_i1[1] - pos_i[1], pos_i1[0] - pos_i[0])

    # Plot waypoints and trajectories
    for i in range(N-1):
        Ps = nodes[waypoints[i]]

        # Plot waypoint
        if flagb[i] != 0:
            ax.plot(Ps[0], Ps[1], 'ro', markersize=4)
        else:
            ax.plot(Ps[0], Ps[1], 'go', markersize=4)

        # Calculate trajectory parameters
        phi_new_opt = phi_opt[i] + flagb[i] * np.pi / 2

        # Get optimization results directly
        r0_opt = r_opt[i] if i < len(r_opt) else 0
        l_seg = l_opt[i] if i < len(l_opt) else 0

        # Plot arc
        theta_start = phi_new_opt + np.pi / 2
        theta_end = phi_opt[i+1] + np.pi / 2

        center_x = r0_opt * np.cos(theta_start)
        center_y = r0_opt * np.sin(theta_start)

        # Generate arc points
        theta = np.linspace(theta_start, theta_end, 100)
        x_arc = r0_opt * np.cos(theta) - center_x + Ps[0]
        y_arc = r0_opt * np.sin(theta) - center_y + Ps[1]
        ax.plot(x_arc, y_arc, 'b-', linewidth=0.8)

        # Plot straight line segment
        x_end_arc = r0_opt * np.cos(theta_end) - center_x
        y_end_arc = r0_opt * np.sin(theta_end) - center_y

        x_line = [x_end_arc + Ps[0], x_end_arc + l_seg * np.cos(phi_opt[i+1]) + Ps[0]]
        y_line = [y_end_arc + Ps[1], y_end_arc + l_seg * np.sin(phi_opt[i+1]) + Ps[1]]
        ax.plot(x_line, y_line, 'b-', linewidth=0.8)

    # Plot final waypoint
    final_pos = nodes[waypoints[-1]]
    ax.plot(final_pos[0], final_pos[1], 'go', markersize=4)

    # Plot safe corridors if available
    if safe_corridors is not None:
        for i in range(N-1):
            start_pos = np.array(nodes[waypoints[i]])
            end_pos = np.array(nodes[waypoints[i+1]])

            slope = safe_corridors[i][0]
            y_min = safe_corridors[i][1]
            y_max = safe_corridors[i][2]

            # Vertical line case (high slope)
            if abs(slope) > 100000:
                x_coords = [start_pos[0] + y_min, start_pos[0] + y_max,
                           end_pos[0] + y_max, end_pos[0] + y_min, start_pos[0] + y_min]
                y_coords = [start_pos[1], start_pos[1], end_pos[1], end_pos[1], start_pos[1]]
            else:
                # Non-vertical line - calculate perpendicular corridor
                length = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                dx = (end_pos[0] - start_pos[0]) / length
                dy = (end_pos[1] - start_pos[1]) / length

                # Perpendicular direction
                perp_dx = -dy
                perp_dy = dx

                # Four corners of corridor
                p1 = [start_pos[0] + y_min * perp_dx, start_pos[1] + y_min * perp_dy]
                p2 = [start_pos[0] + y_max * perp_dx, start_pos[1] + y_max * perp_dy]
                p3 = [end_pos[0] + y_max * perp_dx, end_pos[1] + y_max * perp_dy]
                p4 = [end_pos[0] + y_min * perp_dx, end_pos[1] + y_min * perp_dy]

                x_coords = [p1[0], p2[0], p3[0], p4[0], p1[0]]
                y_coords = [p1[1], p2[1], p3[1], p4[1], p1[1]]

            ax.plot(x_coords, y_coords, 'g--', alpha=0.5, linewidth=1,
                   label='Safe Corridor' if i == 0 else "")


def create_visualization(waypoint_file, optimization_file, graph_file, environment_file, output_file):
    """Create planning result visualization"""
    # Load all data
    waypoint_data = load_json(waypoint_file)
    optimization = load_json(optimization_file)
    nodes = load_graph(graph_file)
    environment = load_json(environment_file)

    waypoints = waypoint_data['Waypoints']
    flagb = waypoint_data['FlagB']

    # Compute safe corridors
    safe_corridors = compute_safe_corridors(waypoints, nodes, environment)

    # Create plot - standard single-column figure size (5cm x 4cm)
    fig, ax = plt.subplots(figsize=(5/2.54, 4/2.54))

    # Draw environment
    plot_environment(ax, environment)

    # Draw trajectory with safe corridors
    plot_trajectory(ax, waypoints, nodes, optimization, flagb, safe_corridors)

    # Finalize - match style from separate_subfigures_visualization
    ax.set_title('Planning Result', fontsize=10, pad=4)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linewidth=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2, pad=1)

    # Add legend below the plot area
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                 ncol=2, frameon=True)

    print(f"Saving figure to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Process all MAP cases and warehouse case"""
    base_path = r"d:\Data_visualization_code\result\MAPS"
    output_base = os.path.join(base_path, "separate_plots")
    os.makedirs(output_base, exist_ok=True)

    # Find all case directories (MAP1-MAP6 and warehouse)
    import glob
    map_dirs = glob.glob(os.path.join(base_path, "MAP*"))
    map_dirs = [d for d in map_dirs if os.path.isdir(d) and not d.endswith('_1')]

    # Add warehouse directory
    warehouse_dir = os.path.join(base_path, "warehouse")
    if os.path.isdir(warehouse_dir):
        map_dirs.append(warehouse_dir)

    print(f"Found {len(map_dirs)} MAP directories")
    print(f"Output directory: {output_base}")

    for map_dir in map_dirs:
        map_name = os.path.basename(map_dir)
        print(f"\n{'='*60}")
        print(f"Processing {map_name}")
        print(f"{'='*60}")

        # Find case-specific files
        case_name = map_name.lower()

        # Try to find files with different patterns
        waypoint_files = glob.glob(os.path.join(map_dir, f"WayPointFlag*{case_name}.json"))
        optimization_files = glob.glob(os.path.join(map_dir, f"Optimization_withSC_path*{case_name}.json"))
        graph_files = glob.glob(os.path.join(map_dir, f"Graph_new_{case_name}.json"))
        if not graph_files:
            graph_files = glob.glob(os.path.join(map_dir, f"Graph_{case_name}.json"))
        environment_files = glob.glob(os.path.join(map_dir, f"environment_{case_name}.json"))

        # Process each combination found
        if waypoint_files and optimization_files and graph_files and environment_files:
            for waypoint_file in waypoint_files:
                for optimization_file in optimization_files:
                    # Extract path number from filenames
                    import re
                    wp_match = re.search(r'WayPointFlag(\d+)', waypoint_file)
                    opt_match = re.search(r'path(\d+)', optimization_file)

                    if wp_match and opt_match and wp_match.group(1) == opt_match.group(1):
                        path_num = wp_match.group(1)

                        # For warehouse, only process path4
                        if map_name.lower() == 'warehouse' and path_num != '4':
                            continue

                        try:
                            output_file = os.path.join(output_base, f"{map_name}_path{path_num}_visualization.png")

                            print(f"  Creating visualization for path {path_num}...")
                            create_visualization(
                                waypoint_file,
                                optimization_file,
                                graph_files[0],
                                environment_files[0],
                                output_file
                            )

                            print(f"  [OK] Created: {output_file}")

                        except Exception as e:
                            print(f"  [FAILED] Error processing path {path_num}: {e}")
                            import traceback
                            traceback.print_exc()
        else:
            print(f"  [SKIP] Missing required files:")
            print(f"    Waypoints: {len(waypoint_files)} found")
            print(f"    Optimization: {len(optimization_files)} found")
            print(f"    Graph: {len(graph_files)} found")
            print(f"    Environment: {len(environment_files)} found")

    print(f"\n{'='*60}")
    print("Visualization complete")


if __name__ == "__main__":
    main()
