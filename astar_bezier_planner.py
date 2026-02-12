"""
A* + Bezier path planner (single feasible path per map).
"""

import json
import math
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from scipy.interpolate import splprep, splev
import yaml


class Config:
    """Configuration parameters."""

    # Coordinate conversion
    PIXEL_TO_METER = 0.0023

    # Grid and safety
    GRID_RESOLUTION = 0.02  # m
    OBSTACLE_EXPANSION = 0.03  # m

    # Bezier smoothing
    PATH_RESOLUTION = 0.003  # m

    @classmethod
    def load_from_yaml(cls, config_path: str) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                cls.PIXEL_TO_METER = config.get("pixel_to_meter_scale", cls.PIXEL_TO_METER)
        except Exception as exc:
            print(f"[WARN] Config load failed, using defaults: {exc}")


def load_environment(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    map_name = Path(json_path).parent.name

    obstacles = []
    for poly_data in data.get("polygons", []):
        vertices = np.array(poly_data["vertices"]) * Config.PIXEL_TO_METER
        obstacles.append(Polygon(vertices))

    if obstacles:
        obstacles_union = unary_union(obstacles)
        obstacles_union_expanded = obstacles_union.buffer(Config.OBSTACLE_EXPANSION)
    else:
        obstacles_union = None
        obstacles_union_expanded = None

    bounds_pixel = data.get("coord_bounds", [0, data["width"], 0, data["height"]])
    bounds = [b * Config.PIXEL_TO_METER for b in bounds_pixel]

    start_pose = data.get("start_pose", [100, 100, 0])
    goal_pose = data.get("goal_pose", [500, 500, 0])

    start = [
        start_pose[0] * Config.PIXEL_TO_METER,
        start_pose[1] * Config.PIXEL_TO_METER,
        start_pose[2] if len(start_pose) > 2 else 0.0,
    ]

    goal = [
        goal_pose[0] * Config.PIXEL_TO_METER,
        goal_pose[1] * Config.PIXEL_TO_METER,
        goal_pose[2] if len(goal_pose) > 2 else 0.0,
    ]

    return {
        "obstacles": obstacles,
        "obstacles_union": obstacles_union,
        "obstacles_union_expanded": obstacles_union_expanded,
        "bounds": bounds,
        "start": start,
        "goal": goal,
        "map_name": map_name,
    }


def check_collision(point: np.ndarray, obstacles_union, bounds: List[float]) -> bool:
    x, y = point[0], point[1]

    if bounds is not None:
        margin = Config.OBSTACLE_EXPANSION
        if x < bounds[0] + margin or x > bounds[1] - margin or y < bounds[2] + margin or y > bounds[3] - margin:
            return True

    if obstacles_union is not None:
        return obstacles_union.contains(Point(point))

    return False


def check_line_collision(p1: np.ndarray, p2: np.ndarray, obstacles_union, bounds: List[float]) -> bool:
    if bounds is not None:
        if check_collision(p1, None, bounds) or check_collision(p2, None, bounds):
            return True

    if obstacles_union is not None:
        line = LineString([p1, p2])
        return line.intersects(obstacles_union)

    return False


def build_occupancy_grid(bounds: List[float], obstacles_union, resolution: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = bounds
    xs = np.arange(x_min, x_max + resolution, resolution)
    ys = np.arange(y_min, y_max + resolution, resolution)

    grid = np.zeros((len(xs), len(ys)), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if obstacles_union is not None and obstacles_union.contains(Point(x, y)):
                grid[i, j] = True

    return grid, xs, ys


def coord_to_index(x: float, y: float, xs: np.ndarray, ys: np.ndarray) -> Tuple[int, int]:
    ix = int(np.clip(round((x - xs[0]) / (xs[1] - xs[0])), 0, len(xs) - 1))
    iy = int(np.clip(round((y - ys[0]) / (ys[1] - ys[0])), 0, len(ys) - 1))
    return ix, iy


def index_to_coord(ix: int, iy: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.array([xs[ix], ys[iy]])


def astar_plan(start: np.ndarray, goal: np.ndarray, bounds: List[float], obstacles_union, resolution: float) -> Optional[List[np.ndarray]]:
    grid, xs, ys = build_occupancy_grid(bounds, obstacles_union, resolution)

    start_idx = coord_to_index(start[0], start[1], xs, ys)
    goal_idx = coord_to_index(goal[0], goal[1], xs, ys)

    if grid[start_idx] or grid[goal_idx]:
        return None

    moves = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start_idx, goal_idx), 0.0, start_idx))

    came_from = {}
    g_score = {start_idx: 0.0}

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [index_to_coord(ix, iy, xs, ys) for ix, iy in path]

        for dx, dy, cost in moves:
            nx, ny = current[0] + dx, current[1] + dy
            if nx < 0 or nx >= grid.shape[0] or ny < 0 or ny >= grid.shape[1]:
                continue
            if grid[nx, ny]:
                continue

            neighbor = (nx, ny)
            tentative_g = current_g + cost
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_idx)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return None


def simplify_path(path: List[np.ndarray], obstacles_union, bounds: List[float]) -> List[np.ndarray]:
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if not check_line_collision(path[i], path[j], obstacles_union, bounds):
                simplified.append(path[j])
                i = j
                break
        else:
            i += 1

    return simplified


def smooth_path_with_bezier(waypoints: List[np.ndarray], obstacles_union, bounds: List[float]) -> Tuple[np.ndarray, bool]:
    if len(waypoints) < 3:
        return np.array(waypoints), False

    waypoints = np.array(waypoints)
    try:
        tck, _ = splprep([waypoints[:, 0], waypoints[:, 1]], s=0, k=min(3, len(waypoints) - 1))
        length = 0.0
        for i in range(len(waypoints) - 1):
            length += np.linalg.norm(waypoints[i + 1] - waypoints[i])
        num_points = max(int(length / Config.PATH_RESOLUTION), len(waypoints) * 2)
        u_new = np.linspace(0, 1, num_points)
        smooth_x, smooth_y = splev(u_new, tck)
        smooth_path = np.column_stack([smooth_x, smooth_y])

        if obstacles_union is not None:
            dense_count = max(int(len(smooth_path) * 0.2), 1)
            for idx, point in enumerate(smooth_path):
                if idx < dense_count or idx % 10 == 0:
                    if check_collision(point, obstacles_union, bounds):
                        return smooth_path, True

        return smooth_path, False
    except Exception as exc:
        print(f"[WARN] Bezier smoothing failed: {exc}")
        return np.array(waypoints), False


def plot_result(env: Dict, raw_path: List[np.ndarray], smooth_path: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for obs in env["obstacles"]:
        if obs.geom_type == "Polygon":
            x, y = obs.exterior.xy
            ax.fill(x, y, color="gray", alpha=0.5, edgecolor="black", linewidth=1)
        elif obs.geom_type == "MultiPolygon":
            for poly in obs.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color="gray", alpha=0.5, edgecolor="black", linewidth=1)

    if raw_path:
        raw_arr = np.array(raw_path)
        ax.plot(raw_arr[:, 0], raw_arr[:, 1], "o--", color="orange", linewidth=2, markersize=4, label="A* Path")

    if smooth_path is not None and len(smooth_path) > 0:
        min_radius = min_radius_from_path(smooth_path)
        line_width = 3.5 if min_radius < 0.2 else 2.5
        ax.plot(smooth_path[:, 0], smooth_path[:, 1], "-r", linewidth=line_width, label="Bezier Path")

    start = env["start"]
    goal = env["goal"]
    ax.plot(start[0], start[1], "go", markersize=12, label="Start")
    ax.plot(goal[0], goal[1], "r*", markersize=14, label="Goal")

    bounds = env["bounds"]
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"A* + Bezier Path - {env['map_name']}")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def min_radius_from_path(path: np.ndarray) -> float:
    if len(path) < 3:
        return float("inf")

    max_kappa = 0.0
    for i in range(1, len(path) - 1):
        p0 = path[i - 1]
        p1 = path[i]
        p2 = path[i + 1]

        v1 = p1 - p0
        v2 = p2 - p1

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        if len_v1 < 1e-9 or len_v2 < 1e-9:
            continue

        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.acos(cos_angle)
        chord = np.linalg.norm(p2 - p0)
        if chord > 1e-9:
            kappa = 2.0 * math.sin(angle / 2.0) / chord
            if abs(kappa) > max_kappa:
                max_kappa = abs(kappa)

    if max_kappa < 1e-9:
        return float("inf")
    return 1.0 / max_kappa


def process_single_map(env_path: str, output_dir: str, verbose: bool = True) -> bool:
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {env_path}")
        print(f"{'='*60}")

    try:
        env = load_environment(env_path)
        bounds = env["bounds"]

        if verbose:
            print(f"[OK] Map: {env['map_name']}")
            print(f"  Start: ({env['start'][0]:.3f}, {env['start'][1]:.3f})")
            print(f"  Goal: ({env['goal'][0]:.3f}, {env['goal'][1]:.3f})")

        raw_path = astar_plan(
            start=np.array(env["start"][:2]),
            goal=np.array(env["goal"][:2]),
            bounds=bounds,
            obstacles_union=env["obstacles_union_expanded"],
            resolution=Config.GRID_RESOLUTION,
        )

        if raw_path is None:
            print("  [FAIL] A* failed to find a path")
            return False

        raw_path[0] = np.array(env["start"][:2])
        raw_path[-1] = np.array(env["goal"][:2])

        simplified = simplify_path(raw_path, env["obstacles_union_expanded"], bounds)
        smooth_path, has_collision = smooth_path_with_bezier(simplified, env["obstacles_union_expanded"], bounds)

        if verbose:
            print(f"  Raw points: {len(raw_path)}, simplified: {len(simplified)}, smooth: {len(smooth_path)}")
            if has_collision:
                print("  [WARN] Bezier path collides; using it anyway for visualization")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        map_name = env["map_name"]

        path_output = {
            "map_name": map_name,
            "coordinate_frame": "world_meter",
            "start_pose": env["start"],
            "goal_pose": env["goal"],
            "waypoints": [p.tolist() for p in simplified],
            "smooth_path": smooth_path.tolist(),
        }

        json_path = output_path / f"astar_path_{map_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(path_output, f, indent=2)

        if verbose:
            print(f"  [OK] Path saved: {json_path}")

        vis_path = output_path / f"astar_path_{map_name}.png"
        plot_result(env, simplified, smooth_path, str(vis_path))

        if verbose:
            print(f"  [OK] Plot saved: {vis_path}")

        return True

    except Exception as exc:
        print(f"\n[FAIL] Processing error: {exc}")
        return False


def batch_process_maps(maps_dir: str, output_base_dir: str) -> None:
    maps_path = Path(maps_dir)
    env_files = sorted(maps_path.glob("*/environment_*.json"))

    if not env_files:
        print(f"No environment files in {maps_dir}")
        return

    print(f"\nFound {len(env_files)} maps")
    print(f"Output dir: {output_base_dir}\n")

    success_count = 0
    failed_maps = []

    for env_file in env_files:
        map_name = env_file.parent.name
        if map_name.startswith("warehouse_"):
            print(f"[SKIP] {map_name}")
            continue

        output_dir = Path(output_base_dir) / map_name
        success = process_single_map(str(env_file), str(output_dir), verbose=True)

        if success:
            success_count += 1
        else:
            failed_maps.append(map_name)

    print(f"\nDone: {success_count}/{len(env_files)} succeeded")
    if failed_maps:
        print(f"Failed: {failed_maps}")


def main() -> None:
    config_path = r"d:\Data_visualization_code\result\romi\config.yaml"
    if Path(config_path).exists():
        Config.load_from_yaml(config_path)

    maps_dir = r"d:\Data_visualization_code\result\MAPS"
    output_dir = r"d:\Data_visualization_code\result\Astar_Results"

    batch_process_maps(maps_dir, output_dir)


if __name__ == "__main__":
    main()
