"""
Check path length and minimum curvature radius for A*+Bezier and RRT*+Dubins results.
Saves a combined JSON report.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ASTAR_RESULTS_DIR = Path(r"d:\Data_visualization_code\result\Astar_Results")
DUBINS_RESULTS_DIR = Path(r"d:\Data_visualization_code\result\RRT_Dubins_Results")
OUTPUT_JSON = Path(r"d:\Data_visualization_code\result\path_metrics.json")


def compute_path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def compute_curvature(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(len(points))

    curvatures = np.zeros(len(points))
    for i in range(1, len(points) - 1):
        p0 = points[i - 1]
        p1 = points[i]
        p2 = points[i + 1]

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
            curvatures[i] = 2.0 * math.sin(angle / 2.0) / chord

    if len(points) >= 3:
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]

    return curvatures


def min_radius_from_curvature(curvatures: np.ndarray) -> float:
    max_kappa = float(np.max(np.abs(curvatures))) if len(curvatures) > 0 else 0.0
    if max_kappa < 1e-9:
        return float("inf")
    return 1.0 / max_kappa


def load_astar_result(path: Path) -> Tuple[np.ndarray, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    smooth_path = np.array(data.get("smooth_path", []), dtype=float)
    if smooth_path.ndim != 2 or smooth_path.shape[1] < 2:
        smooth_path = np.zeros((0, 2))
    return smooth_path[:, :2], data


def load_dubins_result(path: Path) -> Tuple[np.ndarray, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    full_path = np.array(data.get("full_path", []), dtype=float)
    if full_path.ndim == 2 and full_path.shape[1] >= 2:
        return full_path[:, :2], data
    return np.zeros((0, 2)), data


def collect_results() -> Dict:
    results = {
        "astar": {},
        "rrt_dubins": {},
    }

    for json_path in ASTAR_RESULTS_DIR.glob("**/astar_path_*.json"):
        try:
            points, data = load_astar_result(json_path)
            length = compute_path_length(points)
            curvatures = compute_curvature(points)
            min_radius = min_radius_from_curvature(curvatures)

            map_name = data.get("map_name", json_path.stem.replace("astar_path_", ""))
            results["astar"][map_name] = {
                "path_length_m": length,
                "min_turn_radius_m": min_radius,
                "num_points": int(len(points)),
                "source": str(json_path),
            }
        except Exception as exc:
            results["astar"][json_path.stem] = {
                "error": str(exc),
                "source": str(json_path),
            }

    for json_path in DUBINS_RESULTS_DIR.glob("**/dubins_trajectory_*.json"):
        try:
            points, data = load_dubins_result(json_path)
            length = compute_path_length(points)
            curvatures = compute_curvature(points)
            min_radius = min_radius_from_curvature(curvatures)

            map_name = data.get("map_name", json_path.stem.replace("dubins_trajectory_", ""))
            results["rrt_dubins"][map_name] = {
                "path_length_m": length,
                "min_turn_radius_m": min_radius,
                "num_points": int(len(points)),
                "source": str(json_path),
            }
        except Exception as exc:
            results["rrt_dubins"][json_path.stem] = {
                "error": str(exc),
                "source": str(json_path),
            }

    return results


def main() -> None:
    results = collect_results()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
