#!/usr/bin/env python3
"""
Path planning with clean error handling.
No defensive programming bullshit - fail fast and clear.
"""
import json
import os
import sys
import numpy as np
from typing import Dict, Any, Tuple, List

# Import dependencies
import gurobipy as gp
from gurobipy import GRB
from GenerateMatrix import load_reeb_graph_from_file
from Planing_functions import (Initial_Guess, Planning_normalization,
                              get_normalization_prams, get_safe_corridor,
                              Planning_error_withinSC)

sys.path.append('/root/workspace/config')
from config_loader import config
sys.path.append('/root/workspace/src/Replanning/scripts')
from coordinate_transform import get_frame_info


class PathPlanningError(Exception):
    """Single exception type for all path planning failures"""
    pass


class ConfigData:
    """Single source of truth for all configuration"""
    def __init__(self):
        self.case = config.case
        self.N = config.N
        self.arc_range = config.arc_range
        self.phi0 = self._load_phi0()
        self.pixel_to_meter = config.pixel_to_meter_scale

    def _load_phi0(self) -> float:
        """Load phi0 from GA result or use config default"""
        ga_file = config.get_full_path(config.Result_file, use_data_path=True)

        if not os.path.exists(ga_file):
            return config.phi0

        with open(ga_file, 'r') as f:
            data = json.load(f)

        # Try different keys in order of preference
        for key in ['Initial_guess_phi', 'optimized_phi0']:
            if key in data:
                value = data[key]
                return value[0] if isinstance(value, list) else value

        return config.phi0

    def get_file_paths(self) -> Dict[str, str]:
        """Get all required file paths"""
        return {
            'graph': config.get_full_path(config.file_path, use_data_path=True),
            'environment': config.get_full_path(config.environment_file, use_data_path=True),
            'assignment': config.get_full_path(config.assignment_result_file, use_data_path=True),
            'waypoints': config.get_full_path(config.waypoints_file_path, use_data_path=True),
            'normalization': config.get_full_path(config.Normalization_planning_path, use_data_path=True),
            'ga_result': config.get_full_path(config.Result_file, use_data_path=True),
            'matrices': config.get_full_path(f"Estimated_matrices_{self.case}.npz", use_data_path=True),
            'result': config.get_full_path(f"Optimization_withSC_path{self.N}{self.case}.json", use_data_path=True),
            'figure': config.get_full_path(f"Optimization_winthSC_path{self.N}{self.case}.png", use_data_path=True)
        }


def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file or die"""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_waypoints_data(file_path: str) -> Tuple[List, List, List]:
    """Load waypoints data or die"""
    data = load_json(file_path)
    return data['Waypoints'], data['RelayPoints'], data['FlagB']


def load_matrices(file_path: str) -> Tuple[np.ndarray, ...]:
    """Load matrices or die"""
    data = np.load(file_path)
    return data['Ec'], data['El'], data['Ad'], data['Cr'], data['Cl']


def process_waypoints(waypoints: List, relay_points: List, flag_b: List,
                     matrices: Tuple[np.ndarray, ...]) -> Dict[str, Any]:
    """Process waypoints into the format we need"""
    Ec, El, Ad, Cr, Cl = matrices

    # Build waypoint sequence
    original_waypoints = waypoints.copy()
    waypoint_sequence = [relay_points[0][0]]
    flag_sequence = [0]
    relay_point_list = [arc[0] for arc in relay_points]

    # Simple algorithm - no special cases
    while original_waypoints:
        current_end = waypoint_sequence[-1]
        for arc in original_waypoints:
            if arc[0] == current_end:
                waypoint_sequence.append(arc[1])
                original_waypoints.remove(arc)
                flag_sequence.append(1 if arc[1] in relay_point_list else 0)
                break

    # Calculate turn directions
    flag_directions = flag_sequence.copy()
    relay_idx = 0

    for i, is_relay in enumerate(flag_sequence):
        if is_relay == 1:
            flag_directions[i] = -1 if flag_b[relay_idx] == 1 else 1
            relay_idx += 1

    return {
        'Waypoints': waypoint_sequence,
        'Flags': flag_sequence,
        'FlagB': flag_directions
    }


def save_json(data: Dict[str, Any], file_path: str):
    """Save JSON file or die"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def validate_path_parameters(reeb_graph, config_data: ConfigData):
    """Validate path makes physical sense"""
    num_nodes = len(reeb_graph.nodes)
    start = reeb_graph.nodes[num_nodes-2].configuration
    end = reeb_graph.nodes[num_nodes-1].configuration

    distance_pixels = np.linalg.norm(end - start)
    distance_meters = distance_pixels * config_data.pixel_to_meter

    # Reasonable bounds check
    if distance_pixels < 10 or distance_pixels > 2000:
        raise PathPlanningError(f"Unreasonable pixel distance: {distance_pixels}")

    if distance_meters > 5.0:
        raise PathPlanningError(f"Unreasonable physical distance: {distance_meters}m")

    return start, end, distance_pixels


def run_optimization(paths: Dict[str, str], config_data: ConfigData,
                    reeb_graph, safe_corridor):
    """Run the optimization - fail fast if anything goes wrong"""

    # Use GA result if available, otherwise fail
    if not os.path.exists(paths['ga_result']):
        raise PathPlanningError(f"GA result required but not found: {paths['ga_result']}")

    Planning_error_withinSC(
        paths['waypoints'],
        paths['normalization'],
        paths['environment'],
        safe_corridor,
        reeb_graph,
        config_data.phi0,
        paths['ga_result'],
        Result_file=paths['result'],
        figure_file=paths['figure']
    )

    # Add coordinate frame info to result
    frame_info = get_frame_info()
    result_data = load_json(paths['result'])
    result_data.update({
        'coordinate_frames': frame_info,
        'coordinate_frame': 'world_pixel',
        'data_coordinate_frame': 'world_pixel'
    })
    save_json(result_data, paths['result'])


def main():
    """Main function - no try-catch bullshit, just fail cleanly"""

    # Load configuration
    config_data = ConfigData()
    paths = config_data.get_file_paths()

    print(f"Path planning for case {config_data.case} with {config_data.N} robots")
    print(f"Initial angle: {config_data.phi0:.4f} ({config_data.phi0/np.pi:.2f}π)")

    # Load and validate graph
    reeb_graph = load_reeb_graph_from_file(paths['graph'])
    start, end, distance = validate_path_parameters(reeb_graph, config_data)

    print(f"Path: {start} → {end} (distance: {distance:.1f} pixels)")

    # Check assignment result exists
    if not os.path.exists(paths['assignment']):
        raise PathPlanningError(f"Assignment result required: {paths['assignment']}")

    # Process waypoints
    waypoints, relay_points, flag_b = load_waypoints_data(paths['assignment'])
    matrices = load_matrices(paths['matrices'])
    processed_waypoints = process_waypoints(waypoints, relay_points, flag_b, matrices)
    save_json(processed_waypoints, paths['waypoints'])

    # Get safe corridor
    safe_corridor, _, _, _ = get_safe_corridor(reeb_graph, paths['waypoints'], paths['environment'])

    # Run optimization
    run_optimization(paths, config_data, reeb_graph, safe_corridor)

    print(f"✓ Optimization complete: {paths['result']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Path planning failed: {e}")
        sys.exit(1)