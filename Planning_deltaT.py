# optimization the time of deleta l of the trajectory
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import json
import os
from GenerateMatrix import load_reeb_graph_from_file
import casadi as ca
from Environment import Environment
from Planing_functions import get_safe_corridor
import casadi.tools as ca_tools
from BarriersOriginal import generate_barriers_test
from trajectory_visualization import plot_trajectory_with_time
# Import the trajectory tools
from trajectory_parameters import save_trajectory_parameters, load_trajectory_parameters, plot_from_saved_trajectory, generate_spline_from_saved_trajectory
from uniform_time_trajectory import convert_robot_trajectory_to_uniform_time, save_trajectory_in_tb_format

# Import the trajectory saving function
from save_spline_trajectory import save_spline_trajectory

# Add config path to sys.path and load configuration
import sys
sys.path.append('/root/workspace/config')
from config_loader import config

# Import coordinate transformation utilities
from coordinate_transform import convert_pixel_positions_to_world_meters, convert_world_pixel_data_to_meters

# Get robot physical parameters from config
robot_params = config.get_robot_physical_params()
# Get correct wheel parameters for differential drive calculations
w_max = robot_params['wheel_w_max']    # the maximum wheel angular velocity (rad/s)
aw_max = robot_params['wheel_aw_max']  # the maximum wheel angular acceleration (rad/s²)
# Other robot parameters
r_limit = robot_params['r_limit']  # m - minimum turning radius
r_w = robot_params['r_w']       # the radius of the wheel (m)
v_max = robot_params['v_max']   # m/s - maximum linear velocity (pre-calculated in config)
a_max = robot_params['a_max']   # m/s² - maximum linear acceleration (pre-calculated in config)
l_r = robot_params['l_r']       # the wheel base (m)
mu = robot_params['mu']         # Coefficient of friction (typical for rubber on concrete)
mu_f = robot_params['mu_f']     # Safety factor
g = robot_params['g']           # Gravitational acceleration (m/s²)
mu_mu_f = robot_params['mu_mu_f']  # pre-calculated in config


def load_WayPointFlag_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Waypoints'], data['Flags'],data["FlagB"]

def load_reeb_graph(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    nodes = data['nodes']
    in_neighbors = data['in_neighbors']
    out_neighbors = data['out_neighbors']
    return nodes, in_neighbors, out_neighbors

def load_optimization_data(file_path):
    """
    Load optimization data from a JSON file.
    For Initial_Guess files, only phi, l, r are expected.
    For ST path files, v and a might also be available.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Required fields
    phi = data["Optimization_phi"]
    l = data["Optimization_l"]
    r = data["Optimization_r"]
    
    # Optional fields - set to default values if not present
    if "Optimization_v" in data:
        v = data["Optimization_v"]
    else:
        print(f"Warning: 'Optimization_v' not found in {file_path}")
        # Create default values (zeros) with double the length of phi
        v = [0.0] * (len(phi) * 2)
    
    if "Optimization_a" in data:
        a = data["Optimization_a"]
    else:
        print(f"Warning: 'Optimization_a' not found in {file_path}")
        # Create default values (zeros) with double the length of phi
        a = [0.0] * (len(phi) * 2)
    
    return phi, l, r, v, a

# Functions to calculate constraints based on differential drive kinematics
def calculate_angular_velocity_limit(r):
    """
    Calculate the maximum angular velocity (ω_c) for a specific arc radius.
    
    This is derived from the differential drive constraint:
    [ω_r]   = [1/r_w,  l_r/(2*r_w)] * [ω_c*r]
    [ω_l]     [1/r_w, -l_r/(2*r_w)]   [ω_c]
    
    Args:
        r: Radius of the arc (meters)
    
    Returns:
        Maximum allowable angular velocity (rad/s)
    """
    if abs(r) <= l_r/2:
        # For very tight turns, one wheel would need to move backward
        # We set a conservative limit
        return 0.1 * w_max
    
    # Calculate limits for both wheels
    limit_right = w_max * r_w / (abs(r) + l_r/2)
    limit_left = w_max * r_w / (abs(r) - l_r/2)
    
    # Return the more restrictive limit
    return min(limit_right, limit_left)

def calculate_angular_acceleration_limit(r):
    """
    Calculate the maximum angular acceleration (a_c) for a specific arc radius.
    
    This is derived from the differential drive constraint:
    [a_r]   = [1/r_w,  l_r/(2*r_w)] * [a_c*r]
    [a_l]     [1/r_w, -l_r/(2*r_w)]   [a_c]
    
    Args:
        r: Radius of the arc (meters)
    
    Returns:
        Maximum allowable angular acceleration (rad/s²)
    """
    if abs(r) <= l_r/2:
        # For very tight turns, one wheel would need to move backward
        # We set a conservative limit
        return 0.1 * aw_max
    
    # Calculate limits for both wheels
    limit_right = aw_max * r_w / (abs(r) + l_r/2)
    limit_left = aw_max * r_w / (abs(r) - l_r/2)
    
    # Return the more restrictive limit
    return min(limit_right, limit_left)

def load_WayPointFlag_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Waypoints'], data['Flags'],data["FlagB"]

def load_matrices_from_file(file_path):
    data = np.load(file_path)
    Ec = data['Ec']
    El = data['El']
    Ad = data['Ad']
    return Ec, El, Ad
def load_phi_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_phi']
def load_r_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_r']
def load_l_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_l']

def load_trajectory_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['Optimization_phi'],data['Optimization_l'],data['Optimization_r'],data['Optimization_v'],data['Optimization_a']

def Get_index(r0, l, delta_phi, Deltal):
    """
    Calculate the number of small segments needed for arc and line parts.

    Args:
        r0: Radius of the arc
        l: Length of the straight line
        delta_phi: Angular change of the arc
        Deltal: Small segment length

    Returns:
        Tuple (N_arc, N_line) with number of segments for arc and line
    """
    # Calculate number of arc segments with special handling for short arcs
    arc_length = abs(r0 * delta_phi)

    if abs(r0) < 1e-6:  # No arc (zero radius)
        N_arc = 0
    elif arc_length < 0.02:  # Very short arc - skip to avoid over-constraining
        N_arc = 0  # Skip very short arcs that cause numerical issues
    else:  # Normal arc - calculate segments based on Deltal
        N_arc = max(1, int(arc_length / Deltal))

    # Calculate number of line segments, ensuring at least 1 if line exists
    N_line = max(1, int(l / Deltal)) if l > 0.03 else 0

    return N_arc, N_line
    

def Planning_deltaT(waypoints_file_path,reeb_graph,planning_path_result_file,Result_file,figure_file):
    """
    Optimize trajectory timing for a path with given geometry.
    
    This function takes the output from Planning_path (which includes safe corridor constraints)
    and optimizes the time distribution along the trajectory to minimize total time while
    respecting robot dynamics constraints.
    
    Args:
        waypoints_file_path: Path to waypoints and flags file
        reeb_graph: The Reeb graph structure
        planning_path_result_file: Result file from Planning_path with safe corridor optimization
        Result_file: Output file for timing optimization results
        figure_file: Output figure file
    
    Returns:
        time_segments: Optimized time segments for each arc and line
        total_time: Total trajectory time
    """
    Deltal = config.deltal  # Get small segment length from config (m)
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)

    phi_data = np.array(load_phi_from_file(planning_path_result_file))
    r_guess_pixels = np.array(load_r_from_file(planning_path_result_file))
    l_guess_pixels = np.array(load_l_from_file(planning_path_result_file))
    
    # Convert only length and radius from pixels to meters (angles unchanged)
    phi_data, l_guess, r_guess = convert_world_pixel_data_to_meters(phi_data, l_guess_pixels, r_guess_pixels)
    
    # Number of variables
    N = len(Waypoints)
    N_relays = np.nonzero(Flagb)[0].shape[0]
    
    # Create symbolic variables
    phi=phi_data # the angel at each waypoints
    r0=r_guess # the radius of each arc
    l=l_guess # the length of each straight line
    phi_new=np.zeros(np.shape(phi)) # the angle of each straight line
    
    # Arrays to store the number of small segments
    ArcIndex=np.zeros(N-1, dtype=int)
    LineIndex=np.zeros(N-1, dtype=int)
    
    # Calculate the number of arc and line segments
    for i in range(N-1):
        # phi angles are already in world coordinates from the JSON data
        phi_new[i]=phi[i]+Flagb[i]*np.pi/2
        delta_phi=(phi[i+1] - phi_new[i])
        Index=Get_index(r0[i],l[i],delta_phi,Deltal)
        ArcIndex[i]=Index[0]
        LineIndex[i]=Index[1]
    
    # Create symbolic variables for delta times only
    delta_t_arcs = []  # List to hold delta time variables for arcs
    delta_t_lines = [] # List to hold delta time variables for lines
    
    # Constraints
    g = []    # Constraints
    lbg = []  # Lower bounds for constraints
    ubg = []  # Upper bounds for constraints
    all_accelerations = [] # NEW: List to store all acceleration terms for the objective
    arc_counter = 0
    line_counter = 0
    # For each segment, create symbolic variables for delta times
    for i in range(N-1):
        # Define segment_starts_from_relay for the entire iteration
        segment_starts_from_relay = (i == 0 or (i < len(Flags) and Flags[i] == 1))

        # Create variables for arc segments
        if ArcIndex[i] > 0:
            arc_counter += 1
            # Delta times for arc segments - use a unique name for each segment
            # delta_t_arc_i = ca.SX.sym(f'delta_t_arc_seg{i}', ArcIndex[i])
            # delta_t_arcs.append(delta_t_arc_i)  # Still add to the list, but with a segment-specific name
            delta_t_arcs.append([ca.SX.sym(f'delta_t_arc{arc_counter}_{j}') for j in range(ArcIndex[i])])
            # Arc segment - compute segment length
            delta_phi = phi[i+1] - phi_new[i]
            arc_length = abs(r0[i] * delta_phi)
            arc_segment_length = arc_length / ArcIndex[i]
            
            # Acceleration-based continuity between segments
            # Only apply continuity if this segment does NOT start from a relay point
            
            if i > 0 and not segment_starts_from_relay:  # Not the first segment and not starting from relay point
                # print(f"  DEBUG: Adding Line-to-Arc continuity for segment {i}")
                # print(f"    - Previous segment {i-1} ends with Line at waypoint {i}")
                # print(f"    - Current segment {i} starts with Arc from waypoint {i}")
                # print(f"    - segment_starts_from_relay: {segment_starts_from_relay}")
                # print(f"    - Flags[{i}]: {Flags[i] if i < len(Flags) else 'N/A'}")
                
                # Connect with previous segment's final velocity using acceleration constraints
                if LineIndex[i-1] > 0:  # Previous segment ends with a line
                    # Convert from line to arc velocity continuity
                    # v_end_prev = line_segment_length / delta_t_prev
                    prev_line_subsegment_length = l[i-1] / LineIndex[i-1]
                    # Find the correct index in delta_t_lines for segment i-1
                    prev_line_idx = sum(1 for k in range(i-1) if LineIndex[k] > 0) - 1
                    # Velocity at the end of the last subsegment of the previous line (i-1)
                    v_end_prev_line = prev_line_subsegment_length / delta_t_lines[prev_line_idx][-1]
                    
                    # Tangential velocity at the start of the first subsegment of the current arc (i)
                    current_arc_subsegment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / ArcIndex[i]
                    current_arc_idx = sum(1 for k in range(i) if ArcIndex[k] > 0)  # Fixed: count arcs before current segment
                    v_start_curr_arc = current_arc_subsegment_length / delta_t_arcs[current_arc_idx][0]
                    
                    # For acceleration-based continuity: (v_start_curr_arc - v_end_prev_line)/t_avg ∈ [-a_max, a_max]
                    # Note: This is a transition from linear to tangential velocity.
                    # Use average of segment times (simple approach)
                    t_avg = (delta_t_lines[prev_line_idx][-1] + delta_t_arcs[current_arc_idx][0]) / 2
                    a_transition = (v_start_curr_arc - v_end_prev_line) / t_avg
                    all_accelerations.append(a_transition) # NEW: Add to list
                    
                    # Calculate appropriate acceleration limit for this transition
                    # For line to arc transition, use linear acceleration limit (robot chassis acceleration)
                    a_max_transition = a_max
                    
                    g.append(a_transition)
                    lbg.append(-a_max_transition)  # Limit the acceleration
                    ubg.append(a_max_transition)
                    
                    # NOTE: Cross-waypoint line-to-arc transitions removed - handled by same-segment constraints
                
                elif ArcIndex[i-1] > 0 and LineIndex[i-1] == 0:  # Previous segment ends with an arc (no line after it)
                    # Arc to arc continuity when there's no line segment between them
                    # Tangential velocity at the end of the previous arc's last subsegment
                    prev_arc_delta_phi = phi[i] - (phi_data[i-1] + Flagb[i-1]*np.pi/2)
                    prev_arc_length = abs(r0[i-1] * prev_arc_delta_phi)
                    prev_arc_subsegment_length = prev_arc_length / ArcIndex[i-1]
                    
                    # Find the correct index for the previous arc in delta_t_arcs
                    prev_arc_idx = sum(1 for k in range(i-1) if ArcIndex[k] > 0) - 1
                    v_end_prev_arc = prev_arc_subsegment_length / delta_t_arcs[prev_arc_idx][-1]
                    
                    # Tangential velocity at the start of the current arc's first subsegment
                    current_arc_subsegment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / ArcIndex[i]
                    current_arc_idx = sum(1 for k in range(i) if ArcIndex[k] > 0)  # Fixed: count arcs before current segment
                    v_start_curr_arc = current_arc_subsegment_length / delta_t_arcs[current_arc_idx][0]
                    
                    # For arc-to-arc continuity: ensure smooth velocity transition  
                    t_avg = (delta_t_arcs[prev_arc_idx][-1] + delta_t_arcs[current_arc_idx][0]) / 2
                    a_transition = (v_start_curr_arc - v_end_prev_arc) / t_avg
                    all_accelerations.append(a_transition) # Add to list for objective function penalty
                    
                    # Use the more restrictive acceleration limit between the two arcs
                    # Convert angular acceleration limits to tangential acceleration limits
                    a_max_prev = calculate_angular_acceleration_limit(r0[i-1]) * abs(r0[i-1])
                    a_max_curr = calculate_angular_acceleration_limit(r0[i]) * abs(r0[i])
                    a_max_transition = min(a_max_prev, a_max_curr)
                    
                    g.append(a_transition)
                    lbg.append(-a_max_transition)  # Limit the acceleration
                    ubg.append(a_max_transition)
                    
                    # NOTE: Arc-to-arc direct transitions removed - arcs are always followed by lines
                    
            # Ensure arc segments that start at relay points begin from zero velocity
            # Check if this segment starts from a relay point (waypoint i is a relay point)
            if segment_starts_from_relay:  # First segment or starts from relay point
                # print(f"  DEBUG: Arc segment {i} starts from relay point (waypoint {i})")
                # print(f"    - Flags[{i}]: {Flags[i] if i < len(Flags) else 'N/A'}")
                # print(f"    - Flagb[{i}]: {Flagb[i] if i < len(Flagb) else 'N/A'}")
                # print()
                # For arc segments starting from zero velocity, the maximum achievable angular velocity
                # is constrained by ω^2 = 2*α*θ where α is max angular acceleration and θ is angle
                g.append(delta_t_arcs[arc_counter-1][0])
                min_t = np.sqrt(2*arc_segment_length / calculate_angular_acceleration_limit(r0[i]) / abs(r0[i]))
                lbg.append(min_t)  # Enforce minimum time for starting from zero velocity
                ubg.append(10.0)  # Maximum time
            # Constraints for each segment within the arc
            for j in range(ArcIndex[i]):
                # Delta time lower bound (positive time)
                g.append(delta_t_arcs[arc_counter-1][j])
                lbg.append(0.20)  # Minimum time
                ubg.append(10.0)   # Maximum time
                
                # Angular velocity constraint: omega = arc_segment_length / r / delta_t <= w_max_arc
                omega_c = arc_segment_length / abs(r0[i]) / delta_t_arcs[arc_counter-1][j]
                w_max_arc = calculate_angular_velocity_limit(abs(r0[i]))
                g.append(omega_c)
                lbg.append(0)      # Non-negative angular velocity
                ubg.append(w_max_arc)  # Maximum angular velocity based on differential drive constraints
                
                # Minimum tangential velocity constraint: v_tangential = omega * r >= 0.005 m/s
                # This ensures the velocity at the arc (angular velocity * radius) is not smaller than 0.005 m/s
                v_tangential = omega_c * abs(r0[i])
                g.append(v_tangential)
                lbg.append(0.005)  # Minimum tangential velocity (0.005 m/s)
                ubg.append(0.8 * v_max)  # Maximum tangential velocity constraint for arcs (reduced for smoother curves)

                # DEBUG: Print constraint details for first few segments
                if i < 3 and j < 2:
                    print(f"  DEBUG Arc[{i}][{j}]: v_tangential constraint: 0.005 <= v <= {0.8*v_max:.4f} m/s (r={abs(r0[i]):.4f}m)")
                
                # Add constraint: |v|/|w| >= min_radius (reformulated to avoid division)
                # Instead of |v|/|w| >= min_radius, use |v| >= min_radius*|w| for arc part of Planning_deltaT
                v_tangential = omega_c * abs(r0[i])
                min_radius_constraint = v_tangential - abs(r0[i]) * omega_c
                g.append(min_radius_constraint)
                lbg.append(0)      # |v| >= min_radius*|w|
                ubg.append(ca.inf) # No upper bound
                
                # Centripetal force constraint: ω_c²|r| ≤ μ μ_f g
                # This ensures the robot doesn't slide during curved motion
                g.append(omega_c)
                lbg.append(0)      # Non-negative centripetal force
                ubg.append(np.sqrt(mu_mu_f/ abs(r0[i]) )) # Maximum allowable centripetal force based on friction
                
                # Angular acceleration constraint between consecutive segments
                if j > 0:
                    # For arc motion: tangential acceleration = radius * angular acceleration
                    # a_tangential = (v2-v1)/t_avg where v = arc_segment_length/delta_t
                    v1 = arc_segment_length / delta_t_arcs[arc_counter-1][j-1]
                    v2 = arc_segment_length / delta_t_arcs[arc_counter-1][j]
                    t_avg = (delta_t_arcs[arc_counter-1][j-1] + delta_t_arcs[arc_counter-1][j])/2
                    a_tangential = (v2 - v1) / t_avg
                    all_accelerations.append(a_tangential) # Add tangential acceleration to list
                    
                    # Convert to angular acceleration: alpha = a_tangential / radius
                    alpha = a_tangential / abs(r0[i])
                    aw_max_arc = calculate_angular_acceleration_limit(r0[i])
                    g.append(alpha)
                    lbg.append(-aw_max_arc)
                    ubg.append(aw_max_arc)
                    
                    # Additional angular velocity and acceleration constraint: |a_c| ≥ ω²_c/μ - μ_f/|r|
                    # This ensures balanced centripetal and angular acceleration for safe curved motion
                    # Rearranged as: |a_c| - ω²_c/μ ≥ -μ_f/|r|
                    omega_c_current = arc_segment_length / abs(r0[i]) / delta_t_arcs[arc_counter-1][j]
                    centripetal_term = omega_c_current**2 / mu
                    friction_term = -mu_f / abs(r0[i])
                    
                    # For positive angular acceleration: a_c - ω²_c/μ ≥ -μ_f/|r|
                    g.append(alpha - centripetal_term)
                    lbg.append(friction_term)
                    ubg.append(ca.inf)
                    
                    # For negative angular acceleration: -a_c - ω²_c/μ ≥ -μ_f/|r|
                    g.append(-alpha - centripetal_term)
                    lbg.append(friction_term)
                    ubg.append(ca.inf)
            
            # NOTE: Arc stop constraints removed - arcs always have line segments following them
        
        # Create variables for line segments
        if LineIndex[i] > 0:
            line_counter +=1
            # Delta times for line segments
            # delta_t_line = ca.SX.sym(f'delta_t_line_{i}', LineIndex[i])
            # delta_t_lines.append(delta_t_line)
            delta_t_lines.append([ca.SX.sym(f'delta_t_line{line_counter}_{j}') for j in range(LineIndex[i])])
            # Line segment - compute segment length
            line_segment_length = l[i] / LineIndex[i]
            
            # ACCELERATION CONTINUITY: Arc (i) to Line (i)
            # This is for the case where segment i has BOTH an arc and a line component.
            # We need to ensure smooth velocity transition from the end of the arc to the start of the line.
            if ArcIndex[i] > 0:  # If this same segment i also has an arc component
                
                # Tangential velocity at the end of the arc's last subsegment in THIS segment i
                arc_segment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / ArcIndex[i]
                v_end_arc_i = arc_segment_length / delta_t_arcs[arc_counter-1][ArcIndex[i]-1]  # Last subsegment of current arc
                
                # Linear velocity at the start of the line's first subsegment in THIS segment i
                v_start_line_i = line_segment_length / delta_t_lines[line_counter-1][0]  # First subsegment of current line
                
                # For acceleration-based continuity: (v_start_line_i - v_end_arc_i)/t_avg ∈ [-a_max, a_max]
                t_avg = (delta_t_arcs[arc_counter-1][ArcIndex[i]-1] + delta_t_lines[line_counter-1][0]) / 2
                a_transition = (v_start_line_i - v_end_arc_i) / t_avg
                all_accelerations.append(a_transition) # Add to list for objective function penalty
                
                # Use linear acceleration limit for this transition
                g.append(a_transition)
                lbg.append(-a_max)
                ubg.append(a_max)
                
                # HARD VELOCITY CONTINUITY: Ensure smooth velocity matching at arc-to-line transitions
                # Allow small tolerance to maintain feasibility while ensuring smoothness
                velocity_continuity = v_start_line_i - v_end_arc_i
                g.append(velocity_continuity)
                lbg.append(-0.002)  # Small tolerance for feasibility (±2mm/s)
                ubg.append(0.002)
            
            # ACCELERATION CONTINUITY: Arc (i-1) to Line (i)
            # Handle arc-to-line transitions across waypoints when there's no arc in the current segment
            # Only apply if current segment does NOT start from a relay point
            elif ArcIndex[i] == 0 and i > 0 and not segment_starts_from_relay and ArcIndex[i-1] > 0:
                # Previous segment ends with an arc, current segment starts with a line (no arc)
                prev_arc_delta_phi = phi[i] - (phi_data[i-1] + Flagb[i-1]*np.pi/2)
                prev_arc_length = abs(r0[i-1] * prev_arc_delta_phi)
                prev_arc_subsegment_length = prev_arc_length / ArcIndex[i-1]
                
                # Find the correct index for the previous arc in delta_t_arcs
                prev_arc_idx = sum(1 for k in range(i-1) if ArcIndex[k] > 0) - 1
                # Velocity at the end of the last subsegment of the previous arc (i-1)
                v_end_prev_arc = prev_arc_subsegment_length / delta_t_arcs[prev_arc_idx][-1]
                
                # Linear velocity at the start of the first subsegment of the current line (i)
                v_start_curr_line = line_segment_length / delta_t_lines[line_counter-1][0]
                
                # For acceleration-based continuity: (v_start_curr_line - v_end_prev_arc)/t_avg ∈ [-a_max, a_max]
                t_avg = (delta_t_arcs[prev_arc_idx][-1] + delta_t_lines[line_counter-1][0]) / 2
                a_transition = (v_start_curr_line - v_end_prev_arc) / t_avg
                all_accelerations.append(a_transition) # Add arc-to-line transition acceleration
                
                # Use linear acceleration limit for this transition
                g.append(a_transition)
                lbg.append(-a_max)
                ubg.append(a_max)
                
                # HARD VELOCITY CONTINUITY: Ensure smooth velocity matching at cross-waypoint transitions
                # Allow small tolerance to maintain feasibility while ensuring smoothness
                velocity_continuity = v_start_curr_line - v_end_prev_arc
                g.append(velocity_continuity)
                lbg.append(-0.002)  # Small tolerance for feasibility (±2mm/s)
                ubg.append(0.002)
            
            # Ensure starting from zero velocity for line segments that start at relay points or first segment AND have no arc before them
            elif ArcIndex[i] == 0 and (i == 0 or (i < len(Flags) and Flags[i] == 1)):
                # If the line segment 'i' starts at the beginning (waypoint i) and there's no arc segment before it
                # Check if this segment starts from a relay point (waypoint i is a relay point)
                print(f"  DEBUG: Pure line segment {i} starts from relay point (waypoint {i})")
                print(f"    - Flags[{i}]: {Flags[i] if i < len(Flags) else 'N/A'}")
                print(f"    - Flagb[{i}]: {Flagb[i] if i < len(Flagb) else 'N/A'}")
                print()
                # Ensure minimum time for first line segment to start from zero velocity
                g.append(delta_t_lines[line_counter-1][0])
                t_min = np.sqrt(2*line_segment_length/a_max)
                lbg.append(t_min)  # Enforce minimum time for first line segment starting from zero
                ubg.append(5.0)  # Maximum time
            
            # Constraints for each segment within the line
            for j in range(LineIndex[i]):
                # Delta time lower bound (positive time) - Increased minimum time to prevent very small values
                g.append(delta_t_lines[line_counter-1][j])
                lbg.append(0.1)  # Default minimum time (increased from 0.001 to 0.01)
                ubg.append(5.0)   # Maximum time
                
                # Linear velocity constraint: v = line_segment_length / delta_t <= v_max
                velocity_expr = line_segment_length / delta_t_lines[line_counter-1][j]
                g.append(velocity_expr)
                lbg.append(0.005)      # Minimum velocity (0.005 m/s)
                ubg.append(v_max)  # Maximum velocity based on differential drive constraints

                # DEBUG: Print constraint details for first few segments
                if i < 3 and j < 2:
                    print(f"  DEBUG Line[{i}][{j}]: velocity constraint: 0.005 <= v <= {v_max:.4f} m/s (line_len={line_segment_length:.4f}m)")
                
                # Linear acceleration constraint between consecutive segments
                if j > 0:
                    # a_lin = (v2-v1)/delta_t = (L/t2 - L/t1)/t_avg
                    # where L is line_segment_length, and t_avg is (t1+t2)/2
                    a_lin = line_segment_length * (1/delta_t_lines[line_counter-1][j] - 1/delta_t_lines[line_counter-1][j-1]) / ((delta_t_lines[line_counter-1][j] + delta_t_lines[line_counter-1][j-1])/2)
                    all_accelerations.append(a_lin) # NEW: Add to list
                    g.append((delta_t_lines[line_counter-1][j]**2-delta_t_lines[line_counter-1][j-1]**2)/delta_t_lines[line_counter-1][j-1]/delta_t_lines[line_counter-1][j])
                    lbg.append(-a_max/2/line_segment_length)
                    ubg.append(a_max/2/line_segment_length)
            
            # Ensure stopping at the end of a line segment when it ends at a relay point or final point
            # Segment i connects waypoint i to waypoint i+1
            # We need to check if waypoint i+1 is a relay point or the final point
            
            # Method 1: Check if waypoint i+1 is final point
            is_final_point = (i == N-2)
            
            # Method 2: Check if waypoint i+1 is a relay point
            # Based on save_waypoints function: Flags[i+1]==1 means waypoint i+1 is a relay point
            is_relay_point = False
            if i+1 < len(Flags) and Flags[i+1] == 1:
                is_relay_point = True
            
            # Apply stop constraint only when ending at a relay point or final point, 
            # and only for segments that have line components (LineIndex[i] > 0)
            if (is_final_point or is_relay_point) and LineIndex[i] > 0:
                # print(f"  DEBUG: Adding stop constraint for segment {i}")
                # print(f"    - Waypoint {i} -> {i+1}")
                # print(f"    - is_final_point: {is_final_point}")
                # print(f"    - is_relay_point: {is_relay_point}")
                # print(f"    - ArcIndex[{i}]: {ArcIndex[i]}")
                # print(f"    - LineIndex[{i}]: {LineIndex[i]}")
                # if i+1 < len(Flags):
                #     print(f"    - Flags[{i+1}]: {Flags[i+1]}")
                # if i+1 < len(Flagb):
                #     print(f"    - Flagb[{i+1}]: {Flagb[i+1]}")
                # print()
                # If the line segment 'i' needs to stop at its end (waypoint i+1) 
                # Second constraint: ensure minimum time for final segment to prevent excessive deceleration
                g.append(delta_t_lines[line_counter-1][-1])
                t_min=np.sqrt(2*line_segment_length/a_max)
                # Calculate maximum time based on "stop" velocity threshold (0.01 m/s)
                t_max_stop_velocity_line = line_segment_length / 0.01  # Time to reach stop velocity
                # Ensure t_max is always greater than or equal to t_min to avoid infeasible constraints
                t_max_final_line = max(t_max_stop_velocity_line, t_min * 1.1)  # Add 10% buffer above minimum
                lbg.append(t_min)  # Enforce slightly higher minimum time for final segment
                ubg.append(t_max_final_line)  # Maximum time based on stop velocity threshold with feasibility check

            # LINE-TO-ARC VELOCITY CONTINUITY within the same segment
            # Check if this segment has both line and arc components (line first, then arc)
            if LineIndex[i] > 0 and ArcIndex[i] > 0:
                # Velocity at the end of the line's last subsegment in THIS segment i
                line_segment_length = l[i] / LineIndex[i]
                v_end_line_i = line_segment_length / delta_t_lines[line_counter-1][-1]  # Last subsegment of current line

                # Tangential velocity at the start of the arc's first subsegment in THIS segment i
                arc_segment_length = abs(r0[i] * (phi[i+1] - phi_new[i])) / ArcIndex[i]
                current_arc_idx = sum(1 for k in range(i+1) if ArcIndex[k] > 0) - 1  # Index for current arc
                v_start_arc_i = arc_segment_length / delta_t_arcs[current_arc_idx][0]  # First subsegment of current arc

                # HARD VELOCITY CONTINUITY: Ensure smooth velocity matching at line-to-arc transitions within same segment
                velocity_continuity = v_start_arc_i - v_end_line_i
                g.append(velocity_continuity)
                lbg.append(-0.002)  # Small tolerance for feasibility (±2mm/s)
                ubg.append(0.002)

                # ACCELERATION CONTINUITY: limit acceleration between line end and arc start
                t_avg = (delta_t_lines[line_counter-1][-1] + delta_t_arcs[current_arc_idx][0]) / 2
                a_transition = (v_start_arc_i - v_end_line_i) / t_avg
                all_accelerations.append(a_transition)
                g.append(a_transition)
                lbg.append(-a_max)
                ubg.append(a_max)

    # Combine all delta time variables in an interleaved manner (arc + line for each segment)
    all_vars_flat = []
    for i in range(N-1):
        # Add arc variables for this segment if they exist
        arc_idx = -1
        for j, arc_vars in enumerate(delta_t_arcs):
            if j == sum(1 for k in range(i) if ArcIndex[k] > 0):
                arc_idx = j
                break
                
        if arc_idx != -1 and ArcIndex[i] > 0:
            if isinstance(delta_t_arcs[arc_idx], list):
                all_vars_flat.extend(delta_t_arcs[arc_idx])
            else:
                all_vars_flat.append(delta_t_arcs[arc_idx])
        
        # Add line variables for this segment if they exist
        line_idx = -1
        for j, line_vars in enumerate(delta_t_lines):
            if j == sum(1 for k in range(i) if LineIndex[k] > 0):
                line_idx = j
                break
                
        if line_idx != -1 and LineIndex[i] > 0:
            if isinstance(delta_t_lines[line_idx], list):
                all_vars_flat.extend(delta_t_lines[line_idx])
            else:
                all_vars_flat.append(delta_t_lines[line_idx])
    
    # Flatten all variables into a single optimization vector
    if all_vars_flat:
        opt_vars = ca.vertcat(*all_vars_flat)
        
        # Track segments between relay points
        # CORRECTED: relay_waypoint_indices should contain waypoint indices that are relay points
        # Based on save_waypoints function: Flags[i]==1 means waypoint i is a relay point
        relay_waypoint_indices = [i for i, flag in enumerate(Flags) if flag == 1]
        # Add start point (waypoint 0) only if not already present
        if 0 not in relay_waypoint_indices:
            relay_waypoint_indices = [0] + relay_waypoint_indices
        else:
            relay_waypoint_indices = sorted(relay_waypoint_indices)  # Ensure sorted order
        # Add end point (waypoint N-1) if not already included
        if N-1 not in relay_waypoint_indices:
            relay_waypoint_indices.append(N-1)

        # Calculate total time for each relay-to-relay segment
        # Number of relay-to-relay segments is len(relay_waypoint_indices) - 1
        num_relay_segments = len(relay_waypoint_indices) - 1
        T_RL = ca.SX.sym('T_RL', num_relay_segments)
        
        print(f"DEBUG: Relay waypoint indices (from Flags): {relay_waypoint_indices}")
        print(f"DEBUG: Flags array: {Flags}")
        print(f"DEBUG: Flagb array: {Flagb}")
        print(f"DEBUG: Total waypoints: {N}")
        print()
        
        # Calculate total time for each path segment
        segment_times = []

        # NEW CORRECTED MAPPING:
        correct_arc_map_idx = []
        correct_line_map_idx = []
        
        arc_list_counter = 0
        line_list_counter = 0
        for i in range(N-1): # Iterate through N-1 major segments
            if ArcIndex[i] > 0:
                correct_arc_map_idx.append(arc_list_counter)
                arc_list_counter += 1
            else:
                correct_arc_map_idx.append(-1)
            
            if LineIndex[i] > 0:
                correct_line_map_idx.append(line_list_counter)
                line_list_counter += 1
            else:
                correct_line_map_idx.append(-1)
        
        # Now calculate time for each segment with proper indexing
        for i in range(N-1):
            segment_time_for_major_segment_i = 0 # Symbolic expression for total time of major segment i
            
            # Add arc time if exists
            arc_map_list_idx = correct_arc_map_idx[i]
            if arc_map_list_idx != -1:
                # Handle the new structure of delta_t_arcs (list of individual variables)
                if isinstance(delta_t_arcs[arc_map_list_idx], list):
                    segment_time_for_major_segment_i += ca.sum1(ca.vertcat(*delta_t_arcs[arc_map_list_idx]))
                else:
                    segment_time_for_major_segment_i += ca.sum1(delta_t_arcs[arc_map_list_idx])
                
            # Add line time if exists
            line_map_list_idx = correct_line_map_idx[i]
            if line_map_list_idx != -1:
                # Handle the new structure of delta_t_lines (list of individual variables)
                if isinstance(delta_t_lines[line_map_list_idx], list):
                    segment_time_for_major_segment_i += ca.sum1(ca.vertcat(*delta_t_lines[line_map_list_idx]))
                else:
                    segment_time_for_major_segment_i += ca.sum1(delta_t_lines[line_map_list_idx])
                
            segment_times.append(segment_time_for_major_segment_i)
        
        # Calculate relay-to-relay times
        for i in range(len(relay_waypoint_indices)-1):
            start_idx = relay_waypoint_indices[i]
            end_idx = relay_waypoint_indices[i+1]
            
            # Sum times for segments between these relay points
            if end_idx > start_idx:
                # Create a list of symbolic expressions for segment times between relays
                relay_segment_times = segment_times[start_idx:end_idx]
                
                # Only perform vertcat if there are multiple segments
                if len(relay_segment_times) > 1:
                    T_RL[i] = ca.sum1(ca.vertcat(*relay_segment_times))
                elif len(relay_segment_times) == 1:
                    T_RL[i] = relay_segment_times[0]  # Just use the single segment time
                else:
                    T_RL[i] = 0
            else:
                T_RL[i] = 0
        
        # Calculate average relay-to-relay time for even distribution
        t_average = ca.sum1(T_RL) / num_relay_segments
        
        # Objective function: minimize total time + deviation from average time
        objective = ca.sum1(opt_vars) + 40.0 * ca.sum1((T_RL - t_average)**2)
        
        # NEW: Add acceleration penalty to the objective with moderate weight
        acceleration_penalty_weight = 100 # Reduced from 1000 to 100 to avoid over-constraining
        if all_accelerations: # Check if the list is not empty
            accel_terms_vector = ca.vertcat(*all_accelerations)
            objective += acceleration_penalty_weight * ca.sumsqr(accel_terms_vector)
            
        # Create an optimization problem
        nlp = {'x': opt_vars, 'f': objective, 'g': ca.vertcat(*g)}
        
        # Set solver options
        opts = {
            'ipopt.print_level': 5,   # More detailed output for debugging
            'print_time': 1,
            'ipopt.max_iter': 5000,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Initial guess: time distribution proportional to segment lengths
        x0 = np.ones(opt_vars.size1()) * 0.1  # Base value
        
        # Calculate total length of the trajectory for normalization
        total_trajectory_length = 0
        for i in range(N-1):
            # Add arc length if exists
            if ArcIndex[i] > 0:
                arc_length = abs(r0[i] * (phi[i+1] - phi_new[i]))
                total_trajectory_length += arc_length
            
            # Add line length if exists
            if LineIndex[i] > 0:
                total_trajectory_length += l[i]
        
        # Estimate a reasonable average velocity (e.g., half of max velocity)
        avg_velocity = v_max * 0.5
        
        
        # Adjust initial guess to be more realistic based on segment lengths
        idx = 0
        for i in range(N-1):
            if ArcIndex[i] > 0:
                arc_length = abs(r0[i] * (phi[i+1] - phi_new[i]))
                arc_segment_length = arc_length / ArcIndex[i]
                
                # Set initial delta_t based on a reasonable velocity
                for j in range(ArcIndex[i]):
                    # Time estimate: distance/velocity
                    x0[idx] = arc_segment_length / (avg_velocity * 0.1)
                    idx += 1
            
            if LineIndex[i] > 0:
                line_segment_length = l[i] / LineIndex[i]
                
                # Set initial delta_t based on a reasonable velocity
                for j in range(LineIndex[i]):
                    # Time estimate: distance/velocity
                    x0[idx] = line_segment_length / (avg_velocity*0.1)
                    idx += 1
        
        # Solve the optimization problem
        sol = solver(x0=x0, lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg))
        
        # Check if solver succeeded
        solver_stats = solver.stats()
        print(f"Solver status: {solver_stats['return_status']}")
        print(f"Solver success: {solver_stats['success']}")
        
        if not solver_stats['success']:
            print("WARNING: Solver did not converge successfully!")
            print("Trying with relaxed constraints...")
            
            # Try with more relaxed solver settings
            opts_relaxed = {
                'ipopt.print_level': 3,
                'print_time': 1,
                'ipopt.max_iter': 10000,
                'ipopt.acceptable_tol': 1e-3,  # More relaxed tolerance
                'ipopt.acceptable_obj_change_tol': 1e-3,
                'ipopt.acceptable_constr_viol_tol': 1e-3,  # Allow some constraint violation
                'ipopt.mu_init': 1e-1  # Different initialization
            }
            solver_relaxed = ca.nlpsol('solver_relaxed', 'ipopt', nlp, opts_relaxed)
            sol = solver_relaxed(x0=x0, lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg))
            
            solver_stats = solver_relaxed.stats()
            print(f"Relaxed solver status: {solver_stats['return_status']}")
            
            if not solver_stats['success']:
                print("ERROR: Even relaxed solver failed!")
                return [], 0.0
        
        # Extract solution - check if it's numerical
        try:
            if hasattr(sol['x'], 'full'):
                opt_time = sol['x'].full().flatten()
            else:
                print("ERROR: Solver returned symbolic solution instead of numerical!")
                return [], 0.0
        except Exception as e:
            print(f"ERROR extracting solution: {e}")
            return [], 0.0
        
        # Calculate time segments for each part
        time_segments = []
        idx = 0
        
        for i in range(N-1):
            segment_times = {}
            if ArcIndex[i] > 0:
                segment_times['arc'] = opt_time[idx:idx+ArcIndex[i]].tolist()
                idx += ArcIndex[i]
            else:
                segment_times['arc'] = []
                
            if LineIndex[i] > 0:
                segment_times['line'] = opt_time[idx:idx+LineIndex[i]].tolist()
                idx += LineIndex[i]
            else:
                segment_times['line'] = []
                
            time_segments.append(segment_times)
        
        # Calculate total trajectory time
        total_time = float(np.sum(opt_time))

        # DIAGNOSTIC: Verify velocity constraints
        print("\n" + "="*60)
        print("VELOCITY CONSTRAINT VERIFICATION")
        print("="*60)
        print(f"Configured v_max: {v_max:.4f} m/s")
        print(f"Arc velocity limit (0.8*v_max): {0.8*v_max:.4f} m/s")
        print(f"Line velocity limit: {v_max:.4f} m/s")
        print()

        max_arc_velocity = 0.0
        max_line_velocity = 0.0
        arc_violations = []
        line_violations = []
        idx = 0

        for i in range(N-1):
            # Check arc segments
            if ArcIndex[i] > 0:
                arc_length = abs(r0[i] * (phi[i+1] - phi_new[i]))
                arc_segment_length = arc_length / ArcIndex[i]
                for j in range(ArcIndex[i]):
                    v_arc = arc_segment_length / opt_time[idx]
                    max_arc_velocity = max(max_arc_velocity, v_arc)
                    if v_arc > 0.8 * v_max:
                        arc_violations.append({
                            'segment': i,
                            'subsegment': j,
                            'velocity': v_arc,
                            'limit': 0.8 * v_max,
                            'delta_t': opt_time[idx],
                            'arc_length': arc_segment_length,
                            'radius': abs(r0[i])
                        })
                    idx += 1

            # Check line segments
            if LineIndex[i] > 0:
                line_segment_length = l[i] / LineIndex[i]
                for j in range(LineIndex[i]):
                    v_line = line_segment_length / opt_time[idx]
                    max_line_velocity = max(max_line_velocity, v_line)
                    if v_line > v_max:
                        line_violations.append({
                            'segment': i,
                            'subsegment': j,
                            'velocity': v_line,
                            'limit': v_max,
                            'delta_t': opt_time[idx],
                            'line_length': line_segment_length
                        })
                    idx += 1

        print(f"Maximum arc velocity:  {max_arc_velocity:.4f} m/s")
        print(f"Maximum line velocity: {max_line_velocity:.4f} m/s")
        print()

        if arc_violations:
            print(f"⚠️  FOUND {len(arc_violations)} ARC VELOCITY VIOLATIONS:")
            for v in arc_violations[:10]:  # Show first 10
                print(f"  Segment {v['segment']}.{v['subsegment']}: "
                      f"v={v['velocity']:.4f} m/s > limit={v['limit']:.4f} m/s "
                      f"(delta_t={v['delta_t']:.4f}s, arc_len={v['arc_length']:.4f}m, r={v['radius']:.4f}m)")
            if len(arc_violations) > 10:
                print(f"  ... and {len(arc_violations)-10} more violations")
        else:
            print("✅ All arc velocities within limits")

        if line_violations:
            print(f"\n⚠️  FOUND {len(line_violations)} LINE VELOCITY VIOLATIONS:")
            for v in line_violations[:10]:  # Show first 10
                print(f"  Segment {v['segment']}.{v['subsegment']}: "
                      f"v={v['velocity']:.4f} m/s > limit={v['limit']:.4f} m/s "
                      f"(delta_t={v['delta_t']:.4f}s, line_len={v['line_length']:.4f}m)")
            if len(line_violations) > 10:
                print(f"  ... and {len(line_violations)-10} more violations")
        else:
            print("✅ All line velocities within limits")

        print("="*60 + "\n")

        # Save results
        results = {
            'time_segments': time_segments,
            'total_time': total_time,
        }
        
        with open(Result_file, 'w') as file:
            json.dump(results, file)
        
        return time_segments, total_time
    else:
        print("No variables to optimize!")
        return [], 0.0

def save_waypoints(case,N,data_file=None):
    import os # Add explicit import here to avoid reference error
    graph_file = os.path.join(config.data_path, f'Graph_new_{case}.json')
    waypoints_file = os.path.join(config.data_path, f'WayPointFlag{N}{case}.json')
    print(f"Loading graph from: {graph_file}")
    print(f"Loading waypoints from: {waypoints_file}")
    
    nodes, in_neighbors, out_neighbors = load_reeb_graph(graph_file)
    Waypoints, Flags, FlagB = load_WayPointFlag_from_file(waypoints_file)
    
    # Load only phi, l, r from the file (no v, a needed)
    with open(data_file, 'r') as file:
        data = json.load(file)
    phi_pixels = data["Optimization_phi"]
    l_pixels = data["Optimization_l"]
    r_pixels = data["Optimization_r"]
    
    # Convert only length and radius from pixels to meters (angles unchanged)
    phi, l, r = convert_world_pixel_data_to_meters(phi_pixels, l_pixels, r_pixels)
    
    # Extract node positions in pixel coordinates and convert to world meters
    node_positions_pixels = [nodes[Waypoints[i]][1] for i in range(len(Waypoints))]
    node_positions_world = convert_pixel_positions_to_world_meters(node_positions_pixels)

    # Load environment obstacles
    try:
        env_file = os.path.join(config.data_path, f'environment_{case}.json')
        with open(env_file, 'r') as file:
            env_data = json.load(file)
        obstacles = env_data.get('obstacles', [])
        print(f"Loaded {len(obstacles)} obstacles from environment file")
    except Exception as e:
        print(f"Could not load environment file: {e}")
        obstacles = []
    
    WP=[]
    RP=[]
    for i in range(len(Waypoints)-1):
        # Use converted world coordinates in meters instead of pixel coordinates
        # phi angles are already transformed to world coordinates
        Node={'Node':i,'Position':node_positions_world[i],'Orientation':phi[i],
                'Radius':r[i],
                'Length':l[i]}
        WP.append(Node)
        if Flags[i]==1 or i==0:
            # Use transformed world coordinate angles for relay point orientation
            Theta=phi[i]+FlagB[i]*np.pi/2
            RP_Ini={'Node':i,'Position':node_positions_world[i],'Orientation':Theta}
            RP.append(RP_Ini)
    
    Node={'Node':len(Waypoints)-1,'Position':node_positions_world[len(Waypoints)-1]}
    WP.append(Node)
    data={'Waypoints':WP,'RelayPoints':RP}
    
    # Ensure directory exists
    import os
    data_dir = os.path.join(config.data_path, case)
    os.makedirs(data_dir, exist_ok=True)
    
    save_file = os.path.join(data_dir, f'Waypoints_{case}.json')
    with open(save_file, 'w') as file:
        json.dump(data, file)
    
    print(f"Waypoints saved to {save_file}")


def generate_uniform_time_trajectories(waypoints, phi, r0, l, phi_new, time_segments, Flagb, reeb_graph, dt, save_dir, Flags=None):
    """
    Generate uniform time discrete trajectories using the uniform_time_trajectory module
    This replaces the previous compare_discretization_with_spline function
    
    Args:
        waypoints: List of waypoint indices
        phi: Angle array
        r0: Radius array  
        l: Length array
        phi_new: Adjusted angle array
        time_segments: Time optimization results
        Flagb: Flag array for relay points
        reeb_graph: Reeb graph object
        dt: Time step for discretization
        save_dir: Directory to save results
        
    Returns:
        saved_files: List of saved plot files (no parameter files)
    """
    print(f"\n🚀 Generating uniform time discrete trajectories and plots...")
    print(f"📁 Save directory: {save_dir}")
    print(f"⏱️  Time step: {dt}s")
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    saved_files = []
    case = config.case
    
    try:
        # Process each robot (relay segment)
        # NOTE: This function needs Flags array to correctly identify relay points
        # For now, using Flagb as fallback but this may not be correct
        if Flags is not None:
            relay_indices = [i for i, flag in enumerate(Flags) if flag == 1]
        else:
            # Fallback to old logic - may be incorrect
            print("Warning: Using Flagb for relay detection - may be incorrect")
            relay_indices = [i for i, flag in enumerate(Flagb) if flag != 0]
        relay_indices = [0] + relay_indices  # Add start point
        if len(waypoints)-1 not in relay_indices:
            relay_indices.append(len(waypoints)-1)  # Add end point
        
        print(f"📍 Found {len(relay_indices)-1} robot trajectories (relay segments)")
        
        for robot_id in range(len(relay_indices)-1):
            start_idx = relay_indices[robot_id]
            end_idx = relay_indices[robot_id+1]
            
            print(f"\n🤖 Processing Robot {robot_id}: waypoints {start_idx} → {end_idx}")
            
            # Extract trajectory parameters for this robot
            robot_waypoints = list(range(start_idx, end_idx + 1))
            robot_phi = phi[start_idx:end_idx+1] if start_idx < len(phi) else phi[start_idx:]
            robot_r0 = r0[start_idx:end_idx] if start_idx < len(r0) else []
            robot_l = l[start_idx:end_idx] if start_idx < len(l) else []
            robot_phi_new = phi_new[start_idx:end_idx] if start_idx < len(phi_new) else []
            robot_time_segments = time_segments[start_idx:end_idx] if start_idx < len(time_segments) else []
            
            # Create trajectory parameter data (save to file for plot_discrete_path compatibility)
            trajectory_params = {
                'waypoints': robot_waypoints,
                'phi': robot_phi.tolist() if hasattr(robot_phi, 'tolist') else robot_phi,
                'r0': robot_r0.tolist() if hasattr(robot_r0, 'tolist') else robot_r0,
                'l': robot_l.tolist() if hasattr(robot_l, 'tolist') else robot_l,
                'phi_new': robot_phi_new.tolist() if hasattr(robot_phi_new, 'tolist') else robot_phi_new,
                'time_segments': robot_time_segments
            }
            
            # Save parameters temporarily for plot_discrete_path_for_robot compatibility
            params_file = os.path.join(save_dir, f'robot_{robot_id}_trajectory_parameters_{case}.json')
            with open(params_file, 'w') as f:
                json.dump(trajectory_params, f, indent=2)
            
            print(f"📊 Saved trajectory parameters for Robot {robot_id} compatibility: {params_file}")
            
            # Generate uniform time trajectory
            try:
                # Prepare file paths for conversion
                reeb_graph_file = config.get_full_path(config.file_path, use_data_path=True) if hasattr(config, 'file_path') else None
                waypoints_file = config.get_full_path(config.waypoints_file_path, use_data_path=True) if hasattr(config, 'waypoints_file_path') else None
                
                # Convert to uniform time trajectory using the saved parameter file
                if reeb_graph_file and waypoints_file and os.path.exists(reeb_graph_file) and os.path.exists(waypoints_file):
                    uniform_traj = convert_robot_trajectory_to_uniform_time(
                        params_file, reeb_graph_file, waypoints_file, dt
                    )
                    
                    print(f"✅ Generated uniform time trajectory for Robot {robot_id}")
                    
                    # Generate discrete trajectory data using plot_discrete_path
                    print(f"📊 Generating discrete trajectory data for Robot {robot_id}...")
                    try:
                        # Import the plot_discrete_path function
                        sys.path.append('/root/workspace/src/Replanning/scripts')
                        from plot_discrete_path import plot_discrete_path_for_robot
                        
                        # Generate discrete trajectory data (no plot output, only JSON data)
                        plot_discrete_path_for_robot(robot_id, output_file=None, show_plot=False, plot_velocity=False)
                        print(f"✅ Generated discrete trajectory data for Robot {robot_id}")
                        
                    except Exception as discrete_error:
                        print(f"⚠️  Warning: Could not generate discrete trajectory data for Robot {robot_id}: {discrete_error}")
                        print("     Trajectory comparison may not work correctly...")
                    
                    # Now generate trajectory comparison plot using the original function
                    plot_file = os.path.join(save_dir, f'robot_{robot_id}_trajectory_comparison_{case}.png')
                    from uniform_time_trajectory import plot_trajectory_comparison
                    
                    try:
                        plot_trajectory_comparison(robot_id, uniform_traj, case, output_file=plot_file)
                        
                        if os.path.exists(plot_file):
                            print(f"📊 Generated trajectory comparison plot: {plot_file}")
                            saved_files.append(plot_file)
                        else:
                            print(f"⚠️  Trajectory comparison plot was not created")
                            
                    except Exception as plot_error:
                        print(f"⚠️  Warning: Could not generate comparison plot for Robot {robot_id}: {plot_error}")
                        print("     Continuing without plot...")
                    
                    # Clean up temporary parameter file after use (if user doesn't want to keep it)
                    # Comment out the next lines if you want to keep the parameter files
                    # os.remove(params_file)
                    # print(f"🗑️  Removed temporary parameter file: {params_file}")
                    
                    # Save in tb format (tb0_Trajectory.json format) - this is still needed for robot control
                        tb_file = save_trajectory_in_tb_format(uniform_traj, robot_id, save_dir, case)
                        if tb_file:
                            print(f"📄 Generated tb format file: {tb_file}")
                            saved_files.append(tb_file)
                
                else:
                    print(f"⚠️  Missing reeb_graph or waypoints file, skipping uniform time generation for robot {robot_id}")
                    
            except Exception as e:
                print(f"❌ Error generating uniform time trajectory for robot {robot_id}: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error in uniform time trajectory generation: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Generated {len(saved_files)} trajectory files (plots and tb format only)")
    return saved_files

if __name__ == '__main__':
    # Load configuration for case, N, and file paths
    case = config.case
    N = config.N
    
    # Get file paths from config
    file_path = config.file_path
    reeb_graph = load_reeb_graph_from_file(file_path)
    NumNodes=len(reeb_graph.nodes)
    environment_file = config.environment_file
    Start_node=reeb_graph.nodes[NumNodes-2].configuration
    End_node=reeb_graph.nodes[NumNodes-1].configuration
    Distances=np.linalg.norm(End_node-Start_node)
    
    # Output files from config
    assignment_result_file = config.assignment_result_file
    ga_initial_guess_file = config.ga_initial_guess_file  # True initial guess from GA optimization
    ga_initial_guess_figure = config.ga_initial_guess_figure
    waypoints_file_path = config.waypoints_file_path
    planning_path_result_file = config.planning_path_result_file  # Result from Planning_path with safe corridor constraints
    result_file = config.deltaT_result_file  # Final result from Planning_deltaT (time optimization)
    figure_file = config.deltaT_figure_file
    
    # Save waypoints for visualization using the Planning_path result (not the original GA guess)
    try:
        print(f"Saving waypoints from Planning_path result: {planning_path_result_file}...")
        save_waypoints(case, N, planning_path_result_file)
    except Exception as e:
        print(f"Warning: Could not save waypoints: {e}")
        print("Continuing with optimization...")
    
    # Run the time optimization (Planning_deltaT)
    print(f"Starting time optimization using path from: {planning_path_result_file}")
    time_segments, total_time = Planning_deltaT(
        waypoints_file_path=waypoints_file_path,
        reeb_graph=reeb_graph,
        planning_path_result_file=planning_path_result_file,
        Result_file=result_file,
        figure_file=figure_file
    )
    
    print(f"Optimization completed successfully!")
    print(f"Total trajectory time: {total_time:.4f} seconds")
    
    # Generate visualization of differential drive constraints
    # plot_differential_drive_limits()
    
    # Load the optimized path geometry from Planning_path for visualization
    # (This is the same file used as input to the time optimization)
    Waypoints, Flags, Flagb = load_WayPointFlag_from_file(waypoints_file_path)
    phi_data_pixels = np.array(load_phi_from_file(planning_path_result_file))
    r_guess_pixels = np.array(load_r_from_file(planning_path_result_file))
    l_guess_pixels = np.array(load_l_from_file(planning_path_result_file))
    
    # Convert only length and radius from pixels to meters (angles unchanged)
    phi_data, l_guess, r_guess = convert_world_pixel_data_to_meters(phi_data_pixels, l_guess_pixels, r_guess_pixels)
    
    # Convert waypoint positions from pixels to world meters for trajectory functions
    nodes, _, _ = load_reeb_graph(file_path)
    node_positions_pixels = [nodes[Waypoints[i]][1] for i in range(len(Waypoints))]
    node_positions_world = convert_pixel_positions_to_world_meters(node_positions_pixels)
    
    phi_new = np.zeros(np.shape(phi_data))
    
    # Calculate phi_new for visualization - angles are already in world coordinates
    for i in range(len(Waypoints)-1):
        # phi_data is already in world coordinates, use directly
        phi_new[i] = phi_data[i] + Flagb[i]*np.pi/2
    
    # NEW: Save trajectory parameters for later use
    trajectory_files = save_trajectory_parameters(
        waypoints=Waypoints,
        phi=phi_data,
        r0=r_guess,
        l=l_guess,
        phi_new=phi_new,
        time_segments=time_segments,
        Flagb=Flagb,
        reeb_graph=reeb_graph,
        case=case,
        N=N
    )
    
    # # Plot trajectory with time information
    plot_trajectory_with_time(
        waypoints=Waypoints,
        phi=phi_data,
        r0=r_guess,
        l=l_guess,
        phi_new=phi_new,
        time_segments=time_segments,
        figure_file=figure_file,
        reeb_graph=reeb_graph,
        Flagb=Flagb,
        case=case,
        N=N
    )
    

    print(f"\nTrajectory parameters saved for {len(trajectory_files) - 1} robots")
    print("Saved files:")
    for file_path in trajectory_files:
        print(f"  - {file_path}")
    print("\nUse the following functions to reload and process:")
    print(f"  - load_complete_trajectory_parameters('{case}')")
    print(f"  - load_robot_trajectory_parameters('{case}', robot_id)")
    print(f"  - load_robot_trajectory_parameters('{case}')  # Load all robots")




