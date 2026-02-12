"""Quick test of batch visualization for a single case"""
import sys
sys.path.insert(0, 'd:/Data_visualization_code')

from batch_velocity_visualization import load_trajectory_data, plot_case_velocities

# Test with MAP1
print("Testing MAP1...")
data = load_trajectory_data("MAP1")

if data:
    print(f"Loaded {len(data)} robots")
    for robot_id, robot_info in data.items():
        print(f"  Robot {robot_id}: {len(robot_info['waypoint_positions'])} waypoints")

    print("\nGenerating plot...")
    try:
        plot_case_velocities("MAP1", data)
        print("✅ Success!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Failed to load data")
