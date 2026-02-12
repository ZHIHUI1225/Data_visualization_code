"""Debug script to test single map with full error output"""
import sys
import traceback
from pathlib import Path

# Import the RRT Dubins planner
from rrt_dubins_planner import process_single_map, Config

def test_single_map():
    """Test LOOP1 map with full error reporting"""
    
    config_path = r"d:\Data_visualization_code\result\romi\config.yaml"
    if Path(config_path).exists():
        Config.load_from_yaml(config_path)
        print(f"Config loaded from: {config_path}")
    
    print(f"Dubins min radius: {Config.DUBINS_MIN_RADIUS}")
    print(f"RRT max iterations: {Config.RRT_MAX_ITER}")
    print(f"RRT step size: {Config.RRT_STEP_SIZE}")
    print(f"Goal threshold: {Config.RRT_GOAL_THRESHOLD}")
    print()
    
    env_path = r"d:\Data_visualization_code\result\MAPS\LOOP1\environment_LOOP1.json"
    output_dir = r"d:\Data_visualization_code\result\RRT_Dubins_Results\LOOP1_debug"
    
    try:
        success = process_single_map(env_path, output_dir, verbose=True)
        print(f"\n\n{'='*60}")
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        print(f"{'='*60}")
        return success
    except Exception as e:
        print(f"\n\n{'='*60}")
        print(f"EXCEPTION CAUGHT:")
        print(f"{'='*60}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_single_map()
