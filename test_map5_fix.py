"""Test script to verify MAP5 robot count fix"""
import os
import sys

BASE_PATH = "result/MAPS"
case_name = "MAP5"
case_lower = case_name.lower()

# Robot count limit (the fix)
ROBOT_COUNT = {
    "MAP5": 4  # Only 4 robots for MAP5
}

max_robots = ROBOT_COUNT.get(case_name, 10)

print(f"Testing MAP5 robot data loading...")
print(f"Max robots allowed: {max_robots}")
print()

for robot_id in range(max_robots + 2):  # Test up to 6 to see which files exist
    # OLD PATH (wrong - would find MAP5/MAP5/map5/robot_X.json)
    old_path = os.path.join(BASE_PATH, case_name, case_name, case_lower,
                           f"robot_{robot_id}_trajectory_parameters_{case_lower}.json")

    # NEW PATH (correct - finds MAP5/map5/map5/robot_X.json)
    new_path = os.path.join(BASE_PATH, case_name, case_lower, case_lower,
                           f"robot_{robot_id}_trajectory_parameters_{case_lower}.json")

    old_exists = os.path.exists(old_path)
    new_exists = os.path.exists(new_path)

    load_decision = "LOAD" if robot_id < max_robots and new_exists else "SKIP"

    print(f"Robot {robot_id}:")
    print(f"  Old path exists: {old_exists} -> {old_path}")
    print(f"  New path exists: {new_exists} -> {new_path}")
    print(f"  Decision: {load_decision}")
    print()

print("Summary:")
print(f"With ROBOT_COUNT limit = {max_robots}, only robots 0-{max_robots-1} will be loaded")
print(f"Robot 4 (MAP5_2 data) will be SKIPPED")
