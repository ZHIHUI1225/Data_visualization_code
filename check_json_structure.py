import json

# Check RRT structure
with open('result/RRT_Results/MAP1/bezier_trajectory_MAP1.json', 'r') as f:
    rrt_data = json.load(f)

print("RRT JSON top-level keys:")
for key in rrt_data.keys():
    print(f"  - {key}: {type(rrt_data[key]).__name__}")

# Check if metrics are nested
if 'total_distance' in rrt_data:
    print(f"\nDirect access - total_distance: {rrt_data['total_distance']}")
else:
    print("\n'total_distance' not at top level")
    # Search in nested structures
    for key, value in rrt_data.items():
        if isinstance(value, dict) and 'total_distance' in value:
            print(f"Found in '{key}':")
            print(f"  total_distance: {value.get('total_distance')}")
            print(f"  max_curvature: {value.get('max_curvature')}")
            print(f"  total_time: {value.get('total_time')}")