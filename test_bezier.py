import json
from pathlib import Path

# 测试 Bezier 文件读取
bezier_file = Path('result') / 'MAPS' / 'MAP1' / 'bezier_results' / 'bezier_curves_summary.json'

print(f"Looking for file: {bezier_file}")
print(f"File exists: {bezier_file.exists()}")
print(f"Absolute path: {bezier_file.absolute()}")

if bezier_file.exists():
    with open(bezier_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\nFile loaded successfully")
    print(f"Map name: {data.get('map_name')}")
    print(f"Total sequences: {data.get('total_sequences')}")

    sequences = data.get('sequences', [])
    print(f"Sequences found: {len(sequences)}")

    if sequences:
        first_sequence = sequences[0]
        total_distance = first_sequence.get('total_distance', 'NOT FOUND')
        max_curvature = first_sequence.get('max_curvature', 'NOT FOUND')

        print(f"\nFirst sequence:")
        print(f"  Total distance: {total_distance}")
        print(f"  Max curvature: {max_curvature}")

        if isinstance(total_distance, (int, float)) and isinstance(max_curvature, (int, float)):
            min_radius = 1.0 / max_curvature if max_curvature > 0 else float('inf')
            print(f"  Min radius: {min_radius}")
else:
    print("\nFile not found!")