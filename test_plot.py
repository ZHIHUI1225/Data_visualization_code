import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon

try:
    # Load environment data
    json_path = r'd:\Data_visualization_code\result\environment_concept.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['polygons'])} polygons")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot each obstacle polygon
    for i, poly_data in enumerate(data['polygons']):
        vertices = poly_data['vertices']
        polygon = MPLPolygon(vertices,
                            facecolor='gray',
                            edgecolor='black',
                            linewidth=1.5,
                            alpha=0.7)
        ax.add_patch(polygon)
        print(f"Added polygon {i+1}")

    # Set bounds
    bounds = data['coord_bounds']
    ax.set_xlim(bounds[0] - 10, bounds[1] + 10)
    ax.set_ylim(bounds[2] - 10, bounds[3] + 10)

    # Configure plot
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X [units]', fontsize=12)
    ax.set_ylabel('Y [units]', fontsize=12)
    ax.set_title('Environment: Concept Case', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = r'd:\Data_visualization_code\result\environment_concept_plot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"SUCCESS: Plot saved to {output_path}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
